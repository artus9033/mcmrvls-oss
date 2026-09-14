"""UNet-based map segmentation strategy.

Inference notes (performance):
- model is loaded once at configure time, ``eval()``-ed, moved to the fastest
  available device (cuda > mps > cpu) and warmed up with a dummy forward pass
  so the first live frame does not pay lazy-init/autotune costs;
- all forwards run under ``torch.inference_mode()``;
- fp16 weights on CUDA (fp32 elsewhere: BatchNorm on MPS/CPU is only reliably
  fast and accurate in fp32); cuDNN benchmark autotuning is enabled on CUDA;
- pre/post-processing stays in OpenCV (bilinear down to ``inputSize``, nearest
  back up), avoiding PIL round-trips; the input tensor buffer is reused.
"""

from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.tracing import Tracing

from .base import MapSegmentationStrategy, SegmentationMasks

SERVER_DIR = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = "res/models/unet_multihead.pth"
DEFAULT_INPUT_SIZE = 512


class UNetMapSegmentationStrategy(MapSegmentationStrategy):
    name = "unet"

    def __init__(self, params: dict[str, Any] | None = None, logger: Logger | None = None) -> None:
        import torch  # deferred: torch is only required when this strategy is active

        from .unet_model import UNet, UNetMultiHead

        self._torch = torch
        params = params or {}
        self._logger = logger

        model_path = Path(str(params.get("modelPath", DEFAULT_MODEL_PATH)))
        if not model_path.is_absolute():
            model_path = SERVER_DIR / model_path
        if not model_path.exists():
            raise FileNotFoundError(f"UNet model checkpoint not found: {model_path}")

        self._architecture = str(params.get("architecture", "multihead"))
        if self._architecture not in ("multihead", "singlehead"):
            raise ValueError(f"Unknown UNet architecture: {self._architecture}")

        self._input_size = int(params.get("inputSize", DEFAULT_INPUT_SIZE))
        self._mask_markings_to_roads = bool(params.get("maskMarkingsToRoads", True))

        device = str(params.get("device", "auto"))
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        half = params.get("halfPrecision", "auto")
        self._use_half = (self._device.type == "cuda") if half == "auto" else bool(half)

        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        model = UNetMultiHead() if self._architecture == "multihead" else UNet(n_classes=4)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        if self._use_half:
            model.half()
        self._model = model.to(self._device)

        self._input_buffer: np.ndarray | None = None

        if logger is not None:
            logger.info(
                f"UNet map segmentation ready: {self._architecture}, {model_path.name}, "
                f"device={self._device.type}, half={self._use_half}, input={self._input_size}px"
            )

        self.warmup((self._input_size, self._input_size))

    def warmup(self, image_shape: tuple[int, int]) -> None:
        torch = self._torch
        dtype = torch.float16 if self._use_half else torch.float32
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, self._input_size, self._input_size, dtype=dtype, device=self._device)
            self._model(dummy)
            if self._device.type in ("cuda", "mps"):
                torch.cuda.synchronize() if self._device.type == "cuda" else torch.mps.synchronize()

    def _preprocess(self, imageBGR: np.ndarray) -> "Any":
        torch = self._torch
        size = self._input_size
        resized = cv2.resize(imageBGR, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # reuse the float buffer between rounds; matches training: to_tensor => RGB / 255
        if self._input_buffer is None or self._input_buffer.shape[:2] != (size, size):
            self._input_buffer = np.empty((size, size, 3), dtype=np.float32)
        np.multiply(rgb, np.float32(1.0 / 255.0), out=self._input_buffer, casting="unsafe")
        tensor = torch.from_numpy(self._input_buffer).permute(2, 0, 1).unsqueeze(0)
        if self._use_half:
            tensor = tensor.half()
        return tensor.to(self._device, non_blocking=True)

    def segment(
        self,
        imageBGR: np.ndarray,
        robotMask: np.ndarray | None = None,
    ) -> SegmentationMasks | None:
        torch = self._torch
        height, width = imageBGR.shape[:2]

        with Tracing.ScopedZone("UNetMapSegmentation.segment"):
            with torch.inference_mode():
                batch = self._preprocess(imageBGR)

                if self._architecture == "multihead":
                    road_logits, marking_logits = self._model(batch)
                    road_pred = road_logits.argmax(dim=1)
                    marking_pred = marking_logits.argmax(dim=1)
                    if self._mask_markings_to_roads:
                        # markings are only meaningful on road pixels
                        marking_pred = marking_pred * road_pred
                    road_np = road_pred[0].to("cpu", torch.uint8).numpy()
                    marking_np = marking_pred[0].to("cpu", torch.uint8).numpy()
                    roads = road_np
                    red = (marking_np == 1).astype(np.uint8)
                    yellow = (marking_np == 2).astype(np.uint8)
                else:
                    # single-head classes: 0=background, 1=red, 2=yellow, 3=roads.
                    # Markings physically lie on roads, so they count as navigable.
                    pred = self._model(batch).argmax(dim=1)[0].to("cpu", torch.uint8).numpy()
                    roads = (pred > 0).astype(np.uint8)
                    red = (pred == 1).astype(np.uint8)
                    yellow = (pred == 2).astype(np.uint8)

            # scale back to the top-down image resolution (nearest keeps masks crisp)
            interp = cv2.INTER_NEAREST
            roads = cv2.resize(roads, (width, height), interpolation=interp) * 255
            red = cv2.resize(red, (width, height), interpolation=interp) * 255
            yellow = cv2.resize(yellow, (width, height), interpolation=interp) * 255

            if robotMask is not None:
                roads = cv2.bitwise_or(roads, robotMask)

            return SegmentationMasks(
                roadComponents=roads,
                redStopLines=red,
                yellowLines=yellow,
            )
