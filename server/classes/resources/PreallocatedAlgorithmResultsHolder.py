from __future__ import annotations

from typing import Callable

from classes.config import AlgorithmConfig
from classes.marker import CompositeDetection
from classes.robot import RobotDetection
import numpy as np
from utils.adaptive_buffer_allocator import (
    AdaptiveBufferAllocator,
    AdaptiveBufferAllocatorConfig,
    BufferSlot,
)

class PreallocatedAlgorithmResultsHolder:
    # stitching results
    stitchedImg: np.ndarray | None = None
    stitchedImgAnnotated: np.ndarray | None = None

    # top-down view results
    topDownImg: np.ndarray | None = None
    topDownImgAnnotated: np.ndarray | None = None

    # detections
    topDownDetections: list[CompositeDetection]
    robotDetections: list[RobotDetection]

    # map segmentation
    mapSegmentationRedStopLines: np.ndarray | None = None
    mapSegmentationYellowLines: np.ndarray | None = None
    mapSegmentationRoadComponents: np.ndarray | None = None

    # statistics
    fps: float
    topDownCacheHits: int
    topDownCacheMisses: int
    recomputeReasons: dict[str, int]

    # marker visibility telemetry (per-camera + fused view counts)
    visibilityTelemetry: dict | None

    def __init__(
        self,
        config: AlgorithmConfig,
        on_buffer_reallocate: Callable[[str, str, tuple[int, int], tuple[int, int]], None] | None = None,
    ) -> None:
        self.fps = float("nan")
        self.topDownDetections = []
        self.robotDetections = []
        self.topDownCacheHits = 0
        self.topDownCacheMisses = 0
        self.recomputeReasons = {}
        self.visibilityTelemetry = None

        self._adaptiveBufferPreallocation = config.preallocation.adaptiveBufferPreallocation
        alloc_config = AdaptiveBufferAllocatorConfig()
        self._topDownImgAllocator = AdaptiveBufferAllocator(
            allocator_id="topDownImg",
            dtype=np.uint8,
            config=alloc_config,
            on_reallocate=on_buffer_reallocate,
        )
        self._topDownImgAnnotatedAllocator = AdaptiveBufferAllocator(
            allocator_id="topDownImgAnnotated",
            dtype=np.uint8,
            config=alloc_config,
            on_reallocate=on_buffer_reallocate,
        )

    def ensure_top_down_buffer(self, height: int, width: int) -> BufferSlot | None:
        """Get a preallocated buffer for top-down image output. Caller writes into slot.buffer. Returns None if disabled."""
        if not self._adaptiveBufferPreallocation:
            return None
        return self._topDownImgAllocator.ensure_capacity_for(height, width, channels=3)

    def ensure_top_down_annotated_buffer(self, height: int, width: int) -> BufferSlot | None:
        """Get a preallocated buffer for top-down annotated image (same size as topDownImg). Returns None if disabled."""
        if not self._adaptiveBufferPreallocation:
            return None
        return self._topDownImgAnnotatedAllocator.ensure_capacity_for(height, width, channels=3)
