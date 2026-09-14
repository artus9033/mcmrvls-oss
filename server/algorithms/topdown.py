"""
Top-down map generation strategies.

Strategy pattern behind the ``topDownStrategy`` config option:

* ``mosaic`` (:class:`MosaicTopDownStrategy`) - the historical path: warp the
  hierarchically-stitched mosaic into the top-down frame via the corner-tag
  homography ``M`` (piecewise-projective residuals included).
* ``orthorectified`` (:class:`OrthorectifiedTopDownStrategy`) - warp each RAW
  camera directly into the map frame through its exact floor-atlas homography
  and composite there. No piecewise distortion, boundary rectangular by
  construction; robot poses come from the per-camera consensus (the mosaic
  keeps serving the stitched preview and acts as fallback until the atlas
  solves).

Caching mirrors the ``topDownHomographyCache`` semantics: per-camera warp
matrices and blend weights are cached per atlas solve generation and rebuilt
only when the atlas re-solves (or every round when caching is disabled).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING

import cv2
import numpy as np

from algorithms.stitching import topDownWarp
from classes.marker import MarkerDetection
from classes.marker.CompositeDetection import CompositeDetection
from utils.plotting import draw_coasted_robot_pose_on_topdown
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.config import AlgorithmConfig
    from classes.resources.AlgorithmState import AlgorithmState
    from classes.resources.ImagePack import ImagePack
    from classes.resources.InterThreadMemory import InterThreadMemory
    from classes.resources.PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder

ORTHO_OUTPUT_HEIGHT_PX = 800


class TopDownStrategy(ABC):
    """Produces the top-down map image, detections and robot poses for a round."""

    @abstractmethod
    def process(
        self,
        stitchingImagePack: "ImagePack",
        raw_camera_packs: list["ImagePack"],
        state: "AlgorithmState",
        config: "AlgorithmConfig",
        logger: Logger,
        resultsHolder: "PreallocatedAlgorithmResultsHolder",
        memory: "InterThreadMemory | None" = None,
    ) -> None: ...


class MosaicTopDownStrategy(TopDownStrategy):
    """Historical stitched-mosaic path (delegates to topDownWarp)."""

    def process(
        self,
        stitchingImagePack,
        raw_camera_packs,
        state,
        config,
        logger,
        resultsHolder,
        memory=None,
        skipRender: bool = False,
        skipMeasurements: bool = False,
    ) -> None:
        topDownWarp(
            stitchingImagePack=stitchingImagePack,
            state=state,
            config=config,
            logger=logger,
            resultsHolder=resultsHolder,
            memory=memory,
            skipRender=skipRender,
            skipMeasurements=skipMeasurements,
        )


class OrthorectifiedTopDownStrategy(MosaicTopDownStrategy):
    """Atlas-orthorectified fusion: raw cameras composited in the map frame.

    Runs the mosaic path first (stitched preview, Kalman flow, fallback), then
    overrides the top-down outputs with the ortho composite once the floor
    atlas is solved.
    """

    def __init__(self) -> None:
        self._cacheGeneration: int | None = None
        self._cacheKeySizes: tuple | None = None
        # per-camera: (warp matrix map-normalized-scaled, float32 weight mask)
        self._warpMatrices: dict[str, np.ndarray] = {}
        self._weightMasks: dict[str, np.ndarray] = {}
        self._weightSum: np.ndarray | None = None

    def process(
        self,
        stitchingImagePack,
        raw_camera_packs,
        state,
        config,
        logger,
        resultsHolder,
        memory=None,
    ) -> None:
        atlas = getattr(state, "floorAtlas", None)
        usablePacks = (
            [
                pack
                for pack in raw_camera_packs
                if pack.image is not None and str(pack.stitchingStageDescription) in atlas.camera_h
            ]
            if atlas is not None and atlas.ready
            else []
        )
        # when the ortho composite will certainly override the top-down outputs,
        # the mosaic path can skip rendering them; with epipolar off, the whole
        # mosaic measurement machinery (corner extrapolation, M, gates) is
        # bypassed too - the mosaic then serves purely as a preview, and corner
        # positions live exclusively in the ortho/map frame
        willComposite = bool(usablePacks)
        bypassMosaicMeasurements = willComposite and not config.useEpipolarGeometry

        super().process(
            stitchingImagePack,
            raw_camera_packs,
            state,
            config,
            logger,
            resultsHolder,
            memory,
            skipRender=willComposite,
            skipMeasurements=bypassMosaicMeasurements,
        )

        if not willComposite:
            return  # mosaic outputs stand until the atlas solves

        with Tracing.ScopedZone("orthorectifiedTopDown"):
            outH = ORTHO_OUTPUT_HEIGHT_PX
            outW = int(round(outH * config.MAP_RATIO_W_TO_H))

            self._ensureWarpCache(atlas, usablePacks, outW, outH, config)
            if self._weightSum is None or not self._warpMatrices:
                return

            accumulator = np.zeros((outH, outW, 3), dtype=np.float32)
            for pack in usablePacks:
                camera = str(pack.stitchingStageDescription)
                matrix = self._warpMatrices.get(camera)
                mask = self._weightMasks.get(camera)
                if matrix is None or mask is None:
                    continue
                warped = cv2.warpPerspective(
                    pack.image,
                    matrix,
                    (outW, outH),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                accumulator += warped.astype(np.float32) * mask[..., None]

            denominator = np.maximum(self._weightSum, 1e-6)
            orthoImg = (accumulator / denominator[..., None]).astype(np.uint8)

            # synthesized detections at atlas-known positions (static tags only;
            # robots are carried by resultsHolder.robotDetections)
            scale = np.array([outW, outH], dtype=np.float64)
            orthoDetections: list[CompositeDetection] = []
            robotIds = set(int(r) for r in config.configuredRobotIDs)
            for tagId, quadNorm in atlas.atlas_quads_norm.items():
                if int(tagId) in robotIds:
                    continue
                quadPx = np.asarray(quadNorm, dtype=np.float64) * scale
                marker = MarkerDetection(
                    *[(int(p[0]), int(p[1])) for p in quadPx],
                    data=tagId,
                )
                orthoDetections.append(CompositeDetection(tagId, [marker]))

            orthoAnnotated = orthoImg.copy()
            for detection in orthoDetections:
                topLeft = detection.aggregatedTopLeft
                cv2.putText(
                    orthoAnnotated,
                    str(detection.data),
                    (int(topLeft[0]), int(topLeft[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            for det in resultsHolder.robotDetections:
                draw_coasted_robot_pose_on_topdown(
                    orthoAnnotated,
                    rel_x=det.x,
                    rel_y=det.y,
                    azimuth_deg=det.azimuth,
                    label=f"{det.robot.host} ({det.robot.id})",
                    config=config,
                    color_bgr=(0, 255, 0) if det.pose_source == "detection" else (0, 200, 255),
                )

            resultsHolder.topDownImg = orthoImg
            resultsHolder.topDownImgAnnotated = orthoAnnotated
            resultsHolder.topDownDetections = orthoDetections

    def _ensureWarpCache(
        self,
        atlas,
        usablePacks: list["ImagePack"],
        outW: int,
        outH: int,
        config,
    ) -> None:
        """(Re)build per-camera warp matrices and blend weights.

        Keyed by the atlas solve generation and camera image sizes - the same
        freshness policy the mosaic M cache follows; with topDownHomographyCache
        disabled the cache is rebuilt every round.
        """
        sizesKey = tuple(sorted((str(p.stitchingStageDescription), p.image.shape[:2]) for p in usablePacks))
        generation = getattr(atlas, "solve_generation", None)
        cacheEnabled = bool(config.caching.topDownHomographyCache)
        if (
            cacheEnabled
            and self._cacheGeneration == generation
            and self._cacheKeySizes == sizesKey
            and self._warpMatrices
        ):
            return

        scaleMatrix = np.array(
            [[outW, 0, 0], [0, outH, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        self._warpMatrices = {}
        rawMasks: dict[str, np.ndarray] = {}
        for pack in usablePacks:
            camera = str(pack.stitchingStageDescription)
            hCam = atlas.camera_h.get(camera)
            if hCam is None:
                continue
            matrix = scaleMatrix @ np.asarray(hCam, dtype=np.float64)
            # weight favors the camera closest to each floor point: warp a
            # per-pixel proximity prior (max at image center, falling toward
            # the edges) - a cheap proxy for viewing angle/resolution
            h, w = pack.image.shape[:2]
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            prior = 1.0 - 0.8 * np.maximum(np.abs(xs / (w - 1) - 0.5), np.abs(ys / (h - 1) - 0.5)) * 2.0
            mask = cv2.warpPerspective(
                prior,
                matrix,
                (outW, outH),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            self._warpMatrices[camera] = matrix
            rawMasks[camera] = mask

        # Feathered blending: the proximity priors act as smooth per-camera
        # weights, so each region is dominated by (but not exclusively taken
        # from) its best-placed camera, with soft transitions at overlaps.
        # Note: any object above the floor plane (e.g. the robot) still shows
        # parallax-displaced ghost copies where cameras overlap - inherent to
        # blended multi-view composites.
        self._weightMasks = rawMasks
        self._weightSum = (
            np.sum(np.stack(list(rawMasks.values())), axis=0)
            if rawMasks
            else np.zeros((outH, outW), dtype=np.float32)
        )
        self._cacheGeneration = generation
        self._cacheKeySizes = sizesKey


def create_topdown_strategy(config: "AlgorithmConfig") -> TopDownStrategy:
    if getattr(config, "topDownStrategy", "mosaic") == "orthorectified":
        return OrthorectifiedTopDownStrategy()
    return MosaicTopDownStrategy()
