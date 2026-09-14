"""Classical (threshold + morphology) map segmentation strategy.

Thresholds obtained by offline optimization against labeled top-down maps.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.geometry import circle
from utils.tracing import Tracing

from .base import MapSegmentationStrategy, SegmentationMasks

THRESHOLDS = {
    "road": ("Y", [1.0, 153.0]),
    "stop": ("A", [153.0, 204.0]),
    "sep": ("B", [149.0, 207.0]),
}


def applyThresholds(
    channels: dict[str, np.ndarray],
    thresholds: tuple[str, list[float]],
) -> np.ndarray:
    channel_name, (thresh_min, thresh_max) = thresholds
    channel = channels[channel_name]

    M = (channel >= thresh_min) & (channel <= thresh_max)

    return M


def postProcessRoads(MRoad: np.ndarray, K: np.ndarray):
    MRoad = MRoad.astype(np.uint8)

    _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(MRoad.astype(np.uint8), connectivity=8)

    if len(stats) < 2:
        return

    # get the largest component (exclude the background)
    largest_label = (
        # skip the background (label 0)
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    )

    largest_component_mask = np.zeros_like(MRoad, dtype=np.bool_)
    largest_component_mask[labels == largest_label] = 1

    MRoad = MRoad & largest_component_mask

    MRoad = cv2.morphologyEx(MRoad, cv2.MORPH_CLOSE, K, iterations=4)

    return MRoad * 255


def postProcessStop(MStop: np.ndarray, MRoad: np.ndarray, K: np.ndarray):
    MStop = MStop & MRoad

    # post-processing of red lines: remove noise, fill full rectangles
    MStop = cv2.morphologyEx(MStop.astype(np.uint8), cv2.MORPH_OPEN, K)

    # find contours
    contours, _ = cv2.findContours(MStop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MStop = np.zeros_like(MStop)

    # approximate all contours with bounding boxes to fill rectangles (as expected)
    for contour in contours:
        # get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # draw the rectangle on the new mask
        cv2.rectangle(MStop, (x, y), (x + w, y + h), (255,), thickness=-1)

    return MStop


def postProcessSep(
    MSep: np.ndarray,
    MRoad: np.ndarray,
    MStop: np.ndarray,
    KSize: tuple[int, int],
    KLarge: np.ndarray,
    KDenoise: np.ndarray,
):
    KSepMorphOpen = np.ones((3, 3))
    MSep = MSep & MRoad & (~MStop)
    MSep = cv2.morphologyEx(MSep.astype(np.uint8), cv2.MORPH_OPEN, KSepMorphOpen)
    MSep = cv2.morphologyEx(MSep.astype(np.uint8), cv2.MORPH_OPEN, KSepMorphOpen)

    KDiagonalR = np.eye(KSize[0], KSize[1], dtype=np.uint8)
    KDiagonalL = np.fliplr(KDiagonalR)

    MSep = cv2.morphologyEx(
        MSep,
        cv2.MORPH_CLOSE,
        KLarge,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    ).astype(np.bool_)

    diagonalEmphasisIters = 2
    MSep = cv2.morphologyEx(
        MSep.astype(np.uint8),
        cv2.MORPH_DILATE,
        KDiagonalR,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
        iterations=diagonalEmphasisIters,
    ).astype(np.bool_)
    MSep = cv2.morphologyEx(
        MSep.astype(np.uint8),
        cv2.MORPH_DILATE,
        KDiagonalL,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
        iterations=diagonalEmphasisIters,
    ).astype(np.bool_)

    MSep = (
        cv2.morphologyEx(
            MSep.astype(np.uint8),
            cv2.MORPH_ERODE,
            KDenoise,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        * 255
    )

    return MSep


class ClassicalMapSegmentationStrategy(MapSegmentationStrategy):
    """YCrCb/LAB channel thresholding followed by morphological post-processing."""

    name = "classical"

    def __init__(self, params: dict | None = None) -> None:
        # precompute kernels for morphological operations
        self._KSizeSmall = (7, 7)
        self._KSizeLarge = (13, 13)
        self._K = np.ones(self._KSizeSmall, np.uint8)
        self._KLarge = np.ones(self._KSizeLarge, np.uint8)
        self._KDenoise = circle(np.min(self._KSizeSmall))

    def segment(
        self,
        imageBGR: np.ndarray,
        robotMask: np.ndarray | None = None,
    ) -> SegmentationMasks | None:
        with Tracing.ScopedZone("ClassicalMapSegmentation.segment"):
            imgYCrCb = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2YCrCb)
            imgLAB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2LAB)

            [Y, Cr, Cb] = imgYCrCb[:, :, 0], imgYCrCb[:, :, 1], imgYCrCb[:, :, 2]
            Y = cv2.equalizeHist(Y)

            [L, A, B] = imgLAB[:, :, 0], imgLAB[:, :, 1], imgLAB[:, :, 2]

            channels = {
                "Y": Y,
                "Cr": Cr,
                "Cb": Cb,
                "L": L,
                "A": A,
                "B": B,
            }

            # 'raw' thresholding
            MStop = applyThresholds(channels, THRESHOLDS["stop"])
            MSep = applyThresholds(channels, THRESHOLDS["sep"])
            MRoad = applyThresholds(channels, THRESHOLDS["road"])

            # post-processing of road components
            MRoad = postProcessRoads(MRoad, K=self._K)

            if MRoad is None:
                return None

            # Robot positions are navigable road, not obstacles
            if robotMask is not None:
                MRoad = cv2.bitwise_or(MRoad, robotMask)

            # post-processing of red stop lines
            MStop = postProcessStop(MStop, MRoad, self._K)

            # post-processing of yellow lane separator lines
            MSep = postProcessSep(
                MSep=MSep,
                MRoad=MRoad,
                MStop=MStop,
                KLarge=self._KLarge,
                KDenoise=self._KDenoise,
                KSize=self._KSizeSmall,
            )

            return SegmentationMasks(
                roadComponents=MRoad,
                redStopLines=MStop,
                yellowLines=MSep,
            )
