import itertools
from typing import Iterable

from classes.marker.CompositeDetection import CompositeDetection
from classes.marker.MarkerDetection import MarkerDetection
from classes.types import FloatingPoint2D
import cv2
from errors import ProcessingError
import numpy as np
from utils.tracing import Tracing

def detectionToPointsList(
    detection: MarkerDetection | CompositeDetection,
) -> list[FloatingPoint2D]:
    with Tracing.ScopedZone("detectionToPointsList"):
        if isinstance(detection, MarkerDetection):
            return detection.toPointsList()
        else:
            return detection.toAggregatedPointsList()


def calculateHomography(
    detectionsA: Iterable[MarkerDetection | CompositeDetection] | np.ndarray,
    detectionsB: Iterable[MarkerDetection | CompositeDetection] | np.ndarray,
):
    """Calculates the homography matrix H_AB."""
    with Tracing.ScopedZone("calculateHomography"):
        # extract points from markers
        if isinstance(detectionsA, np.ndarray):
            pointsA = detectionsA
        else:
            pointsA = np.array(list(itertools.chain(*[detectionToPointsList(detection) for detection in detectionsA])))

        if isinstance(detectionsB, np.ndarray):
            pointsB = detectionsB
        else:
            pointsB = np.array(list(itertools.chain(*[detectionToPointsList(detection) for detection in detectionsB])))

        # calculate homography matrix
        H, _status = cv2.findHomography(pointsA, pointsB, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        if H is None:
            raise ProcessingError(
                "Homography estimation failed: cv2.findHomography returned None",
                code="HOMOGRAPHY_NONE",
            )

        # ensure normalized homography for better computations
        scale = H[2, 2]
        if scale == 0:
            raise ProcessingError(
                "Homography estimation failed: invalid scale factor (0)",
                code="HOMOGRAPHY_ZERO_SCALE",
            )
        if scale != 1.0:
            H /= scale

        return H
