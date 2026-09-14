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


def calculateAffineHomography(
    pointsA: np.ndarray,
    pointsB: np.ndarray,
) -> np.ndarray:
    """Least-squares affine transform (6 DoF) embedded as a 3x3 homography.

    Use instead of a full projective fit when the correspondences form fewer
    than 4 well-separated clusters (e.g. only 3 corner tags visible): the
    perspective terms (h31, h32) would then be constrained only by the tiny
    within-tag point spread and extrapolate wildly, while an affine fit stays
    stable under extrapolation.
    """
    with Tracing.ScopedZone("calculateAffineHomography"):
        A = np.asarray(pointsA, dtype=np.float64).reshape(-1, 2)
        B = np.asarray(pointsB, dtype=np.float64).reshape(-1, 2)

        design = np.hstack([A, np.ones((A.shape[0], 1))])
        solution, _residuals, rank, _sv = np.linalg.lstsq(design, B, rcond=None)

        if rank < 3:
            raise ProcessingError(
                "Affine estimation failed: correspondences are degenerate (collinear or coincident points)",
                code="AFFINE_DEGENERATE",
            )

        H = np.eye(3, dtype=np.float64)
        H[:2, :] = solution.T
        return H


def calculateHomography(
    detectionsA: Iterable[MarkerDetection | CompositeDetection] | np.ndarray,
    detectionsB: Iterable[MarkerDetection | CompositeDetection] | np.ndarray,
    *,
    useRansac: bool = True,
):
    """Calculates the homography matrix H_AB.

    ``useRansac=False`` fits all correspondences by least squares. Use it when the
    correspondences are matched by identity (no outliers possible) but carry a
    correlated residual - e.g. corner tags on a stitched mosaic, which is only
    piecewise-projective; RANSAC would lock onto a degenerate subset there and
    extrapolate wildly.
    """
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
        if useRansac:
            H, _status = cv2.findHomography(pointsA, pointsB, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        else:
            H, _status = cv2.findHomography(pointsA, pointsB, method=0)

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
