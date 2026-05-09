from typing import Iterable, Sequence

from classes.marker import CompositeDetection, MarkerDetection
from classes.types import FloatingPoint2D, MarkerData, Point2D
import cv2
import numpy as np
from utils.geometry import calculateCentroid
from utils.tracing import Tracing

def getDetections(image: np.ndarray) -> set[MarkerDetection]:
    with Tracing.ScopedZone("getDetections"):
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.detectInvertedMarker = True
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        # arucoParams.cornerRefinementMaxIterations = 100
        # arucoParams.cornerRefinementMinAccuracy = 0.01
        detector = cv2.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)
        (corners, ids, rejected) = detector.detectMarkers(image)

        detections: set[MarkerDetection] = set()

        if ids is None:
            return set()

        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for markerCorner, markerID in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            orderedPoints: list[Point2D | FloatingPoint2D] = [
                topLeft,
                topRight,
                bottomLeft,
                bottomRight,
            ]

            detections.add(MarkerDetection(*orderedPoints, data=markerID))

        return detections


def averageDetections(detections: Sequence[MarkerDetection]) -> MarkerDetection:
    with Tracing.ScopedZone("averageDetections"):
        for i, d in enumerate(detections):
            if d.data != detections[0].data:
                raise ValueError(f"averageDetections: all detections must have the same data, however index {i} has {d.data} != {detections[0].data}")

        avgTopLeft = calculateCentroid([d.topLeft for d in detections])
        avgTopRight = calculateCentroid([d.topRight for d in detections])
        avgBottomLeft = calculateCentroid([d.bottomLeft for d in detections])
        avgBottomRight = calculateCentroid([d.bottomRight for d in detections])

        return MarkerDetection(
            topLeft=avgTopLeft,
            topRight=avgTopRight,
            bottomLeft=avgBottomLeft,
            bottomRight=avgBottomRight,
            data=detections[0].data,
        )


def detectionsToNumpy(
    detections: Iterable[MarkerDetection | CompositeDetection],
) -> tuple[np.ndarray, list[MarkerData]]:
    """Returns a tuple of: a flattened list of detections cast to numpy arrays and a legend carrying IDs of the markers."""
    with Tracing.ScopedZone("detectionsToNumpy"):
        flatDetections: list[MarkerDetection] = []
        markerDataLegendInOrder: list[MarkerData] = []

        for detection in detections:
            if isinstance(detection, CompositeDetection):
                flatDetections.extend(detection.detections)
                markerDataLegendInOrder.extend([detection.data] * len(detection.detections))
            else:
                flatDetections.append(detection)
                markerDataLegendInOrder.append(detection.data)

        return (
            np.array(
                [flatDetection.toPointsList() for flatDetection in flatDetections],
                dtype=np.float32,
            ),
            markerDataLegendInOrder,
        )


def numpyToDetections(
    warpedNpDetections: np.ndarray,
    markerDataLegendInOrder: Iterable[MarkerData],
) -> Iterable[CompositeDetection]:
    with Tracing.ScopedZone("numpyToDetections"):
        detectionsMap: dict[MarkerData, CompositeDetection] = {}

        flatMarkerDetections = [MarkerDetection(*[tuple(p) for p in detectionPoints], data=data) for detectionPoints, data in zip(warpedNpDetections, markerDataLegendInOrder)]

        for flatDetection in flatMarkerDetections:
            if flatDetection.data in detectionsMap:
                detectionsMap[flatDetection.data].appendDetection(flatDetection)
            else:
                detectionsMap[flatDetection.data] = CompositeDetection(data=flatDetection.data, detections=[flatDetection])

        return detectionsMap.values()
