from collections.abc import Callable
from typing import Any, Iterable, Literal, Sequence

from classes.marker import CompositeDetection, MarkerDetection
from classes.types import MarkerData
import numpy as np
from utils.detectors.detector_params import merge_marker_detector_params
from utils.geometry import calculateCentroid
from utils.tracing import Tracing

MarkerDetectorBackend = Literal["apriltag", "aruco"]

_detect_markers: Callable[[np.ndarray], set[MarkerDetection]] | None = None
_active_backend: MarkerDetectorBackend | None = None
_active_params: dict[str, dict[str, Any]] = merge_marker_detector_params(None)


def configure_marker_detector(
    backend: MarkerDetectorBackend,
    params: dict[str, Any] | None = None,
) -> None:
    global _detect_markers, _active_backend, _active_params

    merged = merge_marker_detector_params(params)
    _active_params = merged
    _active_backend = backend

    if backend == "aruco":
        from utils.detectors.aruco_backend import configure_aruco, detect_markers

        configure_aruco(merged["aruco"])
    else:
        from utils.detectors.apriltag_backend import configure_apriltag, detect_markers

        configure_apriltag(merged["apriltag"])

    _detect_markers = detect_markers


def get_active_marker_detector_config() -> tuple[MarkerDetectorBackend, dict[str, dict[str, Any]]]:
    backend = _active_backend or "apriltag"
    return backend, _active_params


def getDetections(image: np.ndarray) -> set[MarkerDetection]:
    with Tracing.ScopedZone("getDetections"):
        if _detect_markers is None:
            configure_marker_detector("apriltag")
        return _detect_markers(image)


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
