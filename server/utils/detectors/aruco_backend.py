"""OpenCV ArUco AprilTag dictionary detector backend."""

from typing import Any

from classes.marker import MarkerDetection
from classes.types import FloatingPoint2D, Point2D
import cv2
import numpy as np

from utils.detectors.detector_params import (
    APRILTAG_CORNER_REFINEMENT_MAP,
    normalize_aruco_params,
    params_cache_key,
)

_detector_cache: dict[tuple[tuple[str, Any], ...], cv2.aruco.ArucoDetector] = {}
_active_params: dict[str, Any] = normalize_aruco_params(None)


def configure_aruco(params: dict[str, Any] | None = None) -> None:
    global _active_params
    _active_params = normalize_aruco_params(params)


def _build_aruco_detector(params: dict[str, Any]) -> cv2.aruco.ArucoDetector:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.detectInvertedMarker = bool(params["detectInvertedMarker"])
    aruco_params.cornerRefinementMethod = APRILTAG_CORNER_REFINEMENT_MAP[
        str(params["cornerRefinementMethod"]).lower()
    ]
    aruco_params.adaptiveThreshWinSizeMin = int(params["adaptiveThreshWinSizeMin"])
    aruco_params.adaptiveThreshWinSizeMax = int(params["adaptiveThreshWinSizeMax"])
    aruco_params.adaptiveThreshWinSizeStep = int(params["adaptiveThreshWinSizeStep"])
    aruco_params.adaptiveThreshConstant = float(params["adaptiveThreshConstant"])
    aruco_params.minMarkerPerimeterRate = float(params["minMarkerPerimeterRate"])
    aruco_params.maxMarkerPerimeterRate = float(params["maxMarkerPerimeterRate"])
    aruco_params.polygonalApproxAccuracyRate = float(params["polygonalApproxAccuracyRate"])
    aruco_params.cornerRefinementWinSize = int(params["cornerRefinementWinSize"])
    aruco_params.cornerRefinementMaxIterations = int(params["cornerRefinementMaxIterations"])
    aruco_params.cornerRefinementMinAccuracy = float(params["cornerRefinementMinAccuracy"])
    aruco_params.errorCorrectionRate = float(params["errorCorrectionRate"])
    aruco_params.aprilTagQuadDecimate = float(params["aprilTagQuadDecimate"])
    aruco_params.aprilTagQuadSigma = float(params["aprilTagQuadSigma"])
    return cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_params)


def get_aruco_detector(params: dict[str, Any] | None = None) -> cv2.aruco.ArucoDetector:
    normalized = normalize_aruco_params(params if params is not None else _active_params)
    key = params_cache_key(normalized)
    if key not in _detector_cache:
        _detector_cache[key] = _build_aruco_detector(normalized)
    return _detector_cache[key]


def clear_aruco_detector_cache() -> None:
    _detector_cache.clear()


def _aruco_corners_to_points(marker_corner: np.ndarray) -> list[Point2D | FloatingPoint2D]:
    # OpenCV returns tag-frame TL, TR, BR, BL; MarkerDetection expects TL, TR, BL, BR.
    corners = marker_corner.reshape((4, 2))
    top_left, top_right, bottom_right, bottom_left = corners
    return [
        _to_point(top_left),
        _to_point(top_right),
        _to_point(bottom_left),
        _to_point(bottom_right),
    ]


def _to_point(corner: np.ndarray) -> Point2D:
    return (int(round(corner[0])), int(round(corner[1])))


def detect_markers(image: np.ndarray) -> set[MarkerDetection]:
    corners, ids, _rejected = get_aruco_detector().detectMarkers(image)
    if ids is None:
        return set()

    detections: set[MarkerDetection] = set()
    for marker_corner, marker_id in zip(corners, ids.flatten()):
        top_left, top_right, bottom_left, bottom_right = _aruco_corners_to_points(marker_corner)
        detections.add(
            MarkerDetection(
                top_left,
                top_right,
                bottom_left,
                bottom_right,
                data=int(marker_id),
            )
        )

    return detections
