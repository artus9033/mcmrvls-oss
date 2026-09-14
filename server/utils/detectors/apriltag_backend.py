"""AprilTag detector backed by vendor/apriltag (built into the server venv)."""

from collections.abc import Sequence
from typing import Any

from apriltag import apriltag as ApriltagDetector
from classes.marker import MarkerDetection
from classes.types import FloatingPoint2D, Point2D
import cv2
import numpy as np

from utils.detectors.detector_params import (
    normalize_apriltag_params,
    params_cache_key,
)

# Matches the previous OpenCV DICT_APRILTAG_36H11 dictionary.
TAG_FAMILY = "tag36h11"

_detector_cache: dict[tuple[tuple[str, Any], ...], ApriltagDetector] = {}
_active_params: dict[str, Any] = normalize_apriltag_params(None)


def configure_apriltag(params: dict[str, Any] | None = None) -> None:
    global _active_params
    _active_params = normalize_apriltag_params(params)


def get_apriltag_detector(params: dict[str, Any] | None = None) -> ApriltagDetector:
    normalized = normalize_apriltag_params(params if params is not None else _active_params)
    key = params_cache_key(normalized)
    if key not in _detector_cache:
        _detector_cache[key] = ApriltagDetector(
            TAG_FAMILY,
            threads=normalized["threads"],
            maxhamming=normalized["maxhamming"],
            decimate=normalized["decimate"],
            blur=normalized["blur"],
            refine_edges=normalized["refine_edges"],
        )
    return _detector_cache[key]


def clear_apriltag_detector_cache() -> None:
    _detector_cache.clear()


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Expected a grayscale or BGR image, got shape {image.shape}")


def detect_apriltags(image: np.ndarray) -> Sequence[dict]:
    return get_apriltag_detector().detect(to_grayscale_uint8(image))


def _apriltag_corners_to_points(corners: np.ndarray) -> list[Point2D | FloatingPoint2D]:
    # Despite the Python binding key name, corners are apriltag p[0..3] in tag frame
    # (TL, TR, BR, BL). Swap TL/TR to match OpenCV ArUco's tag36h11 convention.
    tag_top_left, tag_top_right, tag_bottom_right, tag_bottom_left = corners
    return [
        _to_point(tag_top_right),
        _to_point(tag_top_left),
        _to_point(tag_bottom_right),
        _to_point(tag_bottom_left),
    ]


def _to_point(corner: np.ndarray) -> Point2D:
    return (int(round(corner[0])), int(round(corner[1])))


def detect_markers(image: np.ndarray) -> set[MarkerDetection]:
    best_by_id: dict[int, tuple[float, MarkerDetection]] = {}

    for detection in detect_apriltags(image):
        marker_id = int(detection["id"])
        margin = float(detection["margin"])
        top_left, top_right, bottom_left, bottom_right = _apriltag_corners_to_points(detection["lb-rb-rt-lt"])
        marker = MarkerDetection(
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            data=marker_id,
        )
        if marker_id not in best_by_id or margin > best_by_id[marker_id][0]:
            best_by_id[marker_id] = (margin, marker)

    return {marker for _, marker in best_by_id.values()}
