"""Default and normalization helpers for marker detector hyperparameters."""

from __future__ import annotations

import os
from typing import Any

APRILTAG_CORNER_REFINEMENT_MAP = {
    "none": 0,
    "subpix": 1,
    "contour": 2,
    "apriltag": 3,
}


def default_apriltag_params() -> dict[str, Any]:
    return {
        "threads": None,
        "maxhamming": 2,
        "decimate": 1.0,
        "blur": 0.0,
        "refine_edges": True,
    }


def default_aruco_params() -> dict[str, Any]:
    return {
        "detectInvertedMarker": True,
        "cornerRefinementMethod": "contour",
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 23,
        "adaptiveThreshWinSizeStep": 10,
        "adaptiveThreshConstant": 7,
        "minMarkerPerimeterRate": 0.03,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.03,
        "cornerRefinementWinSize": 5,
        "cornerRefinementMaxIterations": 30,
        "cornerRefinementMinAccuracy": 0.1,
        "errorCorrectionRate": 0.6,
        "aprilTagQuadDecimate": 0.0,
        "aprilTagQuadSigma": 0.0,
    }


def default_marker_detector_params() -> dict[str, dict[str, Any]]:
    return {
        "apriltag": default_apriltag_params(),
        "aruco": default_aruco_params(),
    }


def merge_marker_detector_params(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    defaults = default_marker_detector_params()
    if not raw:
        return defaults
    merged: dict[str, dict[str, Any]] = {}
    for backend in ("apriltag", "aruco"):
        backend_defaults = defaults[backend]
        backend_raw = raw.get(backend) or {}
        merged[backend] = {**backend_defaults, **backend_raw}
    return merged


def normalize_apriltag_params(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**default_apriltag_params(), **(params or {})}
    threads = merged.get("threads")
    if threads is None:
        merged["threads"] = max(1, os.cpu_count() or 1)
    else:
        merged["threads"] = max(1, int(threads))
    merged["maxhamming"] = int(merged["maxhamming"])
    merged["decimate"] = float(merged["decimate"])
    merged["blur"] = float(merged["blur"])
    merged["refine_edges"] = bool(merged["refine_edges"])
    return merged


def normalize_aruco_params(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**default_aruco_params(), **(params or {})}
    method = str(merged["cornerRefinementMethod"]).lower()
    if method not in APRILTAG_CORNER_REFINEMENT_MAP:
        raise ValueError(
            f"Unsupported cornerRefinementMethod: {merged['cornerRefinementMethod']}"
        )
    merged["cornerRefinementMethod"] = method
    merged["detectInvertedMarker"] = bool(merged["detectInvertedMarker"])
    for key in (
        "adaptiveThreshWinSizeMin",
        "adaptiveThreshWinSizeMax",
        "adaptiveThreshWinSizeStep",
        "cornerRefinementWinSize",
        "cornerRefinementMaxIterations",
    ):
        merged[key] = int(merged[key])
    for key in (
        "adaptiveThreshConstant",
        "minMarkerPerimeterRate",
        "maxMarkerPerimeterRate",
        "polygonalApproxAccuracyRate",
        "cornerRefinementMinAccuracy",
        "errorCorrectionRate",
        "aprilTagQuadDecimate",
        "aprilTagQuadSigma",
    ):
        merged[key] = float(merged[key])
    return merged


def params_cache_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((str(k), v) for k, v in params.items()))
