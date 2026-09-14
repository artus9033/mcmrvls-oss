"""On-demand AprilTag calibration from live ImagePack detections."""

from __future__ import annotations

from collections import defaultdict
from logging import Logger
from typing import TYPE_CHECKING, Any

from algorithms.camera_calibration import (
    CameraCalibration,
    corner_tag_object_quads_cm,
    estimate_camera_calibration,
)
import cv2
import numpy as np

if TYPE_CHECKING:
    from classes.config import AlgorithmConfig
    from classes.resources.ImagePack import ImagePack


def _config_dict_for_corner_geometry(config: AlgorithmConfig) -> dict[str, Any]:
    return {
        "dimensions": {
            "mapWidth": config.mapWidth,
            "mapHeight": config.mapHeight,
            "markerWidth": config.markerWidth,
            "markerHeight": config.markerHeight,
        },
        "map": dict(config.mapDefinition),
    }


def _image_quad_from_detection(detection: Any) -> np.ndarray:
    return np.array(
        [
            detection.aggregatedTopLeft,
            detection.aggregatedTopRight,
            detection.aggregatedBottomLeft,
            detection.aggregatedBottomRight,
        ],
        dtype=np.float64,
    )


def accumulate_tag_quads(
    packs: list[ImagePack],
    exclude_tag_ids: set[int],
    buckets: dict[str, dict[int, list[np.ndarray]]],
) -> None:
    """Append per-camera image quads for non-excluded tags (floor tags) into ``buckets``."""
    for pack in packs:
        cam_id = str(pack.stitchingStageDescription)
        cam_bucket = buckets.setdefault(cam_id, defaultdict(list))
        for tag_id, detection in pack.compositeDetectionsStore.items():
            tid = int(tag_id)
            if tid in exclude_tag_ids:
                continue
            cam_bucket[tid].append(_image_quad_from_detection(detection))


def median_quads(
    cam_bucket: dict[int, list[np.ndarray]],
    *,
    min_observations: int = 1,
) -> dict[int, np.ndarray]:
    return {
        tag_id: np.median(np.stack(quads), axis=0)
        for tag_id, quads in cam_bucket.items()
        if len(quads) >= min_observations
    }


def _fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4:
        return None
    # RANSAC: dynamic scenes can include a false/outlier shared tag that
    # otherwise warps the whole camera chain (and poisons the floor-tag atlas).
    H, _mask = cv2.findHomography(
        src.astype(np.float64),
        dst.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
    )
    return H


def _project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


def _solve_cameras_by_chaining(
    static_by_camera: dict[str, dict[int, np.ndarray]],
    corner_quads_xy_cm: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Chain cameras via shared floor tags, then fit one reference->map (cm) H."""
    cameras = list(static_by_camera)
    if not cameras:
        return {}

    pair_h: dict[tuple[str, str], np.ndarray] = {}
    for i, cam_a in enumerate(cameras):
        for cam_b in cameras[i + 1 :]:
            shared = sorted(set(static_by_camera[cam_a]) & set(static_by_camera[cam_b]))
            if len(shared) < 2:
                continue
            src = np.vstack([static_by_camera[cam_a][t] for t in shared])
            dst = np.vstack([static_by_camera[cam_b][t] for t in shared])
            H = _fit_homography(src, dst)
            if H is None:
                continue
            pair_h[(cam_a, cam_b)] = H
            pair_h[(cam_b, cam_a)] = np.linalg.inv(H)

    def corner_count(cam: str) -> int:
        return sum(1 for t in static_by_camera[cam] if t in corner_quads_xy_cm)

    reference = max(cameras, key=lambda c: (corner_count(c), len(static_by_camera[c])))
    to_ref: dict[str, np.ndarray] = {reference: np.eye(3)}
    queue = [reference]
    while queue:
        current = queue.pop(0)
        for other in cameras:
            if other in to_ref or (other, current) not in pair_h:
                continue
            to_ref[other] = to_ref[current] @ pair_h[(other, current)]
            queue.append(other)

    src_points: list[np.ndarray] = []
    dst_points: list[np.ndarray] = []
    for camera, H_to_ref in to_ref.items():
        for tag_id, image_quad in static_by_camera[camera].items():
            if tag_id not in corner_quads_xy_cm:
                continue
            src_points.append(_project(H_to_ref, image_quad))
            dst_points.append(corner_quads_xy_cm[tag_id])
    # Each entry is a 4-corner quad. Homography needs ≥4 points; prefer ≥2
    # distinct corner tags (8 points) — dynamic scenes often miss one corner.
    if len(src_points) < 2:
        return {}
    ref_to_map = _fit_homography(np.vstack(src_points), np.vstack(dst_points))
    if ref_to_map is None:
        return {}
    return {camera: ref_to_map @ H_to_ref for camera, H_to_ref in to_ref.items()}


def build_floor_tag_atlas_cm(
    static_by_camera: dict[str, dict[int, np.ndarray]],
    config: AlgorithmConfig,
    *,
    logger: Logger | None = None,
    map_margin_cm: float = 40.0,
    max_multi_view_std_cm: float = 25.0,
) -> dict[int, np.ndarray]:
    """
    Map-frame object quads (Nx3, z=0, cm) for corner + interior floor tags.

    Overhead cameras often see only one map corner each. Shared interior tags
    let us chain cameras, place interior tags in the map frame, and then give
    every camera a well-spread planar target for PnP / calibrateCamera.

    Interior placements that disagree across views or fall far outside the map
    bounds are discarded (bad pairwise H / false tag matches).
    """
    log = logger.info if logger is not None else (lambda *a, **k: None)
    corner_xyz = corner_tag_object_quads_cm(_config_dict_for_corner_geometry(config))
    corner_xy = {tid: quad[:, :2] for tid, quad in corner_xyz.items()}

    camera_to_map = _solve_cameras_by_chaining(static_by_camera, corner_xy)
    if len(camera_to_map) < 1:
        log("Floor-tag atlas chaining failed; using map corner tags only")
        return corner_xyz

    projected: dict[int, list[np.ndarray]] = defaultdict(list)
    for camera, H in camera_to_map.items():
        for tag_id, image_quad in static_by_camera[camera].items():
            projected[tag_id].append(_project(H, image_quad))

    map_w = float(config.mapWidth)
    map_h = float(config.mapHeight)
    marker_w = float(config.markerWidth)
    marker_h = float(config.markerHeight)

    def _in_bounds(quad_xy: np.ndarray) -> bool:
        center = quad_xy.mean(axis=0)
        return (
            -map_margin_cm <= float(center[0]) <= map_w + map_margin_cm
            and -map_margin_cm <= float(center[1]) <= map_h + map_margin_cm
        )

    def _size_plausible(quad_xy: np.ndarray) -> bool:
        # Allow generous tolerance: chaining H is approximate on sparse corners.
        width = float(np.linalg.norm(quad_xy[1] - quad_xy[0]))
        height = float(np.linalg.norm(quad_xy[2] - quad_xy[0]))
        return (
            0.4 * marker_w <= width <= 2.5 * marker_w
            and 0.4 * marker_h <= height <= 2.5 * marker_h
        )

    atlas_xy: dict[int, np.ndarray] = dict(corner_xy)
    rejected = 0
    for tag_id, quads in projected.items():
        if tag_id in atlas_xy:
            continue
        stack = np.stack(quads)
        median = np.median(stack, axis=0)
        center = median.mean(axis=0)
        if not _in_bounds(median):
            log(
                "Atlas reject tag %s: center (%.1f, %.1f) cm outside map",
                tag_id,
                float(center[0]),
                float(center[1]),
            )
            rejected += 1
            continue
        if not _size_plausible(median):
            width = float(np.linalg.norm(median[1] - median[0]))
            height = float(np.linalg.norm(median[2] - median[0]))
            log(
                "Atlas reject tag %s: projected size %.1fx%.1f cm (expected ~%.0fx%.0f)",
                tag_id,
                width,
                height,
                marker_w,
                marker_h,
            )
            rejected += 1
            continue
        if len(quads) >= 2:
            std = float(stack.std(axis=0).max())
            if std > max_multi_view_std_cm:
                log(
                    "Atlas reject tag %s: multi-view std %.1f cm > %.1f",
                    tag_id,
                    std,
                    max_multi_view_std_cm,
                )
                rejected += 1
                continue
        atlas_xy[tag_id] = median

    atlas_xyz: dict[int, np.ndarray] = {}
    for tag_id, quad_xy in atlas_xy.items():
        pts = np.zeros((4, 3), dtype=np.float64)
        pts[:, :2] = quad_xy
        atlas_xyz[tag_id] = pts

    log(
        "Floor-tag atlas: %d tags (%d corners + %d interior) from %d chained camera(s)%s",
        len(atlas_xyz),
        len(corner_xy),
        len(atlas_xyz) - len(corner_xy),
        len(camera_to_map),
        f", rejected {rejected}" if rejected else "",
    )
    return atlas_xyz


def calibrate_from_accumulated_quads(
    buckets: dict[str, dict[int, list[np.ndarray]]],
    image_sizes: dict[str, tuple[int, int]],
    config: AlgorithmConfig,
    *,
    min_tags: int = 2,
    min_observations: int = 2,
    logger: Logger | None = None,
) -> dict[str, CameraCalibration]:
    """
    Two-pass calibration (free-K then shared-K) from accumulated tag quads.

    Builds a multi-camera floor-tag atlas so cameras that only see one map
    corner can still be calibrated via shared interior tags.

    Returns calibrations keyed by camera stage id (\"0\", \"1\", …).
    """
    log = logger.info if logger is not None else (lambda *a, **k: None)

    static_by_camera: dict[str, dict[int, np.ndarray]] = {}
    for cam_id, cam_bucket in buckets.items():
        quads = median_quads(cam_bucket, min_observations=min_observations)
        if quads:
            static_by_camera[cam_id] = quads
            log(
                "Calibration cam %s: %d floor tag(s) with ≥%d obs: %s",
                cam_id,
                len(quads),
                min_observations,
                sorted(quads),
            )

    if not static_by_camera:
        raise RuntimeError("No floor-tag observations collected for calibration")

    object_quads = build_floor_tag_atlas_cm(static_by_camera, config, logger=logger)
    known_ids = set(object_quads)

    pending: list[dict[str, Any]] = []
    for cam_id, image_quads_all in sorted(static_by_camera.items(), key=lambda item: item[0]):
        image_size = image_sizes.get(cam_id)
        if image_size is None:
            continue
        image_quads = {tid: q for tid, q in image_quads_all.items() if tid in known_ids}
        if len(image_quads) < 1:
            log(
                "Calibration skip cam %s: no atlas tags; seen=%s",
                cam_id,
                sorted(image_quads_all),
            )
            continue

        obj_pts = []
        img_pts = []
        for tag_id, img_quad in sorted(image_quads.items()):
            obj_pts.append(object_quads[tag_id])
            img_pts.append(img_quad)
        pending.append(
            {
                "index": int(cam_id) if str(cam_id).isdigit() else cam_id,
                "cam_id": cam_id,
                "image_size": image_size,
                "object_points": np.vstack(obj_pts),
                "image_points": np.vstack(img_pts),
                "n_tags": len(image_quads),
            }
        )

    if not pending:
        raise RuntimeError(
            "No cameras had enough floor tags for calibration "
            "(need ≥1 atlas tag per camera; check map corner IDs and tag visibility)"
        )

    good: list[tuple[dict[str, Any], CameraCalibration]] = []
    weak: list[dict[str, Any]] = []
    weak_fxs: list[float] = []
    for item in pending:
        # Free-K needs a well-spread target; skip underdetermined single-tag views.
        if item["n_tags"] < min_tags:
            weak.append(item)
            log(
                "Calibration cam %s: only %d atlas tag(s); deferred to shared-K",
                item["cam_id"],
                item["n_tags"],
            )
            continue
        try:
            calib = estimate_camera_calibration(
                item["object_points"],
                item["image_points"],
                item["image_size"],
                camera_index=int(item["index"]) if isinstance(item["index"], int) else None,
            )
            if calib.reprojection_error > 25.0:
                weak.append(item)
                weak_fxs.append(float(calib.camera_matrix[0, 0]))
                log(
                    "Calibration free-K weak cam %s (rms=%.1fpx, tags=%d)",
                    item["cam_id"],
                    calib.reprojection_error,
                    item["n_tags"],
                )
            else:
                good.append((item, calib))
                log(
                    "Calibration free-K OK cam %s rms=%.3fpx fx=%.1f tags=%d",
                    item["cam_id"],
                    calib.reprojection_error,
                    calib.camera_matrix[0, 0],
                    item["n_tags"],
                )
        except Exception as exc:  # noqa: BLE001
            weak.append(item)
            log("Calibration free-K failed cam %s (%s); shared-K retry", item["cam_id"], exc)

    if good:
        shared_fx = float(np.median([float(c.camera_matrix[0, 0]) for _, c in good]))
    elif weak_fxs:
        shared_fx = float(np.median(weak_fxs))
        log("Calibration: using median fx=%.1f from weak free-K attempts", shared_fx)
    else:
        shared_fx = float(pending[0]["image_size"][0])
        log("Calibration: no free-K successes; using image-width prior fx=%.1f", shared_fx)

    results: list[tuple[dict[str, Any], CameraCalibration]] = list(good)
    for item in weak:
        w, h = item["image_size"]
        shared_K = np.array(
            [[shared_fx, 0.0, w / 2.0], [0.0, shared_fx, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        try:
            calib = estimate_camera_calibration(
                item["object_points"],
                item["image_points"],
                item["image_size"],
                camera_index=int(item["index"]) if isinstance(item["index"], int) else None,
                fixed_camera_matrix=shared_K,
            )
            if calib.reprojection_error > 80.0:
                log(
                    "Calibration skip cam %s: shared-K rms=%.1fpx too high (tags=%d)",
                    item["cam_id"],
                    calib.reprojection_error,
                    item["n_tags"],
                )
                continue
            results.append((item, calib))
            log(
                "Calibration shared-K OK cam %s rms=%.3fpx fx=%.1f tags=%d",
                item["cam_id"],
                calib.reprojection_error,
                calib.camera_matrix[0, 0],
                item["n_tags"],
            )
        except Exception as exc:  # noqa: BLE001
            log("Calibration skip cam %s after shared-K failure: %s", item["cam_id"], exc)

    if not results:
        raise RuntimeError("Calibration failed for all cameras")

    return {item["cam_id"]: calib for item, calib in results}
