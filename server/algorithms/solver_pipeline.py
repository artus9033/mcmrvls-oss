"""Integration helpers connecting GeometrySolver to the live stitching pipeline."""

from __future__ import annotations

import itertools
import math
from logging import Logger
from typing import TYPE_CHECKING

import cv2
import numpy as np

from algorithms.robot_kalman_tracker import (
    RobotTagMeasurement,
    compute_tag_measurement_sigmas,
    normalize_tag_geometry,
    quad_area_tl_tr_bl_br,
)
from errors import ProcessingError
from utils.tracing import Tracing

if TYPE_CHECKING:
    from algorithms.pointcloud import PairReconstruction
    from classes.config import AlgorithmConfig
    from classes.marker import CompositeDetection, MarkerData
    from classes.resources.AlgorithmState import AlgorithmState
    from classes.resources.ImagePack import ImagePack


def reset_solver_state(algorithm_state: AlgorithmState) -> None:
    if algorithm_state.solver_state is not None:
        algorithm_state.solver_state.clear()


def process_epipolar_pairs_from_image_packs(
    image_packs: list[ImagePack],
    algorithm_state: AlgorithmState,
    config: AlgorithmConfig,
    logger: Logger,
) -> None:
    """Triangulate 3D points from all raw per-camera pairs before hierarchical stitching."""
    if not config.useEpipolarGeometry:
        return
    if algorithm_state.solver_state is None:
        return

    reset_solver_state(algorithm_state)
    calibrations = getattr(config, "camera_calibrations", None) or {}

    with Tracing.ScopedZone("process_epipolar_pairs"):
        for pack1, pack2 in itertools.combinations(image_packs, 2):
            common_ids = pack1.compositeDetectionsStore.keys() & pack2.compositeDetectionsStore.keys()
            if not common_ids:
                continue
            detections1 = list(pack1.compositeDetectionsStore.values())
            detections2 = list(pack2.compositeDetectionsStore.values())
            cam1_id = pack1.stitchingStageDescription
            cam2_id = pack2.stitchingStageDescription
            calib1 = calibrations.get(str(cam1_id))
            calib2 = calibrations.get(str(cam2_id))
            try:
                algorithm_state.solver.process_camera_pair(
                    detections1,
                    detections2,
                    cam1_id,
                    cam2_id,
                    algorithm_state.solver_state,
                    use_cache=config.caching.stitchingHomographyCache,
                    calib1=calib1,
                    calib2=calib2,
                )
            except TypeError:
                # Homography solver (or older signature) does not accept calib kwargs.
                try:
                    algorithm_state.solver.process_camera_pair(
                        detections1,
                        detections2,
                        cam1_id,
                        cam2_id,
                        algorithm_state.solver_state,
                        use_cache=config.caching.stitchingHomographyCache,
                    )
                except ProcessingError as exc:
                    logger.debug(
                        "Epipolar pair %s x %s skipped: %s",
                        cam1_id,
                        cam2_id,
                        exc,
                    )
            except ProcessingError as exc:
                logger.debug(
                    "Epipolar pair %s x %s skipped: %s",
                    cam1_id,
                    cam2_id,
                    exc,
                )


def _azimuth_from_warped_detection(detection: CompositeDetection) -> float:
    """Same convention as homography robot pose in stitching.py (0 deg = North)."""
    top_left = detection.aggregatedTopLeft
    top_right = detection.aggregatedTopRight
    bottom_left = detection.aggregatedBottomLeft
    bottom_right = detection.aggregatedBottomRight
    front_axle = (
        (top_left[0] + top_right[0]) / 2,
        (top_left[1] + top_right[1]) / 2,
    )
    rear_axle = (
        (bottom_left[0] + bottom_right[0]) / 2,
        (bottom_left[1] + bottom_right[1]) / 2,
    )
    dx = front_axle[0] - rear_axle[0]
    dy = front_axle[1] - rear_axle[1]
    azimuth = math.degrees(math.atan2(dy, dx)) + 90.0
    return (azimuth + 360.0) % 360.0


# A pair whose static tags reconstruct less flatly than this is not trustworthy:
# genuine floor tags are coplanar, so a thick "plane" means the reconstruction
# (or the tag classification) went wrong.
_MAX_PLANE_FLATNESS = 0.05

# Distinct anchor tags required before a pair can be registered. Fewer than
# three clusters leaves a homography's perspective terms constrained only by
# the tiny within-tag corner spread, which extrapolates wildly off-anchor.
_MIN_ANCHOR_TAGS = 3
_MIN_ANCHOR_POINTS = 8
_REGISTRATION_RANSAC_PX = 12.0

# A homography is only trustworthy across the region its correspondences span.
# Anchors sit in a camera pair's overlap, which can be a small part of the
# arena, and a robot outside that span is extrapolated -- in practice to
# positions hundreds of map units away. Reject rather than emit those: the
# reconstruction genuinely does not constrain the robot there.
_MAX_SUPPORT_EXTRAPOLATION = 0.35

# Backstop matching the floor atlas's own sanity gate: a pose this far outside
# the map is not a degraded measurement, it is a failed registration.
_MAP_SANITY_MIN = -0.5
_MAP_SANITY_MAX = 1.5


def _register_pair_to_map(
    pair: "PairReconstruction",
    anchor_quads_px: dict[MarkerData, np.ndarray],
    robot_ids: set[MarkerData],
) -> dict[MarkerData, tuple[float, float]]:
    """
    Register one pair's reconstruction into the map frame.

    An uncalibrated reconstruction is fixed only up to a 3D homography, and a
    full 3D->2D fit is degenerate when every known reference is coplanar --
    which they are here, since the anchors are floor tags. What IS recoverable
    is the floor plane itself, so we fit that, express the anchor tags in a 2D
    in-plane basis, and solve the well-posed 2D homography from there to the
    map. Anchors contribute all four of their corners, giving the fit
    well-separated clusters rather than one point per tag.

    The robot tag is off-plane, so it is cast back through the camera centre
    onto the plane before mapping (plane + parallax); using its reconstructed
    point directly would place the robot where its raised tag floats.

    Returns map-frame pixel positions, empty when the pair cannot be registered.
    """
    from algorithms.epipolar import fit_plane, intersect_ray_with_plane, camera_centres

    def euclidean(point) -> np.ndarray | None:
        try:
            return point.xyz
        except (ValueError, ZeroDivisionError):
            return None

    source_xyz: list[np.ndarray] = []
    target_px: list[tuple[float, float]] = []
    anchor_tags = 0
    for marker_id, quad_px in anchor_quads_px.items():
        if marker_id in robot_ids:
            continue
        corners = pair.corner_points.get(marker_id)
        if not corners:
            continue
        used = 0
        for corner_index, point in corners.items():
            if corner_index < 0 or corner_index >= len(quad_px):
                continue
            xyz = euclidean(point)
            if xyz is None:
                continue
            source_xyz.append(xyz)
            target_px.append((float(quad_px[corner_index][0]), float(quad_px[corner_index][1])))
            used += 1
        if used:
            anchor_tags += 1

    # A homography needs four correspondences, but four corners of a single tag
    # span a few pixels and would extrapolate wildly across the map, so require
    # anchors from distinct tags too.
    if anchor_tags < _MIN_ANCHOR_TAGS or len(source_xyz) < _MIN_ANCHOR_POINTS:
        return {}

    source_arr = np.asarray(source_xyz, dtype=np.float64)

    if pair.metric:
        # Calibrated pairs are already map-frame centimetres; the floor plane is
        # z = 0 by construction, so a plain 2D fit over x/y is enough.
        plane = None
        plane_coords = source_arr[:, :2]
    else:
        try:
            plane = fit_plane(source_arr)
        except ProcessingError:
            return {}
        if plane.flatness > _MAX_PLANE_FLATNESS:
            return {}
        plane_coords = plane.to_plane_coords(source_arr)

    homography, inliers = cv2.findHomography(
        plane_coords.astype(np.float64),
        np.asarray(target_px, dtype=np.float64),
        cv2.RANSAC,
        _REGISTRATION_RANSAC_PX,
    )
    if homography is None:
        return {}
    if inliers is not None and int(inliers.sum()) < _MIN_ANCHOR_POINTS:
        return {}

    centre1 = None
    if not pair.metric and pair.P1 is not None and pair.P2 is not None:
        centre1_h, _ = camera_centres(pair.P1, pair.P2)
        if abs(centre1_h[3]) > 1e-12:
            centre1 = centre1_h[:3] / centre1_h[3]

    def to_map(xyz: np.ndarray) -> tuple[float, float]:
        coords = (
            xyz[:2].reshape(1, 1, 2)
            if plane is None
            else plane.to_plane_coords(xyz.reshape(1, 3)).reshape(1, 1, 2)
        )
        mapped = cv2.perspectiveTransform(coords.astype(np.float64), homography)
        return (float(mapped[0, 0, 0]), float(mapped[0, 0, 1]))

    # Span the anchors cover, used to reject extrapolated robot poses below.
    anchor_min = plane_coords.min(axis=0)
    anchor_max = plane_coords.max(axis=0)
    margin = float(np.linalg.norm(anchor_max - anchor_min)) * _MAX_SUPPORT_EXTRAPOLATION

    registered: dict[MarkerData, tuple[float, float]] = {}
    for marker_id, point in pair.points.items():
        xyz = euclidean(point)
        if xyz is None:
            continue

        is_robot = marker_id in robot_ids
        if is_robot and plane is not None:
            if centre1 is None:
                continue
            footprint = intersect_ray_with_plane(centre1, xyz, plane)
            if footprint is None:
                continue
            xyz = footprint

        if is_robot:
            local = (
                xyz[:2]
                if plane is None
                else plane.to_plane_coords(xyz.reshape(1, 3))[0]
            )
            if np.any(local < anchor_min - margin) or np.any(local > anchor_max + margin):
                continue

        registered[marker_id] = to_map(xyz)

    return registered


def collect_epipolar_robot_measurements(
    algorithm_state: AlgorithmState,
    config: AlgorithmConfig,
    detections: list[CompositeDetection],
    img_width: int,
    img_height: int,
    anchor_quads_px: dict[MarkerData, np.ndarray] | None = None,
    warped_robot_detection: CompositeDetection | None = None,
) -> list[RobotTagMeasurement]:
    """
    Build 2D normalized robot measurements from the epipolar reconstruction.

    Each camera pair is registered into the map frame on its own (see
    `_register_pair_to_map`) and the resulting map-frame positions are fused by
    a coordinate-wise median. Azimuth uses the warped robot detection when
    present, which is the same source the homography solver uses -- so azimuth
    differences between the two solvers reflect their differing warps, not
    differing orientation geometry.
    """
    if algorithm_state.solver_state is None:
        return []

    from algorithms.solvers.epipolar import EpipolarState

    solver_state = algorithm_state.solver_state
    if not isinstance(solver_state, EpipolarState):
        return []

    robot_ids = {robot.id for robot in config.configuredRobots.values()}

    # Register every camera pair independently, then fuse in the MAP frame.
    # Fusing earlier would be meaningless: each pair's reconstruction lives in
    # its own projective frame.
    if not anchor_quads_px:
        return []

    per_pair_robot_positions: dict[MarkerData, list[tuple[float, float]]] = {}
    registered_pairs = 0
    anchor_counts: list[int] = []
    for pair in solver_state.reconstructor.pairs:
        anchor_counts.append(
            sum(
                1 for mid in pair.corner_points
                if mid not in robot_ids and mid in anchor_quads_px
            )
        )
        registered = _register_pair_to_map(pair, anchor_quads_px, robot_ids)
        if not registered:
            continue
        registered_pairs += 1
        for robot_id in robot_ids:
            if robot_id not in registered:
                continue
            x_px, y_px = registered[robot_id]
            # Sanity backstop in map-normalized terms.
            if not (
                _MAP_SANITY_MIN <= x_px / float(img_width) <= _MAP_SANITY_MAX
                and _MAP_SANITY_MIN <= y_px / float(img_height) <= _MAP_SANITY_MAX
            ):
                continue
            per_pair_robot_positions.setdefault(robot_id, []).append((x_px, y_px))

    positions_2d: dict[MarkerData, tuple[float, float]] = {}
    for robot_id, samples in per_pair_robot_positions.items():
        arr = np.asarray(samples, dtype=np.float64)
        positions_2d[robot_id] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

    metric = solver_state.reconstructor.metric
    if config.logger is not None:
        config.logger.info(
            "Epipolar point cloud: %d/%d pair(s) registered; anchors per pair: %s; "
            "robots located: %s; robots configured: %s; metric=%s",
            registered_pairs,
            len(solver_state.reconstructor.pairs),
            anchor_counts,
            sorted(positions_2d.keys()),
            sorted(robot_ids),
            metric,
        )

    if not positions_2d:
        return []
    detections_by_id = {detection.data: detection for detection in detections}

    measurements: list[RobotTagMeasurement] = []
    for robot in config.configuredRobots.values():
        if robot.id not in positions_2d:
            continue

        x_px, y_px = positions_2d[robot.id]
        stitched_detection = detections_by_id.get(robot.id)

        if warped_robot_detection is not None and warped_robot_detection.data == robot.id:
            pose_detection = warped_robot_detection
        else:
            pose_detection = stitched_detection

        if pose_detection is not None:
            top_left = pose_detection.aggregatedTopLeft
            top_right = pose_detection.aggregatedTopRight
            bottom_left = pose_detection.aggregatedBottomLeft
            bottom_right = pose_detection.aggregatedBottomRight
            area_px = quad_area_tl_tr_bl_br(top_left, top_right, bottom_left, bottom_right)
            dx = float(top_right[0] - top_left[0])
            dy = float(top_right[1] - top_left[1])
            baseline_px = float(math.hypot(dx, dy))
            azimuth = _azimuth_from_warped_detection(pose_detection)
        else:
            area_px = 100.0
            baseline_px = 20.0
            azimuth = 0.0

        sx, sy, spsi = compute_tag_measurement_sigmas(
            area_px=area_px,
            baseline_px=baseline_px,
            img_w=img_width,
            img_h=img_height,
            p=robot.kalman_params,
        )
        area_norm, baseline_norm = normalize_tag_geometry(
            area_px=area_px, baseline_px=baseline_px, img_w=img_width, img_h=img_height
        )
        measurements.append(
            RobotTagMeasurement(
                robot=robot,
                x=float(x_px) / float(img_width),
                y=float(y_px) / float(img_height),
                azimuth_deg=float(azimuth),
                sigma_x=sx,
                sigma_y=sy,
                sigma_psi_rad=spsi,
                composite=pose_detection or stitched_detection,
                area_norm=area_norm,
                baseline_norm=baseline_norm,
            )
        )

    return measurements
