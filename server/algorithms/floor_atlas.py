"""
Live floor-tag atlas: per-camera -> map-normalized homographies fitted from
static floor tags accumulated over rounds.

The cameras are statically mounted and every non-robot tag
lies on the floor plane, so each raw (unstitched) camera view maps to the map
frame by a single exact homography - unlike the stitched mosaic, which is only
piecewise-projective. The atlas provides:

  * camera_h: per-camera camera-pixels -> map-normalized homographies
    (used by the per-camera consensus robot solver, `usePerCameraSolver`);
  * interior_centroids_norm: map-normalized centroids of interior floor tags
    (used to augment the stitched top-down fit, `topDownFitUseInteriorTags`,
    and to draw the true piecewise map boundary on the stitched preview).

Robot tags are excluded by configured ID; residual misdetections are rejected
by centroid spread. The atlas re-solves periodically so it survives camera
nudges together with the regular cache invalidation heuristics.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
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

if TYPE_CHECKING:
    from classes.config import AlgorithmConfig
    from classes.resources.ImagePack import ImagePack

# A tag observed with centroid spread beyond this many pixels is considered
# unstable (misdetection or something moved) and is left out of the fit.
MAX_STATIC_CENTROID_STD_PX = 4.0
# Per-camera consensus estimates outside this margin around the map are treated
# as a broken camera fit rather than a plausible robot position.
POSE_NORM_MARGIN = 0.2
# Per-camera robot estimates farther than this (map-normalized) from the
# cross-camera median center are rejected before fusion.
CONSENSUS_OUTLIER_NORM = 0.08


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(src, dst, method=0)
    return H


def project(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def robot_pose_from_quad(quad: np.ndarray) -> tuple[float, float, float]:
    """(x, y, azimuth_deg) from a TL/TR/BL/BR quad; 0 deg = North, clockwise."""
    tl, tr, bl, br = quad
    cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
    front, rear = (tl + tr) / 2.0, (bl + br) / 2.0
    dx, dy = float(front[0] - rear[0]), float(front[1] - rear[1])
    azimuth = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0
    return cx, cy, azimuth


def circular_mean_deg(angles: list[float]) -> float:
    rad = np.radians(angles)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))) % 360.0


def corner_tag_map_quads_norm(config: "AlgorithmConfig") -> dict[int, np.ndarray]:
    """Map-normalized (TL,TR,BL,BR) corner points of each map corner tag.

    (0,0) is the outer corner of the TL tag and (1,1) the outer corner of the
    BR tag - the frame in which the server reports x/y.
    """
    u = float(config.markerWidth) / float(config.mapWidth)
    v = float(config.markerHeight) / float(config.mapHeight)

    def quad(x0: float, y0: float) -> np.ndarray:
        return np.array([(x0, y0), (x0 + u, y0), (x0, y0 + v), (x0 + u, y0 + v)], dtype=np.float64)

    ids = config.mapDefinition
    return {
        int(ids["TL"]): quad(0.0, 0.0),
        int(ids["TR"]): quad(1.0 - u, 0.0),
        int(ids["BL"]): quad(0.0, 1.0 - v),
        int(ids["BR"]): quad(1.0 - u, 1.0 - v),
    }


def _detection_quad(detection) -> np.ndarray:
    return np.array(
        [
            detection.aggregatedTopLeft,
            detection.aggregatedTopRight,
            detection.aggregatedBottomLeft,
            detection.aggregatedBottomRight,
        ],
        dtype=np.float64,
    )


def _solve_by_chaining(
    static_by_camera: dict[str, dict[int, np.ndarray]],
    atlas_quads_norm: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Chain cameras via shared static tags, then fit one reference->map H."""
    cameras = list(static_by_camera)
    pair_h: dict[tuple[str, str], np.ndarray] = {}
    for i, cam_a in enumerate(cameras):
        for cam_b in cameras[i + 1 :]:
            shared = sorted(set(static_by_camera[cam_a]) & set(static_by_camera[cam_b]))
            if len(shared) < 2:
                continue
            src = np.vstack([static_by_camera[cam_a][t] for t in shared])
            dst = np.vstack([static_by_camera[cam_b][t] for t in shared])
            H = fit_homography(src, dst)
            if H is None:
                continue
            pair_h[(cam_a, cam_b)] = H
            pair_h[(cam_b, cam_a)] = np.linalg.inv(H)

    def atlas_count(cam: str) -> int:
        return sum(1 for t in static_by_camera[cam] if t in atlas_quads_norm)

    reference = max(cameras, key=lambda c: (atlas_count(c), len(static_by_camera[c])))
    to_ref: dict[str, np.ndarray] = {reference: np.eye(3)}
    queue = [reference]
    while queue:
        current = queue.pop(0)
        for other in cameras:
            if other in to_ref or (other, current) not in pair_h:
                continue
            to_ref[other] = to_ref[current] @ pair_h[(other, current)]
            queue.append(other)

    src_points, dst_points = [], []
    for camera, H_to_ref in to_ref.items():
        for tag_id, image_quad in static_by_camera[camera].items():
            if tag_id in atlas_quads_norm:
                src_points.append(project(H_to_ref, image_quad))
                dst_points.append(atlas_quads_norm[tag_id])
    # each entry is a 4-vertex quad; >= 3 anchor tags (12 points spread across
    # the map) give a well-conditioned fit even when one corner tag is unseen
    if len(src_points) < 3:
        return {}
    ref_to_map = fit_homography(np.vstack(src_points), np.vstack(dst_points))
    if ref_to_map is None:
        return {}
    return {camera: ref_to_map @ H_to_ref for camera, H_to_ref in to_ref.items()}


class FloorAtlas:
    """Accumulates static-tag observations per raw camera and solves
    camera -> map-normalized homographies plus interior floor-tag positions."""

    def __init__(
        self,
        config: "AlgorithmConfig",
        min_observations: int = 8,
        max_observations: int = 40,
        first_attempt_round: int = 15,
        resolve_interval_rounds: int = 300,
    ) -> None:
        self._robot_ids = set(int(r) for r in config.configuredRobotIDs)
        self._corner_quads_norm = corner_tag_map_quads_norm(config)
        self._min_observations = min_observations
        self._first_attempt_round = first_attempt_round
        self._resolve_interval_rounds = resolve_interval_rounds
        self._quads: dict[str, dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max_observations)))
        self._rounds = 0
        self._last_solve_round: int | None = None

        self.camera_h: dict[str, np.ndarray] = {}
        self.atlas_quads_norm: dict[int, np.ndarray] = {}
        self.interior_centroids_norm: dict[int, tuple[float, float]] = {}
        # map-normalized centroids of the atlas tags each camera actually
        # observes - H_cam is only trustworthy near this support (per-camera
        # fits extrapolate poorly toward the horizon)
        self.camera_support_norm: dict[str, np.ndarray] = {}
        # bumped on every successful solve; consumers key their caches on it
        self.solve_generation = 0

    @property
    def ready(self) -> bool:
        return bool(self.camera_h)

    def invalidate(self) -> None:
        """Drop accumulated observations and solved fits (e.g. cameras moved)."""
        self._quads.clear()
        self._rounds = 0
        self._last_solve_round = None
        self.camera_h = {}
        self.atlas_quads_norm = {}
        self.interior_centroids_norm = {}
        self.camera_support_norm = {}

    def observe(self, raw_camera_packs: list["ImagePack"]) -> None:
        """Accumulate this round's per-camera static-tag quads.

        Genuine camera movement is detected HERE, in the raw frame: a tag whose
        new centroid jumps far from its accumulated median gets its history
        reset, and a camera where most tags jump schedules an early re-solve.
        (Stitched-frame displacement heuristics are deliberately not used - the
        mosaic jitters whenever stitch homographies recalculate.)
        """
        self._rounds += 1
        jumpThreshold = 3.0 * MAX_STATIC_CENTROID_STD_PX
        for pack in raw_camera_packs:
            camera = str(pack.stitchingStageDescription)
            resets = 0
            observed = 0
            for tag_id, detection in pack.compositeDetectionsStore.items():
                if int(tag_id) in self._robot_ids:
                    continue
                observed += 1
                quad = _detection_quad(detection)
                history = self._quads[camera][int(tag_id)]
                if len(history) >= self._min_observations:
                    medianCentroid = np.median(np.stack(history), axis=0).mean(axis=0)
                    if float(np.linalg.norm(quad.mean(axis=0) - medianCentroid)) > jumpThreshold:
                        history.clear()  # tag (or camera) moved - restart its history
                        resets += 1
                history.append(quad)
            if observed >= 2 and resets * 2 >= observed and self.ready:
                # most static tags jumped at once: the camera itself moved -
                # schedule an early re-solve once fresh observations accumulate
                self._last_solve_round = self._rounds - self._resolve_interval_rounds + self._first_attempt_round

    def _static_by_camera(self) -> dict[str, dict[int, np.ndarray]]:
        result: dict[str, dict[int, np.ndarray]] = {}
        for camera, tags in self._quads.items():
            static: dict[int, np.ndarray] = {}
            for tag_id, quads in tags.items():
                if len(quads) < self._min_observations:
                    continue
                stack = np.stack(quads)
                centroids = stack.mean(axis=1)
                if float(centroids.std(axis=0).max()) > MAX_STATIC_CENTROID_STD_PX:
                    continue
                static[tag_id] = np.median(stack, axis=0)
            if static:
                result[camera] = static
        return result

    def maybe_solve(self, logger: Logger | None = None) -> bool:
        """Solve or periodically re-solve the atlas; returns True when solved."""
        if self._last_solve_round is None:
            if self._rounds < self._first_attempt_round:
                return False
        elif self._rounds - self._last_solve_round < self._resolve_interval_rounds:
            return False

        static_by_camera = self._static_by_camera()
        if len(static_by_camera) < 2:
            return False

        corner_h = _solve_by_chaining(static_by_camera, self._corner_quads_norm)
        if len(corner_h) < 2:
            # not solvable yet - retry after more observations accumulate
            self._last_solve_round = self._rounds - self._resolve_interval_rounds + self._first_attempt_round
            return False

        # Interior tags: median map-frame projection across all chained cameras.
        projected: dict[int, list[np.ndarray]] = defaultdict(list)
        for camera, H in corner_h.items():
            for tag_id, image_quad in static_by_camera[camera].items():
                projected[tag_id].append(project(H, image_quad))
        atlas: dict[int, np.ndarray] = dict(self._corner_quads_norm)
        for tag_id, quads in projected.items():
            if tag_id not in atlas:
                atlas[tag_id] = np.median(np.stack(quads), axis=0)

        # Refit each camera directly against the full atlas when it sees >= 3
        # atlas tags (better conditioned than the chain); fill gaps from the chain.
        direct: dict[str, np.ndarray] = {}
        for camera, static_tags in static_by_camera.items():
            shared = sorted(set(static_tags) & set(atlas))
            if len(shared) < 3:
                continue
            H = fit_homography(
                np.vstack([static_tags[t] for t in shared]),
                np.vstack([atlas[t] for t in shared]),
            )
            if H is not None:
                direct[camera] = H

        # Sanity gate: every atlas tag must land plausibly on or near the map.
        # A degenerate chain (thin data after a reset, transient occlusions) can
        # otherwise place tags absurdly far out and poison every consumer; in
        # that case keep the previous solution and retry after more rounds.
        for tag_id, quad in atlas.items():
            centroid = quad.mean(axis=0)
            if not (-0.5 <= centroid[0] <= 1.5 and -0.5 <= centroid[1] <= 1.5):
                if logger is not None:
                    logger.warning(
                        "🗺️ Floor atlas solve rejected: tag %s at implausible map position (%.2f, %.2f); keeping previous solution",
                        tag_id,
                        centroid[0],
                        centroid[1],
                    )
                self._last_solve_round = self._rounds - self._resolve_interval_rounds + self._first_attempt_round
                return False

        self.camera_h = {**corner_h, **direct}
        self.atlas_quads_norm = atlas
        self.interior_centroids_norm = {tag_id: (float(quad[:, 0].mean()), float(quad[:, 1].mean())) for tag_id, quad in atlas.items() if tag_id not in self._corner_quads_norm}
        self.camera_support_norm = {
            camera: np.array(
                [atlas[t].mean(axis=0) for t in static_tags if t in atlas],
                dtype=np.float64,
            )
            for camera, static_tags in static_by_camera.items()
            if camera in self.camera_h
        }
        self._last_solve_round = self._rounds
        self.solve_generation += 1

        if logger is not None:
            logger.info(
                "🗺️ Floor atlas support: %s",
                {cam: np.round(sup, 3).tolist() for cam, sup in self.camera_support_norm.items()},
            )
            logger.info(
                "🗺️ Floor atlas solved: %d camera(s), %d corner + %d interior tag(s)",
                len(self.camera_h),
                len(self._corner_quads_norm),
                len(self.interior_centroids_norm),
            )
        return True

    def robot_measurements(self, raw_camera_packs: list["ImagePack"], config: "AlgorithmConfig") -> dict[int, RobotTagMeasurement]:
        """Multi-camera consensus robot poses in the map-normalized frame.

        Each camera seeing a robot tag projects its quad through that camera's
        exact floor homography. Per-camera estimates differ systematically (the
        tag sits above the floor plane, so each camera displaces it toward its
        own viewpoint - parallax), so fusion must stay stable when the visible
        camera set changes:
        - outlier rejection: estimates far from the median center are dropped;
        - area-weighted mean of the survivors (closer camera = larger tag =
          more weight), instead of a median that flips between cameras;
        - the measurement sigmas are inflated by the cross-camera spread, so
          the Kalman filter smooths harder exactly when cameras disagree.
        """
        if not self.ready:
            return {}

        estimates_by_robot: dict[int, list[tuple[tuple[float, float, float], np.ndarray, float]]] = defaultdict(list)
        for pack in raw_camera_packs:
            camera = str(pack.stitchingStageDescription)
            H = self.camera_h.get(camera)
            if H is None:
                continue
            for tag_id, detection in pack.compositeDetectionsStore.items():
                if int(tag_id) not in self._robot_ids:
                    continue
                quad_px = _detection_quad(detection)
                quad_norm = project(H, quad_px)
                cx, cy = float(quad_norm[:, 0].mean()), float(quad_norm[:, 1].mean())
                if not (-POSE_NORM_MARGIN <= cx <= 1 + POSE_NORM_MARGIN and -POSE_NORM_MARGIN <= cy <= 1 + POSE_NORM_MARGIN):
                    continue  # broken per-camera fit, not a plausible robot position
                areaPx = quad_area_tl_tr_bl_br(
                    tuple(quad_px[0]), tuple(quad_px[1]), tuple(quad_px[2]), tuple(quad_px[3])
                )
                weight = max(float(areaPx), 1.0)
                estimates_by_robot[int(tag_id)].append((robot_pose_from_quad(quad_norm), quad_norm, weight))

        measurements: dict[int, RobotTagMeasurement] = {}
        for robot_id, estimates in estimates_by_robot.items():
            robot = config.findRobotWithID(robot_id)
            if robot is None:
                continue

            centers = np.array([[p[0], p[1]] for p, _, _ in estimates], dtype=np.float64)
            medianCenter = np.median(centers, axis=0)
            survivors = [
                e for e, c in zip(estimates, centers)
                if float(np.hypot(c[0] - medianCenter[0], c[1] - medianCenter[1])) <= CONSENSUS_OUTLIER_NORM
            ] or estimates  # never end up empty

            weights = np.array([w for _, _, w in survivors], dtype=np.float64)
            weights /= weights.sum()
            x = float(sum(w * p[0] for (p, _, _), w in zip(survivors, weights)))
            y = float(sum(w * p[1] for (p, _, _), w in zip(survivors, weights)))
            azimuthRad = [math.radians(p[2]) for p, _, _ in survivors]
            azimuth = float(
                math.degrees(
                    math.atan2(
                        sum(w * math.sin(a) for a, w in zip(azimuthRad, weights)),
                        sum(w * math.cos(a) for a, w in zip(azimuthRad, weights)),
                    )
                )
            ) % 360.0
            fused_quad = np.einsum("i,ijk->jk", weights, np.stack([q for _, q, _ in survivors]))

            # Map-normalized quad treated as a unit-square image for R diagonals.
            tl, tr, bl, br = (tuple(map(float, corner)) for corner in fused_quad)
            area = quad_area_tl_tr_bl_br(tl, tr, bl, br)
            baseline = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
            sx, sy, spsi = compute_tag_measurement_sigmas(
                area_px=max(area, 1e-8),
                baseline_px=max(baseline, 1e-6),
                img_w=1,
                img_h=1,
                p=robot.kalman_params,
            )
            area_norm, baseline_norm = normalize_tag_geometry(
                area_px=max(area, 1e-8), baseline_px=max(baseline, 1e-6), img_w=1, img_h=1
            )
            # Disagreement-aware R: cross-camera spread (parallax) directly
            # inflates the sigmas, so the EKF trusts consensus less exactly in
            # the rounds where the camera set switching would cause flicker.
            if len(survivors) >= 2:
                spreadPos = float(
                    max(
                        math.hypot(p[0] - x, p[1] - y)
                        for p, _, _ in survivors
                    )
                )
                spreadPsi = float(
                    max(
                        abs(((p[2] - azimuth + 180.0) % 360.0) - 180.0)
                        for p, _, _ in survivors
                    )
                )
                sx = max(sx, spreadPos)
                sy = max(sy, spreadPos)
                spsi = max(spsi, math.radians(spreadPsi))
            measurements[robot_id] = RobotTagMeasurement(
                robot=robot,
                x=x,
                y=y,
                azimuth_deg=azimuth,
                sigma_x=sx,
                sigma_y=sy,
                sigma_psi_rad=spsi,
                # No composite: the pose lives in the map frame, not in
                # top-down pixels, so top-down drawing must use the coasted path.
                composite=None,
                area_norm=area_norm,
                baseline_norm=baseline_norm,
            )
        return measurements
