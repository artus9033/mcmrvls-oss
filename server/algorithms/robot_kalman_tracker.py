"""
Extended Kalman filter for AprilTag robot poses on the top-down map.

Unicycle (differential-drive) kinematics in normalized image coordinates:
  d x / dt = v * sin(psi)
  d y / dt = -v * cos(psi)
  d psi / dt = omega
where psi is heading in radians with 0 = North (toward decreasing y), increasing clockwise
— matching server azimuth convention (degrees) used in stitching.

Process noise is a random walk on linear speed v and yaw rate omega (physically motivated
for bounded wheel torques / lane-following, as in typical Duckiebot-style control).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from classes.robot.RobotConfig import RobotConfig
from classes.robot.RobotDetection import RobotDetection
from classes.robot.RobotKalmanParams import RobotKalmanParams

if TYPE_CHECKING:
    from classes.marker.CompositeDetection import CompositeDetection


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _deg_to_rad_same_convention(azimuth_deg: float) -> float:
    """Map API azimuth in degrees [0,360) to radians in (-pi, pi]."""
    r = math.radians(float(azimuth_deg))
    return _wrap_pi(r)


def _rad_to_deg_api(psi_rad: float) -> float:
    deg = math.degrees(_wrap_pi(psi_rad))
    if deg < 0:
        deg += 360.0
    return deg % 360.0


def quad_area_tl_tr_bl_br(
    tl: tuple[float, float],
    tr: tuple[float, float],
    bl: tuple[float, float],
    br: tuple[float, float],
) -> float:
    """Polygon area for corners ordered TL, TR, BR, BL around the perimeter."""
    ordered = np.array([tl, tr, br, bl], dtype=np.float64)
    x = ordered[:, 0]
    y = ordered[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def normalize_tag_geometry(
    *,
    area_px: float,
    baseline_px: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float]:
    """
    Scale-free tag geometry: area as a fraction of the frame, baseline as a
    fraction of the frame diagonal. These are the sole inputs of the sigma model,
    so emitting them is enough to refit its coefficients offline.
    """
    area_norm = max(area_px / float(max(img_w * img_h, 1)), 1e-12)
    diag_px = float(np.hypot(img_w, img_h))
    baseline_norm = max(baseline_px / max(diag_px, 1e-6), 1e-4)
    return area_norm, baseline_norm


def compute_tag_measurement_sigmas(
    *,
    area_px: float,
    baseline_px: float,
    img_w: int,
    img_h: int,
    p: RobotKalmanParams,
) -> tuple[float, float, float]:
    """Return (sigma_x, sigma_y, sigma_psi_rad) for the measurement diagonal R."""
    denom_area, baseline_norm = normalize_tag_geometry(area_px=area_px, baseline_px=baseline_px, img_w=img_w, img_h=img_h)
    sigma_pos = p.position_sigma_from_area_coefficient / math.sqrt(denom_area)
    sigma_pos = min(max(sigma_pos, p.position_sigma_min), p.position_sigma_max)

    sigma_psi_deg = p.azimuth_sigma_from_baseline_coefficient_deg / baseline_norm
    sigma_psi_deg = min(max(sigma_psi_deg, p.azimuth_sigma_min_deg), p.azimuth_sigma_max_deg)
    sigma_psi_rad = math.radians(sigma_psi_deg)
    return sigma_pos, sigma_pos, sigma_psi_rad


@dataclass
class RobotTagMeasurement:
    robot: RobotConfig
    x: float
    y: float
    azimuth_deg: float
    sigma_x: float
    sigma_y: float
    sigma_psi_rad: float
    composite: CompositeDetection | None
    # Calibration diagnostics: the two geometric quantities the sigma model above
    # is a function of, already normalized (area as a fraction of the frame,
    # baseline as a fraction of the frame diagonal). Carried through to the API
    # so calibration can refit the sigma coefficients
    # offline against ground truth. Not used by the filter itself.
    area_norm: float | None = None
    baseline_norm: float | None = None


@dataclass
class _Track:
    x: np.ndarray  # shape (5,)  x, y, psi, v, omega
    P: np.ndarray  # (5,5)
    missed: int
    last_composite: CompositeDetection | None


class RobotKalmanTracker:
    def __init__(self, robots_by_id: dict[int, RobotConfig]) -> None:
        self._robots_by_id = robots_by_id
        self._tracks: dict[int, _Track] = {}
        self.last_innovation: dict[int, tuple[float, float, float]] = {}
        """
        Measurement innovation of the most recent update per robot, (dx, dy, dpsi_rad).
        Whitening is left to the consumer: its lag-1 autocorrelation separates an
        undersized Q from an oversized R, which mean NIS alone cannot. Diagnostic only.
        """
        self.last_nis: dict[int, float] = {}
        """
        Normalized innovation squared of the most recent update per robot, chi-square
        distributed with 3 DoF when Q and R are consistent with the real motion and
        measurement noise. Diagnostic only: read by the offline tuning tool in
        calibration, never by the filter.
        """

    def step(
        self,
        measurements: list[RobotTagMeasurement],
        dt: float,
    ) -> list[RobotDetection]:
        """
        One filtering round. `measurements` contains at most one entry per configured robot
        that was detected this frame. Missing robots are predicted and optionally kept alive
        during the grace period.
        """
        dt = max(float(dt), 1e-4)
        self.last_nis = {}
        self.last_innovation = {}

        meas_by_id: dict[int, RobotTagMeasurement] = {m.robot.id: m for m in measurements}

        for rid, tr in list(self._tracks.items()):
            p = self._robots_by_id[rid].kalman_params
            if not p.enabled:
                del self._tracks[rid]
                continue
            if rid not in meas_by_id:
                self._predict_track(tr, dt, p)
                tr.missed += 1
                tr.last_composite = None
                if tr.missed > p.grace_period_rounds:
                    del self._tracks[rid]

        for m in measurements:
            p = m.robot.kalman_params
            if not p.enabled:
                continue
            rid = m.robot.id
            z = np.array(
                [m.x, m.y, _deg_to_rad_same_convention(m.azimuth_deg)],
                dtype=np.float64,
            )
            R = np.diag([m.sigma_x**2, m.sigma_y**2, m.sigma_psi_rad**2])

            if rid not in self._tracks:
                self._init_track(m.robot, z, R, p, m.composite)
            else:
                tr = self._tracks[rid]
                self._predict_track(tr, dt, p)
                self.last_nis[rid] = self._update_track(tr, z, R, p)
                self.last_innovation[rid] = self._last_nu
                tr.missed = 0
                tr.last_composite = m.composite

        out: list[RobotDetection] = []
        for rid, tr in sorted(self._tracks.items(), key=lambda kv: kv[0]):
            robot = self._robots_by_id[rid]
            p = robot.kalman_params
            if not p.enabled:
                continue
            x, y = float(tr.x[0]), float(tr.x[1])
            azimuth = _rad_to_deg_api(float(tr.x[2]))
            pose_source = "detection" if tr.missed == 0 else "estimation"
            raw_x: float | None = None
            raw_y: float | None = None
            raw_azimuth: float | None = None
            if tr.missed == 0 and rid in meas_by_id:
                m = meas_by_id[rid]
                raw_x, raw_y, raw_azimuth = m.x, m.y, m.azimuth_deg
            det = RobotDetection(
                robot,
                x,
                y,
                azimuth,
                pose_source=pose_source,
                raw_x=raw_x,
                raw_y=raw_y,
                raw_azimuth=raw_azimuth,
            )
            if tr.missed == 0 and rid in meas_by_id:
                det.set_measurement_diagnostics(meas_by_id[rid])
            det.backing_composite = tr.last_composite if tr.missed == 0 else None
            out.append(det)

        return out

    def passthrough(self, measurements: list[RobotTagMeasurement]) -> list[RobotDetection]:
        """When filtering is disabled, emit raw measurements as detections."""
        out: list[RobotDetection] = []
        for m in measurements:
            det = RobotDetection(
                m.robot,
                m.x,
                m.y,
                m.azimuth_deg,
                pose_source="detection",
                raw_x=m.x,
                raw_y=m.y,
                raw_azimuth=m.azimuth_deg,
            )
            det.set_measurement_diagnostics(m)
            det.backing_composite = m.composite
            out.append(det)
        return out

    def _init_track(
        self,
        robot: RobotConfig,
        z: np.ndarray,
        R: np.ndarray,
        p: RobotKalmanParams,
        composite: CompositeDetection | None,
    ) -> None:
        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        x0 = np.array([z[0], z[1], z[2], 0.0, 0.0], dtype=np.float64)
        P0 = np.diag(
            [
                float(R[0, 0]),
                float(R[1, 1]),
                float(R[2, 2]),
                (v_max / 3.0) ** 2,
                (w_max / 3.0) ** 2,
            ]
        )
        self._tracks[robot.id] = _Track(x=x0, P=P0, missed=0, last_composite=composite)

    def _predict_track(self, tr: _Track, dt: float, p: RobotKalmanParams) -> None:
        x, y, psi, v, w = (float(tr.x[i]) for i in range(5))
        s, c = math.sin(psi), math.cos(psi)
        xn = x + v * s * dt
        yn = y - v * c * dt
        psin = _wrap_pi(psi + w * dt)

        F = np.eye(5, dtype=np.float64)
        F[0, 2] = v * c * dt
        F[0, 3] = s * dt
        F[1, 2] = v * s * dt
        F[1, 3] = -c * dt
        F[2, 4] = dt

        sig_v = p.process_noise_linear_velocity_std
        sig_w = math.radians(p.process_noise_angular_rate_std_deg)
        Q = np.zeros((5, 5), dtype=np.float64)
        Q[3, 3] = (sig_v**2) * dt
        Q[4, 4] = (sig_w**2) * dt

        tr.x = np.array([xn, yn, psin, v, w], dtype=np.float64)
        tr.P = F @ tr.P @ F.T + Q
        self._symmetrize(tr.P)

        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        tr.x[3] = float(min(max(tr.x[3], -v_max), v_max))
        tr.x[4] = float(min(max(tr.x[4], -w_max), w_max))

    def _update_track(self, tr: _Track, z: np.ndarray, R: np.ndarray, p: RobotKalmanParams) -> float:
        """Applies the measurement and returns the NIS of this update."""
        H = np.zeros((3, 5), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        x = tr.x
        nu = z - H @ x
        nu[2] = _wrap_pi(float(nu[2]))

        S = H @ tr.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        nis = float(nu @ Sinv @ nu)
        self._last_nu = (float(nu[0]), float(nu[1]), float(nu[2]))

        K = tr.P @ H.T @ Sinv
        tr.x = x + K @ nu
        tr.x[2] = _wrap_pi(float(tr.x[2]))

        I = np.eye(5, dtype=np.float64)
        tr.P = (I - K @ H) @ tr.P
        self._symmetrize(tr.P)

        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        tr.x[3] = float(min(max(tr.x[3], -v_max), v_max))
        tr.x[4] = float(min(max(tr.x[4], -w_max), w_max))
        return nis

    @staticmethod
    def _symmetrize(P: np.ndarray) -> None:
        P[:] = 0.5 * (P + P.T)
