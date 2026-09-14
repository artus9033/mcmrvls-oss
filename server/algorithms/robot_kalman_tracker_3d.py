"""
Extended Kalman filter for 3D robot poses with epipolar geometry.

This extends the original 2D Kalman filter to work with 3D positions from
triangulated AprilTag detections. The state is now 7D: (x, y, z, psi, theta, v, omega)
where:
  - (x, y, z): 3D position
  - psi: yaw angle (heading)
  - theta: pitch angle (for non-planar motion)
  - v: linear velocity
  - omega: yaw rate

For robots constrained to ground plane, z and theta can be kept constant or filtered out.
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


@dataclass
class RobotTagMeasurement3D:
    """3D measurement of a robot AprilTag."""
    robot: RobotConfig
    x: float
    y: float
    z: float
    azimuth_deg: float
    pitch_deg: float  # Pitch angle (0 = horizontal)
    sigma_x: float
    sigma_y: float
    sigma_z: float
    sigma_psi_rad: float
    sigma_theta_rad: float
    composite: CompositeDetection | None


@dataclass
class _Track3D:
    """3D Kalman filter track for a robot."""
    x: np.ndarray  # shape (7,): x, y, z, psi, theta, v, omega
    P: np.ndarray  # (7, 7) covariance
    missed: int
    last_composite: CompositeDetection | None


class RobotKalmanTracker3D:
    """Extended Kalman filter for 3D robot tracking."""
    
    def __init__(self, robots_by_id: dict[int, RobotConfig]) -> None:
        self._robots_by_id = robots_by_id
        self._tracks: dict[int, _Track3D] = {}
    
    def step(
        self,
        measurements: list[RobotTagMeasurement3D],
        dt: float,
    ) -> list[RobotDetection]:
        """
        One filtering round with 3D measurements.
        
        Args:
            measurements: 3D measurements for detected robots
            dt: Time delta since last update
        
        Returns:
            List of RobotDetection with updated poses
        """
        dt = max(float(dt), 1e-4)
        
        meas_by_id: dict[int, RobotTagMeasurement3D] = {m.robot.id: m for m in measurements}
        
        # Predict and age tracks
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
        
        # Update tracks with new measurements
        for m in measurements:
            p = m.robot.kalman_params
            if not p.enabled:
                continue
            rid = m.robot.id
            
            z = np.array(
                [
                    m.x,
                    m.y,
                    m.z,
                    _deg_to_rad_same_convention(m.azimuth_deg),
                    math.radians(m.pitch_deg),
                ],
                dtype=np.float64,
            )
            R = np.diag([
                m.sigma_x**2,
                m.sigma_y**2,
                m.sigma_z**2,
                m.sigma_psi_rad**2,
                m.sigma_theta_rad**2,
            ])
            
            if rid not in self._tracks:
                self._init_track(m.robot, z, R, p, m.composite)
            else:
                tr = self._tracks[rid]
                self._predict_track(tr, dt, p)
                self._update_track(tr, z, R, p)
                tr.missed = 0
                tr.last_composite = m.composite
        
        # Generate output
        out: list[RobotDetection] = []
        for rid, tr in sorted(self._tracks.items(), key=lambda kv: kv[0]):
            robot = self._robots_by_id[rid]
            p = robot.kalman_params
            if not p.enabled:
                continue
            
            # For compatibility, project to 2D (x, y, azimuth)
            x, y = float(tr.x[0]), float(tr.x[1])
            azimuth = _rad_to_deg_api(float(tr.x[3]))
            
            source = "detection" if tr.missed == 0 else "estimation"
            
            out.append(
                RobotDetection(
                    robot=robot,
                    x=x,
                    y=y,
                    azimuth=azimuth,
                    poseSource=source,
                )
            )
        
        return out
    
    def _init_track(
        self,
        robot: RobotConfig,
        z: np.ndarray,
        R: np.ndarray,
        p: RobotKalmanParams,
        composite: CompositeDetection | None,
    ) -> None:
        """Initialize a new track from first measurement."""
        x, y, z_pos, psi, theta = z
        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        
        x0 = np.array([x, y, z_pos, psi, theta, 0.0, 0.0], dtype=np.float64)
        P0 = np.diag(
            [
                float(R[0, 0]),
                float(R[1, 1]),
                float(R[2, 2]),
                float(R[3, 3]),
                float(R[4, 4]),
                (v_max / 3.0) ** 2,
                (w_max / 3.0) ** 2,
            ]
        )
        self._tracks[robot.id] = _Track3D(x=x0, P=P0, missed=0, last_composite=composite)
    
    def _predict_track(self, tr: _Track3D, dt: float, p: RobotKalmanParams) -> None:
        """Predict track forward in time using motion model."""
        x, y, z, psi, theta, v, w = (float(tr.x[i]) for i in range(7))
        
        # 3D unicycle model (moving on a potentially tilted plane)
        s_psi, c_psi = math.sin(psi), math.cos(psi)
        c_theta = math.cos(theta)
        
        # Update position in 3D
        xn = x + v * s_psi * c_theta * dt
        yn = y - v * c_psi * c_theta * dt
        zn = z + v * math.sin(theta) * dt  # Height change due to pitch
        psin = _wrap_pi(psi + w * dt)
        thetan = theta  # Assume pitch doesn't change in prediction
        
        # Jacobian of motion model
        F = np.eye(7, dtype=np.float64)
        F[0, 3] = v * c_psi * c_theta * dt  # dx/dpsi
        F[0, 4] = -v * s_psi * math.sin(theta) * dt  # dx/dtheta
        F[0, 5] = s_psi * c_theta * dt  # dx/dv
        F[1, 3] = v * s_psi * c_theta * dt  # dy/dpsi
        F[1, 4] = v * c_psi * math.sin(theta) * dt  # dy/dtheta
        F[1, 5] = -c_psi * c_theta * dt  # dy/dv
        F[2, 4] = v * c_theta * dt  # dz/dtheta
        F[2, 5] = math.sin(theta) * dt  # dz/dv
        F[3, 6] = dt  # dpsi/domega
        
        # Process noise (random walk on v and omega)
        sig_v = p.process_noise_linear_velocity_std
        sig_w = math.radians(p.process_noise_angular_rate_std_deg)
        Q = np.zeros((7, 7), dtype=np.float64)
        Q[5, 5] = (sig_v**2) * dt
        Q[6, 6] = (sig_w**2) * dt
        
        tr.x = np.array([xn, yn, zn, psin, thetan, v, w], dtype=np.float64)
        tr.P = F @ tr.P @ F.T + Q
        self._symmetrize(tr.P)
        
        # Velocity constraints
        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        tr.x[5] = float(min(max(tr.x[5], -v_max), v_max))
        tr.x[6] = float(min(max(tr.x[6], -w_max), w_max))
    
    def _update_track(
        self,
        tr: _Track3D,
        z: np.ndarray,
        R: np.ndarray,
        p: RobotKalmanParams,
    ) -> None:
        """Update track with new measurement."""
        # Measurement model: H = [I_5x5 | 0_5x2] (observe x, y, z, psi, theta)
        H = np.zeros((5, 7), dtype=np.float64)
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        H[3, 3] = 1.0  # psi
        H[4, 4] = 1.0  # theta
        
        x = tr.x
        nu = z - H @ x
        nu[3] = _wrap_pi(float(nu[3]))  # Wrap angle difference
        nu[4] = _wrap_pi(float(nu[4]))
        
        # Innovation covariance
        S = H @ tr.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        
        # Kalman gain
        K = tr.P @ H.T @ Sinv
        tr.x = x + K @ nu
        tr.x[3] = _wrap_pi(float(tr.x[3]))
        tr.x[4] = _wrap_pi(float(tr.x[4]))
        
        # Update covariance
        I = np.eye(7, dtype=np.float64)
        tr.P = (I - K @ H) @ tr.P
        self._symmetrize(tr.P)
        
        # Velocity constraints
        v_max = p.max_speed_norm_per_sec
        w_max = math.radians(p.max_yaw_rate_deg_per_sec)
        tr.x[5] = float(min(max(tr.x[5], -v_max), v_max))
        tr.x[6] = float(min(max(tr.x[6], -w_max), w_max))
    
    @staticmethod
    def _symmetrize(P: np.ndarray) -> None:
        """Enforce covariance symmetry."""
        P[:] = 0.5 * (P + P.T)
