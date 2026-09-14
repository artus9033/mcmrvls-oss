from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from utils.lang import ensureNumericPrimitive

from .RobotConfig import RobotConfig

if TYPE_CHECKING:
    from algorithms.robot_kalman_tracker import RobotTagMeasurement
    from classes.marker.CompositeDetection import CompositeDetection

PoseSource = Literal["detection", "estimation"]


class RobotDetection:
    robot: RobotConfig
    x: float
    y: float
    azimuth: float
    raw_x: float | None
    raw_y: float | None
    raw_azimuth: float | None
    pose_source: PoseSource
    """Set on detections from the Kalman pipeline; used only for top-down annotation (not in API)."""
    backing_composite: CompositeDetection | None
    """
    Diagnostics of the tag measurement that produced this pose: the R diagonal the
    filter used, and the normalized tag geometry it was derived from. Present only
    on live detections (not on coasted estimates). Consumed by
    calibration to recalibrate the sigma model offline.
    """
    meas_sigma_x: float | None
    meas_sigma_y: float | None
    meas_sigma_psi_deg: float | None
    meas_area_norm: float | None
    meas_baseline_norm: float | None

    def __init__(
        self,
        robot: RobotConfig,
        x: float,
        y: float,
        azimuth: float,
        *,
        pose_source: PoseSource = "detection",
        raw_x: float | None = None,
        raw_y: float | None = None,
        raw_azimuth: float | None = None,
    ) -> None:
        self.robot = robot
        self.x = x
        self.y = y
        self.azimuth = azimuth
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.raw_azimuth = raw_azimuth
        self.pose_source = pose_source
        self.backing_composite = None
        self.meas_sigma_x = None
        self.meas_sigma_y = None
        self.meas_sigma_psi_deg = None
        self.meas_area_norm = None
        self.meas_baseline_norm = None

    def set_measurement_diagnostics(self, measurement: RobotTagMeasurement) -> None:
        self.meas_sigma_x = measurement.sigma_x
        self.meas_sigma_y = measurement.sigma_y
        self.meas_sigma_psi_deg = math.degrees(measurement.sigma_psi_rad)
        self.meas_area_norm = measurement.area_norm
        self.meas_baseline_norm = measurement.baseline_norm

    def toDTO(self, include_robot_info: bool, azimuthOffsetDeg: float = 0):
        dto: dict = {
            **({"robot": self.robot.toDTO()} if include_robot_info else {}),
            "x": ensureNumericPrimitive(self.x),
            "y": ensureNumericPrimitive(self.y),
            "azimuth": ensureNumericPrimitive(self.azimuth + azimuthOffsetDeg),
            "poseSource": self.pose_source,
        }
        if self.raw_x is not None:
            dto["raw_x"] = ensureNumericPrimitive(self.raw_x)
        if self.raw_y is not None:
            dto["raw_y"] = ensureNumericPrimitive(self.raw_y)
        if self.raw_azimuth is not None:
            dto["raw_azimuth"] = ensureNumericPrimitive(self.raw_azimuth + azimuthOffsetDeg)
        for key, value in (
            ("meas_sigma_x", self.meas_sigma_x),
            ("meas_sigma_y", self.meas_sigma_y),
            ("meas_sigma_psi_deg", self.meas_sigma_psi_deg),
            ("meas_area_norm", self.meas_area_norm),
            ("meas_baseline_norm", self.meas_baseline_norm),
        ):
            if value is not None:
                dto[key] = ensureNumericPrimitive(value)
        return dto
