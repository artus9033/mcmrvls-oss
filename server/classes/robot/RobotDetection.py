from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from utils.lang import ensureNumericPrimitive

from .RobotConfig import RobotConfig

if TYPE_CHECKING:
    from classes.marker.CompositeDetection import CompositeDetection

PoseSource = Literal["detection", "estimation"]


class RobotDetection:
    robot: RobotConfig
    x: float
    y: float
    azimuth: float
    pose_source: PoseSource
    """Set on detections from the Kalman pipeline; used only for top-down annotation (not in API)."""
    backing_composite: CompositeDetection | None

    def __init__(
        self,
        robot: RobotConfig,
        x: float,
        y: float,
        azimuth: float,
        *,
        pose_source: PoseSource = "detection",
    ) -> None:
        self.robot = robot
        self.x = x
        self.y = y
        self.azimuth = azimuth
        self.pose_source = pose_source
        self.backing_composite = None

    def toDTO(self, include_robot_info: bool, azimuthOffsetDeg: float = 0):
        return {
            **({"robot": self.robot.toDTO()} if include_robot_info else {}),
            "x": ensureNumericPrimitive(self.x),
            "y": ensureNumericPrimitive(self.y),
            "azimuth": ensureNumericPrimitive(self.azimuth + azimuthOffsetDeg),
            "poseSource": self.pose_source,
        }
