"""
Common robot measurement resolution + Kalman tracking, shared by every
measurement path (stitched-mosaic, epipolar, per-camera consensus /
orthorectified). Applies the per-camera consensus substitution and runs the
unicycle EKF (or passthrough), independent of which top-down strategy produced
the base measurements.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from algorithms.robot_kalman_tracker import RobotTagMeasurement
from classes.robot import RobotDetection
from utils.tracing import Tracing

if TYPE_CHECKING:
    from classes.config import AlgorithmConfig
    from classes.resources.AlgorithmState import AlgorithmState
    from classes.resources.InterThreadMemory import InterThreadMemory
    from classes.resources.PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder


def resolve_robot_detections(
    robot_measurements: list[RobotTagMeasurement],
    state: "AlgorithmState",
    config: "AlgorithmConfig",
    resultsHolder: "PreallocatedAlgorithmResultsHolder",
    memory: "InterThreadMemory | None" = None,
) -> list[RobotDetection]:
    """Base measurements -> tracked robot detections (all solver paths).

    1. Per-camera consensus substitution: when usePerCameraSolver or the
       orthorectified strategy is active, atlas-consensus measurements replace
       the base solver's measurement per robot (base kept as fallback for
       robots the consensus could not see this round).
    2. Kalman: the unicycle EKF steps enabled robots (coasting through
       dropouts); disabled robots pass through unfiltered.
    """
    if config.usePerCameraSolver or config.topDownStrategy == "orthorectified":
        percamMeasurements = getattr(state, "percamRobotMeasurements", None) or {}
        if percamMeasurements:
            measurementsByRobot = {m.robot.id: m for m in robot_measurements}
            measurementsByRobot.update(percamMeasurements)
            robot_measurements = list(measurementsByRobot.values())
            Tracing.message(
                f"Per-camera solver supplied {len(percamMeasurements)} robot measurement(s)"
            )

    fps_val = resultsHolder.fps
    if not math.isfinite(fps_val) or fps_val <= 0:
        kalman_dt = 1.0 / 30.0
    else:
        kalman_dt = 1.0 / float(fps_val)

    robotDetections: list[RobotDetection]
    if memory is not None:
        kalman_active = any(r.kalman_params.enabled for r in config.configuredRobots.values())
        if kalman_active:
            enabled_meas = [m for m in robot_measurements if m.robot.kalman_params.enabled]
            disabled_meas = [m for m in robot_measurements if not m.robot.kalman_params.enabled]
            robotDetections = memory.robot_kalman_tracker.step(enabled_meas, kalman_dt)
            robotDetections.extend(memory.robot_kalman_tracker.passthrough(disabled_meas))
            robotDetections.sort(key=lambda d: d.robot.id)
        else:
            robotDetections = memory.robot_kalman_tracker.passthrough(robot_measurements)
    else:
        robotDetections = []
        for m in robot_measurements:
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
            det.backing_composite = m.composite
            robotDetections.append(det)
    return robotDetections
