from abc import ABC
from typing import List, NamedTuple, Optional, Tuple

from threading import RLock

import numpy as np
from scipy.spatial.transform import Rotation


class PlannerLoggerProvider(ABC):
    @staticmethod
    def loginfo(msg): ...

    @staticmethod
    def logdebug(msg): ...

    @staticmethod
    def logwarn(msg): ...

    @staticmethod
    def logerr(msg): ...


class PlannerPosition(NamedTuple):
    x: float
    y: float


class PlannerQuaternion(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class PlannerPose:
    def __init__(self, position: PlannerPosition, quaternion: PlannerQuaternion):
        self.position = position
        self.quaternion = quaternion

    def __str__(self):
        return f"({self.position.x:.2f}, {self.position.y:.2f}, z: {np.rad2deg(Rotation.from_quat((self.quaternion.x, self.quaternion.y, self.quaternion.z, self.quaternion.w)).as_euler('xyz')[2]):.2f})"

    def __repr__(self):
        return self.__str__()


PlannerPath = List[PlannerPose]


def group_close_poses(
    path: PlannerPath, closeDistanceThreshold: float, singleGroupMaxLength: float
) -> PlannerPath:
    """
    Groups consecutive poses if:
    - distance to previous < closeDistanceThreshold, AND
    - distance from first in cluster < singleGroupMaxLength.

    Each group is replaced with its literal middle pose.
    """
    if not path:
        return []

    result: PlannerPath = []
    cluster: List[PlannerPose] = [path[0]]

    def pick_middle_pose(cluster: List[PlannerPose]) -> PlannerPose:
        mid_index = len(cluster) // 2
        return cluster[mid_index]

    for pose in path[1:]:
        prev = cluster[-1]
        first = cluster[0]

        # distance to previous
        dx_prev = pose.position.x - prev.position.x
        dy_prev = pose.position.y - prev.position.y
        close = np.hypot(dx_prev, dy_prev) < closeDistanceThreshold

        # span distance from first
        dx_span = pose.position.x - first.position.x
        dy_span = pose.position.y - first.position.y
        span_ok = np.hypot(dx_span, dy_span) < singleGroupMaxLength

        if close and span_ok:
            cluster.append(pose)
        else:
            result.append(pick_middle_pose(cluster))
            cluster = [pose]

    # finalize last cluster
    if cluster:
        result.append(pick_middle_pose(cluster))

    return result


class DiffDriveLocalPlanner:
    path: Optional[PlannerPath] = None
    currentPathIndex: int = 0
    wasPositionReached: bool = False
    isRotatingToNextGoal: bool = False
    isPathDirty: bool = True
    setNewPathOrientationImputationInvertDirections: bool = False

    def __init__(
        self,
        posTolerance: float,
        finalWaypointPosTolerance: float,
        logger: PlannerLoggerProvider,
        angleTolerance=np.deg2rad(5),
        vLimit=0.5,
        omegaLimit=2.0,
        trajectoryAdherenceFactor=1.0,
        crossTrackErrorMinThreshold=0.02,
        lookAheadIndicesLength=3,
        angularErrorThresholdAboveWhichToStopAndAdjustOrientation=np.deg2rad(10),
        groupingThreshold=None,
        groupLengthThreshold=None,
        disableOrientationMatching=False,
        minWaypointDistance=0.2,
        setNewPathOrientationImputationInvertDirections=False,
    ):
        """
        path: trajectory of poses to follow
        posTolerance: distance tolerance to consider waypoint reached
        finalWaypointPosTolerance: distance tolerance to consider final waypoint reached
        logger: logger instance
        angleTolerance: angular tolerance (rad)
        vLimit: maximum linear velocity
        omegaLimit: maximum angular velocity
        trajectoryAdherenceFactor: weight factor for trajectory adherence (0.0 = ignore trajectory, higher = stronger adherence)
        crossTrackErrorMinThreshold: minimum cross-track error to start reducing speed
        lookAheadIndicesLength: number of waypoints to look ahead when selecting optimal waypoint
        angularErrorThresholdAboveWhichToStopAndAdjustOrientation: angular error threshold (rad) above which to stop and adjust orientation at waypoint
        groupingThreshold: distance threshold to group close waypoints together
        disableOrientationMatching: if True, robot will drive through waypoints without stopping to match orientation
        minWaypointDistance: minimum distance from robot to waypoint to consider it for targeting
        setNewPathOrientationImputationInvertDirections: if True, the imputed path poses orientations are inverted (useful for robots with reversed coordinate systems)
        """
        self.pathLock = RLock()

        self.posTolerance = posTolerance
        self.finalWaypointPosTolerance = finalWaypointPosTolerance
        self.logger = logger
        self.angleTolerance = angleTolerance
        self.vLimit = vLimit
        self.omegaLimit = omegaLimit
        self.trajectoryAdherenceFactor = trajectoryAdherenceFactor
        self.crossTrackErrorMinThreshold = crossTrackErrorMinThreshold
        self.lookAheadIndicesLength = lookAheadIndicesLength
        self.angularErrorThresholdAboveWhichToStopAndAdjustOrientation = (
            angularErrorThresholdAboveWhichToStopAndAdjustOrientation
        )
        self.groupingThreshold = groupingThreshold
        self.groupLengthThreshold = groupLengthThreshold
        self.disableOrientationMatching = disableOrientationMatching
        self.minWaypointDistance = minWaypointDistance
        self.setNewPathOrientationImputationInvertDirections = (
            setNewPathOrientationImputationInvertDirections
        )

        self.reset()

    def reset(self):
        with self.pathLock:
            self.path = None
            self.currentPathIndex = 0
            self.wasPositionReached = False
            self.isRotatingToNextGoal = False
            self.isPathDirty = True

    def _find_nearest_waypoint_index(
        self, robotPosition: PlannerPosition, startFromIndex: int = 0
    ) -> int:
        """Find the nearest waypoint index in the path starting from startFromIndex (forward-only)."""
        if not self.path or len(self.path) == 0:
            return 0

        robot_pos = np.array(robotPosition)
        min_distance = float("inf")
        nearest_index = startFromIndex

        # First pass: search for waypoints that meet minimum distance requirement
        for i in range(startFromIndex, len(self.path)):
            waypoint_pos = np.array(self.path[i].position)
            distance = float(np.linalg.norm(robot_pos - waypoint_pos))

            # Prefer waypoints that are at least minWaypointDistance units away
            if distance >= self.minWaypointDistance and distance < min_distance:
                min_distance = distance
                nearest_index = i

        # If no waypoint meets the minimum distance requirement, find the furthest one available
        if min_distance == float("inf"):
            max_distance = 0.0
            for i in range(startFromIndex, len(self.path)):
                waypoint_pos = np.array(self.path[i].position)
                distance = float(np.linalg.norm(robot_pos - waypoint_pos))

                if distance > max_distance:
                    max_distance = distance
                    nearest_index = i

        return nearest_index

    def _find_optimal_forward_waypoint(
        self, robotPosition: PlannerPosition, startFromIndex: int = 0
    ) -> int:
        """Find the optimal waypoint to target, considering both distance and path progress."""
        if not self.path or len(self.path) == 0:
            return 0

        robot_pos = np.array(robotPosition)

        # Look ahead window - consider a few waypoints ahead to avoid local minima
        look_ahead_window = min(
            self.lookAheadIndicesLength, len(self.path) - startFromIndex
        )

        best_index = startFromIndex
        best_score = float("inf")

        # find waypoints that meet minimum distance requirement
        for i in range(
            startFromIndex, min(startFromIndex + look_ahead_window, len(self.path))
        ):
            waypoint_pos = np.array(self.path[i].position)
            distance = float(np.linalg.norm(robot_pos - waypoint_pos))

            # Skip waypoints that are too close (less than minWaypointDistance units away)
            if distance < self.minWaypointDistance:
                continue

            # Score combines distance penalty with progress reward
            # Encourage forward progress along the path
            progress_bonus = max(
                self.posTolerance * 3, (-(i - startFromIndex) * 0.2)
            )  # Small bonus for further waypoints
            score = distance + progress_bonus

            if score < best_score:
                best_score = score
                best_index = i

        return best_index

    def setNewPath(self, path: PlannerPath, imputeOrientations: bool = False):
        """Set a new path and optionally start from the nearest waypoint to the robot's current position."""

        with self.pathLock:
            self.reset()

            if imputeOrientations:
                prevPos: Optional[PlannerPosition] = None
                for i, pose in enumerate(path):
                    if len(path) > 1:
                        # if no next, use previous
                        maybeNextPos = (
                            path[i + 1].position if i + 1 < len(path) else None
                        )
                        angle = (
                            np.arctan2(
                                pose.position.x - maybeNextPos.x,
                                pose.position.y - maybeNextPos.y,
                            )
                            if maybeNextPos
                            else np.arctan2(
                                prevPos.x - pose.position.x,
                                prevPos.y - pose.position.y,
                            )
                        )
                        # normalize the angle
                        angle = (angle) % (2 * np.pi)

                        if self.setNewPathOrientationImputationInvertDirections:
                            angle = (angle + np.pi) % (2 * np.pi)

                        rawQuatArr = Rotation.from_euler(
                            "xyz",
                            [
                                0,
                                0,
                                angle,
                            ],
                            degrees=False,
                        ).as_quat()

                        orientation = PlannerQuaternion(
                            x=rawQuatArr[0],
                            y=rawQuatArr[1],
                            z=rawQuatArr[2],
                            w=rawQuatArr[3],
                        )
                    else:
                        # if last, just no change in orientation
                        orientation = PlannerQuaternion(
                            0,
                            0,
                            0,
                            1,
                        )

                    # tuples are immutable so reassign a new instance (PlannerPose is a NamedTuple)
                    path[i] = PlannerPose(pose.position, orientation)

                    prevPos = pose.position

            if self.groupingThreshold and self.groupLengthThreshold:
                path = group_close_poses(
                    path,
                    closeDistanceThreshold=self.groupingThreshold,
                    singleGroupMaxLength=self.groupLengthThreshold,
                )

            self.path = path

            return path

    def _calculate_cross_track_error(self, robotPosition: PlannerPosition) -> float:
        """
        Calculate the cross-track error (perpendicular distance from robot to path segment)
        """
        if not self.path or self.currentPathIndex >= len(self.path):
            return 0.0

        # Get current and previous waypoints to define the path segment
        if self.currentPathIndex == 0:
            # For the first waypoint, use it as both start and end
            p1 = np.array(self.path[0].position)
            p2 = np.array(self.path[0].position)
        else:
            p1 = np.array(self.path[self.currentPathIndex - 1].position)
            p2 = np.array(self.path[self.currentPathIndex].position)

        robot_pos = np.array(robotPosition)

        # If p1 and p2 are the same (first waypoint case), return distance to point
        if np.allclose(p1, p2):
            return float(np.linalg.norm(robot_pos - p1))

        # Calculate cross-track error using point-to-line distance formula
        # Vector from p1 to p2
        path_vector = p2 - p1
        # Vector from p1 to robot
        robot_vector = robot_pos - p1

        # Project robot vector onto path vector
        path_length_sq = np.dot(path_vector, path_vector)
        if path_length_sq < 1e-10:  # Avoid division by zero
            return float(np.linalg.norm(robot_vector))

        t = np.dot(robot_vector, path_vector) / path_length_sq
        t = np.clip(t, 0.0, 1.0)  # Clamp to segment

        # Find closest point on segment
        closest_point = p1 + t * path_vector

        # Return perpendicular distance
        return float(np.linalg.norm(robot_pos - closest_point))

    def _normalize_angle_error(
        self, target_angle: float, current_angle: float
    ) -> float:
        """
        Calculate normalized angle error between target and current angle.
        Returns the shortest angular distance between the two angles.
        """
        return np.arctan2(
            np.sin(target_angle - current_angle), np.cos(target_angle - current_angle)
        )

    def _get_robot_yaw(self, robot_quaternion: PlannerQuaternion) -> float:
        """
        Extract yaw angle from robot quaternion.
        """
        return Rotation.from_quat(robot_quaternion).as_euler("xyz", degrees=False)[2]

    def _get_target_yaw(self, target_quaternion: PlannerQuaternion) -> float:
        """
        Extract yaw angle from target quaternion.
        """
        return Rotation.from_quat(target_quaternion).as_euler("xyz", degrees=False)[2]

    def _calculate_proportional_control(self, angle_error: float) -> float:
        """
        Apply proportional control with P-gain and clamp to omega limits.
        """
        omega = 2.0 * angle_error
        return np.clip(omega, -self.omegaLimit, self.omegaLimit)

    def _calculate_cross_track_correction(
        self, robotPosition: PlannerPosition, robotYaw: float, angle_to_target: float
    ) -> float:
        """
        Calculate cross-track correction angle for trajectory adherence.
        """
        cross_track_error = self._calculate_cross_track_error(robotPosition)

        if (
            self.trajectoryAdherenceFactor <= 0.0
            or cross_track_error <= 0.01
            or not self.path
        ):
            return 0.0

        # Find the direction to the path (perpendicular to path segment)
        if self.currentPathIndex == 0:
            # For first waypoint, correction points toward the waypoint
            correction_direction = angle_to_target
        else:
            # Get path segment direction
            prev_pos = np.array(self.path[self.currentPathIndex - 1].position)
            curr_pos = np.array(self.path[self.currentPathIndex].position)
            path_vector = curr_pos - prev_pos

            if np.linalg.norm(path_vector) > 1e-10:
                # Find closest point on path segment
                robotPos = np.array(robotPosition)
                robot_to_prev = robotPos - prev_pos
                path_length_sq = np.dot(path_vector, path_vector)
                t = np.clip(
                    np.dot(robot_to_prev, path_vector) / path_length_sq,
                    0.0,
                    1.0,
                )
                closest_point = prev_pos + t * path_vector

                # Direction from robot to closest point on path
                to_path_vector = closest_point - robotPos
                if np.linalg.norm(to_path_vector) > 1e-10:
                    correction_direction = np.arctan2(
                        to_path_vector[1], to_path_vector[0]
                    )
                else:
                    # Path direction angle as fallback
                    correction_direction = np.arctan2(path_vector[1], path_vector[0])
            else:
                correction_direction = angle_to_target

        # Calculate correction angle error
        correction_angle_error = self._normalize_angle_error(
            correction_direction, robotYaw
        )

        # Scale correction by cross-track error and adherence factor
        return (
            self.trajectoryAdherenceFactor
            * correction_angle_error
            * np.tanh(cross_track_error)
        )

    def _step(
        self,
        robotPosition: Optional[PlannerPosition],
        robotQuaternion: Optional[PlannerQuaternion],
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """
        Perform one control iteration.
        robot_pos: (x, y)
        robot_quat: (qx, qy, qz, qw)
        returns: (v, omega, dist, theta, currentPathIndex) or None if finished
        """

        with self.pathLock:
            if not self.path or not robotPosition or not robotQuaternion:
                return None

            if self.isPathDirty:
                self.isPathDirty = False

                self.currentPathIndex = self._find_nearest_waypoint_index(robotPosition)
                self.logger.loginfo(
                    f"[DiffDriveLocalPlanner] Starting from nearest waypoint #{self.currentPathIndex} "
                    f"at position ({self.path[self.currentPathIndex].position.x:.2f}, "
                    f"{self.path[self.currentPathIndex].position.y:.2f})"
                )

            if self.currentPathIndex >= len(self.path):
                self.logger.logdebug(
                    "[DiffDriveLocalPlanner] Not driving anymore, all points on trajectory have been exhausted"
                )
                self.reset()
                return None

            isCurrentTargetTheLastWaypoint = self.currentPathIndex < len(self.path) - 1

            # Check if there's a better waypoint ahead in the path using forward-only constraint
            if isCurrentTargetTheLastWaypoint:
                optimal_index = self._find_optimal_forward_waypoint(
                    robotPosition, self.currentPathIndex
                )
                if optimal_index > self.currentPathIndex:
                    self.logger.loginfo(
                        f"[DiffDriveLocalPlanner] Chosen new target waypoint: moving from index "
                        f"{self.currentPathIndex} to {optimal_index}"
                    )
                    self.currentPathIndex = optimal_index
                    self.wasPositionReached = False
                    self.isRotatingToNextGoal = False

            target = self.path[self.currentPathIndex]
            targetPosition = np.array(target.position)

            # distance to target
            dist = np.linalg.norm(targetPosition - np.array(robotPosition))

            positionReached = (
                dist
                <= (
                    self.posTolerance
                    if isCurrentTargetTheLastWaypoint
                    else self.finalWaypointPosTolerance
                )
            ).__bool__()
            try:
                robotYaw = self._get_robot_yaw(robotQuaternion)

                # Calculate yaw error based on whether orientation matching is disabled
                if self.disableOrientationMatching:
                    # Point towards target position (azimuth)
                    robotPos = np.array(robotPosition)
                    angle_to_target = np.arctan2(
                        targetPosition[1] - robotPos[1], targetPosition[0] - robotPos[0]
                    )
                    yawError = self._normalize_angle_error(angle_to_target, robotYaw)
                else:
                    # Match target's quaternion orientation
                    targetYaw = self._get_target_yaw(target.quaternion)
                    yawError = self._normalize_angle_error(targetYaw, robotYaw)
                    print(
                        f"Target {np.rad2deg(targetYaw)}, robot {np.rad2deg(robotYaw)}"
                    )

                # only stop and adjust orientation before approaching the next waypoint, if the angular difference is bigger than 35 degrees
                # and orientation matching is not disabled
                shouldStopToAdjustOrientation = not self.disableOrientationMatching and (
                    self.currentPathIndex >= len(self.path) - 1  # either last waypoint
                    # or threshold exceeded
                    or (
                        abs(yawError)
                        > self.angularErrorThresholdAboveWhichToStopAndAdjustOrientation
                    )
                )

                # if we reached position, check orientation now
                if positionReached:
                    if not self.wasPositionReached and shouldStopToAdjustOrientation:
                        self.logger.loginfo(
                            f"[DiffDriveLocalPlanner] Reached waypoint position {targetPosition} (distance: {dist:.2f}), now adjusting orientation."
                        )

                    if (
                        abs(yawError) < self.angleTolerance
                        or not shouldStopToAdjustOrientation
                        or self.disableOrientationMatching
                    ):
                        # we reached the current waypoint: both position & orientation
                        self.currentPathIndex = self._find_optimal_forward_waypoint(
                            robotPosition,
                            self.currentPathIndex + 1,
                        )

                        self.logger.loginfo(
                            f"[DiffDriveLocalPlanner] Reached waypoint #{self.currentPathIndex} at ({targetPosition[0]:.2f}, {targetPosition[1]:.2f}) with distance {dist:.2f} and yaw error {yawError:.2f}, now moving to the next one."
                        )

                        if self.currentPathIndex >= len(self.path):
                            self.logger.loginfo(
                                "[DiffDriveLocalPlanner] Journey finished successfully!"
                            )
                            self.reset()
                            return None
                        else:
                            if shouldStopToAdjustOrientation:
                                # Set flag to rotate to next goal before moving
                                self.isRotatingToNextGoal = True
                                self.logger.loginfo(
                                    "[DiffDriveLocalPlanner] Now rotating to face the next goal."
                                )

                        # Calculate angle to next waypoint
                        target = self.path[self.currentPathIndex]
                        targetPosition = np.array(target.position)

                        # Robot heading and angle to next target
                        robotYaw = self._get_robot_yaw(robotQuaternion)

                        # Calculate yaw error for next waypoint based on orientation matching setting
                        if self.disableOrientationMatching:
                            # Point towards next target position (azimuth)
                            robotPos = np.array(robotPosition)
                            angle_to_next_target = np.arctan2(
                                targetPosition[1] - robotPos[1],
                                targetPosition[0] - robotPos[0],
                            )
                            yawError = self._normalize_angle_error(
                                angle_to_next_target, robotYaw
                            )
                        else:
                            # Match next target's quaternion orientation
                            targetYaw = self._get_target_yaw(target.quaternion)
                            yawError = self._normalize_angle_error(targetYaw, robotYaw)

                    if shouldStopToAdjustOrientation:
                        # if position reached but not orientation, rotate in place
                        omega = self._calculate_proportional_control(yawError)
                        return (
                            0.0,
                            omega,
                            float(dist),
                            yawError,
                            self.currentPathIndex,
                        )

                # Check if we need to rotate to face the next goal
                elif self.isRotatingToNextGoal and self.currentPathIndex < len(
                    self.path
                ):
                    if shouldStopToAdjustOrientation:
                        # Check if we're oriented correctly toward next goal
                        if abs(yawError) < self.angleTolerance:
                            self.isRotatingToNextGoal = False
                            self.wasPositionReached = False
                            self.logger.loginfo(
                                "[DiffDriveLocalPlanner] Finished rotating to next goal, now moving on."
                            )
                        else:
                            # Rotate in place to face next goal
                            omega = self._calculate_proportional_control(yawError)
                            return (
                                0.0,
                                omega,
                                float(dist),
                                yawError,
                                self.currentPathIndex,
                            )

                # Normal movement toward current target (only if not in special rotation state)
                if not self.isRotatingToNextGoal:
                    # Get angle to target for cross-track correction
                    robotPos = np.array(robotPosition)
                    if self.disableOrientationMatching:
                        # Use azimuth to target position
                        angle_to_target = np.arctan2(
                            targetPosition[1] - robotPos[1],
                            targetPosition[0] - robotPos[0],
                        )

                        # correct the azimuth by taking into account robot's yaw
                        angle_to_target = self._normalize_angle_error(
                            angle_to_target, robotYaw
                        )
                    else:
                        # Use target's quaternion orientation
                        angle_to_target = self._get_target_yaw(target.quaternion)
                        # take into account robot's yaw
                        angle_to_target = self._normalize_angle_error(
                            angle_to_target, robotYaw
                        )

                    # calculate cross-track correction for trajectory adherence
                    cross_track_correction = self._calculate_cross_track_correction(
                        robotPosition, robotYaw, angle_to_target
                    )

                    # Combine target-seeking and trajectory adherence
                    total_angle_error = yawError + cross_track_correction

                    # simple proportional control with trajectory adherence
                    v = self.vLimit * np.clip(dist, 0.2, 1.0)  # slow down near target
                    omega = self._calculate_proportional_control(total_angle_error)

                    # reduce speed when far from trajectory
                    cross_track_error = self._calculate_cross_track_error(robotPosition)
                    if (
                        self.trajectoryAdherenceFactor > 0.0
                        and cross_track_error > self.crossTrackErrorMinThreshold
                    ):
                        speed_reduction = np.exp(
                            -cross_track_error * self.trajectoryAdherenceFactor
                        )
                        v *= max(0.3, speed_reduction)  # don't reduce speed below 30%
                    return float(v), omega, float(dist), yawError, self.currentPathIndex

                # if rotating to next goal, don't move but update position state
                else:
                    return (
                        0.0,
                        0.0,  # stop while rotating to next goal
                        float(dist),
                        0.0,  # stop while rotating to next goal
                        self.currentPathIndex,
                    )
            finally:
                self.wasPositionReached = positionReached

    def step(
        self,
        robotPosition: Optional[PlannerPosition],
        robotQuaternion: Optional[PlannerQuaternion],
    ) -> Optional[Tuple[float, float, float, float, float]]:
        control = self._step(robotPosition, robotQuaternion)

        if control is None:
            return control
        else:
            v, omega, dist, theta, currentPathIndex = control

            # constrain a full range angle to [-pi, pi] (-90, 90)
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # v = v * (1 - min(1.0, abs(omega) / self.omegaLimit))
            v = dist * np.cos(theta)
            v = np.clip(v, -self.vLimit, self.vLimit)
            omega = np.clip(omega, -self.omegaLimit, self.omegaLimit)

            return (v, omega, dist, theta, currentPathIndex)
