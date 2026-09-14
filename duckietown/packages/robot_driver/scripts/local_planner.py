#!/usr/bin/env python3


from time import sleep
from typing import Optional

import numpy as np
import rospy
from DiffDriveLocalPlanner import (
    DiffDriveLocalPlanner,
    PlannerLoggerProvider,
    PlannerPose,
    PlannerPosition,
    PlannerQuaternion,
    PlannerPath,
)
from duckietown_msgs.msg import Twist2DStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from duckietown.dtros import DTROS, NodeType

# constant: control to stop the wheels
MSG_STOP_TWIST2DSTAMPED = Twist2DStamped()
MSG_STOP_TWIST2DSTAMPED.v = 0
MSG_STOP_TWIST2DSTAMPED.omega = 0


class RospyLoggerProvider(PlannerLoggerProvider):
    @staticmethod
    def loginfo(msg):
        print(msg)

    @staticmethod
    def logdebug(msg):
        print(msg)

    @staticmethod
    def logwarn(msg):
        print(msg)

    @staticmethod
    def logerr(msg):
        print(msg)


class LocalPlannerNode(DTROS):
    poseStamped: Optional[PoseStamped] = None
    path: PlannerPath = []
    last_path_received_time: Optional[rospy.Time] = None
    path_timeout: float = 2.0  # Stop if no path update for 2 seconds

    def __init__(self) -> None:
        super(LocalPlannerNode, self).__init__(
            node_name="local_planner",
            node_type=NodeType.GENERIC,
        )

        self.veh = rospy.get_param("~veh", None)

        # Subscribers
        self.poseSub = rospy.Subscriber(
            f"/mcmrvls/robot_pose",
            PoseStamped,
            self.pose_callback,
            queue_size=1,
        )
        self.pathSub = rospy.Subscriber(
            f"/{self.veh}/move_base/NavfnROS/plan",
            Path,
            self.path_callback,
            queue_size=1,
        )

        # Publishers
        self.currentTargetPub = rospy.Publisher(
            f"/{self.veh}/local_planner/current_target", PoseStamped, queue_size=1
        )
        self.driverPub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1
        )

        # Internal state
        self.plannerRate = rospy.Rate(20)
        self.debugPrintRate = rospy.Rate(2)
        self.planner = DiffDriveLocalPlanner(
            posTolerance=0.05,
            finalWaypointPosTolerance=0.12,
            crossTrackErrorMinThreshold=0.05,
            logger=RospyLoggerProvider(),
            vLimit=2,
            omegaLimit=2,
            trajectoryAdherenceFactor=2,
            disableOrientationMatching=True,
            angularErrorThresholdAboveWhichToStopAndAdjustOrientation=np.deg2rad(40),
            groupingThreshold=0.014,
            groupLengthThreshold=0.05,
            minWaypointDistance=0.035,
            setNewPathOrientationImputationInvertDirections=False,
            lookAheadIndicesLength=4,
        )

    def pose_callback(self, msg):
        self.poseStamped = msg.pose

    def path_callback(self, msg):
        if not msg.poses:
            rospy.logwarn("[LocalPlannerNode] Received empty path from global planner (aborted/failed), STOPPING robot immediately.")
            self.planner.reset()
            self.path = []  # Clear path to stop main loop from driving
            self.last_path_received_time = None  # Clear timestamp
            # Immediately publish stop command multiple times for safety
            self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
            self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
            self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
            return

        # Update timestamp for valid path
        self.last_path_received_time = rospy.Time.now()

        rospy.loginfo(
            f"New raw path of length {len(msg.poses)} received from global planner"
        )

        # NavFNRos does not provide orientations, so we impute them from the neighboring pose
        self.path = self.planner.setNewPath(
            [
                PlannerPose(
                    PlannerPosition(pose.pose.position.x, pose.pose.position.y),
                    PlannerQuaternion(
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z,
                        pose.pose.orientation.w,
                    ),
                )
                for pose in msg.poses
            ],
            imputeOrientations=True,
        )

        rospy.loginfo(
            f"Path length after processing by local planner: {len(self.path)}, first pose is {self.path[0]}, last pose is {self.path[-1]}"
        )

    def run_blocking(self):
        twist2DStamped = Twist2DStamped()

        lastWasDriving: bool = False

        while not rospy.is_shutdown():
            # Check for path timeout (global planner failed/stalled without sending empty path)
            if (
                self.last_path_received_time is not None
                and len(self.path) > 0
                and (rospy.Time.now() - self.last_path_received_time).to_sec() > self.path_timeout
            ):
                rospy.logerr(
                    f"[LocalPlannerNode] Path timeout exceeded ({self.path_timeout}s) - global planner may have failed. STOPPING robot."
                )
                self.planner.reset()
                self.path = []
                self.last_path_received_time = None
                self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
                self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
                self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)
                lastWasDriving = False
                self.plannerRate.sleep()
                continue

            if self.poseStamped:
                control = self.planner.step(
                    PlannerPosition(
                        x=self.poseStamped.position.x, y=self.poseStamped.position.y
                    ),
                    PlannerQuaternion(
                        x=self.poseStamped.orientation.x,
                        y=self.poseStamped.orientation.y,
                        z=self.poseStamped.orientation.z,
                        w=self.poseStamped.orientation.w,
                    ),
                )

                # log state transitions
                if lastWasDriving and not control:
                    rospy.loginfo(
                        "[LocalPlannerNode] Stopping the robot, reached the goal!"
                    )

                if not lastWasDriving and control:
                    rospy.loginfo("[LocalPlannerNode] Starting to drive!")

                # actuate
                if control and len(self.path) > 0:
                    v, omega, dist, ang_err, _currWaypointIndex = control

                    target = self.path[_currWaypointIndex]
                    targetROS = PoseStamped()
                    targetROS.header.stamp = rospy.Time.now()
                    targetROS.header.frame_id = "map"
                    targetROS.pose.position.x = target.position.x
                    targetROS.pose.position.y = target.position.y
                    targetROS.pose.position.z = 0.0
                    targetROS.pose.orientation.x = target.quaternion.x
                    targetROS.pose.orientation.y = target.quaternion.y
                    targetROS.pose.orientation.z = target.quaternion.z
                    targetROS.pose.orientation.w = target.quaternion.w
                    self.currentTargetPub.publish(targetROS)

                    now = rospy.Time.now()
                    twist2DStamped.header.stamp = now
                    twist2DStamped.v = v
                    twist2DStamped.omega = omega

                    # if self.debugPrintRate.remaining() <= rospy.Duration(0, 0):
                    rospy.loginfo(
                        f"[LocalPlannerNode] Driving with v={v:.2f} m/s, omega={omega:.2f} rad/s, dist to goal: {dist:.2f} m, angle error: {np.rad2deg(ang_err):.2f} degrees"
                    )
                    # self.debugPrintRate.sleep()

                    self.driverPub.publish(twist2DStamped)
                else:
                    self.driverPub.publish(MSG_STOP_TWIST2DSTAMPED)

                lastWasDriving = control is not None

            self.plannerRate.sleep()

    def on_shutdown(self):
        # stop listening
        self.pathSub.unregister()
        self.poseSub.unregister()

        # publish the 'stop' control multiple times for sure
        twist2DStamped = MSG_STOP_TWIST2DSTAMPED

        self.driverPub.publish(twist2DStamped)
        self.driverPub.publish(twist2DStamped)
        sleep(0.5)
        self.driverPub.publish(twist2DStamped)
        self.driverPub.publish(twist2DStamped)


if __name__ == "__main__":
    try:
        node = LocalPlannerNode()
        node.run_blocking()
    except rospy.ROSInterruptException:
        pass
