#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

from duckietown_msgs.msg import BoolStamped
from duckietown.dtros import DTROS, NodeType


class StatusListenerNode(DTROS):
    def __init__(self) -> None:
        super(StatusListenerNode, self).__init__(
            node_name="multicamera_localization_status_listener_node",
            node_type=NodeType.GENERIC,
        )

        self.veh = rospy.get_param("~veh", None)

        # Timeout threshold (2 seconds)
        self.timeout_threshold = 2.0

        # Track last received times and states
        self.last_map_time = None
        self.last_robot_pose_time = None
        self.map_is_down = False
        self.robot_pose_is_down = False

        self.map_sub = rospy.Subscriber("map", OccupancyGrid, self.mapCallback)
        self.robot_pose_sub = rospy.Subscriber(
            "robot_pose", PoseStamped, self.robotPoseCallback
        )

        self.emergencyStopPub = rospy.Publisher(
            f"{self.veh}/emergency_stop", BoolStamped, queue_size=1, latch=True
        )

        # Timer to check for timeouts
        self.check_timer = rospy.Timer(rospy.Duration(0.5), self.check_timeouts)

    def mapCallback(self, msg):
        """Update map info using OccupancyGrid message."""
        current_time = rospy.Time.now()

        # Check if this is the first message or if we were previously down
        if self.last_map_time is None:
            rospy.loginfo("[StatusListenerNode] Map topic connected")
        elif self.map_is_down:
            rospy.loginfo("[StatusListenerNode] Map topic alive again")
            self.map_is_down = False

        self.last_map_time = current_time

    def robotPoseCallback(self, msg):
        """Update robot pose info using PoseStamped message."""
        current_time = rospy.Time.now()

        # Check if this is the first message or if we were previously down
        if self.last_robot_pose_time is None:
            rospy.loginfo("[StatusListenerNode] Robot pose topic connected")
        elif self.robot_pose_is_down:
            rospy.loginfo("[StatusListenerNode] Robot pose topic alive again")
            self.robot_pose_is_down = False

        self.last_robot_pose_time = current_time

    def check_timeouts(self, event):
        """Check if any topics have timed out."""
        current_time = rospy.Time.now()

        dirty = False

        # Check map topic timeout
        if self.last_map_time is not None:
            time_since_map = (current_time - self.last_map_time).to_sec()
            if time_since_map > self.timeout_threshold and not self.map_is_down:
                rospy.logwarn("[StatusListenerNode] Map topic down")
                self.map_is_down = True

                dirty = True

        # Check robot pose topic timeout
        if self.last_robot_pose_time is not None:
            time_since_pose = (current_time - self.last_robot_pose_time).to_sec()
            if time_since_pose > self.timeout_threshold and not self.robot_pose_is_down:
                rospy.logwarn("[StatusListenerNode] Robot pose topic down")
                self.robot_pose_is_down = True

                dirty = True

        if dirty:
            rospy.loginfo(
                f"[StatusListenerNode] New statuses - Map: {'DOWN' if self.map_is_down else 'UP'}, Robot Pose: {'DOWN' if self.robot_pose_is_down else 'UP'}"
            )

            if self.robot_pose_is_down or self.map_is_down:
                rospy.logwarn(
                    "[StatusListenerNode] A critical topic is down, localization may be compromised; performing emergency stop"
                )

                msg = BoolStamped()
                msg.header.stamp = rospy.Time.now()
                msg.data = True
                self.emergencyStopPub.publish(msg)


if __name__ == "__main__":
    try:
        node = StatusListenerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
