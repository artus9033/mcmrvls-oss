#!/usr/bin/env python3

from typing import Optional

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from SubscriberNodeBase import SubscriberNodeBase

MAP_TOPIC = "map"


class LocalizationNode(SubscriberNodeBase):
    mapWidth: Optional[float] = None
    mapHeight: Optional[float] = None
    mapResolution: Optional[float] = None

    def __init__(self) -> None:
        super(LocalizationNode, self).__init__(
            "multicamera_localization_robot_localization_node"
        )

        # robot localization message-related singletons
        self.poseStampedPub = rospy.Publisher("robot_pose", PoseStamped, queue_size=1)

        self.poseStamped = self.generatePoseStamped()

        # parameters loading
        self.robotId = rospy.get_param("~robot_id", None)
        self.veh = rospy.get_param("~veh", None)

        # parameters validation
        LocalizationNode.assertParameterPresent("robot_id", self.robotId)
        LocalizationNode.assertParameterPresent("veh", self.veh)

        self.tfBroadcaster = tf.TransformBroadcaster()

        self.mapSub = rospy.Subscriber(MAP_TOPIC, OccupancyGrid, self.mapCallback)

        rospy.loginfo(
            "[LocalizationNode] Waiting for topic %s to become active...", MAP_TOPIC
        )
        rospy.wait_for_message(MAP_TOPIC, OccupancyGrid)
        rospy.loginfo(
            "[LocalizationNode] Topic %s is now active, launching this node", MAP_TOPIC
        )

    def mapCallback(self, msg: OccupancyGrid):
        self.mapWidth = msg.info.width
        self.mapHeight = msg.info.height
        self.mapResolution = msg.info.resolution
        self.mapOriginX = msg.info.origin.position.x
        self.mapOriginY = msg.info.origin.position.y

    def connect(self):
        super(LocalizationNode, self).connect()

        rospy.loginfo(
            f"[LocalizationNode] Connected to Socket.io server, subscribing to stream for robot {self.robotId}"
        )

        self.sioEmit("subscribe_single_robot", self.robotId)

    def generatePoseStamped(self):
        header = Header()
        header.frame_id = "map"
        header.stamp = rospy.Time.now()

        poseStamped = PoseStamped()
        poseStamped.header = header
        poseStamped.pose = Pose()

        return poseStamped

    def processSioEvent(self, event, data):
        if event == "single_robot_results":
            if (
                self.mapWidth is None
                or self.mapHeight is None
                or self.mapWidth <= 0
                or self.mapHeight <= 0
            ):
                rospy.logwarn(
                    "[LocalizationNode] mapWidth or mapHeight is None, cannot process robot pose yet - dropping message"
                )
                return

            detection = data["detection"]
            x = detection["x"] * self.mapWidth * self.mapResolution
            y = (1 - detection["y"]) * self.mapHeight * self.mapResolution
            qRot = Quaternion(
                *tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    # below: yaw deg -> rad and convert from clockwise-0-at-east to counterclockwise-0-at-east
                    # note: the value already is 0-at-east because the backend serves the data in such format
                    np.deg2rad(360 - detection["azimuth"]),
                )
            )

            # broadcast pose
            self.poseStamped.pose.position.x = x
            self.poseStamped.pose.position.y = y
            self.poseStamped.pose.orientation = qRot

            self.poseStamped.header.seq += 1
            self.poseStamped.header.stamp = rospy.Time.now()

            self.poseStampedPub.publish(self.poseStamped)

            # broadcast tf footprint -> map
            self.tfBroadcaster.sendTransform(
                (x, y, 0),  # translation
                (qRot.x, qRot.y, qRot.z, qRot.w),  # rotation
                rospy.Time.now(),
                f"{self.veh}/footprint",  # child frame
                "map",  # parent frame
            )


if __name__ == "__main__":
    try:
        node = LocalizationNode()
        node.run_blocking()
    except rospy.ROSInterruptException:
        pass
