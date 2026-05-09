#!/usr/bin/env python3


import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Header

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

from time import sleep


class CmdVelDuckietownMuxNode(DTROS):
    def __init__(self) -> None:
        super(CmdVelDuckietownMuxNode, self).__init__(
            node_name="cmd_vel_dt_mux_node",
            node_type=NodeType.GENERIC,
        )

        self.veh = rospy.get_param("~veh", None)

        self.dtPub = rospy.Publisher(
            f"/{self.veh}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1
        )
        self.moveBaseSub = rospy.Subscriber(
            f"/{self.veh}/planner_cmd_vel", Twist, self.cmdVelCallback, queue_size=1
        )

        self.initMessage()

    def initMessage(self):
        self.twist2DStamped = Twist2DStamped()
        self.twist2DStamped.header = Header()
        self.twist2DStamped.header.frame_id = "map"

    def cmdVelCallback(self, msg):
        """Convert geometry_msgs.Twist to duckietown_msgs.Twist2DStamped."""

        self.twist2DStamped.header.stamp = rospy.Time.now()
        self.twist2DStamped.v = msg.linear.x
        self.twist2DStamped.omega = msg.angular.z

        self.dtPub.publish(self.twist2DStamped)

    def on_shutdown(self):
        # stop listening & updating control message
        self.moveBaseSub.unregister()

        # set control to stop the wheels
        self.twist2DStamped.v = 0
        self.twist2DStamped.omega = 0

        # publish the 'stop' control multiple times for sure
        self.dtPub.publish(self.twist2DStamped)
        self.dtPub.publish(self.twist2DStamped)
        sleep(0.5)
        self.dtPub.publish(self.twist2DStamped)
        self.dtPub.publish(self.twist2DStamped)


if __name__ == "__main__":
    try:
        node = CmdVelDuckietownMuxNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
