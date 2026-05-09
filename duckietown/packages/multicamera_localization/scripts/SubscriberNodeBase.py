from abc import ABC, abstractmethod
from typing import Any, Optional
from threading import RLock

import rospy
import socketio

from duckietown.dtros import DTROS, NodeType


class SubscriberNodeBase(ABC, DTROS):
    _sio: Optional[socketio.SimpleClient] = None

    def __init__(self, node_name: str) -> None:
        super(SubscriberNodeBase, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC
        )

        self.node_name = node_name

        # parameters loading
        self.serverAddress = rospy.get_param("~server_address", None)
        self.serverPort = rospy.get_param("~server_port", None)

        # parameters validation
        SubscriberNodeBase.assertParameterPresent("server_address", self.serverAddress)
        SubscriberNodeBase.assertParameterPresent("server_port", self.serverPort)

        # operational variables
        self._sioLock = RLock()

    @staticmethod
    def assertParameterPresent(param_name: str, param: Any) -> None:
        """
        Asserts that a parameter has been passed to the node.

        @raises RuntimeError if the parameter is None
        """

        if param is None:
            raise RuntimeError(
                f"FATAL: {param_name} param has not been passed to localization_node.py"
            )

    def connect(self):
        """
        Connects to the socket.io server in blocking mode.
        """

        if self._sio and self._sio.connected:
            try:
                self._sio.disconnect()
            except:
                # no-op
                pass

        self._sio = socketio.SimpleClient()

        rospy.loginfo(f"[{self.node_name}] Establishing Socket.io connection...")
        self._sio.connect(f"ws://{self.serverAddress}:{self.serverPort}")

    def sioEmit(self, event: str, *args) -> bool:
        """
        Emits an event to the socket.io server in a thread-safe manner.

        :param event: The event name.
        :param args: The event arguments.
        :return: True if the event was emitted, False otherwise (if socket was not initialized yet).
        """

        if not self._sio:
            return False

        with self._sioLock:
            self._sio.emit(event, *args)
            return True

    def run_blocking(self):
        """
        Runs the node in blocking mode until it should exit.
        """
        reconnectionRate = rospy.Rate(1)  # 1hz
        messageQueueRate = rospy.Rate(4)  # 4hz

        while not rospy.is_shutdown():
            try:
                with self._sioLock:
                    if self._sio and self._sio.connected_event.is_set():
                        # message queue in connected state loop branch
                        while not rospy.is_shutdown():
                            try:
                                # below: short timeout that this node can check if it has been interrupted & should exit
                                [event, data] = self._sio.receive(timeout=0.75)

                                self.processSioEvent(event, data)

                            except socketio.exceptions.TimeoutError:
                                break

                        messageQueueRate.sleep()

                    else:
                        # reconnect loop branch
                        rospy.loginfo(
                            f"[{self.node_name}] Socket.io client not connected, re-connecting"
                        )
                        self.connect()

            except socketio.exceptions.ConnectionError as conn_error:
                rospy.logerr(
                    f"[{self.node_name}] Socket.io connection error: {conn_error}"
                )
                reconnectionRate.sleep()

    @abstractmethod
    def processSioEvent(self, event: str, data: Any) -> None:
        """
        Process an event received from the socket.io server.

        :param event: The event name.
        :param data: The event data.
        """

        ...
