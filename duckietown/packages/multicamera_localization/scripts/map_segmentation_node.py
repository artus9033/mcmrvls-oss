#!/usr/bin/env python3


from threading import RLock
from typing import Any, Dict, Tuple, Union

import numpy as np
import rospy
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap, GetMapResponse
from std_msgs.msg import Header
from SubscriberNodeBase import SubscriberNodeBase


class MapSegmentationNode(SubscriberNodeBase):
    def __init__(self) -> None:
        super(MapSegmentationNode, self).__init__(
            "multicamera_localization_map_segmentation_node"
        )

        # map server message singletons
        self.mapPub = rospy.Publisher("map", OccupancyGrid, queue_size=1)

        self.mapGrid = self.generateMapGridSingleton()
        self.mapGridLock = RLock()

        self.mapService = rospy.Service("static_map", GetMap, self.handle_get_map)

    @staticmethod
    def assertParameterPresent(param_name, param):
        if param is None:
            raise RuntimeError(
                f"FATAL: {param_name} param has not been passed to localization_node.py"
            )

    def connect(self):
        super(MapSegmentationNode, self).connect()

        rospy.loginfo(
            "[MapSegmentationNode] Connected to Socket.io server, subscribing to map stream"
        )

        self.sioEmit("subscribe_map_numpy_feed")

    def _convertArrayBufferToArrayInROSOrder(
        self, buffer, bitmapSize2D: Tuple[int], bitmapSize1D: int
    ) -> np.ndarray:
        """
        Typically, an image has its top-left corner as the origin (0,0),
        with the y-axis pointing downwards. In contrast, ROS OccupancyGrid
        expects the origin to be at the bottom-left corner, with the y-axis
        pointing upwards.
        This function performs buffer-to-ndarray parsing, dtype conversion
        and it also flips the image vertically to convert it to ROS's
        coordinate system.
        """

        return np.flipud(
            np.frombuffer(buffer, dtype=np.uint8)[:bitmapSize1D]
            .astype(np.bool_)
            .reshape(bitmapSize2D)
        )

    def processSioEvent(self, event, data: Dict[str, Any]):
        if event == "map_segmentation_numpy":
            # metadata
            metersPerPixel: Union[float, int] = data["metersPerPixel"]
            widthPx: int = data["widthPx"]
            heightPx: int = data["heightPx"]
            bitmapSize2D = (heightPx, widthPx)
            bitmapSize1D = widthPx * heightPx

            # bitmaps
            roadsMaskROS = self._convertArrayBufferToArrayInROSOrder(
                data["roadsMask"], bitmapSize2D, bitmapSize1D
            )
            stopLinesMaskROS = self._convertArrayBufferToArrayInROSOrder(
                data["stopLinesMask"], bitmapSize2D, bitmapSize1D
            )
            laneSeparatorLinesMaskROS = self._convertArrayBufferToArrayInROSOrder(
                data["laneSeparatorLinesMask"], bitmapSize2D, bitmapSize1D
            )

            # combine masks for OccupancyGrid, but with non-standard, custom value meaning:
            # - 0 => asphalt
            # - 33 => stop line
            # - 66 => lane separator line
            # - 100 => not-on-asphalt (non-road, i.e., treat-as-obstacle)

            # start with everything as "obstacle"
            occupancyMapGrid = np.full(roadsMaskROS.shape, 100, dtype=np.uint8)

            # asphalt (where roadsMask == 1)
            occupancyMapGrid = np.where(roadsMaskROS == 1, 0, occupancyMapGrid)

            # FIXME: use the below in an advanced case later

            # # lane separators override asphalt
            # occupancyMapGrid = np.where(
            #     laneSeparatorLinesMaskROS == 1, 66, occupancyMapGrid
            # )

            # # stop lines override everything
            # occupancyMapGrid = np.where(stopLinesMaskROS == 1, 33, occupancyMapGrid)

            # as per https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
            with self.mapGridLock:
                self.mapGrid.header.seq += 1
                self.mapGrid.header.stamp = rospy.Time.now()
                self.mapGrid.info.resolution = metersPerPixel
                self.mapGrid.info.width = widthPx
                self.mapGrid.info.height = heightPx
                self.mapGrid.data = (
                    occupancyMapGrid.flatten().tolist()
                )  # row-major order flattening
                self.mapPub.publish(self.mapGrid)

    def handle_get_map(self, req: GetMap) -> GetMapResponse:
        with self.mapGridLock:
            return GetMapResponse(map=self.mapGrid)

    def generateMapGridSingleton(self) -> OccupancyGrid:
        origin = Pose(Point(0.0, 0.0, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))

        # populate the OccupancyGrid message
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = "map"
        grid.info.resolution = 0
        grid.info.width = 0
        grid.info.height = 0
        grid.info.origin = origin
        grid.data = np.zeros((0, 0), dtype=np.int8).flatten().tolist()

        return grid


if __name__ == "__main__":
    try:
        node = MapSegmentationNode()
        node.run_blocking()
    except rospy.ROSInterruptException:
        pass
