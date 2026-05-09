import logging

from utils.os import listWorkingCameraPorts

from ..VideoCaptureWrapper import (
    AbstractVideoCaptureWrapper,
    LiveRTSPVideoCaptureWrapper,
    LiveUSBCamVideoCaptureWrapper,
)
from . import AbstractVideoCaptureWrapperFactory

class LiveVideoCaptureWrapperFactory(AbstractVideoCaptureWrapperFactory):
    @staticmethod
    def createVideoCaptureWrappers(
        logger: logging.Logger,
        rtspSources: list[str],
    ) -> list[AbstractVideoCaptureWrapper]:
        wrappers: list[AbstractVideoCaptureWrapper] = []

        cameraPorts = listWorkingCameraPorts(logger)

        for cameraPort in cameraPorts:
            logger.info(f"Opening camera {cameraPort}")

            wrappers.append(LiveUSBCamVideoCaptureWrapper(logger=logger, camera=cameraPort))

        for rtspSource in rtspSources:
            logger.info(f"Opening RTSP stream {rtspSource}")

            wrappers.append(LiveRTSPVideoCaptureWrapper(logger=logger, rtspSource=rtspSource))

        return wrappers
