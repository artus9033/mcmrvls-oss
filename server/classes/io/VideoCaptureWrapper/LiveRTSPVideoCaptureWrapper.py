import logging

import cv2
import numpy as np
import rtsp

from . import AbstractVideoCaptureWrapper
from .errors import VideoCaptureError

class LiveRTSPVideoCaptureWrapper(AbstractVideoCaptureWrapper):
    def __init__(self, logger: logging.Logger, rtspSource: str):
        self.logger = logger
        self.rtspSource = rtspSource

        self.logger.info(f"Opening RTSP {rtspSource}")

        self.client = rtsp.Client(rtsp_server_uri=rtspSource, verbose=True)

    def read(self) -> np.ndarray:
        frame = self.client.read()

        if not frame:
            self.logger.error(f"Error: Unable to capture image from RTSP {self.rtspSource}")

            raise VideoCaptureError()

        return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    def getIdentifier(self) -> str:
        return self.rtspSource

    def release(self):
        self.client.close()
