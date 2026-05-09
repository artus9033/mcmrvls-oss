import logging

import cv2
import numpy as np

from . import AbstractCV2VideoCaptureWrapper
from .errors import VideoCaptureError

class LiveUSBCamVideoCaptureWrapper(AbstractCV2VideoCaptureWrapper):
    def __init__(self, logger: logging.Logger, camera: tuple[int, str]):
        (camera_index, camera_name) = camera
        self.logger = logger
        self.camera_index = camera_index

        self.logger.info(f"Opening camera {camera_index}")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if "A4Tech".lower() in camera_name.lower():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            # self.cap.set(cv2.CAP_PROP_FPS, 0.5)
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()

        if not ret:
            self.logger.error(f"Error: Unable to capture image from camera {self.camera_index}")

            raise VideoCaptureError()

        return frame

    def getIdentifier(self) -> str:
        return self.camera_index.__str__()
