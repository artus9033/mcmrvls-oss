from __future__ import annotations

import logging

import cv2
import numpy as np

from . import AbstractCV2VideoCaptureWrapper
from .errors.VideoCaptureError import VideoCaptureError
from .errors.VideoEndedException import VideoEndedException

class PlaybackVideoCaptureWrapper(AbstractCV2VideoCaptureWrapper):
    cap: cv2.VideoCapture | None = None

    def __init__(self, logger: logging.Logger, video_path: str, capture_index: int, loop: bool):
        self.logger = logger
        self.video_path = video_path
        self.capture_index = capture_index
        self.loop = loop

        self.logger.info(f"Opening playback file {video_path}")

        self.reinitializeCapture()

    def reinitializeCapture(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.video_path)

    def read(self) -> np.ndarray:
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                # loop from the beginning if playing a video file
                self.logger.debug(f"Video stream {self.capture_index} ended, rewinding")

                self.reinitializeCapture()

                ret, frame = self.cap.read()

                if not ret:
                    self.logger.error(f"Error: Unable to capture image from video file {self.video_path}")

                    raise VideoCaptureError()

                return frame
            else:
                raise VideoEndedException(f"Playback of {self.video_path} ended")

        return frame

    def getIdentifier(self) -> str:
        return self.capture_index.__str__()
