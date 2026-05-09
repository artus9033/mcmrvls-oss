import cv2

from .AbstractVideoCaptureWrapper import AbstractVideoCaptureWrapper

class AbstractCV2VideoCaptureWrapper(AbstractVideoCaptureWrapper):
    cap: cv2.VideoCapture

    def release(self):
        self.cap.release()
