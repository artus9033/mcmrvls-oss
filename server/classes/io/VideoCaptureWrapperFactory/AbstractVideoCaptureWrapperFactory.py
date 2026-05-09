from abc import ABC, abstractmethod

from ..VideoCaptureWrapper import AbstractVideoCaptureWrapper

class AbstractVideoCaptureWrapperFactory(ABC):
    @staticmethod
    @abstractmethod
    def createVideoCaptureWrappers(*args) -> list[AbstractVideoCaptureWrapper]: ...
