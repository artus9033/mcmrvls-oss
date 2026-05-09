from abc import ABC, abstractmethod

import numpy as np

class AbstractVideoCaptureWrapper(ABC):
    @abstractmethod
    def __init__(self, *args): ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def getIdentifier(self) -> str: ...

    @abstractmethod
    def release(self) -> None: ...
