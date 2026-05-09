from typing import Sized, cast

from utils.detections import averageDetections

from ..marker import MarkerDetection
from ..types import MarkerData

class CornerDetection:
    def __init__(
        self,
        markerDetections: list[MarkerDetection] | None,
        data: MarkerData,
        isTopLeft: bool,
        isTopRight: bool,
        isBottomLeft: bool,
        isBottomRight: bool,
    ) -> None:
        self.markerDetections = markerDetections
        self.data = data
        self.isTopLeft = isTopLeft
        self.isTopRight = isTopRight
        self.isBottomLeft = isBottomLeft
        self.isBottomRight = isBottomRight

        self.averagedMarkerDetection = None if self.markerDetections is None else averageDetections(self.markerDetections)

    def __key__(self):
        return self.data

    def __hash__(self):
        return hash(self.__key__())

    def __lt__(self, other):
        return self.__key__() < other.__key__()

    def __gt__(self, other):
        return self.__key__() > other.__key__()

    def __lte__(self, other):
        return self.__key__() <= other.__key__()

    def __gte__(self, other):
        return self.__key__() >= other.__key__()

    def __str__(self) -> str:
        return f"CornerDetection(averagedMarkerDetection={self.averagedMarkerDetection}, len(markerDetections)={None if self.markerDetections else len(cast(Sized, self.markerDetections))}, isTopLeft={self.isTopLeft}, isTopRight={self.isTopRight}, isBottomLeft={self.isBottomLeft}, isBottomRight={self.isBottomRight})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarkerDetection):
            return False

        return self.__key__() == other.__key__()
