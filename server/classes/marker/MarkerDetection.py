from typing import TypeVar

from utils.geometry import calculateCentroid
from utils.lang import ensureNumericPrimitive, ensureNumericPrimitiveTuple

from ..types import FloatingPoint2D, MarkerData, Point2D

TPoint2D = TypeVar("TPoint2D", Point2D, FloatingPoint2D)


class MarkerDetection:
    def __init__(
        self,
        topLeft: Point2D | FloatingPoint2D,
        topRight: Point2D | FloatingPoint2D,
        bottomLeft: Point2D | FloatingPoint2D,
        bottomRight: Point2D | FloatingPoint2D,
        data: MarkerData,
    ) -> None:
        [self.topLeft, self.topRight, self.bottomLeft, self.bottomRight] = [
            topLeft,
            topRight,
            bottomLeft,
            bottomRight,
        ]
        self.centroid = calculateCentroid([topLeft, topRight, bottomLeft, bottomRight])
        self.data = data

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
        return f"MarkerDetection(topLeft={self.topLeft}, topRight={self.topRight}, bottomLeft={self.bottomLeft}, bottomRight={self.bottomRight}, data={self.data})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MarkerDetection):
            return False

        return self.__key__() == other.__key__()

    def toPointsList(self):
        return [self.topLeft, self.topRight, self.bottomLeft, self.bottomRight]

    def toDTO(self):
        return {
            "topLeft": ensureNumericPrimitiveTuple(self.topLeft),
            "topRight": ensureNumericPrimitiveTuple(self.topRight),
            "bottomLeft": ensureNumericPrimitiveTuple(self.bottomLeft),
            "bottomRight": ensureNumericPrimitiveTuple(self.bottomRight),
            "data": ensureNumericPrimitive(self.data),
        }
