from logging import getLogger

import numpy as np
from utils.geometry import calculateCentroid
from utils.lang import ensureNumericPrimitiveTuple

from ..types import FloatingPoint2D, MarkerData
from .MarkerDetection import MarkerDetection

class CompositeDetection:
    _centroidDirty: bool
    _tlDirty: bool
    _trDirty: bool
    _blDirty: bool
    _brDirty: bool

    _centroid: FloatingPoint2D
    _topLeft: FloatingPoint2D
    _topRight: FloatingPoint2D
    _bottomLeft: FloatingPoint2D
    _bottomRight: FloatingPoint2D

    def __init__(self, data: MarkerData, detections: list[MarkerDetection]) -> None:
        self.detections = detections
        self.data = data
        self.logger = getLogger("CompositeDetection")

        self._setAllCalculatedPropsDirty()

    def appendDetection(self, detection: MarkerDetection):
        self.detections.append(detection)

        self._setAllCalculatedPropsDirty()

    def extendBy(self, stitchedDetection: "CompositeDetection"):
        self.assertOtherIsCompatible(stitchedDetection)

        self.detections.extend(stitchedDetection.detections)

        self._setAllCalculatedPropsDirty()

    def _setAllCalculatedPropsDirty(self):
        self._centroidDirty = True
        self._tlDirty = True
        self._trDirty = True
        self._blDirty = True
        self._brDirty = True

    @property
    def aggregatedCentroid(self):
        if self._centroidDirty:
            self._centroid = calculateCentroid([d.centroid for d in self.detections])

            self._centroidDirty = False

        return self._centroid

    @property
    def aggregatedTopLeft(self):
        if self._tlDirty:
            self._topLeft = calculateCentroid([d.topLeft for d in self.detections])

            self._tlDirty = False

        return self._topLeft

    @property
    def aggregatedTopRight(self):
        if self._trDirty:
            self._topRight = calculateCentroid([d.topRight for d in self.detections])

            self._trDirty = False

        return self._topRight

    @property
    def aggregatedBottomLeft(self):
        if self._blDirty:
            self._bottomLeft = calculateCentroid([d.bottomLeft for d in self.detections])

            self._blDirty = False

        return self._bottomLeft

    @property
    def aggregatedBottomRight(self):
        if self._brDirty:
            self._bottomRight = calculateCentroid([d.bottomRight for d in self.detections])

            self._brDirty = False

        return self._bottomRight

    @property
    def aggregatedWidth(self):
        return np.linalg.norm(np.array(self.aggregatedTopRight) - np.array(self.aggregatedTopLeft))

    @property
    def aggregatedHeight(self):
        return np.linalg.norm(np.array(self.aggregatedBottomLeft) - np.array(self.aggregatedTopLeft))

    def toAggregatedPointsList(self):
        return [
            self.aggregatedTopLeft,
            self.aggregatedTopRight,
            self.aggregatedBottomLeft,
            self.aggregatedBottomRight,
        ]

    def assertOtherIsCompatible(self, other: "CompositeDetection"):
        if self.data != other.data:
            raise ValueError(f"Can only extend same marker composite detections (this one is {self.data}, other is {other.data})")

    def __str__(self) -> str:
        return f"CompositeDetection(data={self.data}, detections={[np.round(d.centroid, decimals=2) for d in self.detections]})"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompositeDetection):
            return False

        return self.__key__() == other.__key__()

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

    def __len__(self):
        return len(self.detections)

    def toDTO(self):
        return MarkerDetection(
            topLeft=ensureNumericPrimitiveTuple(self.aggregatedTopLeft),
            topRight=ensureNumericPrimitiveTuple(self.aggregatedTopRight),
            bottomLeft=ensureNumericPrimitiveTuple(self.aggregatedBottomLeft),
            bottomRight=ensureNumericPrimitiveTuple(self.aggregatedBottomRight),
            data=self.data,
        ).toDTO()
