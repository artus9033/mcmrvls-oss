from typing import Sequence

from classes.types import CornerPointsList, CornerSpec, FloatingPoint2D, Point2D
import numpy as np

def calculateCentroid(points: Sequence[Point2D | FloatingPoint2D]) -> FloatingPoint2D:
    length = len(points)

    return (
        sum([p[0] for p in points]) / length,
        sum([p[1] for p in points]) / length,
    )


def getOrderedCornerPointsRotationShift(whichCornerCurrently: CornerSpec, whichCornerDesired: CornerSpec) -> int:
    order: list[CornerSpec] = ["TL", "TR", "BR", "BL"]
    currentIdx = order.index(whichCornerCurrently)
    desiredIdx = order.index(whichCornerDesired)

    shiftBy = currentIdx - desiredIdx

    if shiftBy < 0:
        shiftBy += 4

    return shiftBy


def shiftOrderedCornerPointsBy(cornerPoints: CornerPointsList, shiftBy: int) -> CornerPointsList:
    # first translate our TL, TR, BL, BR to a continuous square's TL, TR, BR, BL ordering
    cornerPoints[2], cornerPoints[3] = cornerPoints[3], cornerPoints[2]

    shiftedPoints = [*cornerPoints[shiftBy:], *cornerPoints[:shiftBy]]

    # finally translate the result's continuous square's TL, TR, BR, BL to our TL, TR, BR, BL ordering
    shiftedPoints[2], shiftedPoints[3] = shiftedPoints[3], shiftedPoints[2]

    # undo translation of cornerPoints
    cornerPoints[2], cornerPoints[3] = cornerPoints[3], cornerPoints[2]

    return shiftedPoints


def rotateOrderedCornerPoints(
    cornerPoints: CornerPointsList,
    whichCornerCurrently: CornerSpec,
    whichCornerDesired: CornerSpec,
) -> CornerPointsList:
    """Rotates the ordered (TL, TR, BL, BR) list of points of a square by a given offset so that they match a different perspective's ordering."""
    shiftBy = getOrderedCornerPointsRotationShift(whichCornerCurrently=whichCornerCurrently, whichCornerDesired=whichCornerDesired)

    shiftedPoints = shiftOrderedCornerPointsBy(cornerPoints, shiftBy)

    return shiftedPoints


def pointsDistance(p1: Point2D | FloatingPoint2D, p2: Point2D | FloatingPoint2D) -> float | int:
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def circle(radius: int | float | np.number) -> np.ndarray:
    size = 2 * radius + 1
    y, x = np.ogrid[:size, :size]

    distance_from_center = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)

    return (distance_from_center <= radius).astype(np.uint8)
