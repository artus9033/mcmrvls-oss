from classes.config import AlgorithmConfig
from classes.map import CornerDetection
from classes.marker import CompositeDetection, MarkerDetection
from classes.types import CornerPointsList, FloatingPoint2D, MarkerData

def genCorner(xOffset: float, yOffset: float, width: float = 1.0, height: float = 1.0) -> CornerPointsList:
    return [
        (xOffset, yOffset),
        (width + xOffset, yOffset),
        (xOffset, height + yOffset),
        (width + xOffset, height + yOffset),
    ]


def genRectOfCornerRects(width: float, height: float, config: AlgorithmConfig) -> list[CornerPointsList]:
    corners: list[CornerPointsList] = []
    width /= 1.5
    height /= 1.5
    cornerMarkerW = int(width / config.RATIO_MAP_EDGE_TO_MARKER_EDGE_W)
    cornerMarkerH = int(height / config.RATIO_MAP_EDGE_TO_MARKER_EDGE_H)
    for xOffset, yOffset in [(0, 0), (width, 0), (0, height), (width, height)]:
        corners.append(genCorner(xOffset, yOffset, width=cornerMarkerW, height=cornerMarkerH))

    return corners


def packCornerDetection(
    detectionOrData: CompositeDetection | MarkerDetection | MarkerData,
    config: AlgorithmConfig,
):
    if isinstance(detectionOrData, MarkerDetection):
        data = detectionOrData.data
        detections = [detectionOrData]
    elif isinstance(detectionOrData, CompositeDetection):
        data = detectionOrData.data
        detections = detectionOrData.detections
    else:
        data = detectionOrData
        detections = None

    isTopLeft = data == config.mapDefinition["TL"]
    isTopRight = data == config.mapDefinition["TR"]
    isBottomLeft = data == config.mapDefinition["BL"]
    isBottomRight = data == config.mapDefinition["BR"]

    return CornerDetection(
        markerDetections=detections,
        data=data,
        isTopLeft=isTopLeft,
        isTopRight=isTopRight,
        isBottomLeft=isBottomLeft,
        isBottomRight=isBottomRight,
    )


def getSquareExtremeCornerPoints(
    squareMarkers: list[MarkerDetection],
) -> list[FloatingPoint2D]:
    if len(squareMarkers) != 4:
        raise ValueError("getSquareExtremeCornerPoints: corners must have exactly 4 elements")

    squareCentroids = [m.centroid for m in squareMarkers]
    squareMarkerOrderingMapping = {i: squareCentroids.index(centroid) for i, centroid in enumerate(squareCentroids)}

    points: list[FloatingPoint2D] = []

    for i, marker in enumerate(squareMarkers):
        sortedIdx = squareMarkerOrderingMapping[i]  # in TL, TR, BL, BR order
        match sortedIdx:
            case 0:
                points.append(marker.topLeft)

            case 1:
                points.append(marker.topRight)

            case 2:
                points.append(marker.bottomLeft)

            case 3:
                points.append(marker.bottomRight)

    return points
