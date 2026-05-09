from typing import Annotated, Literal

Point2D = tuple[int, int]

FloatingPoint2D = tuple[float, float]

MarkerData = int

CornerSpec = Literal["TL"] | Literal["TR"] | Literal["BL"] | Literal["BR"]

DimensionsConfigKeys = Literal["mapWidth"] | Literal["mapHeight"] | Literal["markerWidth"] | Literal["markerHeight"]

CornerPointsList = Annotated[list[Point2D | FloatingPoint2D], 4]
"""A list of 4 corner points of a square, in the order of TL, TR, BL, BR."""
