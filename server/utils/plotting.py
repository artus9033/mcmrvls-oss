import math
from typing import Iterable, Optional

from classes.config import AlgorithmConfig
from classes.marker import CompositeDetection, MarkerDetection
import cv2
from matplotlib import colormaps
import numpy as np
from utils.constants import neutralBorderColorBGR

# cache the most common sets
uniqueColorsCache: dict[int, list[list[int]]] = {}
MAX_CACHED_COLORMAPS: int = 8
CHAR_BOX_SIZE: int = 10
FONT_SCALE: float = 1.8
PLOTTING_TEXT_OFFSET: int = math.ceil(CHAR_BOX_SIZE * FONT_SCALE * 3)


def genUniqueColors(N: int) -> list[list[int]]:
    if N < MAX_CACHED_COLORMAPS and N in uniqueColorsCache:
        return uniqueColorsCache[N]

    colors = colormaps["viridis"](np.linspace(0, 1, N, endpoint=False)) * 255

    # trim the last value for the color to be 3D instead of 4D
    cmap = [[color[0], color[1], color[2]] for color in colors]

    return cmap


def putText(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int,
    scale: float,
    colorBGR: tuple[int, int, int],
    thickness: int,
):
    # first the border
    cv2.putText(img, text, pos, font, scale, neutralBorderColorBGR, thickness + 3)
    # then the final text
    cv2.putText(img, text, pos, font, scale, colorBGR, thickness)


def drawDetectionsOnImage(
    image: np.ndarray,
    detections: Iterable[MarkerDetection | CompositeDetection],
    config: AlgorithmConfig,
    colorBGR: Optional[tuple[int, int, int]] = None,
    bDrawLabels: bool = True,
    offset: tuple[int, int] = (0, 0),
):
    if colorBGR is None:
        colorBGR = (0, 255, 0)

    for detection in detections:
        if isinstance(detection, CompositeDetection):
            pointsList = detection.toAggregatedPointsList()
            centroid = detection.aggregatedCentroid
        else:
            pointsList = detection.toPointsList()
            centroid = detection.centroid

        [topLeft, topRight, bottomLeft, bottomRight] = [tuple(map(lambda x: int(x), p)) for p in pointsList]
        centroid = [int(coord) for coord in centroid]
        topLeft = (topLeft[0] + offset[0], topLeft[1] + offset[1])
        topRight = (topRight[0] + offset[0], topRight[1] + offset[1])
        bottomLeft = (bottomLeft[0] + offset[0], bottomLeft[1] + offset[1])
        bottomRight = (bottomRight[0] + offset[0], bottomRight[1] + offset[1])
        centroid = (centroid[0] + offset[0], centroid[1] + offset[1])

        markerID = detection.data

        cv2.drawMarker(
            image,
            topLeft,
            colorBGR,
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=20,
            thickness=8,
        )

        # draw the bounding box of the detection
        cv2.line(image, topLeft, topRight, colorBGR, config.plottingLineThickness)
        cv2.line(image, topRight, bottomRight, colorBGR, config.plottingLineThickness)
        cv2.line(image, bottomRight, bottomLeft, colorBGR, config.plottingLineThickness)
        cv2.line(image, bottomLeft, topLeft, colorBGR, config.plottingLineThickness)
        # compute and draw the center (x, y)-coordinates of the marker
        [cX, cY] = centroid
        cv2.circle(image, (cX, cY), 5, (255, 127, 0), -1)

        # draw the marker ID on the image
        if bDrawLabels:
            coordY = max(0, topLeft[1] - CHAR_BOX_SIZE)

            if coordY - PLOTTING_TEXT_OFFSET < 0:
                coordY = min(image.shape[1], topLeft[1] + PLOTTING_TEXT_OFFSET)

            label = str(markerID)
            putText(
                image,
                label,
                (
                    max(0, topLeft[0] - CHAR_BOX_SIZE * len(label) * 2),
                    coordY,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 255, 0),
                4,
            )

    return image


def draw_coasted_robot_pose_on_topdown(
    image: np.ndarray,
    *,
    rel_x: float,
    rel_y: float,
    azimuth_deg: float,
    label: str,
    config: AlgorithmConfig,
    color_bgr: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw centroid, heading arrow, and label for a Kalman-coasted robot (no tag polygon)."""
    h, w = image.shape[:2]
    cx = int(rel_x * w)
    cy = int(rel_y * h)
    arrow_len = max(24.0, 0.06 * float(min(w, h)))
    rad = math.radians(azimuth_deg)
    ex = int(cx + arrow_len * math.sin(rad))
    ey = int(cy - arrow_len * math.cos(rad))
    r = max(8, config.plottingLineThickness * 2)
    thickness = max(2, config.plottingLineThickness // 2)
    cv2.circle(image, (cx, cy), r, color_bgr, thickness)
    cv2.arrowedLine(image, (cx, cy), (ex, ey), color_bgr, thickness, tipLength=0.25)
    putText(
        image,
        label,
        (min(max(0, cx + 12), w - 1), min(max(PLOTTING_TEXT_OFFSET, cy - 8), h - 1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color_bgr,
        2,
    )
