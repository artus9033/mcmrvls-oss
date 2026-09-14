"""Map segmentation entry point.

The actual segmentation is delegated to the strategy selected via the
``mapSegmentation`` config section (see ``algorithms.segmentation``); this
module keeps the results-holder plumbing and the strategy-independent robot
exclusion mask.
"""

from logging import Logger

import cv2
import numpy as np

from algorithms.segmentation import get_map_segmentation_strategy
from classes.resources.PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder
from utils.tracing import Tracing


def buildRobotExclusionMask(
    algorithmResultsHolder: PreallocatedAlgorithmResultsHolder,
    imgShape: tuple[int, ...],
) -> np.ndarray | None:
    """Circle mask over robot positions: robots are navigable, not obstacles."""
    robotDetections = algorithmResultsHolder.robotDetections
    if not robotDetections:
        return None

    imgHeight, imgWidth = imgShape[:2]
    robotMask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)

    # Estimate robot radius in pixels (robot marker is typically ~14cm, map is ~300cm)
    # This gives us approximately 4.6% of the image width/height
    # Add some padding to ensure the entire robot is excluded
    robotRadiusPx = int(max(imgWidth, imgHeight) * 0.06)  # ~6% for safety margin

    for robotDet in robotDetections:
        # Convert relative coordinates to pixel coordinates
        robotX = int(robotDet.x * imgWidth)
        robotY = int(robotDet.y * imgHeight)

        # Draw a circle at the robot position
        cv2.circle(robotMask, (robotX, robotY), robotRadiusPx, 255, -1)

    return robotMask


def updateMapSegmentation(
    algorithmResultsHolder: PreallocatedAlgorithmResultsHolder,
    logger: Logger,
):
    with Tracing.ScopedZone("updateMapSegmentation"):
        im = algorithmResultsHolder.topDownImg

        if im is None:
            return

        strategy = get_map_segmentation_strategy()
        robotMask = buildRobotExclusionMask(algorithmResultsHolder, im.shape)

        masks = strategy.segment(im, robotMask=robotMask)

        if masks is None:
            logger.warning(f"Map segmentation ({strategy.name}) produced no result")
            return

        algorithmResultsHolder.mapSegmentationRedStopLines = masks.redStopLines
        algorithmResultsHolder.mapSegmentationYellowLines = masks.yellowLines
        algorithmResultsHolder.mapSegmentationRoadComponents = masks.roadComponents
