from logging import Logger
from classes.resources.PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder
import cv2
import numpy as np
from utils.geometry import circle
from utils.tracing import Tracing

# NOTE: below thresholds obtained from classical-map-segmentation-threshold-opt.ipynb
THRESHOLDS = {
    "road": ("Y", [1.0, 153.0]),
    "stop": ("A", [153.0, 204.0]),
    "sep": ("B", [149.0, 207.0]),
}


def applyThresholds(
    channels: dict[str, np.ndarray],
    thresholds: tuple[str, list[float]],
) -> np.ndarray:
    channel_name, (thresh_min, thresh_max) = thresholds
    channel = channels[channel_name]

    M = (channel >= thresh_min) & (channel <= thresh_max)

    return M


def postProcessRoads(MRoad: np.ndarray, K: np.ndarray):
    MRoad = MRoad.astype(np.uint8)

    _num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(MRoad.astype(np.uint8), connectivity=8)

    if len(stats) < 2:
        return

    # get the largest component (exclude the background)
    largest_label = (
        # skip the background (label 0)
        np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    )

    largest_component_mask = np.zeros_like(MRoad, dtype=np.bool_)
    largest_component_mask[labels == largest_label] = 1

    MRoad = MRoad & largest_component_mask

    MRoad = cv2.morphologyEx(MRoad, cv2.MORPH_CLOSE, K, iterations=4)

    return MRoad * 255


def postProcessStop(MStop: np.ndarray, MRoad: np.ndarray, K: np.ndarray):
    MStop = MStop & MRoad

    # post-processing of red lines: remove noise, fill full rectangles
    MStop = cv2.morphologyEx(MStop.astype(np.uint8), cv2.MORPH_OPEN, K)

    # find contours
    contours, _ = cv2.findContours(MStop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    MStop = np.zeros_like(MStop)

    # approximate all contours with bounding boxes to fill rectangles (as expected)
    for contour in contours:
        # get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # draw the rectangle on the new mask
        cv2.rectangle(MStop, (x, y), (x + w, y + h), (255,), thickness=-1)

    return MStop


def postProcessSep(
    MSep: np.ndarray,
    MRoad: np.ndarray,
    MStop: np.ndarray,
    KSize: tuple[int, int],
    KLarge: np.ndarray,
    KDenoise: np.ndarray,
):
    KSepMorphOpen = np.ones((3, 3))
    MSep = MSep & MRoad & (~MStop)
    MSep = cv2.morphologyEx(MSep.astype(np.uint8), cv2.MORPH_OPEN, KSepMorphOpen)
    MSep = cv2.morphologyEx(MSep.astype(np.uint8), cv2.MORPH_OPEN, KSepMorphOpen)

    KDiagonalR = np.eye(KSize[0], KSize[1], dtype=np.uint8)
    KDiagonalL = np.fliplr(KDiagonalR)

    MSep = cv2.morphologyEx(
        MSep,
        cv2.MORPH_CLOSE,
        KLarge,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    ).astype(np.bool_)

    diagonalEmphasisIters = 2
    MSep = cv2.morphologyEx(
        MSep.astype(np.uint8),
        cv2.MORPH_DILATE,
        KDiagonalR,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
        iterations=diagonalEmphasisIters,
    ).astype(np.bool_)
    MSep = cv2.morphologyEx(
        MSep.astype(np.uint8),
        cv2.MORPH_DILATE,
        KDiagonalL,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(0,),
        iterations=diagonalEmphasisIters,
    ).astype(np.bool_)

    MSep = (
        cv2.morphologyEx(
            MSep.astype(np.uint8),
            cv2.MORPH_ERODE,
            KDenoise,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(0,),
        )
        * 255
    )

    return MSep


def updateMapSegmentation(
    algorithmResultsHolder: PreallocatedAlgorithmResultsHolder,
    logger: Logger,
):
    with Tracing.ScopedZone("updateMapSegmentation"):
        im = algorithmResultsHolder.topDownImg

        if im is None:
            return

        imgYCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
        imgLAB = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

        [Y, Cr, Cb] = imgYCrCb[:, :, 0], imgYCrCb[:, :, 1], imgYCrCb[:, :, 2]
        Y = cv2.equalizeHist(Y)

        [L, A, B] = imgLAB[:, :, 0], imgLAB[:, :, 1], imgLAB[:, :, 2]

        channels = {
            "Y": Y,
            "Cr": Cr,
            "Cb": Cb,
            "L": L,
            "A": A,
            "B": B,
        }

        # 'raw' thresholding
        MStop = applyThresholds(channels, THRESHOLDS["stop"])
        MSep = applyThresholds(channels, THRESHOLDS["sep"])
        MRoad = applyThresholds(channels, THRESHOLDS["road"])

        # precompute kernels for morphological operations
        KSizeSmall = (7, 7)
        KSizeLarge = (13, 13)
        K = np.ones(KSizeSmall, np.uint8)
        KLarge = np.ones(KSizeLarge, np.uint8)
        KDenoise = circle(np.min(KSizeSmall))

        # post-processing of road components
        MRoad = postProcessRoads(MRoad, K=K)

        if MRoad is None:
            logger.warning("No road components found")
            return

        # post-processing of red stop lines
        MStop = postProcessStop(MStop, MRoad, K)

        # post-processing of yellow lane separator lines
        MSep = postProcessSep(
            MSep=MSep,
            MRoad=MRoad,
            MStop=MStop,
            KLarge=KLarge,
            KDenoise=KDenoise,
            KSize=KSizeSmall,
        )

        algorithmResultsHolder.mapSegmentationRedStopLines = MStop
        algorithmResultsHolder.mapSegmentationYellowLines = MSep
        algorithmResultsHolder.mapSegmentationRoadComponents = MRoad
