import itertools
import math
from logging import Logger
from typing import Generator, Iterable, cast

import cv2
import numpy as np
from algorithms.homography import calculateHomography
from algorithms.robot_kalman_tracker import (
    RobotTagMeasurement,
    compute_tag_measurement_sigmas,
    quad_area_tl_tr_bl_br,
)
from algorithms.transformations import stitchImages
from algorithms.transformationValidators import validateAspectRatioClose
from classes.config import AlgorithmConfig
from classes.map import CornerDetection
from classes.marker import MarkerDetection
from classes.marker.CompositeDetection import CompositeDetection
from classes.resources.AlgorithmState import AlgorithmState
from classes.resources.InterThreadMemory import CacheEvent, InterThreadMemory
from classes.resources.ImagePack import ImagePack
from classes.resources.PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder
from classes.resources.PreallocatedStitchingResultsHolder import PreallocatedStitchingResultsHolder
from classes.resources.StitchingStageDescription import StitchingStageDescription
from classes.resources.StitchingState import StitchingState
from classes.robot import RobotDetection
from classes.types import CornerPointsList, FloatingPoint2D, Point2D
from errors import ProcessingError, StitchingStrategyInfeasibleError
from utils.constants import CROP_IMAGE_TO_ROI_IF_EXCEEDS_BASE_IMAGE_SIZE_PERCENT, RECALC_HEURISTIC_DISPLACED_DETECTIONS_MIN_DIFF_NORM_PERCENT_THRESH, neutralBorderColorBGR
from utils.detections import detectionsToNumpy, numpyToDetections
from utils.geometry import pointsDistance
from utils.map import genCorner, genRectOfCornerRects, getSquareExtremeCornerPoints, packCornerDetection
from utils.math import minmax
from utils.plotting import draw_coasted_robot_pose_on_topdown, drawDetectionsOnImage, genUniqueColors, putText
from utils.tracing import Tracing

# since the order of points in our system is TL, TR, BL, BR and here for a continuous square it is TL, TR, BR, BL,
# we need to sort the points in the order of TL, TR, BR, BL:
cornerIdxOrderingMappingTLTRBRBL = {i: i for i in range(4)}
cornerIdxOrderingMappingTLTRBRBL[2], cornerIdxOrderingMappingTLTRBRBL[3] = (
    cornerIdxOrderingMappingTLTRBRBL[3],
    cornerIdxOrderingMappingTLTRBRBL[2],
)


StitchingPairOrSingle = tuple[ImagePack, ImagePack] | tuple[ImagePack]


def matchPairsForStitching(
    imagePacks: list[ImagePack],
) -> Generator[StitchingPairOrSingle, None, None]:
    n = len(imagePacks)
    indices = list(range(n))

    # Precompute edge weights (common detections)
    weights = {}
    for i, j in itertools.combinations(indices, 2):
        w = len(imagePacks[i].compositeDetectionsStore.keys() & imagePacks[j].compositeDetectionsStore.keys())
        a, b = sorted((i, j))
        weights[(a, b)] = w

    def pairings(indices_left: list[int]):
        """Generate all possible pairings (brute force, order-stable)."""
        if not indices_left:
            yield []
        else:
            i = indices_left[0]
            for j in indices_left[1:]:
                a, b = sorted((i, j))
                rest_indices = [x for x in indices_left if x not in (a, b)]
                for rest in pairings(rest_indices):
                    yield [(a, b)] + rest

    # Generate all pairings (exponential, but fine for small sets)
    # Only keep pairings where every pair has > 0 common detections
    all_pairings = []
    for pairs in pairings(indices if n % 2 == 0 else indices[:-1]):
        if not pairs:
            continue
        min_weight = min(weights[p] for p in pairs)
        if min_weight == 0:
            continue  # Discard pairings with any 0-common pair
        total_weight = sum(weights[p] for p in pairs)
        all_pairings.append((min_weight, total_weight, pairs))

    if not all_pairings:
        raise StitchingStrategyInfeasibleError(
            None,
            None,
            "No valid pairings without zero-common-detection pairs could be produced for stitching",
        )

    # Pick pairing with best balance (maximize min, then total)
    _, _, best_pairs = max(all_pairings, key=lambda x: (x[0], x[1]))

    used: set[int] = set()
    pairs_out: list[StitchingPairOrSingle] = []

    for a, b in best_pairs:
        used.add(a)
        used.add(b)
        pairs_out.append((imagePacks[a], imagePacks[b]))

    # If odd, add leftover
    if n % 2 == 1:
        leftover = set(indices) - used
        if leftover:
            pairs_out.append((imagePacks[min(leftover)],))

    # Sort results by stitchingStageDescription (first element in each tuple)
    pairs_out.sort(key=lambda pair: pair[0].stitchingStageDescription)

    for p in pairs_out:
        yield p


def stitchImagePacks(
    imagePacks: list[ImagePack],
    config: AlgorithmConfig,
    state: StitchingState,
    stitchingStageDescription: StitchingStageDescription,
    logger: Logger,
    resultsHolder: PreallocatedStitchingResultsHolder,
    memory: InterThreadMemory | None = None,
) -> ImagePack:
    try:
        with Tracing.ScopedZone(f"stitch {stitchingStageDescription.__str__()}") as stitchImagePacksZone:
            if len(imagePacks) < 2:
                raise ProcessingError(
                    "stitchImagePacks expects at least 2 image packs",
                    code="STITCH_IMAGE_PACKS_TOO_FEW",
                    value=f"count={len(imagePacks)}",
                )
            logger.debug(f"Stitching stage '{stitchingStageDescription}' started")

            img1 = imagePacks[0].image
            img1Desc = imagePacks[0].stitchingStageDescription.__str__()

            img2 = imagePacks[1].image
            img2Desc = imagePacks[1].stitchingStageDescription.__str__()

            [camera1Detections, camera2Detections] = [pack.compositeDetectionsStore.values() for pack in imagePacks]

            camera1Store = imagePacks[0].compositeDetectionsStore
            camera2Store = imagePacks[1].compositeDetectionsStore
            commonDetectionKeys = list(set(camera1Store.keys()) & set(camera2Store.keys()))

            # if any of the source image packs is dirty, it means that its processing in stitchImagePacks resulted in the
            # homography being recalculated, so we need to recalculate the homography here as well
            anySourceImagePackDirty: bool = False
            for pack in imagePacks:
                if pack.dirty:
                    anySourceImagePackDirty = True

            if config.streamProcessingVisualizations["COMMON_FEATURES_IMAGE"]:
                annotatedImages: list[np.ndarray] = []
                for pack in imagePacks:
                    detections = pack.compositeDetectionsStore.values()
                    imageAnnotated = drawDetectionsOnImage(pack.image.copy(), detections, config)
                    annotatedImages.append(imageAnnotated)

                # ensure same common images preview size for all images
                maxImgHeight = max([img1.shape[0], img2.shape[0]])
                maxImgWidth = max([img1.shape[1], img2.shape[1]])
                imgOffsets: list[tuple[int, int]] = []
                for i, img in enumerate(annotatedImages):
                    [imgH, imgW, _] = img.shape
                    ox = maxImgWidth - imgW
                    oy = maxImgHeight - imgH
                    imgOffsets.append((ox, oy))

                with Tracing.ScopedZone("visualizeCommonFeatures"):
                    annotatedImagesResized = [img.copy() for img in annotatedImages]
                    for i, (img, (ox, oy)) in enumerate(zip(annotatedImagesResized, imgOffsets)):
                        annotatedImagesResized[i] = cv2.copyMakeBorder(
                            img,
                            0,
                            oy,
                            0,
                            ox,
                            borderType=cv2.BORDER_CONSTANT,
                            value=0,  # type: ignore
                        )

                    commonFeaturesImg = np.concatenate(annotatedImagesResized, axis=1)

                    for key, colorBGR in zip(
                        commonDetectionKeys,
                        genUniqueColors(len(commonDetectionKeys)),
                    ):
                        detectionImageA = camera1Store[key]
                        detectionImageB = camera2Store[key]

                        # imageAOffsetX = sum(
                        #     [pack[1].shape[1] for pack in imagePacks[:cameraIndex1]]
                        # )
                        # imageBOffsetX = sum(
                        #     [pack[1].shape[1] for pack in imagePacks[:cameraIndex2]]
                        # )
                        # since the border added to image A is bottom and right only, we only need to add it to image B, not A
                        # unused due to reason described above: + imgOffsets[1][0]
                        imageBOffsetX = img1.shape[1] + imgOffsets[0][0]
                        # unused due to reason described above: + imgOffsets[1][1]
                        # imageBOffsetY = imgOffsets[0][0]

                        # calculate the start and end points of the line
                        for dimenAccessor in [
                            "aggregatedTopLeft",
                            "aggregatedTopRight",
                            "aggregatedBottomLeft",
                            "aggregatedBottomRight",
                        ]:
                            dimenA: FloatingPoint2D = getattr(detectionImageA, dimenAccessor)
                            dimenB: FloatingPoint2D = getattr(detectionImageB, dimenAccessor)
                            start_point: Point2D = (int(dimenA[0]), int(dimenA[1]))
                            end_point: Point2D = (
                                int(dimenB[0]) + imageBOffsetX,
                                int(dimenB[1]),
                            )

                            cv2.line(
                                img=commonFeaturesImg,
                                pt1=start_point,
                                pt2=end_point,
                                color=colorBGR,
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )

                    resultsHolder.commonFeaturesImg = commonFeaturesImg

            camera1CommonDetections = [camera1Store[key] for key in commonDetectionKeys]
            camera2CommonDetections = [camera2Store[key] for key in commonDetectionKeys]

            # if we have enough detections, filter
            nonRobotCommonDetectionIds = {detection.data for detection in camera1CommonDetections if detection.data not in config.configuredRobotIDs}

            if len(nonRobotCommonDetectionIds) > 2:
                logger.debug(f"Using only non-robot detections in the stitching process for stability since there are enough of them {len(camera1CommonDetections)} vs {len([detection for detection in camera1CommonDetections if detection.data in nonRobotCommonDetectionIds])}")

                camera1CommonDetections = [detection for detection in camera1CommonDetections if detection.data in nonRobotCommonDetectionIds]
                camera2CommonDetections = [detection for detection in camera2CommonDetections if detection.data in nonRobotCommonDetectionIds]

            # if pre-stitching detections are displaced w.r.t. previous ones,
            # the camera must've been moved and therefore the cache should be invalidated
            for key, img, detections in [
                ("img1", img1, camera1CommonDetections),
                ("img2", img2, camera2CommonDetections),
            ]:
                if len(detections) == 0:
                    continue
                detectionsById = {detection.data: detection for detection in detections}
                maybeChangeReason = state.markingCheckHaveDetectionsChanged(
                    detectionsById,
                    config,
                )
                if maybeChangeReason:
                    reason = f"{key}: {maybeChangeReason}"
                    msg = f"Marking stitching state '{stitchingStageDescription}' as dirty due to {reason}"
                    logger.info(msg)
                    Tracing.message(msg)

                    resultsHolder.lastRecalcReason = reason
                    state.markNeedsHomographyRecalculation("⛓️‍💥 " + reason)
                    if memory is not None:
                        memory.emit_cache_event(
                            CacheEvent(
                                event_type="homography_cache_invalidated",
                                data={
                                    "stage": stitchingStageDescription.__str__(),
                                    "reason": reason,
                                    "cache_type": "stitching",
                                },
                            )
                        )
                    break

            cacheDisabled = not config.caching.stitchingHomographyCache
            shouldRecalculateHomography = cacheDisabled or state.shouldRecalculateHomography()
            # check if the homography needs to be revalidated
            if shouldRecalculateHomography or anySourceImagePackDirty:
                stitchImagePacksZone.color(Tracing.COLOR_ORANGE)

                if cacheDisabled:
                    reasonText = "cache disabled"
                else:
                    reasonText = "StitchingState requires recalculation" if shouldRecalculateHomography else "source images were dirty"
                msg = f"Updating homography for stitching '{stitchingStageDescription}' (reason: {reasonText})"
                logger.info(f"🔢 {msg} 🔢")
                Tracing.message(msg)
                resultsHolder.cacheMisses += 1
                if cacheDisabled:
                    resultsHolder.lastRecalcReason = "cache_disabled"
                else:
                    resultsHolder.lastRecalcReason = "state_recalc" if shouldRecalculateHomography else "source_dirty"

                with Tracing.ScopedZone("recalculateStitchingHomography"):
                    if len(camera1CommonDetections) == 0 or len(camera2CommonDetections) == 0:
                        raise ProcessingError(
                            f"No common detections have been found between processed cameras {img1Desc} & {img2Desc}",
                            code="NO_COMMON_DETECTIONS",
                        )

                    H = calculateHomography(camera2CommonDetections, camera1CommonDetections)

                    bHomographyUpdated = True

                    state.updateCachedHomography(H)

                    # heuristic: reprojection error check for marker points post-stitching
                    marker_src_pts = np.float32([detection.aggregatedCentroid for detection in camera1CommonDetections]).reshape(-1, 2)
                    marker_dst_pts = np.float32([detection.aggregatedCentroid for detection in camera2CommonDetections]).reshape(-1, 2)
                    if len(marker_src_pts) >= 4 and len(marker_dst_pts) >= 4:
                        # Project marker_dst_pts using H and compare to marker_src_pts
                        marker_dst_pts_hom = np.concatenate(
                            [marker_dst_pts, np.ones((marker_dst_pts.shape[0], 1))],
                            axis=1,
                        )
                        projected = (H @ marker_dst_pts_hom.T).T
                        projected = projected[:, :2] / projected[:, 2:3]
                        errors = np.linalg.norm(projected - marker_src_pts, axis=1)
                        mean_error = np.mean(errors)
                        if mean_error > min(*img1.shape[:2]) * RECALC_HEURISTIC_DISPLACED_DETECTIONS_MIN_DIFF_NORM_PERCENT_THRESH:
                            reason = f"high marker reprojection error: {mean_error:.2f}"
                            msg = f"Marking stitching state '{stitchingStageDescription}' as dirty due to {reason}"
                            Tracing.message(msg)
                            logger.info("⛓️‍💥 " + msg)
                            state.markNeedsHomographyRecalculation(reason)
                            resultsHolder.lastRecalcReason = reason
                            if memory is not None:
                                memory.emit_cache_event(
                                    CacheEvent(
                                        event_type="homography_cache_invalidated",
                                        data={
                                            "stage": stitchingStageDescription.__str__(),
                                            "reason": reason,
                                            "cache_type": "stitching",
                                        },
                                    )
                                )
            else:
                # use cached homography
                H = state.homography

                bHomographyUpdated = False

                msg = f"Reusing cached homography for stitching '{stitchingStageDescription}'"
                stitchImagePacksZone.color(Tracing.COLOR_GREEN)
                logger.info(f"🍀 {msg} 🍀")
                Tracing.message(msg)
                resultsHolder.cacheHits += 1

            npDetectionsCamera1, legendForNpDetectionsCamera1 = detectionsToNumpy(camera1Detections)
            npDetectionsCamera2, legendForNpDetectionsCamera2 = detectionsToNumpy(camera2Detections)

            try:
                (
                    stitchedImg,
                    warpedCamera1NumpyDetections,
                    warpedCamera2NumpyDetections,
                ) = stitchImages(
                    img1,
                    img2,
                    cast(np.ndarray, H),
                    npDetectionsImg1=npDetectionsCamera1,
                    npDetectionsImg2=npDetectionsCamera2,
                )
            except ProcessingError as e:
                state.markNeedsHomographyRecalculation(f"ProcessingError caught: {e.code}, {e.fullExplanation}")
                Tracing.message(f"Marking stitching state '{stitchingStageDescription}' as dirty due to error {e.code}, {e.fullExplanation}")
                if memory is not None:
                    memory.emit_cache_event(
                        CacheEvent(
                            event_type="homography_cache_invalidated",
                            data={
                                "stage": stitchingStageDescription.__str__(),
                                "reason": f"{e.code}: {e.fullExplanation}",
                                "cache_type": "stitching",
                            },
                        )
                    )
                raise e

            with Tracing.ScopedZone("warpDetectionsStitching"):
                camera1DetectionsOnStitched: set[CompositeDetection] = set(numpyToDetections(warpedCamera1NumpyDetections, legendForNpDetectionsCamera1))
                camera2DetectionsOnStitched: set[CompositeDetection] = set(numpyToDetections(warpedCamera2NumpyDetections, legendForNpDetectionsCamera2))

                result = ImagePack(
                    image=stitchedImg,
                    stitchingStageDescription=stitchingStageDescription,
                    isStitchingResult=True,
                    dirty=bHomographyUpdated,
                )

                result.appendStitchedDetections(camera1DetectionsOnStitched)
                result.appendStitchedDetections(camera2DetectionsOnStitched)

                logger.debug(f"Stitching stage '{stitchingStageDescription}' finished")

            return result
    except Exception as e:
        logger.error(f"Error in stitchImagePacks for stage '{stitchingStageDescription}': {e}")
        raise e


def topDownWarp(
    stitchingImagePack: ImagePack,
    state: AlgorithmState,
    config: AlgorithmConfig,
    logger: Logger,
    resultsHolder: PreallocatedAlgorithmResultsHolder,
    memory: InterThreadMemory | None = None,
):
    with Tracing.ScopedZone("topDownWarp") as topDownWarpZone:
        logger.debug(" == Processing sample == ")

        stitchedImg = stitchingImagePack.image
        detectionsOnStitched = stitchingImagePack.compositeDetectionsStore.values()
        if len(detectionsOnStitched) == 0:
            raise ProcessingError(
                "No detections found on stitched image",
                code="NO_DETECTIONS_ON_STITCHED_IMAGE",
            )

        maybeTransformUpdateReason = state.maybeGetTransformUpdateReason()

        # if any of the source image packs is dirty, it means that its processing in stitchImagePacks resulted in the
        # homography being recalculated, so we need to recalculate the homography here as well
        if maybeTransformUpdateReason is not None:
            if memory is not None:
                memory.emit_cache_event(
                    CacheEvent(
                        event_type="homography_cache_invalidated",
                        data={
                            "stage": "top_down",
                            "reason": maybeTransformUpdateReason,
                            "cache_type": "top_down",
                        },
                    )
                )
            resultsHolder.topDownCacheMisses += 1
            resultsHolder.recomputeReasons[maybeTransformUpdateReason] = resultsHolder.recomputeReasons.get(maybeTransformUpdateReason, 0) + 1
            topDownWarpZone.color(Tracing.COLOR_ORANGE)
            topDownWarpZone.text(f"Update reason: {maybeTransformUpdateReason}")

            msg = f"Recalculating top-down transformation (reason: {maybeTransformUpdateReason})"
            logger.info(f"🔢 {msg} 🔢")
            Tracing.message(msg)

            # try to estimate a transformation from the current stitched image perspective to a top-down perspective, using the detected corner markers
            foundMarkerIDs = set(map(lambda x: x.data, detectionsOnStitched))
            missingMarkerIDs = set(config.mapDefinition.values()).difference(foundMarkerIDs)

            foundCornerDetections: set[CornerDetection] = set([packCornerDetection(detection, config) for detection in detectionsOnStitched if detection.data not in missingMarkerIDs])

            top_left_corner: CornerPointsList | None = None
            top_right_corner: CornerPointsList | None = None
            bottom_left_corner: CornerPointsList | None = None
            bottom_right_corner: CornerPointsList | None = None

            for detection in foundCornerDetections:
                if detection.averagedMarkerDetection is not None:
                    corner = detection.averagedMarkerDetection.toPointsList()

                    if detection.isTopLeft:
                        top_left_corner = corner
                    elif detection.isTopRight:
                        top_right_corner = corner
                    elif detection.isBottomLeft:
                        bottom_left_corner = corner
                    elif detection.isBottomRight:
                        bottom_right_corner = corner

            if len(foundCornerDetections) == 0:
                raise ProcessingError(
                    "No corner detections have been found that are distinguishable in the stitched image; ensure that the corners seen by cameras aren't lying perfectly on the same line (move them a bit)",
                    code="NO_CORNER_DETECTIONS",
                )

            maxSize = max(*stitchedImg.shape)
            unfilteredHomographyCornersRect = genRectOfCornerRects(width=maxSize, height=maxSize / config.MAP_RATIO_W_TO_H, config=config)

            unfilteredHomographyCornersPointListsInWarped = [
                top_left_corner,
                top_right_corner,
                bottom_left_corner,
                bottom_right_corner,
            ]
            unfilteredHomographyCornersData = [
                config.mapDefinition["TL"],
                config.mapDefinition["TR"],
                config.mapDefinition["BL"],
                config.mapDefinition["BR"],
            ]

            filteredHomographyCornersPointListsInWarped = [corner for corner in unfilteredHomographyCornersPointListsInWarped if corner is not None]

            if len(filteredHomographyCornersPointListsInWarped) < 1:
                raise ProcessingError(
                    "Expected to find at least 1 corner detection in warped image.",
                    code="TOO_LITTLE_CORNERS_DETECTED",
                    value=f"detected corners - {len(filteredHomographyCornersPointListsInWarped)}",
                )

            filteredHomographyCornersData = [
                data
                for corner, data in zip(
                    unfilteredHomographyCornersPointListsInWarped,
                    unfilteredHomographyCornersData,
                )
                if corner is not None
            ]

            stitchedImgPreview = stitchedImg.copy()

            for i, (cornerPoints, data) in enumerate(
                zip(
                    filteredHomographyCornersPointListsInWarped,
                    filteredHomographyCornersData,
                )
            ):
                tempCorner = MarkerDetection(*cornerPoints, data=data)
                cornerCentroid = tempCorner.centroid

                desiredCorner = config.markerDataToCornerSpec[data]

                putText(
                    stitchedImgPreview,
                    f"{data}, {desiredCorner}",
                    (int(cornerCentroid[0]), int(cornerCentroid[1] - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    3,
                )

                resultsHolder.stitchedImg = stitchedImgPreview

            filteredHomographyCornersRect = [cornerPointsList for i, cornerPointsList in enumerate(unfilteredHomographyCornersRect) if unfilteredHomographyCornersPointListsInWarped[i] is not None]

            presentHomographyCornersInWarped = np.array(filteredHomographyCornersPointListsInWarped).flatten().reshape(-1, 2)

            presentHomographyCornersRect = np.array([cornerPointsList for cornerPointsList in filteredHomographyCornersRect]).flatten().reshape(-1, 2)

            if (not config.caching.topDownHomographyCache) or state.cornersH is None or stitchingImagePack.dirty:
                if memory is not None and maybeTransformUpdateReason is None:
                    memory.emit_cache_event(
                        CacheEvent(
                            event_type="homography_cache_invalidated",
                            data={
                                "stage": "top_down",
                                "reason": "cornersH recalc (dirty or None)",
                                "cache_type": "top_down",
                            },
                        )
                    )
                with Tracing.ScopedZone("recalculateTopDownHomography"):
                    # initialize or recalculate the homography matrix based on corner points
                    cornersH = calculateHomography(
                        presentHomographyCornersRect,
                        presentHomographyCornersInWarped,
                    )

                    if config.caching.topDownHomographyCache:
                        if state._homographyBufferPreallocation and state._cornersHBuffer is not None:
                            np.copyto(state._cornersHBuffer, cornersH)
                            state.cornersH = state._cornersHBuffer
                        else:
                            state.cornersH = cornersH.copy()
            else:
                # use cached corner-warping homography
                cornersH = state.cornersH

            squareCornersWarped: Iterable[CornerPointsList] = cv2.perspectiveTransform(
                np.array(
                    [corner for corner in unfilteredHomographyCornersRect],
                    dtype=np.float32,
                ),
                cornersH,
            )

            state.allCornerMarkers = [
                MarkerDetection(
                    *[(int(p[0]), int(p[1])) for p in [tuple(lst) for lst in corner]],
                    data=unfilteredHomographyCornersData[idx],
                )
                for idx, corner in enumerate(squareCornersWarped)
            ]

            # clip outlier corners so as not to produce extremely large images
            for corner in state.allCornerMarkers:
                for cornerKey in ["topLeft", "topRight", "bottomLeft", "bottomRight"]:
                    point = getattr(corner, cornerKey)

                    origType = type(point[0])
                    threshAbsXY = (
                        origType(stitchedImg.shape[1] * (1 + CROP_IMAGE_TO_ROI_IF_EXCEEDS_BASE_IMAGE_SIZE_PERCENT)),
                        origType(stitchedImg.shape[0] * (1 + CROP_IMAGE_TO_ROI_IF_EXCEEDS_BASE_IMAGE_SIZE_PERCENT)),
                    )

                    if (abs(point[0]), abs(point[1])) >= threshAbsXY:
                        setattr(
                            corner,
                            cornerKey,
                            (
                                max(-threshAbsXY[0], min(point[0], threshAbsXY[0])),
                                max(-threshAbsXY[1], min(point[1], threshAbsXY[1])),
                            ),
                        )

        # == Visualization block (stitchedImgAnnotated) - start ==
        stitchedImgAnnotated = drawDetectionsOnImage(
            stitchedImg.copy(),
            detectionsOnStitched,
            config,
            (90, 127, 255),  # (255, 0, 0),
            False,
        )
        allPointsIter = list(itertools.chain(*[corner.toPointsList() for corner in state.allCornerMarkers]))
        (minX, maxX) = minmax([p[0] for p in allPointsIter])
        (minY, maxY) = minmax([p[1] for p in allPointsIter])
        offsetLeft = int(math.ceil(-min(0, minX)))
        offsetTop = int(math.ceil(-min(0, minY)))
        offsetBottom = int(math.ceil(max(0, maxY - stitchedImgAnnotated.shape[0])))
        offsetRight = int(math.ceil(max(0, maxX - stitchedImgAnnotated.shape[1])))
        stitchedImgAnnotatedBorderOffset = (offsetLeft, offsetTop)

        # if the extrapolated corners are out-of-bounds, extend the image to be able to see them
        if offsetBottom > 0 or offsetRight > 0 or offsetLeft > 0 or offsetTop > 0:
            stitchedImgAnnotated = cv2.copyMakeBorder(
                stitchedImgAnnotated,
                top=offsetTop,
                bottom=offsetBottom,
                left=offsetLeft,
                right=offsetRight,
                borderType=cv2.BORDER_CONSTANT,
                value=neutralBorderColorBGR,
            )

        stitchedImgAnnotated = drawDetectionsOnImage(
            stitchedImgAnnotated,
            state.allCornerMarkers,
            config,
            (90, 255, 255),
            offset=stitchedImgAnnotatedBorderOffset,
        )

        # == Visualization block (stitchedImgAnnotated) - end; some visualizations are inside the code below ==

        if maybeTransformUpdateReason is not None:
            # draw markers on extreme points that will be boundaries of the new map after warping the perspective
            state.extremeCornerPoints = getSquareExtremeCornerPoints(state.allCornerMarkers)

            rawPt = state.extremeCornerPoints[cornerIdxOrderingMappingTLTRBRBL[3]]
            prevPt = (
                int(rawPt[0] + stitchedImgAnnotatedBorderOffset[0]),
                int(rawPt[1] + stitchedImgAnnotatedBorderOffset[1]),
            )
            for i in range(4):
                rawPt = state.extremeCornerPoints[cornerIdxOrderingMappingTLTRBRBL[i]]
                pt = (
                    int(rawPt[0] + stitchedImgAnnotatedBorderOffset[0]),
                    int(rawPt[1] + stitchedImgAnnotatedBorderOffset[1]),
                )
                cv2.line(stitchedImgAnnotated, prevPt, pt, (255, 0, 255), 2)
                cv2.circle(stitchedImgAnnotated, pt, 10, (255, 0, 255), -1)
                prevPt = pt

            for marker in state.allCornerMarkers:
                cv2.drawMarker(
                    stitchedImgAnnotated,
                    [
                        int(marker.topLeft[0] + stitchedImgAnnotatedBorderOffset[0]),
                        int(marker.topLeft[1] + stitchedImgAnnotatedBorderOffset[1]),
                    ],
                    (255, 255, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=20,
                    thickness=8,
                )

            # crop & transform the stitched image to the top-down perspective
            state.transformedWidth = max(
                # top-left corner's top-left vertex to top-right corner's top-right vertex
                abs(pointsDistance(state.extremeCornerPoints[0], state.extremeCornerPoints[1])),
                # bottom-left corner's bottom-left vertex to bottom-right corner's bottom-right vertex
                abs(pointsDistance(state.extremeCornerPoints[2], state.extremeCornerPoints[3])),
            )
            state.transformedHeight = max(
                # top-left corner's top-left vertex to bottom-left corner's bottom-left vertex
                abs(pointsDistance(state.extremeCornerPoints[0], state.extremeCornerPoints[2])),
                # top-right corner's top-right vertex to bottom-right corner's bottom-right vertex
                abs(pointsDistance(state.extremeCornerPoints[1], state.extremeCornerPoints[3])),
            )

            resultsHolder.stitchedImgAnnotated = stitchedImgAnnotated

            # heuristic: evaluate if the transformation may even be near successful and fail otherwise, based on the aspect ratio of the transformed image
            stitchedImgAspectRatio = state.transformedWidth / state.transformedHeight
            if not validateAspectRatioClose(
                targetAspect=config.MAP_RATIO_W_TO_H,
                actualAspect=stitchedImgAspectRatio,
                tolerancePercent=(1 - config.toleranceAspectRatio) * 100,
            ):
                raise ProcessingError(
                    (
                        "The transformation of the stitched image to top-down "
                        "perspective resulted in a malformed image - most likely, "
                        "arrangement of keypoints is not viable to compute a precise "
                        "enough homography to properly stitch images (aspect ratio "
                        f"of {stitchedImgAspectRatio:.1f} is over "
                        f"{config.toleranceAspectRatio * 100:.1f}% different than "
                        f"the expected value, which is {config.MAP_RATIO_W_TO_H:.1f})."
                    ),
                    code="INVALID_ASPECT_RATIO",
                    value=(f"actual aspect = {stitchedImgAspectRatio:.1f}, target is {config.MAP_RATIO_W_TO_H:.1f}, tolerance = {config.toleranceAspectRatio * 100:.1f}%"),
                )

            if state.transformedWidth > state.transformedHeight:
                state.transformedHeight = state.transformedWidth / config.MAP_RATIO_W_TO_H
            else:
                state.transformedWidth = state.transformedHeight * config.MAP_RATIO_W_TO_H

            topDownSquare = genCorner(
                xOffset=0,
                yOffset=0,
                width=state.transformedWidth,
                height=state.transformedHeight,
            )

            m = cv2.getPerspectiveTransform(
                np.array(state.extremeCornerPoints, dtype=np.float32),
                np.array(topDownSquare, dtype=np.float32),
            )
            if state._homographyBufferPreallocation and state._MBuffer is not None:
                np.copyto(state._MBuffer, m)
                state.M = state._MBuffer
            else:
                state.M = m

            state.handleTransformUpdated()
        else:
            msg = "Reusing top-down transformation state"
            logger.info(f"🍀 {msg} 🍀")
            topDownWarpZone.color(Tracing.COLOR_GREEN)
            Tracing.message(msg)
            resultsHolder.topDownCacheHits += 1

        if state.transformedWidth is None or state.transformedHeight is None:
            raise ProcessingError(
                "Transformed width or height is None; this should not happen",
                code="TRANSFORMED_DIMENSIONS_NOT_SET",
            )

        dsize = (
            int(math.ceil(state.transformedWidth)),
            int(math.ceil(state.transformedHeight)),
        )
        dstH, dstW = int(math.ceil(state.transformedHeight)), int(math.ceil(state.transformedWidth))

        try:
            preallocSlot = resultsHolder.ensure_top_down_buffer(dstH, dstW)
            if preallocSlot is not None:
                dstBuf = preallocSlot.buffer
                topDownImg = cv2.warpPerspective(
                    stitchedImg,
                    cast(np.ndarray, state.M),
                    (dstW, dstH),
                    dst=dstBuf,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=neutralBorderColorBGR,
                )
            else:
                topDownImg = cv2.warpPerspective(
                    stitchedImg,
                    cast(np.ndarray, state.M),
                    dsize,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=neutralBorderColorBGR,
                )
        except cv2.error as e:
            raise ProcessingError(
                "cv2.error caught during topDownImg cv2.warpPerspective call",
                code="OPENCV_ERROR",
                value=e.__str__(),
            )

        npUnionOfDetectionsOnStitched, legendForNpUnionOfDetectionsOnStitched = detectionsToNumpy(detectionsOnStitched)

        topDownUnionOfDetections = numpyToDetections(
            cv2.perspectiveTransform(
                npUnionOfDetectionsOnStitched,
                cast(np.ndarray, state.M),
            ),
            legendForNpUnionOfDetectionsOnStitched,
        )

        # evaluate if the transformation may even be near successful and fail otherwise, based on whether any of the detections are over DETECTION_BOUNDS_TOLERANCE% outside the bounds of the top-down image
        # width-then-height
        topDownImgBounds = topDownImg.shape[1], topDownImg.shape[0]
        topDownUnionOfDetectionsCentroids = [d.aggregatedCentroid for d in topDownUnionOfDetections]

        for centroid in topDownUnionOfDetectionsCentroids:
            if centroid[0] < -config.detectionBoundsTolerance * topDownImgBounds[0] or centroid[0] > (1 + config.detectionBoundsTolerance) * topDownImgBounds[0] or centroid[1] < -config.detectionBoundsTolerance * topDownImgBounds[1] or centroid[1] > (1 + config.detectionBoundsTolerance) * topDownImgBounds[1]:
                raise ProcessingError(
                    (
                        "The transformation of the stitched image to top-down "
                        "perspective resulted in a malformed image - most likely, "
                        "arrangement of keypoints is not viable to compute a precise "
                        "enough homography to warp stitched image properly "
                        f"(some detections are over {config.detectionBoundsTolerance * 100:.1f}% "
                        "outside the bounds of the top-down image)."
                    ),
                    code="INVALID_DETECTIONS_BOUNDS",
                    value=(f"centroid = {centroid}, bounds = {topDownImgBounds}, tolerance = {config.detectionBoundsTolerance * 100:.1f}%"),
                )

        nonCornerDetections = set(topDownUnionOfDetections).difference(state.allCornerMarkers)

        preallocAnnotatedSlot = resultsHolder.ensure_top_down_annotated_buffer(dstH, dstW)
        if preallocAnnotatedSlot is not None:
            np.copyto(preallocAnnotatedSlot.buffer, topDownImg)
            topDownImgAnnotated = preallocAnnotatedSlot.buffer
        else:
            topDownImgAnnotated = topDownImg.copy()

        robot_measurements: list[RobotTagMeasurement] = []

        for detection in nonCornerDetections:
            foundRobot = config.findRobotWithID(detection.data)
            isRobot = foundRobot is not None
            topLeft = detection.aggregatedTopLeft
            topRight = detection.aggregatedTopRight
            bottomLeft = detection.aggregatedBottomLeft
            bottomRight = detection.aggregatedBottomRight
            [centroidX, centroidY] = detection.aggregatedCentroid

            if isRobot:
                imgRobotRelX = centroidX / topDownImg.shape[1]
                imgRobotRelY = centroidY / topDownImg.shape[0]

                frontAxle = (
                    (topLeft[0] + topRight[0]) / 2,
                    (topLeft[1] + topRight[1]) / 2,
                )
                rearAxle = (
                    (bottomLeft[0] + bottomRight[0]) / 2,
                    (bottomLeft[1] + bottomRight[1]) / 2,
                )
                dx = frontAxle[0] - rearAxle[0]
                dy = frontAxle[1] - rearAxle[1]
                azimuth = np.rad2deg(np.arctan2(dy, dx)) + 90  # 0 deg is North, increasing clockwise
                azimuth = (azimuth + 360) % 360

                if azimuth < 0:
                    azimuth += 360

                ih, iw = topDownImg.shape[0], topDownImg.shape[1]
                area_px = quad_area_tl_tr_bl_br(
                    (float(topLeft[0]), float(topLeft[1])),
                    (float(topRight[0]), float(topRight[1])),
                    (float(bottomLeft[0]), float(bottomLeft[1])),
                    (float(bottomRight[0]), float(bottomRight[1])),
                )
                baseline_px = float(np.hypot(dx, dy))
                sx, sy, spsi = compute_tag_measurement_sigmas(
                    area_px=area_px,
                    baseline_px=baseline_px,
                    img_w=iw,
                    img_h=ih,
                    p=foundRobot.kalman_params,
                )
                robot_measurements.append(
                    RobotTagMeasurement(
                        robot=foundRobot,
                        x=float(imgRobotRelX),
                        y=float(imgRobotRelY),
                        azimuth_deg=float(azimuth),
                        sigma_x=sx,
                        sigma_y=sy,
                        sigma_psi_rad=spsi,
                        composite=detection,
                    )
                )
            else:
                drawDetectionsOnImage(
                    topDownImgAnnotated,
                    [detection],
                    config,
                    colorBGR=(0, 0, 255),
                    bDrawLabels=False,
                )
                putText(
                    topDownImgAnnotated,
                    str(detection.data),
                    (int(topLeft[0]), int(topLeft[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    4,
                )

        fps_val = resultsHolder.fps
        if not math.isfinite(fps_val) or fps_val <= 0:
            kalman_dt = 1.0 / 30.0
        else:
            kalman_dt = 1.0 / float(fps_val)

        robotDetections: list[RobotDetection]
        if memory is not None:
            kalman_active = any(r.kalman_params.enabled for r in config.configuredRobots.values())
            if kalman_active:
                enabled_meas = [m for m in robot_measurements if m.robot.kalman_params.enabled]
                disabled_meas = [m for m in robot_measurements if not m.robot.kalman_params.enabled]
                robotDetections = memory.robot_kalman_tracker.step(enabled_meas, kalman_dt)
                robotDetections.extend(memory.robot_kalman_tracker.passthrough(disabled_meas))
                robotDetections.sort(key=lambda d: d.robot.id)
            else:
                robotDetections = memory.robot_kalman_tracker.passthrough(robot_measurements)
        else:
            robotDetections = []
            for m in robot_measurements:
                det = RobotDetection(m.robot, m.x, m.y, m.azimuth_deg, pose_source="detection")
                det.backing_composite = m.composite
                robotDetections.append(det)

        for det in robotDetections:
            imgRobotRelX = det.x
            imgRobotRelY = det.y
            azimuth = det.azimuth
            foundRobot = det.robot
            if det.backing_composite is not None:
                topLeft = det.backing_composite.aggregatedTopLeft
                drawDetectionsOnImage(
                    topDownImgAnnotated,
                    [det.backing_composite],
                    config,
                    colorBGR=(0, 255, 0),
                    bDrawLabels=False,
                )
                putText(
                    topDownImgAnnotated,
                    f"{foundRobot.host} ({foundRobot.id})",
                    (int(topLeft[0]), int(topLeft[1] - 18 * 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.25,
                    (255, 255, 255),
                    3,
                )
                putText(
                    topDownImgAnnotated,
                    f"az: {azimuth:.2f} deg",
                    (int(topLeft[0]), int(topLeft[1] - 18 * 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    4,
                )
                putText(
                    topDownImgAnnotated,
                    f"({imgRobotRelX:.2f}, {imgRobotRelY:.2f})",
                    (int(topLeft[0]), int(topLeft[1] - 18 * 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    4,
                )
            else:
                draw_coasted_robot_pose_on_topdown(
                    topDownImgAnnotated,
                    rel_x=imgRobotRelX,
                    rel_y=imgRobotRelY,
                    azimuth_deg=azimuth,
                    label=f"{foundRobot.host} ({foundRobot.id})",
                    config=config,
                    color_bgr=(0, 200, 255),
                )
                putText(
                    topDownImgAnnotated,
                    f"az: {azimuth:.2f} deg (est.)",
                    (int(imgRobotRelX * topDownImg.shape[1]) + 14, int(imgRobotRelY * topDownImg.shape[0]) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    3,
                )
                putText(
                    topDownImgAnnotated,
                    f"({imgRobotRelX:.2f}, {imgRobotRelY:.2f})",
                    (int(imgRobotRelX * topDownImg.shape[1]) + 14, int(imgRobotRelY * topDownImg.shape[0]) + 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    3,
                )

            Tracing.plot(
                f"Robot {foundRobot.id} pos. X",
                round(imgRobotRelX * 100),
                Tracing.PlotType.Percentage,
            )
            Tracing.plot(
                f"Robot {foundRobot.id} pos. Y",
                round(imgRobotRelY * 100),
                Tracing.PlotType.Percentage,
            )
            Tracing.plot(
                f"Robot {foundRobot.id} azimuth deg",
                round(azimuth, 1),
                Tracing.PlotType.Number,
            )

        resultsHolder.topDownImg = topDownImg
        resultsHolder.topDownImgAnnotated = topDownImgAnnotated
        resultsHolder.topDownDetections = list(topDownUnionOfDetections)
        resultsHolder.robotDetections = robotDetections
