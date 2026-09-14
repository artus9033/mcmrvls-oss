import itertools
import math
from logging import Logger
from typing import Generator, Iterable, cast

import cv2
import numpy as np
from algorithms.homography import calculateAffineHomography, calculateHomography
from algorithms.robot_kalman_tracker import (
    RobotTagMeasurement,
    compute_tag_measurement_sigmas,
    normalize_tag_geometry,
    quad_area_tl_tr_bl_br,
)
from algorithms.robot_tracking import resolve_robot_detections
from algorithms.solver_pipeline import collect_epipolar_robot_measurements
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


# A camera's floor homography is exact only near its observed tag support -
# per-camera fits extrapolate poorly away from it, and because all cameras are
# chained through the same tag set their extrapolation errors are CORRELATED,
# so multi-camera agreement cannot vouch for far points. Votes are therefore
# only accepted within this tight map-normalized radius of a camera's nearest
# support tag; everything further is an honest gap.
CAMERA_SUPPORT_RADIUS_NORM = 0.12


def _cameraSupportDistance(atlas, camera: str, pointNorm) -> float:
    support = getattr(atlas, "camera_support_norm", {}).get(camera)
    if support is None or len(support) == 0:
        return float("inf")
    return float(np.linalg.norm(support - np.asarray(pointNorm, dtype=np.float64), axis=1).min())


def _exactProjectPointToMosaic(
    atlas,
    stitchingImagePack: ImagePack,
    pointNorm,
    agreementTolPx: float,
) -> tuple[float, float] | None:
    """Project a map-normalized floor point onto the mosaic via the camera model.

    Each calibrated camera imaging the point (in-bounds, positive w, within its
    tight tag-support radius) casts a vote W_cam @ H_cam^-1 @ p. With multiple
    votes, they must agree within agreementTolPx (midpoint returned) - a
    consistency check against a stale warp, not a substitute for support.
    Returns None otherwise - an honest gap beats a wrong line.
    """
    votes: list[tuple[float, np.ndarray]] = []  # (support distance, stitched px)
    for camera, hCam in atlas.camera_h.items():
        warp = stitchingImagePack.cameraWarps.get(camera)
        size = stitchingImagePack.cameraSizes.get(camera)
        if warp is None or size is None:
            continue
        supportDist = _cameraSupportDistance(atlas, camera, pointNorm)
        if supportDist > CAMERA_SUPPORT_RADIUS_NORM:
            continue
        try:
            hCamInv = np.linalg.inv(hCam)
        except np.linalg.LinAlgError:
            continue
        pCamArr, camValid = _projectWithCheirality(hCamInv, np.array([pointNorm]))
        if not bool(camValid[0]):
            continue  # behind this camera's horizon (negative w)
        pCam = pCamArr[0]
        w, h = size
        if not (0 <= pCam[0] <= w and 0 <= pCam[1] <= h):
            continue
        pStitchedArr, warpValid = _projectWithCheirality(warp, pCamArr)
        if not bool(warpValid[0]):
            continue
        votes.append((supportDist, pStitchedArr[0]))

    if not votes:
        return None
    votes.sort(key=lambda v: v[0])
    if len(votes) >= 2:
        a, b = votes[0][1], votes[1][1]
        if float(np.linalg.norm(a - b)) <= agreementTolPx:
            mid = (a + b) / 2.0
            return (float(mid[0]), float(mid[1]))
        return None  # cameras disagree - the fit is not trustworthy here
    point = votes[0][1]
    return (float(point[0]), float(point[1]))


def _projectWithCheirality(H: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Homogeneous projection returning (projected Nx2, valid mask).

    cv2.perspectiveTransform divides by w regardless of its sign, so points
    behind a camera's horizon can land on plausible in-bounds pixels. A point is
    genuinely imaged only when w > 0; the mask flags those.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = H @ np.vstack([pts.T, np.ones(len(pts))])
    w = homogeneous[2]
    valid = w > 1e-9
    safeW = np.where(valid, w, 1.0)
    projected = (homogeneous[:2] / safeW).T
    return projected, valid


def _exactCornerQuadFromCameraModel(
    atlas,
    stitchingImagePack: ImagePack,
    cornerData,
) -> list[tuple[float, float]] | None:
    """Place a missing corner tag on the mosaic via the exact camera model.

    The atlas camera->map homography is exact for every floor point inside a
    camera's FOV (tags there or not), and the stitcher tracks each camera's
    cumulative warp into the mosaic. So the corner's mosaic position is
    W_cam @ H_cam^-1 @ corner_map - a model evaluation, not an extrapolation.
    Picks the camera that sees the corner with the largest interior margin;
    returns None when no calibrated camera images the corner region.
    """
    targetQuad = atlas.atlas_quads_norm.get(int(cornerData))
    if targetQuad is None:
        return None

    mosaicDiag = math.hypot(stitchingImagePack.image.shape[1], stitchingImagePack.image.shape[0]) if stitchingImagePack.image is not None else 2000.0
    points: list[tuple[float, float]] = []
    for vertexNorm in np.asarray(targetQuad, dtype=np.float64):
        point = _exactProjectPointToMosaic(
            atlas,
            stitchingImagePack,
            (float(vertexNorm[0]), float(vertexNorm[1])),
            agreementTolPx=0.02 * mosaicDiag,
        )
        if point is None:
            return None  # every vertex must be trustworthy
        points.append(point)
    return points


def _stabilizeEstimatedCorner(
    state,
    spec: str,
    point: tuple[float, float] | None,
    jumpThresholdPx: float,
    adoptAfter: int = 5,
) -> tuple[float, float] | None:
    """Temporal gate for estimated corner points.

    Frame-local fits occasionally produce a wild corner for a single round;
    a jump far from the previously drawn point is ignored until it persists
    for `adoptAfter` consecutive rounds (a genuine change, e.g. after an
    atlas re-solve). Nearby updates are adopted with light smoothing.
    """
    history = getattr(state, "estimatedCornerHistory", None)
    if history is None:
        history = {}
        state.estimatedCornerHistory = history

    previous = history.get(spec)
    if point is None:
        return previous[0] if previous is not None else None
    if previous is None:
        history[spec] = (point, 0)
        return point
    prevPoint, streak = previous
    jump = math.hypot(point[0] - prevPoint[0], point[1] - prevPoint[1])
    if jump <= jumpThresholdPx:
        smoothed = (
            prevPoint[0] * 0.7 + point[0] * 0.3,
            prevPoint[1] * 0.7 + point[1] * 0.3,
        )
        history[spec] = (smoothed, 0)
        return smoothed
    if streak + 1 >= adoptAfter:
        history[spec] = (point, 0)  # the new location persisted - adopt it
        return point
    history[spec] = (prevPoint, streak + 1)
    return prevPoint


def _drawEstimatedMapQuad(
    img: np.ndarray,
    atlas,
    stitchingImagePack: ImagePack,
    detectionsOnStitched,
    config: AlgorithmConfig,
    state=None,
) -> None:
    """Estimated map quad on the ortho preview: dim, explicitly "(est.)".

    Corner estimates come from the gated atlas chain only (exact camera model,
    then the local RANSAC fit) - no mosaic fit is involved. Chords between the
    available corners sketch the estimated boundary; corners the model cannot
    trustworthily place are left out rather than invented.
    """
    mosaicDiag = math.hypot(img.shape[1], img.shape[0])
    robotIDs = set(config.configuredRobotIDs)
    staticDetections = [d for d in detectionsOnStitched if d.data not in robotIDs]
    detectionByTag = {int(d.data): d for d in staticDetections}
    detectedCornerIDs = {int(d.data) for d in staticDetections if d.data in set(config.mapDefinition.values())}
    # rough map center in mosaic pixels, for picking a tag's outer vertex
    mapCenterPx = (
        np.mean([d.aggregatedCentroid for d in staticDetections], axis=0)
        if staticDetections
        else np.array([img.shape[1] / 2.0, img.shape[0] / 2.0])
    )

    def outerVertex(quadArr: np.ndarray) -> tuple[float, float]:
        outer = quadArr[int(np.argmax(np.linalg.norm(quadArr - mapCenterPx, axis=1)))]
        return (float(outer[0]), float(outer[1]))

    cornerPoints: dict[str, tuple[float, float] | None] = {}
    cornerEstimated: dict[str, bool] = {}
    for spec, mapCornerNorm in (("TL", (0.0, 0.0)), ("TR", (1.0, 0.0)), ("BR", (1.0, 1.0)), ("BL", (0.0, 1.0))):
        tagId = int(config.mapDefinition[spec])
        point: tuple[float, float] | None = None
        if tagId in detectedCornerIDs:
            # measurement first: the detected tag's outer vertex IS the corner
            detection = detectionByTag[tagId]
            point = outerVertex(
                np.array(
                    [detection.aggregatedTopLeft, detection.aggregatedTopRight, detection.aggregatedBottomLeft, detection.aggregatedBottomRight],
                    dtype=np.float64,
                )
            )
        if point is None:
            point = _exactProjectPointToMosaic(
                atlas,
                stitchingImagePack,
                mapCornerNorm,
                agreementTolPx=0.02 * mosaicDiag,
            )
        if point is None:
            quad = _extrapolateCornerQuadFromAtlas(
                atlas,
                detectionsOnStitched,
                tagId,
                maxResidualPx=0.03 * mosaicDiag,
            )
            if quad is not None:
                point = outerVertex(np.asarray(quad, dtype=np.float64))
        estimated = tagId not in detectedCornerIDs
        if estimated and state is not None:
            # single-frame wild fits must not make the corner flicker around
            point = _stabilizeEstimatedCorner(state, spec, point, jumpThresholdPx=0.08 * mosaicDiag)
        cornerPoints[spec] = point
        cornerEstimated[spec] = estimated

    magenta = (255, 0, 255)
    dimGray = (190, 190, 190)
    order = ["TL", "TR", "BR", "BL"]
    for specA, specB in zip(order, order[1:] + order[:1]):
        a, b = cornerPoints[specA], cornerPoints[specB]
        if a is None or b is None:
            continue
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), magenta, 2, cv2.LINE_AA)
    for spec in order:
        point = cornerPoints[spec]
        if point is None:
            continue
        pt = (int(point[0]), int(point[1]))
        cv2.circle(img, pt, 10, magenta, -1)
        if cornerEstimated[spec]:
            cv2.drawMarker(img, pt, dimGray, markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=3)
            labelPos = (
                min(max(pt[0] + 14, 8), img.shape[1] - 220),
                min(max(pt[1] + 8, 34), img.shape[0] - 12),
            )
            putText(img, f"{config.mapDefinition[spec]} (est.)", labelPos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, dimGray, 2)


def _drawExactMapBoundary(
    img: np.ndarray,
    atlas,
    stitchingImagePack: ImagePack,
    offset: tuple[int, int],
) -> bool:
    """Draw the map boundary on the mosaic EXACTLY via the camera model.

    Each boundary sample (map-normalized perimeter point) is projected into the
    raw camera that images it (atlas H_cam is exact for the floor plane) and
    then through that camera's cumulative stitch warp: W_cam @ H_cam^-1 @ p.
    No fitting involved; segments where no camera sees the boundary are left
    as honest gaps. Returns False when the required warps are unavailable.
    """
    if not stitchingImagePack.cameraWarps or not atlas.camera_h:
        return False

    mosaicDiag = math.hypot(img.shape[1], img.shape[0])

    # closed perimeter of the unit square, TL -> TR -> BR -> BL -> TL
    perEdge = 17
    ts = np.linspace(0.0, 1.0, perEdge)
    perimeter: list[tuple[float, float]] = (
        [(float(t), 0.0) for t in ts]
        + [(1.0, float(t)) for t in ts]
        + [(float(1 - t), 1.0) for t in ts]
        + [(0.0, float(1 - t)) for t in ts]
    )

    projected: list[tuple[int, int] | None] = []
    for pNorm in perimeter:
        point = _exactProjectPointToMosaic(
            atlas,
            stitchingImagePack,
            pNorm,
            agreementTolPx=0.02 * mosaicDiag,
        )
        if point is None:
            projected.append(None)
        else:
            projected.append((int(point[0] + offset[0]), int(point[1] + offset[1])))

    drewAnything = False
    segments = 0
    for a, b in zip(projected, projected[1:] + projected[:1]):
        if a is None or b is None:
            continue  # honest gap: no camera images this stretch of boundary
        cv2.line(img, a, b, (0, 255, 255), 2)
        segments += 1
        drewAnything = True
    Tracing.message(f"Exact map boundary: {segments}/{len(projected)} segments drawn")
    return drewAnything


def _extrapolateCornerQuadFromAtlas(
    atlas,
    detectionsOnStitched,
    cornerData,
    maxResidualPx: float,
) -> list[tuple[float, float]] | None:
    """Extrapolate a missing corner tag's stitched-pixel quad from the floor atlas.

    Fits a local map-normalized -> stitched-pixels homography over the detected
    static tags nearest to the missing corner. On the piecewise-projective mosaic
    this is far more precise than the global fit fallback, which carries a large
    stable bias on extrapolated corners. Returns None (caller falls back) when
    support is insufficient or the fit residual exceeds maxResidualPx.
    """
    targetQuad = atlas.atlas_quads_norm.get(int(cornerData))
    if targetQuad is None:
        return None
    cornerCenter = (float(targetQuad[:, 0].mean()), float(targetQuad[:, 1].mean()))

    # One correspondence per SUB-detection: on a ghosted mosaic the aggregated
    # centroid averages duplicate copies of the same tag warped in by different
    # cameras, which systematically biases the fit. Sub-detections are each a
    # single camera's real warped observation; RANSAC then keeps the mutually
    # consistent subset (one warp regime) instead of splitting the difference.
    srcNorm: list[tuple[float, float]] = []
    dstPx: list[tuple[float, float]] = []
    for detection in detectionsOnStitched:
        quadNorm = atlas.atlas_quads_norm.get(int(detection.data))
        if quadNorm is None or int(detection.data) == int(cornerData):
            continue
        tagCenterNorm = (float(quadNorm[:, 0].mean()), float(quadNorm[:, 1].mean()))
        subDetections = getattr(detection, "detections", None) or [detection]
        for sub in subDetections:
            centroid = getattr(sub, "centroid", None)
            if centroid is None:
                centroid = detection.aggregatedCentroid
            srcNorm.append(tagCenterNorm)
            dstPx.append((float(centroid[0]), float(centroid[1])))
    if len(set(srcNorm)) < 4:
        return None

    # local support: correspondences of the 8 tags nearest to the missing corner
    nearestTags = sorted(
        set(srcNorm),
        key=lambda p: (p[0] - cornerCenter[0]) ** 2 + (p[1] - cornerCenter[1]) ** 2,
    )[:8]
    keep = [i for i, p in enumerate(srcNorm) if p in set(nearestTags)]
    src = np.asarray([srcNorm[i] for i in keep], dtype=np.float64)
    dst = np.asarray([dstPx[i] for i in keep], dtype=np.float64)
    H, inlierMask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=maxResidualPx / 2)
    if H is None or inlierMask is None or int(inlierMask.sum()) < 4:
        return None
    inliers = inlierMask.ravel().astype(bool)
    reprojected = cv2.perspectiveTransform(src[inliers].reshape(-1, 1, 2), H).reshape(-1, 2)
    residualRms = float(np.sqrt(np.mean(np.sum((reprojected - dst[inliers]) ** 2, axis=1))))
    if residualRms > maxResidualPx:
        return None
    points = cv2.perspectiveTransform(
        np.asarray(targetQuad, dtype=np.float64).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    return [(float(p[0]), float(p[1])) for p in points]


def _drawPiecewiseMapBoundary(
    img: np.ndarray,
    atlas,
    detectionsOnStitched,
    extremeCornerPoints,
    offset: tuple[int, int],
) -> None:
    """Draw the map boundary as it actually runs through the piecewise-projective
    stitched mosaic (yellow), next to the straight magenta chord quad.

    The mosaic is not a single homography of the floor, so straight chords between
    the corner tags need not follow the physical map edge. For each edge we fit a
    local map-normalized -> stitched-pixels homography from the floor tags detected
    near that edge (atlas positions x stitched detections) and sample it into a
    bent polyline. Divergence between yellow and magenta is per-region stitching
    distortion made visible; it does not affect the reported poses.
    """
    corrNorm: list[tuple[float, float]] = []
    corrPx: list[tuple[float, float]] = []
    for detection in detectionsOnStitched:
        quad = atlas.atlas_quads_norm.get(int(detection.data))
        if quad is None:
            continue
        centroid = detection.aggregatedCentroid
        corrNorm.append((float(quad[:, 0].mean()), float(quad[:, 1].mean())))
        corrPx.append((float(centroid[0]), float(centroid[1])))
    # The detected outer corner vertices anchor the edge endpoints exactly.
    for (nx, ny), pt in zip([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)], extremeCornerPoints):
        corrNorm.append((nx, ny))
        corrPx.append((float(pt[0]), float(pt[1])))

    edges = [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (1.0, 1.0)), ((1.0, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 0.0))]
    for (x0, y0), (x1, y1) in edges:
        edgeNorm: list[tuple[float, float]] = []
        edgePx: list[tuple[float, float]] = []
        ex, ey = x1 - x0, y1 - y0  # unit-length edges of the unit square
        for (nx, ny), px in zip(corrNorm, corrPx):
            t = max(0.0, min(1.0, (nx - x0) * ex + (ny - y0) * ey))
            distToEdge = math.hypot(nx - (x0 + t * ex), ny - (y0 + t * ey))
            if distToEdge <= 0.45:
                edgeNorm.append((nx, ny))
                edgePx.append(px)
        if len(edgeNorm) < 4:
            continue  # not enough local support; the magenta chord stays the best guess
        H, _ = cv2.findHomography(
            np.asarray(edgeNorm, dtype=np.float64),
            np.asarray(edgePx, dtype=np.float64),
            method=0,
        )
        if H is None:
            continue
        samples = np.array(
            [[x0 + ex * t, y0 + ey * t] for t in np.linspace(0.0, 1.0, 17)],
            dtype=np.float64,
        )
        projected = cv2.perspectiveTransform(samples.reshape(-1, 1, 2), H).reshape(-1, 2)

        # Degeneracy gate: near-collinear support makes the local fit
        # ill-conditioned and the polyline can fly far past its own anchors.
        # A healthy least-squares fit passes close to the exact corner anchors
        # and stays close to the chord in total length; otherwise skip the edge.
        anchorByNormCorner = {(0.0, 0.0): 0, (1.0, 0.0): 1, (0.0, 1.0): 2, (1.0, 1.0): 3}
        anchorStart = np.asarray(extremeCornerPoints[anchorByNormCorner[(x0, y0)]], dtype=np.float64)
        anchorEnd = np.asarray(extremeCornerPoints[anchorByNormCorner[(x1, y1)]], dtype=np.float64)
        chordLen = float(np.linalg.norm(anchorEnd - anchorStart))
        polylineLen = float(np.sum(np.linalg.norm(np.diff(projected, axis=0), axis=1)))
        if (
            chordLen <= 1.0
            or float(np.linalg.norm(projected[0] - anchorStart)) > 0.2 * chordLen
            or float(np.linalg.norm(projected[-1] - anchorEnd)) > 0.2 * chordLen
            or polylineLen > 2.0 * chordLen
        ):
            continue
        prevPt: tuple[int, int] | None = None
        for p in projected:
            pt = (int(p[0] + offset[0]), int(p[1] + offset[1]))
            if prevPt is not None:
                cv2.line(img, prevPt, pt, (0, 255, 255), 2)
            prevPt = pt


def _sub_detection_for_common_features_viz(
    composite: CompositeDetection,
    *,
    facing_right: bool,
) -> MarkerDetection:
    """Pick the sub-detection closest to the stitching partner.

    A CompositeDetection may aggregate several sub-detections spread across a
    previously-stitched image; its aggregated corner points are an average that can
    land far from any real tag, producing correspondence lines that lead nowhere.
    """
    if facing_right:
        return max(composite.detections, key=lambda detection: detection.centroid[0])
    return min(composite.detections, key=lambda detection: detection.centroid[0])


def _cropStitchedToContentBoundingBox(
    stitchedImg: np.ndarray,
    detectionArrays: list[np.ndarray],
    *,
    marginPx: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop the stitched canvas to the bounding box of its actual content.

    The stitching canvas is sized to hold the full frames of both source images, which for
    nested stitches accumulates large black regions. This crops to the extreme corners of the
    visible (non-black) pixels, extended to also cover all detection points, so downstream
    stages and previews are not dominated by empty canvas. Returns the cropped image and the
    (x, y) crop offset to subtract from any coordinates expressed in the old canvas frame.
    """
    # single-channel OR of the color planes (a pixel is content iff any channel is non-zero);
    # cheaper than materializing a full-canvas 3-channel boolean mask in numpy
    blue, green, red = cv2.split(stitchedImg)
    contentMask = cv2.bitwise_or(cv2.bitwise_or(blue, green), red)
    contentColumns = np.flatnonzero(contentMask.any(axis=0))
    contentRows = np.flatnonzero(contentMask.any(axis=1))
    if len(contentColumns) == 0 or len(contentRows) == 0:
        return stitchedImg, (0, 0)

    x0, x1 = int(contentColumns[0]), int(contentColumns[-1])
    y0, y1 = int(contentRows[0]), int(contentRows[-1])

    # never crop away detection points, even if they sit on black pixels
    for detectionArray in detectionArrays:
        if len(detectionArray) == 0:
            continue
        points = detectionArray.reshape(-1, 2)
        x0 = min(x0, int(math.floor(float(np.min(points[:, 0])))))
        x1 = max(x1, int(math.ceil(float(np.max(points[:, 0])))))
        y0 = min(y0, int(math.floor(float(np.min(points[:, 1])))))
        y1 = max(y1, int(math.ceil(float(np.max(points[:, 1])))))

    imgH, imgW = stitchedImg.shape[:2]
    x0 = max(0, x0 - marginPx)
    y0 = max(0, y0 - marginPx)
    x1 = min(imgW - 1, x1 + marginPx)
    y1 = min(imgH - 1, y1 + marginPx)

    if (x0, y0) == (0, 0) and (x1, y1) == (imgW - 1, imgH - 1):
        return stitchedImg, (0, 0)

    return stitchedImg[y0 : y1 + 1, x0 : x1 + 1], (x0, y0)


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
                        detectionImageA = _sub_detection_for_common_features_viz(
                            camera1Store[key],
                            facing_right=True,
                        )
                        detectionImageB = _sub_detection_for_common_features_viz(
                            camera2Store[key],
                            facing_right=False,
                        )

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
                            "topLeft",
                            "topRight",
                            "bottomLeft",
                            "bottomRight",
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

            # always filter out robot detections from homography calculations
            nonRobotCommonDetectionIds = {detection.data for detection in camera1CommonDetections if detection.data not in config.configuredRobotIDs}

            logger.debug(f"Filtering robot detections from homography calculation: {len(camera1CommonDetections)} total -> {len(nonRobotCommonDetectionIds)} non-robot detections")

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
                    canvasTranslation,
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

            # crop the stitched canvas to its content bounding box, shifting detections accordingly
            stitchedImg, cropOffset = _cropStitchedToContentBoundingBox(
                stitchedImg,
                [warpedCamera1NumpyDetections, warpedCamera2NumpyDetections],
            )
            if cropOffset != (0, 0):
                cropOffsetArr = np.array(cropOffset, dtype=warpedCamera1NumpyDetections.dtype)
                warpedCamera1NumpyDetections = warpedCamera1NumpyDetections - cropOffsetArr
                warpedCamera2NumpyDetections = warpedCamera2NumpyDetections - cropOffsetArr

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

                # compose cumulative raw-camera -> stitched-frame warps: img1
                # content moved by the canvas translation, img2 content by the
                # canvas translation after H; both then shifted by the content crop
                tCanvas = np.array(
                    [[1, 0, canvasTranslation[0] - cropOffset[0]], [0, 1, canvasTranslation[1] - cropOffset[1]], [0, 0, 1]],
                    dtype=np.float64,
                )
                w1Total = tCanvas
                w2Total = tCanvas @ np.asarray(H, dtype=np.float64)
                result.cameraWarps = {
                    **{cam: w1Total @ warp for cam, warp in imagePacks[0].cameraWarps.items()},
                    **{cam: w2Total @ warp for cam, warp in imagePacks[1].cameraWarps.items()},
                }
                result.cameraSizes = {**imagePacks[0].cameraSizes, **imagePacks[1].cameraSizes}

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
    skipRender: bool = False,
    skipMeasurements: bool = False,
):
    """skipRender: keep all geometry/measurement math (M, bounds gates, mosaic
    fallback measurements, Kalman) but skip rendering the top-down image and its
    annotations - used by the orthorectified strategy, which overrides the
    rendered outputs anyway.

    skipMeasurements: bypass the mosaic measurement machinery entirely (corner
    extrapolation, M, gates, mosaic robot measurements) - only the stitched
    preview annotations (measured tags + exact atlas boundary) and the common
    measurement/Kalman flow run. Used by the orthorectified strategy when the
    atlas is solved and epipolar (which needs M) is off; corner positions then
    come exclusively from the ortho/map frame."""
    with Tracing.ScopedZone("topDownWarp") as topDownWarpZone:
        logger.debug(" == Processing sample == ")

        stitchedImg = stitchingImagePack.image
        detectionsOnStitched = stitchingImagePack.compositeDetectionsStore.values()
        if len(detectionsOnStitched) == 0:
            raise ProcessingError(
                "No detections found on stitched image",
                code="NO_DETECTIONS_ON_STITCHED_IMAGE",
            )

        if skipMeasurements:
            # Orthorectified fast path: the mosaic serves as a preview only.
            # Measured tags and the exact atlas boundary are annotated; corner
            # positions live exclusively in the ortho/map frame (where they are
            # definitional), so no extrapolation is drawn or computed here.
            stitchedImgAnnotated = drawDetectionsOnImage(
                stitchedImg.copy(),
                detectionsOnStitched,
                config,
                (90, 127, 255),
                False,
            )
            cornerTagIDs = set(config.mapDefinition.values())
            detectedCornerDetections = [d for d in detectionsOnStitched if d.data in cornerTagIDs]
            if detectedCornerDetections:
                stitchedImgAnnotated = drawDetectionsOnImage(
                    stitchedImgAnnotated,
                    detectedCornerDetections,
                    config,
                    (90, 255, 255),
                )
            floorAtlasPreview = getattr(state, "floorAtlas", None)
            if floorAtlasPreview is not None and floorAtlasPreview.ready:
                _drawExactMapBoundary(
                    stitchedImgAnnotated,
                    floorAtlasPreview,
                    stitchingImagePack,
                    (0, 0),
                )
                _drawEstimatedMapQuad(
                    stitchedImgAnnotated,
                    floorAtlasPreview,
                    stitchingImagePack,
                    detectionsOnStitched,
                    config,
                    state=state,
                )
            resultsHolder.stitchedImgAnnotated = stitchedImgAnnotated

            # no mosaic measurements: the consensus substitution in the common
            # resolver supplies the poses; Kalman coasts through dropouts
            resultsHolder.robotDetections = resolve_robot_detections(
                [],
                state,
                config,
                resultsHolder,
                memory,
            )
            topDownWarpZone.color(Tracing.COLOR_GREEN)
            return

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
                    # corner correspondences are matched by tag identity, so there are
                    # no outliers to reject - but the stitched mosaic is only
                    # piecewise-projective, so RANSAC would discard most points as
                    # "outliers" and extrapolate from a degenerate subset
                    # with < 4 tags a projective fit would recover perspective only
                    # from the tags' shapes, but on the stitched mosaic those are
                    # systematically distorted by the pairwise stitching warps
                    # (measured live: ~150px+ stable bias on the extrapolated
                    # corner); the trustworthy signal is the tag centroids, so fit
                    # affine - on centroids when >= 3 tags, else on sub-corners
                    numDetectedCornerTags = len(filteredHomographyCornersPointListsInWarped)
                    if numDetectedCornerTags < 4:
                        logger.info(f"Only {numDetectedCornerTags} corner tag(s) detected; using affine (6 DoF) fit for top-down transform")
                        if numDetectedCornerTags >= 3:
                            fitSrc = np.array([np.mean(np.asarray(r, dtype=np.float64).reshape(-1, 2), axis=0) for i, r in enumerate(unfilteredHomographyCornersRect) if unfilteredHomographyCornersPointListsInWarped[i] is not None])
                            fitDst = np.array([np.mean(np.asarray(c, dtype=np.float64).reshape(-1, 2), axis=0) for c in filteredHomographyCornersPointListsInWarped])
                        else:
                            fitSrc = presentHomographyCornersRect
                            fitDst = presentHomographyCornersInWarped
                        cornersH = calculateAffineHomography(fitSrc, fitDst)
                    else:
                        cornersH = calculateHomography(
                            presentHomographyCornersRect,
                            presentHomographyCornersInWarped,
                            useRansac=False,
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

            # use the actual detected corner positions wherever available - the
            # global fit carries a residual (the mosaic is piecewise-projective),
            # so replacing exact measurements with fit outputs would skew all
            # four map boundary points; only truly missing corners are synthetic
            cornerAtlas = getattr(state, "floorAtlas", None)
            allCornerMarkers: list[MarkerDetection] = []
            extrapolatedCornerIDs: set = set()
            for idx, corner in enumerate(squareCornersWarped):
                pointsInWarped = unfilteredHomographyCornersPointListsInWarped[idx]
                if pointsInWarped is None:
                    extrapolatedCornerIDs.add(unfilteredHomographyCornersData[idx])
                    # missing corner tag: prefer an atlas-local extrapolation
                    # (fit over the nearest detected floor tags), which avoids
                    # the large stable bias of the global-fit fallback on the
                    # piecewise-projective mosaic
                    globalPoints = [tuple(lst) for lst in corner]
                    atlasPoints = None
                    atlasCornerEnabled = getattr(config, "useAtlasCornerResolution", True)
                    if atlasCornerEnabled and cornerAtlas is not None and cornerAtlas.ready:
                        mosaicDiag = math.hypot(stitchedImg.shape[1], stitchedImg.shape[0])
                        # 1st choice: exact camera-model placement (no fitting at all)
                        atlasPoints = _exactCornerQuadFromCameraModel(
                            cornerAtlas,
                            stitchingImagePack,
                            unfilteredHomographyCornersData[idx],
                        )
                        if atlasPoints is not None:
                            logger.debug(
                                f"Corner tag {unfilteredHomographyCornersData[idx]} placed via exact camera model"
                            )
                    if atlasPoints is None and atlasCornerEnabled and cornerAtlas is not None and cornerAtlas.ready:
                        atlasPoints = _extrapolateCornerQuadFromAtlas(
                            cornerAtlas,
                            detectionsOnStitched,
                            unfilteredHomographyCornersData[idx],
                            maxResidualPx=0.03 * mosaicDiag,
                        )
                        # Consistency gate: local and global extrapolations are
                        # independent estimators; when the support tags cluster
                        # far from the missing corner the local fit can land
                        # anywhere. If the two disagree badly, trust the global
                        # one (bounded, well-understood bias) over the local.
                        if atlasPoints is not None:
                            localCentroid = np.mean(np.asarray(atlasPoints, dtype=np.float64), axis=0)
                            globalCentroid = np.mean(np.asarray(globalPoints, dtype=np.float64), axis=0)
                            if float(np.linalg.norm(localCentroid - globalCentroid)) > 0.12 * mosaicDiag:
                                logger.info(
                                    f"Atlas corner extrapolation for tag {unfilteredHomographyCornersData[idx]} "
                                    "disagrees with the global fit; keeping the global fallback"
                                )
                                atlasPoints = None
                    pointsInWarped = atlasPoints if atlasPoints is not None else globalPoints
                allCornerMarkers.append(
                    MarkerDetection(
                        *[(int(p[0]), int(p[1])) for p in pointsInWarped],
                        data=unfilteredHomographyCornersData[idx],
                    )
                )
            state.allCornerMarkers = allCornerMarkers
            state.extrapolatedCornerIDs = extrapolatedCornerIDs

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

        # detected corners are measurements (solid yellow); extrapolated ones are
        # model guesses in a region with no observations - draw them dim, with an
        # explicit "est." suffix, so the preview cannot be read as a measurement
        _extrapolatedCornerIDs = getattr(state, "extrapolatedCornerIDs", set())
        detectedCornerMarkers = [m for m in state.allCornerMarkers if m.data not in _extrapolatedCornerIDs]
        extrapolatedCornerMarkers = [m for m in state.allCornerMarkers if m.data in _extrapolatedCornerIDs]
        stitchedImgAnnotated = drawDetectionsOnImage(
            stitchedImgAnnotated,
            detectedCornerMarkers,
            config,
            (90, 255, 255),
            offset=stitchedImgAnnotatedBorderOffset,
        )
        if extrapolatedCornerMarkers:
            stitchedImgAnnotated = drawDetectionsOnImage(
                stitchedImgAnnotated,
                extrapolatedCornerMarkers,
                config,
                (160, 160, 160),
                offset=stitchedImgAnnotatedBorderOffset,
            )
            for marker in extrapolatedCornerMarkers:
                putText(
                    stitchedImgAnnotated,
                    "(est.)",
                    (
                        int(marker.topLeft[0] + stitchedImgAnnotatedBorderOffset[0]),
                        int(marker.topLeft[1] + stitchedImgAnnotatedBorderOffset[1] + 46),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (160, 160, 160),
                    3,
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
                # white cross = measured corner; dim thin cross = extrapolated
                isExtrapolated = marker.data in getattr(state, "extrapolatedCornerIDs", set())
                cv2.drawMarker(
                    stitchedImgAnnotated,
                    [
                        int(marker.topLeft[0] + stitchedImgAnnotatedBorderOffset[0]),
                        int(marker.topLeft[1] + stitchedImgAnnotatedBorderOffset[1]),
                    ],
                    (160, 160, 160) if isExtrapolated else (255, 255, 255),
                    markerType=cv2.MARKER_TILTED_CROSS,
                    markerSize=20,
                    thickness=3 if isExtrapolated else 8,
                )

            # true (bent) map boundary through the piecewise-projective mosaic:
            # prefer the exact camera-model projection (atlas H_cam + cumulative
            # stitch warps, no fitting); fall back to the per-edge local fits
            floorAtlas = getattr(state, "floorAtlas", None)
            if floorAtlas is not None and floorAtlas.ready:
                drewExact = _drawExactMapBoundary(
                    stitchedImgAnnotated,
                    floorAtlas,
                    stitchingImagePack,
                    stitchedImgAnnotatedBorderOffset,
                )
                logger.debug(f"Map boundary rendering: exact={drewExact}")
                if not drewExact:
                    _drawPiecewiseMapBoundary(
                        stitchedImgAnnotated,
                        floorAtlas,
                        detectionsOnStitched,
                        state.extremeCornerPoints,
                        stitchedImgAnnotatedBorderOffset,
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

            # Augment the 4-corner fit with interior floor tags whose map
            # positions are known from the live atlas: a least-squares fit over
            # corners + interior centroids averages the mosaic's piecewise
            # distortion across the whole map instead of pinning it (exactly and
            # only) at the corners.
            floorAtlas = getattr(state, "floorAtlas", None)
            if (
                config.topDownFitUseInteriorTags
                and floorAtlas is not None
                and floorAtlas.ready
                and floorAtlas.interior_centroids_norm
            ):
                fitSrcPts: list[list[float]] = [[float(p[0]), float(p[1])] for p in state.extremeCornerPoints]
                fitDstPts: list[list[float]] = [[float(p[0]), float(p[1])] for p in topDownSquare]
                for detection in detectionsOnStitched:
                    centroidNorm = floorAtlas.interior_centroids_norm.get(int(detection.data))
                    if centroidNorm is None:
                        continue
                    centroid = detection.aggregatedCentroid
                    fitSrcPts.append([float(centroid[0]), float(centroid[1])])
                    fitDstPts.append([
                        centroidNorm[0] * state.transformedWidth,
                        centroidNorm[1] * state.transformedHeight,
                    ])
                if len(fitSrcPts) >= 5:
                    mAugmented, _ = cv2.findHomography(
                        np.asarray(fitSrcPts, dtype=np.float64),
                        np.asarray(fitDstPts, dtype=np.float64),
                        method=0,
                    )
                    if mAugmented is not None:
                        m = mAugmented
                        logger.info(
                            f"Top-down fit augmented with {len(fitSrcPts) - 4} interior floor tag(s)"
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

        topDownImg: np.ndarray | None = None
        if not skipRender:
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

        # Fail only when robot detections (the measured quantity) land far outside the
        # top-down canvas. Interior/extraneous tags can project near the vanishing line
        # after a nested stitch and should not abort an otherwise usable map frame.
        topDownImgBounds = dstW, dstH
        robot_ids = set(config.configuredRobotIDs)
        for detection in topDownUnionOfDetections:
            if detection.data not in robot_ids:
                continue
            centroid = detection.aggregatedCentroid
            if centroid[0] < -config.detectionBoundsTolerance * topDownImgBounds[0] or centroid[0] > (1 + config.detectionBoundsTolerance) * topDownImgBounds[0] or centroid[1] < -config.detectionBoundsTolerance * topDownImgBounds[1] or centroid[1] > (1 + config.detectionBoundsTolerance) * topDownImgBounds[1]:
                raise ProcessingError(
                    (
                        "The transformation of the stitched image to top-down "
                        "perspective resulted in a malformed image - most likely, "
                        "arrangement of keypoints is not viable to compute a precise "
                        "enough homography to warp stitched image properly "
                        f"(robot detection {detection.data} is over {config.detectionBoundsTolerance * 100:.1f}% "
                        "outside the bounds of the top-down image)."
                    ),
                    code="INVALID_DETECTIONS_BOUNDS",
                    value=(f"centroid = {centroid}, bounds = {topDownImgBounds}, tolerance = {config.detectionBoundsTolerance * 100:.1f}%"),
                )

        nonCornerDetections = set(topDownUnionOfDetections).difference(state.allCornerMarkers)

        topDownImgAnnotated: np.ndarray | None = None
        if not skipRender and topDownImg is not None:
            preallocAnnotatedSlot = resultsHolder.ensure_top_down_annotated_buffer(dstH, dstW)
            if preallocAnnotatedSlot is not None:
                np.copyto(preallocAnnotatedSlot.buffer, topDownImg)
                topDownImgAnnotated = preallocAnnotatedSlot.buffer
            else:
                topDownImgAnnotated = topDownImg.copy()

        robot_measurements: list[RobotTagMeasurement] = []

        if config.useEpipolarGeometry:
            # Registration anchors for the epipolar point cloud: map-frame
            # quads of every static floor tag, taken from the floor atlas.
            # The atlas solves per-camera map homographies from RAW camera
            # views, not from the stitched mosaic under test, so both solvers
            # register against the same map references and differ only in how
            # the robot pose itself is derived. Using the interior tags' warped
            # centroids instead would hand the epipolar path positions produced
            # by its own benchmark opponent. The four map-corner tags alone
            # cannot serve: no camera pair sees more than one of them.
            anchor_quads_px: dict = {}
            epipolarAtlas = getattr(state, "floorAtlas", None)
            if epipolarAtlas is not None and epipolarAtlas.ready:
                for tagID, quadNorm in epipolarAtlas.atlas_quads_norm.items():
                    if config.findRobotWithID(int(tagID)) is not None:
                        continue
                    anchor_quads_px[int(tagID)] = np.asarray(quadNorm, dtype=np.float64) * np.array(
                        [dstW, dstH], dtype=np.float64
                    )
            robot_measurements = collect_epipolar_robot_measurements(
                algorithm_state=state,
                config=config,
                detections=list(topDownUnionOfDetections),
                img_width=dstW,
                img_height=dstH,
                anchor_quads_px=anchor_quads_px,
            )
            msg = f"Epipolar solver produced {len(robot_measurements)} robot measurement(s)"
            logger.info(msg)
            Tracing.message(msg)

            for detection in nonCornerDetections:
                if topDownImgAnnotated is None:
                    break  # skipRender: nothing to draw on
                if config.findRobotWithID(detection.data) is not None:
                    continue
                topLeft = detection.aggregatedTopLeft
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
        else:
            for detection in nonCornerDetections:
                foundRobot = config.findRobotWithID(detection.data)
                isRobot = foundRobot is not None
                topLeft = detection.aggregatedTopLeft
                topRight = detection.aggregatedTopRight
                bottomLeft = detection.aggregatedBottomLeft
                bottomRight = detection.aggregatedBottomRight
                [centroidX, centroidY] = detection.aggregatedCentroid

                if isRobot:
                    imgRobotRelX = centroidX / dstW
                    imgRobotRelY = centroidY / dstH

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

                    ih, iw = dstH, dstW
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
                    area_norm, baseline_norm = normalize_tag_geometry(
                        area_px=area_px, baseline_px=baseline_px, img_w=iw, img_h=ih
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
                            area_norm=area_norm,
                            baseline_norm=baseline_norm,
                        )
                    )
                elif topDownImgAnnotated is not None:
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

        # common measurement resolution + Kalman flow (shared by all paths)
        robotDetections = resolve_robot_detections(
            robot_measurements,
            state,
            config,
            resultsHolder,
            memory,
        )

        for det in robotDetections:
            imgRobotRelX = det.x
            imgRobotRelY = det.y
            azimuth = det.azimuth
            foundRobot = det.robot
            if topDownImgAnnotated is None:
                pass  # skipRender: telemetry plots below still run
            elif det.backing_composite is not None:
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
                    (int(imgRobotRelX * dstW) + 14, int(imgRobotRelY * dstH) + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 200, 255),
                    3,
                )
                putText(
                    topDownImgAnnotated,
                    f"({imgRobotRelX:.2f}, {imgRobotRelY:.2f})",
                    (int(imgRobotRelX * dstW) + 14, int(imgRobotRelY * dstH) + 52),
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

        if not skipRender:
            resultsHolder.topDownImg = topDownImg
            resultsHolder.topDownImgAnnotated = topDownImgAnnotated
        resultsHolder.topDownDetections = list(topDownUnionOfDetections)
        resultsHolder.robotDetections = robotDetections
