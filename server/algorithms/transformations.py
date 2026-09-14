import cv2
from errors import ProcessingError
import numpy as np
from utils.constants import neutralBorderColorBGR
from utils.tracing import Tracing

# limit on how much (as a multiple of img1's dimensions) the stitched canvas may extend beyond
# img1's bounds in each direction; guards against a degenerate homography exploding memory usage
MAX_CANVAS_EXPANSION_FACTOR = 2.0


def stitchImages(
    img1: np.ndarray | None,
    img2: np.ndarray | None,
    H: np.ndarray,
    npDetectionsImg1: np.ndarray,
    npDetectionsImg2: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, tuple[int, int]]:
    """Warp img2 into img1's frame with homography H and composite both onto a canvas large
    enough to hold their union, so no warped pixels are cropped away. Detections of both images
    are translated into the canvas' coordinate system. Returns the stitched image, the warped
    detections for img1 and img2, and the (tx, ty) canvas translation applied to img1's frame
    (img1 content maps by T(tx,ty); img2 content by T(tx,ty) @ H)."""
    with Tracing.ScopedZone("stitchImages"):
        if len(npDetectionsImg1) == 0:
            raise ProcessingError(
                "stitchImages() received empty npDetectionsImg1",
                "STITCH_IMAGES_NO_DETECTIONS_IMG1",
            )

        if len(npDetectionsImg2) == 0:
            raise ProcessingError(
                "stitchImages() received empty npDetectionsImg2",
                "STITCH_IMAGES_NO_DETECTIONS_IMG2",
            )

        # warp img2 detections into img1's coordinate frame
        warpedNpDetectionsImg2 = cv2.perspectiveTransform(npDetectionsImg2, H)

        if img1 is None or img2 is None:
            return (None, npDetectionsImg1, warpedNpDetectionsImg2, (0, 0))

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # compute the bounding box of img1 ∪ warped img2 in img1's frame
        img2Corners = np.array(
            [[[0, 0]], [[w2, 0]], [[0, h2]], [[w2, h2]]],
            dtype=np.float64,
        )
        warpedImg2Corners = cv2.perspectiveTransform(img2Corners, H).reshape(-1, 2)

        # clamp the canvas extents so a degenerate homography cannot explode memory usage
        maxExpandX = MAX_CANVAS_EXPANSION_FACTOR * w1
        maxExpandY = MAX_CANVAS_EXPANSION_FACTOR * h1
        xMin = max(min(0.0, float(np.min(warpedImg2Corners[:, 0]))), -maxExpandX)
        yMin = max(min(0.0, float(np.min(warpedImg2Corners[:, 1]))), -maxExpandY)
        xMax = min(max(float(w1), float(np.max(warpedImg2Corners[:, 0]))), w1 + maxExpandX)
        yMax = min(max(float(h1), float(np.max(warpedImg2Corners[:, 1]))), h1 + maxExpandY)

        # translation moving the union bounding box into positive coordinates
        tx, ty = int(np.floor(-xMin)), int(np.floor(-yMin))
        Ht = np.array(
            [[1, 0, tx], [0, 1, ty], [0, 0, 1]],
            dtype=np.float64,
        )

        canvasW = int(np.ceil(xMax)) + tx
        canvasH = int(np.ceil(yMax)) + ty

        try:
            warpedImg2 = cv2.warpPerspective(
                img2,
                Ht @ H,
                (canvasW, canvasH),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=neutralBorderColorBGR,
            )
        except cv2.error as e:
            raise ProcessingError(
                "cv2.error caught during stitchImages() cv2.warpPerspective call",
                code="OPENCV_ERROR",
                value=e.__str__(),
            )

        # composite: outside img1's ROI the canvas is exactly warpedImg2 (content or black),
        # so reuse its buffer and only touch img1's ROI. Within the ROI, blend where warpedImg2
        # has valid (non-black) pixels - blending everywhere would darken non-overlapping areas -
        # and place img1 as-is elsewhere. Masks and blending are ROI-sized, not canvas-sized.
        stitchedImage = warpedImg2
        roiWarped = warpedImg2[ty : ty + h1, tx : tx + w1]
        overlapRoi = np.any(roiWarped != 0, axis=2)
        blendedRoi = cv2.addWeighted(img1, 0.5, roiWarped, 0.5, 0)
        roiOut = stitchedImage[ty : ty + h1, tx : tx + w1]
        np.copyto(roiOut, img1, where=~overlapRoi[..., None])
        np.copyto(roiOut, blendedRoi, where=overlapRoi[..., None])

        # translate both detection sets into the canvas' coordinate system
        offset = np.array([tx, ty], dtype=npDetectionsImg1.dtype)
        return (
            stitchedImage,
            npDetectionsImg1 + offset,
            warpedNpDetectionsImg2 + offset,
            (tx, ty),
        )
