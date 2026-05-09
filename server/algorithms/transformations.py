import cv2
from errors import ProcessingError
import numpy as np
from utils.constants import neutralBorderColorBGR
from utils.tracing import Tracing

def stitchImages(
    img1: np.ndarray | None,
    img2: np.ndarray | None,
    H: np.ndarray,
    npDetectionsImg1: np.ndarray,
    npDetectionsImg2: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Warp img2 to img1 with homography H, as well as the detections for img1 and img2. Returns the stitched image, warped detections for img1 and img2, and the translation matrix Ht."""
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

        # warp img2 detections using translated homography matrix
        warpedNpDetectionsImg2 = cv2.perspectiveTransform(npDetectionsImg2, H)

        if img1 is not None and img2 is not None:
            try:
                warpedImg2 = cv2.warpPerspective(
                    img2,
                    H,
                    list(reversed(img1.shape[:2])),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=neutralBorderColorBGR,
                )
            except cv2.error as e:
                raise ProcessingError(
                    "cv2.error caught during stitchImages() cv2.warpPerspective call",
                    code="OPENCV_ERROR",
                    value=e.__str__(),
                )

            # below: just blending the images would darken possible non-overlapping areas, therefore
            # this operation is masked so that blending occurs only in the overlapping ROI

            # create mask where img2 has valid pixels (non-black)
            mask2 = np.any(warpedImg2 != 0, axis=2)

            # blend only overlapping areas (OpenCV path is typically faster)
            blended = cv2.addWeighted(img1, 0.5, warpedImg2, 0.5, 0)
            stitchedImage = img1.copy()
            stitchedImage[mask2] = blended[mask2]
        else:
            stitchedImage = None

        return (
            stitchedImage,
            npDetectionsImg1,
            warpedNpDetectionsImg2,
        )
