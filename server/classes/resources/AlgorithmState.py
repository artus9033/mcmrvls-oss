from classes.marker import MarkerDetection
from classes.types import FloatingPoint2D
import numpy as np

from .ImagePack import ImagePack

class AlgorithmState:
    prevH: np.ndarray | None = None
    pPrev: np.ndarray | None = None
    extremeCornerPoints: list[FloatingPoint2D] | None = None
    transformedWidth: float | None = None
    transformedHeight: float | None = None
    allCornerMarkers: list[MarkerDetection]
    M: np.ndarray | None = None
    cornersH: np.ndarray | None = None
    _MBuffer: np.ndarray
    _cornersHBuffer: np.ndarray

    """Set to None to disable forcing a new homography matrix on the next iteration; set to a str describing tge reason to force it"""
    nextIterForceRecalcReason: str | None = None

    """
    Tracks whether the process ImagePack is dirty (i.e., its homography has been updated) and will therefore cause the current algorithm
    stage to recalculate as well recalculated

    Note: this REQUIRES that handleBeforeRound(stitchingImagePack) is called before processing the pack, otherwise will malfunction!
    """
    imagePackDirty: bool = False

    def __init__(self, homographyBufferPreallocation: bool = True):
        self.allCornerMarkers = []
        self._homographyBufferPreallocation = homographyBufferPreallocation
        self._MBuffer = np.empty((3, 3), dtype=np.float64) if homographyBufferPreallocation else None
        self._cornersHBuffer = np.empty((3, 3), dtype=np.float64) if homographyBufferPreallocation else None

    def handleBeforeStage(self, stitchingImagePack: ImagePack):
        """Call this before processing the pack to ensure that the state is in sync with ImagePack state"""
        self.imagePackDirty = stitchingImagePack.dirty

    def maybeGetTransformUpdateReason(self) -> str | None:
        """Get the reason why the transform may need to be recalculated or `None` if no recalculation is needed"""
        if self.nextIterForceRecalcReason is not None:
            return self.nextIterForceRecalcReason

        if self.imagePackDirty:
            return "ImagePack dirty"

        if self.transformedWidth is None or self.transformedHeight is None:
            return "Transformed dimensions not set"

        return None

    def handleTransformUpdated(self):
        self.nextIterForceRecalcReason = None  # reset the flag
