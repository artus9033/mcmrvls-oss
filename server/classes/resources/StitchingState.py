from logging import getLogger

from classes.config.AlgorithmConfig import AlgorithmConfig
from classes.marker import CompositeDetection
from classes.types import MarkerData
import numpy as np
from utils.tracing import Tracing

from .MarkerAppearanceStateMonitor import MarkerAppearanceStateMonitor
from .RoundCacheResource import RoundCacheResource

class StitchingState(RoundCacheResource):
    """State for multi-round stitching, including homography and tracking need for recalculation."""

    homography: np.ndarray | None = None
    _homographyBuffer: np.ndarray
    _markerStatesMonitor: MarkerAppearanceStateMonitor

    _needsHomographyRecalculation: bool = True

    def __init__(self, description: str, logLevel: int | str, homographyBufferPreallocation: bool = True):
        super().__init__()

        self._markerStatesMonitor = MarkerAppearanceStateMonitor(logLevel=logLevel)

        self.logger = getLogger(f"StitchingState_{description}")
        self.logger.setLevel(logLevel)

        self.description = description
        self._homographyBufferPreallocation = homographyBufferPreallocation
        self._homographyBuffer = np.empty((3, 3), dtype=np.float64) if homographyBufferPreallocation else None

    def shouldRecalculateHomography(self) -> bool:
        return not self.hasCachedHomography() or self._needsHomographyRecalculation

    def markingCheckHaveDetectionsChanged(
        self,
        newCompositeDetectionsStore: dict[MarkerData, CompositeDetection],
        config: AlgorithmConfig,
    ) -> str | None:
        """Heuristic for detecting movement / change between presence of markers in images.

        Checks for changes in marker detections that could cause a homography recalculation, using consecutive rounds logic for both key set and value changes.

        To not over-react to acute, transient 'flips' in marker presence (e.g. due to instantaneous video quality drops,
        focus losses, etc.) that may mean a marker becomes visible or invisible, monitor is tracking extinguishing / appearing
        markers and the consecutive rounds within which a given state change is occurring are thresholded by the 'threshold' argument.

        Returns a string describing the reason for the change, or None if no change was detected.
        """

        with Tracing.ScopedZone("markingCheckHaveDetectionsChanged"):
            # below: exclude robots' markers (they may be moving)
            newNonRobotKeys = set([k for k in newCompositeDetectionsStore.keys() if k not in config.configuredRobotIDs])

            reasons: list[str] = []
            for changedMarker, reason in self._markerStatesMonitor.step(
                newNonRobotKeys,
                markerPositions=newCompositeDetectionsStore,
                appearingConsecutiveRoundsDelay=config.recalcHeuristicAppearingMarkerConsecutiveRoundsDelay,
                extinguishingConsecutiveRoundsDelay=config.recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay,
                ewmaAlpha=config.markerVisibilityEwmaAlpha,
                hysteresisThresholdHigh=config.markerVisibilityThresholdHigh,
                hysteresisThresholdLow=config.markerVisibilityThresholdLow,
            ):
                reasons.append(f"Marker {changedMarker} - action: {reason}")

            return ", ".join([str(reason) for reason in reasons]) if reasons else None

    def updateCachedHomography(self, H: np.ndarray):
        if self._homographyBufferPreallocation and self._homographyBuffer is not None:
            np.copyto(self._homographyBuffer, H)
            self.homography = self._homographyBuffer
        else:
            self.homography = H.copy()
        self._needsHomographyRecalculation = False

    def hasCachedHomography(self) -> bool:
        return self.homography is not None

    def markNeedsHomographyRecalculation(self, reason: str):
        Tracing.message(f"Marking StitchingState {self.description.__str__()} as needing homography recalculation, reason: {reason}")
        self._needsHomographyRecalculation = True
