from logging import Logger
from typing import Any

from algorithms.segmentation import MapSegmentationStrategyName, configure_map_segmentation
from utils.detections import MarkerDetectorBackend, configure_marker_detector

from ..robot import RobotConfig
from ..types import CornerSpec, DimensionsConfigKeys, MarkerData

class PreallocationConfig:
    def __init__(
        self,
        adaptiveBufferPreallocation: bool,
        homographyBufferPreallocation: bool,
    ) -> None:
        self.adaptiveBufferPreallocation = adaptiveBufferPreallocation
        self.homographyBufferPreallocation = homographyBufferPreallocation

    @staticmethod
    def from_dict(config: dict[str, Any] | None) -> "PreallocationConfig":
        cfg = config or {}
        return PreallocationConfig(
            adaptiveBufferPreallocation=bool(cfg.get("adaptiveBufferPreallocation", True)),
            homographyBufferPreallocation=bool(cfg.get("homographyBufferPreallocation", True)),
        )


class CacheConfig:
    def __init__(
        self,
        enabled: bool,
        stitchingHomographyCache: bool,
        topDownHomographyCache: bool,
        mapSegmentationThrottle: bool,
        mapSegmentationIntervalSeconds: int,
    ) -> None:
        self.enabled = enabled
        self.stitchingHomographyCache = stitchingHomographyCache
        self.topDownHomographyCache = topDownHomographyCache
        self.mapSegmentationThrottle = mapSegmentationThrottle
        self.mapSegmentationIntervalSeconds = mapSegmentationIntervalSeconds

    @staticmethod
    def from_dict(config: dict[str, Any] | None) -> "CacheConfig":
        cfg = config or {}
        enabled = bool(cfg.get("enabled", True))

        stitchingHomographyCache = bool(cfg.get("stitchingHomographyCache", True))
        topDownHomographyCache = bool(cfg.get("topDownHomographyCache", True))
        mapSegmentationThrottle = bool(cfg.get("mapSegmentationThrottle", True))
        mapSegmentationIntervalSeconds = max(0, int(cfg.get("mapSegmentationIntervalSeconds", 3)))

        if not enabled:
            stitchingHomographyCache = False
            topDownHomographyCache = False
            mapSegmentationThrottle = False

        return CacheConfig(
            enabled=enabled,
            stitchingHomographyCache=stitchingHomographyCache,
            topDownHomographyCache=topDownHomographyCache,
            mapSegmentationThrottle=mapSegmentationThrottle,
            mapSegmentationIntervalSeconds=mapSegmentationIntervalSeconds,
        )


class AlgorithmConfig:
    def __init__(
        self,
        mapDefinition: dict[CornerSpec, MarkerData],
        dimensions: dict[DimensionsConfigKeys, int],
        configuredRobots: list[RobotConfig],
        recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay: int,
        caching: CacheConfig,
        preallocation: "PreallocationConfig",
        logger: Logger,
        recalcHeuristicAppearingMarkerConsecutiveRoundsDelay: int = 4,
        markerVisibilityEwmaAlpha: float = 0.35,
        markerVisibilityThresholdHigh: float = 0.75,
        markerVisibilityThresholdLow: float = 0.25,
        markerDetector: MarkerDetectorBackend = "apriltag",
        markerDetectorParams: dict[str, Any] | None = None,
        mapSegmentation: dict[str, Any] | None = None,
        useEpipolarGeometry: bool = True,
        usePerCameraSolver: bool = False,
        topDownFitUseInteriorTags: bool = True,
        useAtlasCornerResolution: bool = True,
        topDownStrategy: str = "mosaic",
        camera_calibrations: dict[str, Any] | None = None,
        autoCalibrateOnStart: bool | None = None,
    ) -> None:
        self.logger = logger

        self.mapDefinition = mapDefinition
        self.useEpipolarGeometry = useEpipolarGeometry
        self.usePerCameraSolver = usePerCameraSolver
        self.topDownFitUseInteriorTags = topDownFitUseInteriorTags
        self.useAtlasCornerResolution = useAtlasCornerResolution
        self.topDownStrategy = topDownStrategy
        self.camera_calibrations = camera_calibrations or {}
        # Default: auto-calibrate at start whenever the epipolar solver is active.
        self.autoCalibrateOnStart = (
            bool(useEpipolarGeometry) if autoCalibrateOnStart is None else bool(autoCalibrateOnStart)
        )
        self.markerDetector = markerDetector
        self.markerDetectorParams = markerDetectorParams or {}
        configure_marker_detector(markerDetector, self.markerDetectorParams)

        self.mapSegmentation = mapSegmentation or {}
        self.mapSegmentationStrategy = str(self.mapSegmentation.get("strategy", "classical"))
        configure_map_segmentation(
            self.mapSegmentationStrategy,  # type: ignore[arg-type]
            self.mapSegmentation,
            logger=logger,
        )

        self.detectionBoundsTolerance = 1
        self.toleranceAspectRatio = 0.18

        self.mapWidth = dimensions["mapWidth"]
        self.mapHeight = dimensions["mapHeight"]

        self.markerWidth = dimensions["markerWidth"]
        self.markerHeight = dimensions["markerHeight"]

        self.configuredRobots = {robot.id: robot for robot in configuredRobots}
        self.configuredRobotIDs = set([robot.id for robot in configuredRobots])

        self.plottingLineThickness = 6

        self.streamProcessingVisualizations = {
            "COMMON_FEATURES_IMAGE": True,
        }

        self.writeCapturesToDisk = False

        self.recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay = recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay
        self.recalcHeuristicAppearingMarkerConsecutiveRoundsDelay = recalcHeuristicAppearingMarkerConsecutiveRoundsDelay
        self.markerVisibilityEwmaAlpha = markerVisibilityEwmaAlpha
        self.markerVisibilityThresholdHigh = markerVisibilityThresholdHigh
        self.markerVisibilityThresholdLow = markerVisibilityThresholdLow
        self.caching = caching
        self.preallocation = preallocation

        self._postprocess()

    def _postprocess(self):
        self.markerDataToCornerSpec = {v: k for k, v in self.mapDefinition.items()}

        self.RATIO_MAP_EDGE_TO_MARKER_EDGE_W = self.mapWidth / self.markerWidth
        self.RATIO_MAP_EDGE_TO_MARKER_EDGE_H = self.mapHeight / self.markerHeight
        self.MAP_RATIO_W_TO_H = self.RATIO_MAP_EDGE_TO_MARKER_EDGE_W / self.RATIO_MAP_EDGE_TO_MARKER_EDGE_H
        self.mapWidthMeters = self.mapWidth / 100
        self.mapHeightMeters = self.mapHeight / 100

    logger: Logger

    mapDefinition: dict[CornerSpec, MarkerData]
    markerDataToCornerSpec: dict[MarkerData, CornerSpec]

    """normalized [0-1] percentage"""
    detectionBoundsTolerance: float
    """normalized [0-1] percentage"""
    toleranceAspectRatio: float

    # map size
    mapWidth: int
    """map width in centimeters"""
    mapHeight: int
    """map height in centimeters"""

    # corner marker size
    markerWidth: int
    """corner marker width in centimeters"""
    markerHeight: int
    """corner marker height in centimeters"""

    configuredRobots: dict[MarkerData, RobotConfig]
    configuredRobotIDs: set[MarkerData]

    RATIO_MAP_EDGE_TO_MARKER_EDGE_W: float
    RATIO_MAP_EDGE_TO_MARKER_EDGE_H: float
    MAP_RATIO_W_TO_H: float

    useEpipolarGeometry: bool
    """If True, use epipolar geometry for non-coplanar markers instead of homography"""

    usePerCameraSolver: bool
    """If True, robot poses come from per-raw-camera floor-atlas homographies fused
    across cameras (median position, circular-mean azimuth), overriding the active
    base solver's measurement per robot (base stays as fallback). Complementary to
    both the homography-mosaic and epipolar solvers; the mosaic keeps serving
    visualization"""

    topDownFitUseInteriorTags: bool
    """If True, the stitched->top-down homography is least-squares fitted over the
    corner tags plus atlas-known interior floor tags instead of the 4 corner points only"""

    useAtlasCornerResolution: bool
    """If True (default), undetected map-corner tags are placed on the mosaic by the
    gated atlas estimators (exact camera model, then local atlas fit) before falling
    back to the legacy global corner fit. Disable together with
    topDownFitUseInteriorTags for a pre-atlas ablation baseline."""

    topDownStrategy: str
    """Top-down map generation strategy: "mosaic" (warp of the stitched mosaic via M)
    or "orthorectified" (raw cameras composited directly in the map frame via the
    floor atlas; implies per-camera consensus robot poses; falls back to mosaic
    until the atlas solves)"""

    camera_calibrations: dict[str, Any]
    """Optional per-camera calibrations keyed by stage id (\"0\", \"1\", …) or video stem"""

    autoCalibrateOnStart: bool
    """When True and epipolar is active, estimate calibrations from live frames (memory-only)."""

    markerDetector: MarkerDetectorBackend
    """Marker detector backend: apriltag (vendor) or aruco (OpenCV)"""

    markerDetectorParams: dict[str, Any]
    """Optional per-backend detector hyperparameters from config.yaml"""

    mapSegmentation: dict[str, Any]
    """Map segmentation config section: strategy selection + per-strategy params"""

    mapSegmentationStrategy: MapSegmentationStrategyName
    """Selected map segmentation strategy: classical (thresholds) or unet"""

    plottingLineThickness: int

    writeCapturesToDisk: bool

    recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay: int
    recalcHeuristicAppearingMarkerConsecutiveRoundsDelay: int
    markerVisibilityEwmaAlpha: float
    markerVisibilityThresholdHigh: float
    markerVisibilityThresholdLow: float
    caching: CacheConfig
    preallocation: PreallocationConfig

    def findRobotWithID(self, data: MarkerData) -> RobotConfig | None:
        return self.configuredRobots.get(data, None)
