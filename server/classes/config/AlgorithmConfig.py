from logging import Logger
from typing import Any

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
    ) -> None:
        self.logger = logger

        self.mapDefinition = mapDefinition

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
