from classes.config import AlgorithmConfig
import numpy as np

from .RoundCacheResource import RoundCacheResource

class PreallocatedStitchingResultsHolder(RoundCacheResource):
    commonFeaturesImg: np.ndarray | None = None
    cacheHits: int
    cacheMisses: int
    lastRecalcReason: str | None

    def __init__(self, config: AlgorithmConfig) -> None:
        # TODO: preallocate bitmap buffers & use them instead of re-allocating each time!
        self.cacheHits = 0
        self.cacheMisses = 0
        self.lastRecalcReason = None
