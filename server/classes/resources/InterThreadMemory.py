from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger, getLogger
import queue
import threading

from algorithms.robot_kalman_tracker import RobotKalmanTracker
from classes.config import AlgorithmConfig
from classes.resources.StitchingStageDescription import StitchingStageDescription
from utils.tracing import Tracing
from werkzeug.serving import BaseWSGIServer

from .AlgorithmState import AlgorithmState
from .PreallocatedAlgorithmResultsHolder import PreallocatedAlgorithmResultsHolder
from .PreallocatedStitchingResultsHolder import PreallocatedStitchingResultsHolder
from .StitchingState import StitchingState

@dataclass
class CacheEvent:
    """Event to emit over WebSocket when caches/buffers change."""

    event_type: str  # "homography_invalidated" | "buffer_reallocated"
    data: dict = field(default_factory=dict)


class InterThreadMemory:
    """Since Python global variables are slower than local variables, we use a class to store them
    and pass it to the algorithm thread as a reference
    """

    _threadShutdownEvent: threading.Event
    isHttpServerReady: bool
    algorithmState: AlgorithmState

    """True when the main algorithm loop failed (stitching/processing error); displayed data is stale."""
    dataStale: bool

    stitchingStatesByStitchingStages: dict[StitchingStageDescription, StitchingState]
    """mapping for stitchingDescription -> StitchingState"""

    stitchingResultsHoldersByStitchingStages: dict[StitchingStageDescription, PreallocatedStitchingResultsHolder]
    """mapping for stitchingDescription -> PreallocatedStitchingResultsHolder"""

    algorithmConfig: AlgorithmConfig
    algorithmLogger: Logger
    algorithmThreadLogger: Logger
    sioLogger: Logger
    rosLogger: Logger
    httpServerLogger: Logger
    _logger: Logger
    algorithmResultsHolder: PreallocatedAlgorithmResultsHolder
    sioDataDirty: threading.Event
    rosDataDirty: threading.Event

    cacheEventsQueue: queue.Queue[CacheEvent]
    """Thread-safe queue of cache/buffer events to emit over WebSocket."""

    _httpServer: BaseWSGIServer | None = None
    _stateLock: threading.RLock
    _resultsLock: threading.RLock

    def __init__(
        self,
        algorithmConfig: AlgorithmConfig,
        algorithmLogger: Logger,
        algorithmThreadLogger: Logger,
        logLevel: int | str,
    ) -> None:
        self._threadShutdownEvent = threading.Event()
        self._stateLock = threading.RLock()
        self._resultsLock = threading.RLock()
        self.isHttpServerReady = False

        self.stitchingStatesByStitchingStages = {}
        self.stitchingResultsHoldersByStitchingStages = {}

        self.algorithmState = AlgorithmState(
            homographyBufferPreallocation=algorithmConfig.preallocation.homographyBufferPreallocation,
        )
        self.dataStale = True  # Stale until first successful run
        self.algorithmConfig = algorithmConfig

        self.algorithmLogger = algorithmLogger
        self.algorithmThreadLogger = algorithmThreadLogger

        self.sioLogger = getLogger("SIO")
        self.sioLogger.setLevel(logLevel)

        self.rosLogger = getLogger("ROS")
        self.rosLogger.setLevel(logLevel)

        self.httpServerLogger = getLogger("HTTPServer")
        self.httpServerLogger.setLevel(logLevel)

        self._logger = getLogger("InterThreadMemory")
        self._logger.setLevel(logLevel)

        self.logLevel = logLevel

        def _on_buffer_reallocate(
            allocator_id: str,
            direction: str,
            old_shape: tuple[int, ...],
            new_shape: tuple[int, ...],
        ) -> None:
            with Tracing.ScopedZone("buffer_reallocated"):
                Tracing.message(f"Buffer reallocated: {allocator_id} {direction} {old_shape} -> {new_shape}")
                self.emit_cache_event(
                    CacheEvent(
                        event_type="buffer_reallocated",
                        data={
                            "allocator_id": allocator_id,
                            "direction": direction,
                            "old_shape": list(old_shape),
                            "new_shape": list(new_shape),
                        },
                    )
                )

        self.algorithmResultsHolder = PreallocatedAlgorithmResultsHolder(
            config=algorithmConfig,
            on_buffer_reallocate=_on_buffer_reallocate,
        )

        self.robot_kalman_tracker = RobotKalmanTracker(algorithmConfig.configuredRobots)

        self.sioDataDirty = threading.Event()
        self.rosDataDirty = threading.Event()
        self.cacheEventsQueue = queue.Queue()

    def markHttpServerReady(self, server: BaseWSGIServer):
        self.isHttpServerReady = True
        self._httpServer = server

    def emit_cache_event(self, event: CacheEvent) -> None:
        """Queue a cache event for WebSocket emission and wake the SIO emitter."""
        self.cacheEventsQueue.put(event)
        self.sioDataDirty.set()

    def shutdown(self):
        self._logger.info("Shutting down threads via event...")
        self._threadShutdownEvent.set()

        if self._httpServer is not None:
            self._logger.info("Shutting down HTTP server...")
            self._httpServer.shutdown()
            self._httpServer = None

        self._logger.info("Shutdown completed")

    def isRunning(self):
        return not self._threadShutdownEvent.is_set()

    def getStitchingStateForStitchingStagePossiblyCreating(self, stitchingStageDescription: StitchingStageDescription) -> StitchingState:
        with self._stateLock:
            if stitchingStageDescription not in self.stitchingStatesByStitchingStages:
                self.stitchingStatesByStitchingStages[stitchingStageDescription] = StitchingState(
                    stitchingStageDescription.__str__(),
                    self.logLevel,
                    homographyBufferPreallocation=self.algorithmConfig.preallocation.homographyBufferPreallocation,
                )

            state = self.stitchingStatesByStitchingStages[stitchingStageDescription]
            state.markUsage()
            return state

    def getStitchingResultsHolderForStitchingStagePossiblyCreating(self, stitchingStageDescription: StitchingStageDescription) -> PreallocatedStitchingResultsHolder:
        with self._stateLock:
            if stitchingStageDescription not in self.stitchingResultsHoldersByStitchingStages:
                self.stitchingResultsHoldersByStitchingStages[stitchingStageDescription] = PreallocatedStitchingResultsHolder(
                    config=self.algorithmConfig,
                )

            holder = self.stitchingResultsHoldersByStitchingStages[stitchingStageDescription]
            holder.markUsage()
            return holder

    def markRound(self):
        with self._stateLock:
            for holder in self.stitchingResultsHoldersByStitchingStages.values():
                holder.markRoundAndCheckIfNeeded()

            for state in self.stitchingStatesByStitchingStages.values():
                state.markRoundAndCheckIfNeeded()
