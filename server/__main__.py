from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from copy import copy
from logging import basicConfig, getLogger
import os
import queue
import signal
import threading
import time
from typing import Any, Dict, cast
from wsgiref.types import WSGIApplication

from algorithms.floor_atlas import FloorAtlas
from algorithms.mapSegmentation import updateMapSegmentation
from algorithms.topdown import create_topdown_strategy
from algorithms.runtime_calibration import (
    accumulate_tag_quads,
    calibrate_from_accumulated_quads,
)
from algorithms.solver_pipeline import process_epipolar_pairs_from_image_packs
from algorithms.stitching import matchPairsForStitching, stitchImagePacks
from classes.config import AlgorithmConfig, CacheConfig, PreallocationConfig
from pathlib import Path
from classes.io.VideoCaptureWrapper.AbstractVideoCaptureWrapper import AbstractVideoCaptureWrapper
from classes.io.VideoCaptureWrapper.errors import VideoCaptureError, VideoEndedException
from classes.io.VideoCaptureWrapperFactory.LiveVideoCaptureWrapperFactory import LiveVideoCaptureWrapperFactory
from classes.io.VideoCaptureWrapperFactory.PlaybackVideoCaptureWrapperFactory import PlaybackVideoCaptureWrapperFactory
from classes.marker.CompositeDetection import CompositeDetection
from classes.misc.Timer import Timer
from classes.resources.ImagePack import ImagePack
from classes.resources.InterThreadMemory import CacheEvent, InterThreadMemory
from classes.resources.StitchingStageDescription import StitchingStageDescription
from classes.robot import RobotConfig, RobotKalmanParams
from classes.server.SioRoomsManager import SioRoomsStateTracker
from config import ConfigReader
import cv2
from errors import NextIterError, ProcessingError, StitchingStrategyInfeasibleError
from flask import Flask, abort as flask_abort
import numpy as np
from pynput import keyboard
import socketio
from utils.constants import DEFAULT_RECALC_HEURISTIC_EXTINGUISHING_MARKER_CONSECUTIVE_ROUNDS_DELAY, RECALC_HEURISTIC_DISPLACED_DETECTIONS_MIN_DIFF_NORM_PERCENT_THRESH, serverRootPath
from utils.plotting import drawDetectionsOnImage, putText
from utils.tracing import Tracing
from utils.geometry import pointsDistance
from werkzeug.serving import WSGIRequestHandler, is_running_from_reloader, make_server

# below: flag that controls at runtime whether to run on a pre-recorded video file (then
# the catalog containing camera captures shall be passed in) or None to run live from available cameras
RUN_FROM_PLAYBACK_DIRECTORY: str | None = os.environ.get("RUN_FROM_PLAYBACK_DIRECTORY", None)

configReader = ConfigReader()
readConfig = configReader.readConfig(playback_scenario=RUN_FROM_PLAYBACK_DIRECTORY)
httpConfig = cast(Dict, readConfig.get("http"))
isDebug: bool = cast(bool, readConfig.get("debug"))
isProfiling: bool = cast(bool, readConfig.get("profiling"))
cacheConfig = CacheConfig.from_dict(cast(Dict, readConfig.get("caching", {})))
preallocationConfig = PreallocationConfig.from_dict(cast(Dict, readConfig.get("preallocation", {})))

OUTPUT_JPEG_QUALITY: int = cast(int, readConfig.get("algorithmJPEGQuality"))

basicConfig(
    format="%(asctime)s - [%(levelname)s] %(name)s - %(message)s",
    level=cast(Any, readConfig.get("otherLogLevel")),
)

rootLogger = getLogger("M-C M-R VLS")
rootLogger.setLevel(cast(Any, readConfig.get("serverLogLevel")))

httpLogger = getLogger("HTTP")
httpLogger.setLevel(cast(Any, readConfig.get("httpLogLevel")))

algorithmLogger = getLogger("Algorithm")
algorithmLogger.setLevel(cast(Any, readConfig.get("algorithmLogLevel")))

algorithmThreadLogger = getLogger("AlgorithmInstrumenter")
algorithmThreadLogger.setLevel(cast(Any, readConfig.get("algorithmInstrumenterLogLevel")))

Tracing.initialize(isProfiling)

flaskApp = Flask(
    __name__,
    static_folder=None if isDebug else os.path.join(serverRootPath, "frontend_built"),
    static_url_path=None if isDebug else "/",
)
sio = socketio.Server(cors_allowed_origins=("*" if isDebug else None), async_mode="threading")
app = socketio.WSGIApp(sio, flaskApp)


if not isDebug:
    rootLogger.info("Running in production mode & serving prebuilt frontend")
else:
    rootLogger.info("Running in debug mode")

rootLogger.info("⌨️⌨️⌨️  Press 'r' to force manual cache invalidation ⌨️⌨️⌨️")

if isProfiling:
    rootLogger.info("Profiling is enabled, traces will be collected")


def genSingleRobotRoom(robotId: str | int) -> str:
    return f"robot_{robotId}"


@flaskApp.get("/status")
def status():
    return {"status": "ok"}


@flaskApp.route("/")
def root():
    if isDebug:
        flask_abort(404)
    else:
        return flaskApp.send_static_file("index.html")


@sio.on("connect")  # pyright: ignore[reportOptionalCall]
def onConnect(sid, environ):
    httpLogger.info(f"> Client connected: {sid}")
    SioRoomsStateTracker.handleConnected(sid)
    # memory is assigned later in module load; connect only fires after the server is up.
    try:
        sio.emit("system_status", {"calibration": memory.calibration_telemetry()}, to=sid)
    except NameError:
        pass


@sio.on("disconnect")  # pyright: ignore[reportOptionalCall]
def onDisconnect(sid):
    httpLogger.info(f"> Client disconnected: {sid}")
    SioRoomsStateTracker.handleDisconnected(sid)


@sio.on("subscribe_all_detections")  # pyright: ignore[reportOptionalCall]
def onSubscribeAllDetections(sid):
    httpLogger.info(f"> Client {sid} subscribed to all detections")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS)


@sio.on("unsubscribe_all_detections")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeAllDetections(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from all detections")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS)


@sio.on("subscribe_algorithm_previews")  # pyright: ignore[reportOptionalCall]
def onSubscribeAlgorithmPreviews(sid):
    httpLogger.info(f"> Client {sid} subscribed to algorithm previews")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS)


@sio.on("unsubscribe_algorithm_previews")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeAlgorithmPreviews(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from algorithm previews")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS)


@sio.on("subscribe_stitching_previews")  # pyright: ignore[reportOptionalCall]
def onSubscribeStitchingPreviews(sid):
    httpLogger.info(f"> Client {sid} subscribed to stitching previews")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS)


@sio.on("unsubscribe_stitching_previews")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeStitchingPreviews(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from stitching previews")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS)


@sio.on("subscribe_map_numpy_feed")  # pyright: ignore[reportOptionalCall]
def onSubscribeMapNumpyFeed(sid):
    httpLogger.info(f"> Client {sid} subscribed to map numpy feed")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY)


@sio.on("unsubscribe_map_numpy_feed")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeMapNumpyFeed(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from map numpy feed")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY)


@sio.on("subscribe_cache_events")  # pyright: ignore[reportOptionalCall]
def onSubscribeCacheEvents(sid):
    httpLogger.info(f"> Client {sid} subscribed to cache events")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS)


@sio.on("unsubscribe_cache_events")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeCacheEvents(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from cache events")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS)


@sio.on("subscribe_map_segmentation_previews")  # pyright: ignore[reportOptionalCall]
def onSubscribeMapSegmentationPreviews(sid):
    httpLogger.info(f"> Client {sid} subscribed to map segmentation previews")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS)


@sio.on("subscribe_camera_previews")  # pyright: ignore[reportOptionalCall]
def onSubscribeCameraSegmentationPreviews(sid):
    httpLogger.info(f"> Client {sid} subscribed to map segmentation previews")

    sio.enter_room(sid, SioRoomsStateTracker.SIO_ROOM_CAMERA_PREVIEWS)
    SioRoomsStateTracker.handleRoomJoinedBy(sid, SioRoomsStateTracker.SIO_ROOM_CAMERA_PREVIEWS)


@sio.on("unsubscribe_map_segmentation_previews")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeMapSegmentationPreviews(sid):
    httpLogger.info(f"> Client {sid} unsubscribed from map segmentation previews")

    sio.leave_room(sid, SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS)
    SioRoomsStateTracker.handleRoomLeftBy(sid, SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS)


@sio.on("subscribe_single_robot")  # pyright: ignore[reportOptionalCall]
def onSubscribeRobot(sid, robotId):
    httpLogger.info(f"> Client {sid} subscribed to detection data for robot {robotId}")

    sio.enter_room(sid, genSingleRobotRoom(robotId))


@sio.on("unsubscribe_single_robot")  # pyright: ignore[reportOptionalCall]
def onUnsubscribeRobot(sid, robotId):
    httpLogger.info(f"> Client {sid} unsubscribed from detection data for robot {robotId}")

    sio.leave_room(sid, genSingleRobotRoom(robotId))


@sio.on("invalidate_caches")  # pyright: ignore[reportOptionalCall]
def onInvalidateCaches(sid):
    httpLogger.info(f"> Client {sid} requested cache invalidation")
    memory.invalidate_caches(reason="frontend request")
    sio.emit("invalidate_caches_ack", {"ok": True}, to=sid)


@sio.on("recalibrate_cameras")  # pyright: ignore[reportOptionalCall]
def onRecalibrateCameras(sid):
    httpLogger.info(f"> Client {sid} requested camera recalibration")
    ok = memory.request_recalibrate()
    if not ok:
        sio.emit(
            "recalibrate_cameras_ack",
            {"ok": False, "error": "Recalibration requires the epipolar solver"},
            to=sid,
        )
        return
    sio.emit("recalibrate_cameras_ack", {"ok": True}, to=sid)


_global_robot_kalman = RobotKalmanParams.from_config_dict(readConfig.get("robotKalmanFilter"))
configured_robots_list = [
    RobotConfig(
        id=int(robot_entry["id"]),
        host=str(robot_entry["host"]),
        kalman_params=_global_robot_kalman.merged_with(robot_entry.get("kalman")),
    )
    for robot_entry in readConfig["robots"]
]

camera_calibrations: dict = {}
_auto_cal_cfg = readConfig.get("autoCalibrateOnStart")
_use_epipolar = bool(readConfig.get("useEpipolarGeometry", False))
_auto_calibrate = _use_epipolar if _auto_cal_cfg is None else bool(_auto_cal_cfg)

algorithmConfig = AlgorithmConfig(
    mapDefinition=readConfig["map"],
    dimensions=readConfig["dimensions"],
    configuredRobots=configured_robots_list,
    recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay=readConfig.get(
        "recalcHeuristicExtinguishingMarkerConsecutiveRoundsDelay",
        DEFAULT_RECALC_HEURISTIC_EXTINGUISHING_MARKER_CONSECUTIVE_ROUNDS_DELAY,
    ),
    recalcHeuristicAppearingMarkerConsecutiveRoundsDelay=readConfig.get(
        "recalcHeuristicAppearingMarkerConsecutiveRoundsDelay",
        4,
    ),
    markerVisibilityEwmaAlpha=float(readConfig.get("markerVisibilityEwmaAlpha", 0.35)),
    markerVisibilityThresholdHigh=float(readConfig.get("markerVisibilityThresholdHigh", 0.75)),
    markerVisibilityThresholdLow=float(readConfig.get("markerVisibilityThresholdLow", 0.25)),
    markerDetector=str(readConfig.get("markerDetector", "apriltag")),
    markerDetectorParams=readConfig.get("markerDetectorParams"),
    mapSegmentation=readConfig.get("mapSegmentation"),
    caching=cacheConfig,
    preallocation=preallocationConfig,
    logger=algorithmLogger,
    useEpipolarGeometry=_use_epipolar,
    usePerCameraSolver=bool(readConfig.get("usePerCameraSolver", False)),
    topDownFitUseInteriorTags=bool(readConfig.get("topDownFitUseInteriorTags", True)),
    useAtlasCornerResolution=bool(readConfig.get("useAtlasCornerResolution", True)),
    topDownStrategy=str(readConfig.get("topDownStrategy", "mosaic")),
    camera_calibrations=camera_calibrations,
    autoCalibrateOnStart=_auto_calibrate if _use_epipolar else False,
)

memory = InterThreadMemory(
    algorithmConfig=algorithmConfig,
    algorithmLogger=algorithmLogger,
    algorithmThreadLogger=algorithmThreadLogger,
    logLevel=cast(Any, readConfig.get("interThreadMemoryLogLevel")),
)


def initializeAlgorithmWorkerThread(i: int):
    Tracing.threadName(f"algorithm worker #{i}")
    # keep this thread busy for some time so that the tasks are scheduled on all threads
    time.sleep(0.6)


def mainAlgorithmWorker(memory: InterThreadMemory):
    Tracing.threadName("mainAlgorithmWorker")

    while (readConfig["algorithmWaitForHTTPServerReady"] and not memory.isHttpServerReady) and memory.isRunning():
        pass

    captures: list[AbstractVideoCaptureWrapper]

    enforcedFrameDelay: float | None = None

    if RUN_FROM_PLAYBACK_DIRECTORY is not None:
        captures = PlaybackVideoCaptureWrapperFactory.createVideoCaptureWrappers(
            logger=memory.algorithmThreadLogger,
            videosDirectoryName=RUN_FROM_PLAYBACK_DIRECTORY,
            loop=readConfig.get("playbackModeLoop", True),
        )

        # try to read if FPS should be enforced
        fpsFilePath = PlaybackVideoCaptureWrapperFactory.getPlaybackDirPath(os.path.join(RUN_FROM_PLAYBACK_DIRECTORY, "fps.txt"))

        if os.path.exists(fpsFilePath):
            with open(fpsFilePath, "r") as fpsFile:
                enforcedFPS = float(fpsFile.read().strip())

                algorithmLogger.info(f"Enforcing FPS: {enforcedFPS} (read from {fpsFilePath})")

                enforcedFrameDelay = 1 / enforcedFPS

    else:
        captures = LiveVideoCaptureWrapperFactory.createVideoCaptureWrappers(
            logger=memory.algorithmThreadLogger,
            rtspSources=readConfig.get("rtspSources", []),
        )

    fpsAverageFrameCounter = 0
    averageFps = 0
    fpsAveragerTimer = Timer(duration_seconds=3)
    mapSegmentationIntervalSeconds = max(0, int(memory.algorithmConfig.caching.mapSegmentationIntervalSeconds))
    mapRecalculationTimer = Timer(
        duration_seconds=mapSegmentationIntervalSeconds,
        initial_elapsed_state=True,
    )
    mapRecalculationFutureHolder: Future | None = None

    nAlgorithmWorkerThreads = max(1, min((os.cpu_count() or 1) - 3, len(captures)))
    threadPoolExecutor = ThreadPoolExecutor(max_workers=nAlgorithmWorkerThreads, thread_name_prefix="processing_")
    miscTaskExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="misc_task_pool_")

    for i in range(nAlgorithmWorkerThreads):
        threadPoolExecutor.submit(initializeAlgorithmWorkerThread, i)

    markerPositionsLastRound: dict[int, CompositeDetection] = {}

    # On-demand / startup calibration accumulation
    calib_accum_active = False
    calib_accum_frames = 0
    calib_accum_target_frames = 20
    calib_quad_buckets: dict = {}
    calib_image_sizes: dict[str, tuple[int, int]] = {}
    # Robots move — exclude them from the floor-tag atlas / calibration target.
    calib_exclude_tag_ids = set(int(r) for r in memory.algorithmConfig.configuredRobotIDs)

    # Live floor-tag atlas: exact per-raw-camera -> map homographies, feeding the
    # per-camera consensus solver, the interior-tag top-down fit, and the
    # piecewise map-boundary preview overlay.
    floorAtlas = FloorAtlas(memory.algorithmConfig)
    memory.algorithmState.floorAtlas = floorAtlas
    topDownStrategy = create_topdown_strategy(memory.algorithmConfig)
    algorithmLogger.info(f"Top-down strategy: {type(topDownStrategy).__name__}")

    while memory.isRunning():
        try:
            startTime = time.time()

            imagePacks: list[ImagePack] = []

            with Tracing.ScopedZone("capture_cameras"):
                for i, captureWrapper in enumerate(captures):
                    try:
                        frame = captureWrapper.read()
                    except VideoCaptureError:
                        algorithmLogger.warning(f"Failed to read frame from camera {captureWrapper.getIdentifier()}")
                        raise NextIterError()
                    except VideoEndedException as e:
                        algorithmLogger.warning(e)
                        exit(0)

                    imagePacks.append(
                        ImagePack(
                            image=frame,
                            stitchingStageDescription=StitchingStageDescription(str(i)),
                        )
                    )

            stitchingFailed: bool = False
            raw_camera_packs = list(imagePacks)

            with Tracing.ScopedZone("floor_atlas"):
                floorAtlas.observe(raw_camera_packs)
                if floorAtlas.maybe_solve(logger=memory.algorithmThreadLogger):
                    memory.emit_cache_event(
                        CacheEvent(
                            event_type="floor_atlas_resolved",
                            data={"generation": floorAtlas.solve_generation, "cameras": len(floorAtlas.camera_h)},
                        )
                    )
                if memory.algorithmConfig.usePerCameraSolver or memory.algorithmConfig.topDownStrategy == "orthorectified":
                    # Per-camera consensus robot poses (map-normalized); consumed
                    # by topDownWarp in place of the mosaic-derived measurements.
                    memory.algorithmState.percamRobotMeasurements = floorAtlas.robot_measurements(
                        raw_camera_packs,
                        memory.algorithmConfig,
                    )

            # Start or continue on-demand / startup calibration collection
            if memory.algorithmConfig.useEpipolarGeometry:
                if not calib_accum_active and memory.take_recalibrate_request():
                    calib_accum_active = True
                    calib_accum_frames = 0
                    calib_quad_buckets = {}
                    calib_image_sizes = {}
                    algorithmLogger.info("📷 Camera calibration: collecting frames…")

                if calib_accum_active:
                    accumulate_tag_quads(raw_camera_packs, calib_exclude_tag_ids, calib_quad_buckets)
                    for pack in raw_camera_packs:
                        if pack.image is None:
                            continue
                        h, w = pack.image.shape[:2]
                        calib_image_sizes[str(pack.stitchingStageDescription)] = (w, h)
                    calib_accum_frames += 1

                    if calib_accum_frames >= calib_accum_target_frames:
                        try:
                            with Tracing.ScopedZone("runtime_calibration"):
                                new_calibrations = calibrate_from_accumulated_quads(
                                    calib_quad_buckets,
                                    calib_image_sizes,
                                    memory.algorithmConfig,
                                    min_tags=2,
                                    min_observations=2,
                                    logger=algorithmLogger,
                                )
                            memory.algorithmConfig.camera_calibrations = {
                                **memory.algorithmConfig.camera_calibrations,
                                **new_calibrations,
                            }
                            algorithmLogger.info(
                                "📷 Camera calibration complete: %d camera(s) (memory-only)",
                                len(new_calibrations),
                            )
                            memory.set_calibration_result(ok=True)
                            memory.invalidate_caches(reason="camera recalibration")
                        except Exception as exc:  # noqa: BLE001
                            algorithmLogger.error("📷 Camera calibration failed: %s", exc)
                            memory.set_calibration_result(ok=False, error=str(exc))
                        finally:
                            calib_accum_active = False
                            calib_accum_frames = 0
                            calib_quad_buckets = {}
                            calib_image_sizes = {}

            if memory.algorithmConfig.useEpipolarGeometry:
                process_epipolar_pairs_from_image_packs(
                    raw_camera_packs,
                    memory.algorithmState,
                    memory.algorithmConfig,
                    memory.algorithmThreadLogger,
                )

            # emit camera previews just to the clients that are subscribed to them
            with Tracing.ScopedZone("emit_previews"):
                if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_CAMERA_PREVIEWS) > 0:
                    data = {}
                    for i, imagePack in enumerate(imagePacks):
                        if imagePack.image is not None:
                            img = imagePack.image.copy()

                            for detection in imagePack.compositeDetectionsStore.values():
                                topLeft = detection.aggregatedTopLeft

                                drawDetectionsOnImage(
                                    img,
                                    [detection],
                                    algorithmConfig,
                                    colorBGR=(0, 0, 255),
                                    bDrawLabels=False,
                                )
                                putText(
                                    img,
                                    str(detection.data),
                                    (int(topLeft[0]), int(topLeft[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2,
                                    (0, 0, 255),
                                    4,
                                )
                        else:
                            img = np.zeros((400, 400, 3), dtype=np.uint8)
                            putText(
                                img,
                                "No image",
                                (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2.0,
                                (0, 0, 255),
                                4,
                            )

                        data[i] = encodeJpegForSIO(img, debugLabel=f"camera_{i}")

                    sio.emit(
                        event="camera_previews",
                        room=SioRoomsStateTracker.SIO_ROOM_CAMERA_PREVIEWS,
                        data=data,
                    )

            with Tracing.ScopedZone("stitching") as stitchingZone:
                while len(imagePacks) > 1 and memory.isRunning():
                    futures: list[Future] = []

                    anythingRecalculated: bool = False
                    carryOver: ImagePack | None = None

                    try:
                        # submit a worker for each pair of items
                        for chunk in matchPairsForStitching(imagePacks=imagePacks):
                            if len(chunk) == 1:
                                carryOver = chunk[0]
                            else:
                                [imagePack1, imagePack2] = chunk

                                stitchingStageDescription = StitchingStageDescription(
                                    imagePack1.stitchingStageDescription,
                                    imagePack2.stitchingStageDescription,
                                )

                                futures.append(
                                    threadPoolExecutor.submit(
                                        stitchImagePacks,
                                        imagePacks=[imagePack1, imagePack2],
                                        stitchingStageDescription=stitchingStageDescription,
                                        state=memory.getStitchingStateForStitchingStagePossiblyCreating(stitchingStageDescription),
                                        config=memory.algorithmConfig,
                                        logger=memory.algorithmThreadLogger,
                                        resultsHolder=memory.getStitchingResultsHolderForStitchingStagePossiblyCreating(stitchingStageDescription),
                                        memory=memory,
                                    )
                                )

                    except StitchingStrategyInfeasibleError as e:
                        if e.pack1 is not None and e.pack2 is not None:
                            pairDesc = f"{e.pack1.stitchingStageDescription} x {e.pack2.stitchingStageDescription}"
                        else:
                            pairDesc = "unknown pair"
                        reason = f", reason: {e.reason}" if e.reason else ""
                        memory.algorithmThreadLogger.warning(f"Stitching strategy infeasible for image ({pairDesc}){reason} - insufficient pairs for stitching, dropping synced capture frame")
                        memory.algorithmState.imagePackDirty = True  # recalculate next iteration for possible recovery
                        memory.dataStale = True
                        memory.sioDataDirty.set()
                        stitchingFailed = True
                        anythingRecalculated = True
                        break  # break the outer while loop to ensure new samples are collected; 'continue' here would result in an inifite loop with just this sample

                    # next iteration should work on the cumulated results from this iteration
                    # note: calling result() on a Future propagates a possible error to the caller context if it occurred inside the Future
                    try:
                        results = [
                            *[future.result() for future in wait(futures).done],
                            *([carryOver] if carryOver is not None else []),
                        ]

                        if None in results:
                            raise ProcessingError(
                                "One of the stitching tasks failed silently without raising an exception",
                                code="STITCHING_TASK_FAILED",
                            )

                        imagePacks = sorted(results)

                        if len(imagePacks) == 0:
                            raise ProcessingError(
                                "No valid image packs collected from futures",
                                code="NO_VALID_IMAGE_PACKS",
                            )
                    except ProcessingError as e:
                        memory.algorithmLogger.error("❌ Stitching failed ❌")
                        memory.algorithmLogger.error(f"\t> Full explanation: {e}")
                        memory.algorithmLogger.error(f"\t> Error code: {e.code}")
                        memory.algorithmLogger.error(f"\t> Erroneous value: {e.value}")
                        memory.algorithmState.imagePackDirty = True
                        memory.dataStale = True
                        memory.sioDataDirty.set()
                        for imagePack in imagePacks:
                            stitchingStageDescription = StitchingStageDescription(imagePack.stitchingStageDescription)
                            try:
                                memory.stitchingStatesByStitchingStages[stitchingStageDescription].markNeedsHomographyRecalculation("stitching failed")
                                memory.emit_cache_event(
                                    CacheEvent(
                                        event_type="homography_cache_invalidated",
                                        data={"stage": stitchingStageDescription.__str__(), "reason": "stitching failed", "cache_type": "stitching"},
                                    )
                                )
                            except KeyError:
                                pass  # apparently that stitching stage description had never been cached yet
                        Tracing.message("Stitching failed: " + str(e))
                        stitchingZone.color(Tracing.COLOR_RED)
                        memory.algorithmState.nextIterForceRecalcReason = e.code
                        stitchingFailed = True
                        anythingRecalculated = True
                        break
                    else:
                        memory.algorithmLogger.debug(f"📎 Stitching {imagePacks[0].stitchingStageDescription} successful 📎")
                        Tracing.message(f"Stitching {imagePacks[0].stitchingStageDescription} successful")
                        stitchingZone.color(Tracing.COLOR_GREEN)
                    finally:
                        memory.algorithmLogger.debug("== Finished stitching samples == ")

            # possibly dispose unused cached resources & states
            memory.markRound()

            if not memory.isRunning():
                break

            if not stitchingFailed:
                try:
                    with Tracing.ScopedZone("processing") as processingZone:
                        processingZoneColor = Tracing.COLOR_GREEN
                        stitchingImagePack = imagePacks[0]

                        memory.algorithmState.handleBeforeStage(stitchingImagePack)

                        # main algorithm work
                        with memory._resultsLock:
                            memory.algorithmResultsHolder.visibilityTelemetry = {
                                "perCamera": [
                                    {
                                        "camera": camera_index,
                                        "markers": list(image_pack.compositeDetectionsStore.keys()),
                                    }
                                    for camera_index, image_pack in enumerate(raw_camera_packs)
                                ],
                            }
                            topDownStrategy.process(
                                stitchingImagePack=stitchingImagePack,
                                raw_camera_packs=raw_camera_packs,
                                state=memory.algorithmState,
                                config=memory.algorithmConfig,
                                logger=memory.algorithmLogger,
                                resultsHolder=memory.algorithmResultsHolder,
                                memory=memory,
                            )

                        # heuristic 2: distance between aggregated centroids of the same marker data
                        for (
                            markerData,
                            detection,
                        ) in stitchingImagePack.compositeDetectionsStore.items():
                            otherDetection = markerPositionsLastRound.get(markerData)

                            if otherDetection is None or memory.algorithmResultsHolder.topDownImg is None:
                                continue  # no previous position to compare to or no valid top-down image

                            distance = pointsDistance(
                                detection.aggregatedCentroid,
                                otherDetection.aggregatedCentroid,
                            )
                            displacementThreshold = min(*memory.algorithmResultsHolder.topDownImg.shape[:2]) * RECALC_HEURISTIC_DISPLACED_DETECTIONS_MIN_DIFF_NORM_PERCENT_THRESH

                            if distance >= displacementThreshold:
                                msg = f"Detections for marker {markerData} in after stitching are displaced far enough to invalidate cached homography. Distance: {distance:.2f}, max threshold: {displacementThreshold:.2f}"
                                memory.algorithmLogger.debug(msg)
                                Tracing.message(msg)

                                memory.algorithmState.nextIterForceRecalcReason = "DIFF_DISPLACED_DETECTIONS_POST_STITCHING"
                                memory.algorithmState.imagePackDirty = True
                                memory.stitchingStatesByStitchingStages.clear()
                                # NOTE: the floor atlas is deliberately NOT
                                # invalidated here - this heuristic measures
                                # displacement in the STITCHED frame, which
                                # jitters whenever stitch homographies are
                                # recalculated and says nothing about raw
                                # camera geometry. The atlas detects genuine
                                # camera movement itself, in the raw frame
                                # (see FloorAtlas.observe).
                                memory.dataStale = True
                                memory.sioDataDirty.set()
                                processingZoneColor = Tracing.COLOR_DARKGREEN

                        markerPositionsLastRound = copy(stitchingImagePack.compositeDetectionsStore)

                        processingZone.color(processingZoneColor)

                    if not memory.isRunning():
                        break

                    if mapRecalculationFutureHolder is not None and mapRecalculationFutureHolder.done():
                        try:
                            mapRecalculationFutureHolder.result()
                            memory.algorithmLogger.debug("Done recalculating map")
                        except Exception as e:  # noqa: BLE001
                            memory.algorithmLogger.exception(f"Map recalculation failed: {e}")
                        finally:
                            mapRecalculationFutureHolder = None

                    # map recalculation
                    if memory.algorithmConfig.caching.mapSegmentationThrottle:
                        shouldRecalculateMap = mapRecalculationTimer.hasElapsed()
                    else:
                        shouldRecalculateMap = True

                    if shouldRecalculateMap and mapRecalculationFutureHolder is None:
                        memory.algorithmLogger.debug("Recalculating map...")

                        def _recalculate_map():
                            with Tracing.ScopedZone("map_segmentation"):
                                with memory._resultsLock:
                                    updateMapSegmentation(memory.algorithmResultsHolder, memory.algorithmLogger)

                        mapRecalculationFutureHolder = miscTaskExecutor.submit(_recalculate_map)
                        if memory.algorithmConfig.caching.mapSegmentationThrottle:
                            mapRecalculationTimer.start()
                    # no need to do memory.sioDataDirty.set() - it will happen in try-else branch

                except ProcessingError as e:
                    memory.algorithmLogger.error("❌ Processing failed ❌")
                    memory.algorithmLogger.error(f"\t> Full explanation: {e}")
                    memory.algorithmLogger.error(f"\t> Error code: {e.code}")
                    memory.algorithmLogger.error(f"\t> Erroneous value: {e.value}")
                    memory.algorithmState.nextIterForceRecalcReason = e.code
                    memory.stitchingStatesByStitchingStages.clear()
                    memory.dataStale = True
                    memory.sioDataDirty.set()
                    processingZone.color(Tracing.COLOR_RED)
                    stitchingFailed = True
                    anythingRecalculated = True
                else:
                    memory.algorithmLogger.debug("✅ Processing successful ✅")
                    memory.dataStale = False
                    memory.sioDataDirty.set()
                finally:
                    memory.algorithmLogger.debug("== Finished processing sample == ")

            if anythingRecalculated:
                memory.algorithmState.imagePackDirty = True

            if enforcedFrameDelay is not None:
                time.sleep(max(0, enforcedFrameDelay - (time.time() - startTime)))

            if fpsAveragerTimer.hasElapsed():
                fps_averager_elapsed_time = fpsAveragerTimer.getTimeSinceStart()
                averageFps = fpsAverageFrameCounter / fps_averager_elapsed_time

                memory.algorithmResultsHolder.fps = averageFps

                memory.algorithmThreadLogger.debug(f"> New avg. FPS: {averageFps:.2f}fps, averaged over {fpsAverageFrameCounter} frames in {fps_averager_elapsed_time:.2f}s")

                fpsAverageFrameCounter = 0
                fpsAveragerTimer.start()

            fpsAverageFrameCounter += 1

            if memory.algorithmResultsHolder.topDownImgAnnotated is not None:
                Tracing.frameImage(memory.algorithmResultsHolder.topDownImgAnnotated)

        except NextIterError:
            # placeholder to run continue in this while loop
            continue

        except Exception as e:
            memory.algorithmThreadLogger.exception(e)
            os._exit(-1)

        finally:
            Tracing.frameMark()

    memory.algorithmThreadLogger.info("Algorithm worker thread cleaning up...")

    if mapRecalculationFutureHolder is not None:
        mapRecalculationFutureHolder.cancel()
    miscTaskExecutor.shutdown(wait=False)

    for cap in captures:
        memory.algorithmThreadLogger.info(f"Releasing camera {cap.getIdentifier()}")
        cap.release()

    memory.algorithmThreadLogger.info("Algorithm worker thread finished")


threads: list[threading.Thread] = []

mainAlgorithmThread = threading.Thread(target=mainAlgorithmWorker, args=(memory,), name="Algorithm")
mainAlgorithmThread.daemon = False
mainAlgorithmThread.start()

threads.append(mainAlgorithmThread)


def encodeJpegForSIO(image: np.ndarray | None, debugLabel: str) -> bytes | None:
    if image is None:
        return None

    try:
        _, jpeg = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
        )
    except:  # noqa: E722
        httpLogger.warning(f"Failed to encode image '{debugLabel}' as JPEG")
        return None

    return jpeg.tobytes()


def sioEmitterWorker(memory: InterThreadMemory):
    Tracing.threadName("sioEmitterWorker")

    while not memory.isHttpServerReady and memory.isRunning():
        pass

    while memory.isRunning():
        if not memory.sioDataDirty.wait(timeout=0.5):
            continue

        # drain and emit cache events (homography invalidations, buffer reallocations)
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS) > 0:
            try:
                while True:
                    ev = memory.cacheEventsQueue.get_nowait()
                    sio.emit(
                        event="cache_event",
                        room=SioRoomsStateTracker.SIO_ROOM_CACHE_EVENTS,
                        data={"type": ev.event_type, **ev.data},
                    )
            except queue.Empty:
                pass

        # emit all results just to the clients that are subscribed to them
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS) > 0:
            with memory._resultsLock:
                topDownDetections = [det.toDTO() for det in memory.algorithmResultsHolder.topDownDetections]
                robotDetections = [det.toDTO(include_robot_info=True) for det in memory.algorithmResultsHolder.robotDetections]
                fps = memory.algorithmResultsHolder.fps
            try:
                visibility = memory.algorithmResultsHolder.visibilityTelemetry
                sio.emit(
                    event="all_detections",
                    room=SioRoomsStateTracker.SIO_ROOM_ALL_DETECTIONS,
                    data={
                        "topDownDetections": topDownDetections,
                        "detections": robotDetections,
                        "fps": fps,
                        "dataStale": memory.dataStale,
                        "visibility": visibility,
                        "calibration": memory.calibration_telemetry(),
                    },
                )
            except Exception as e:  # noqa: BLE001
                memory.sioLogger.warning(f"SIO emit failed (all_detections): {e}")

        # emit algorithm previews just to the clients that are subscribed to them
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS) > 0:
            with memory._resultsLock:
                topDownImg = memory.algorithmResultsHolder.topDownImg
                topDownImgAnnotated = memory.algorithmResultsHolder.topDownImgAnnotated
                stitchedImg = memory.algorithmResultsHolder.stitchedImg
                stitchedImgAnnotated = memory.algorithmResultsHolder.stitchedImgAnnotated
            topDownJPEG = encodeJpegForSIO(topDownImg, debugLabel="topDownImg")
            topDownAnnotatedJPEG = encodeJpegForSIO(topDownImgAnnotated, debugLabel="topDownImgAnnotated")
            stitchedJPEG = encodeJpegForSIO(stitchedImg, debugLabel="stitchedImg")
            stitchedAnnotatedJPEG = encodeJpegForSIO(stitchedImgAnnotated, debugLabel="stitchedImgAnnotated")
            try:
                sio.emit(
                    event="algorithm_previews",
                    room=SioRoomsStateTracker.SIO_ROOM_ALGORITHM_PREVIEWS,
                    data={
                        "topDownImg": topDownJPEG,
                        "topDownImgAnnotated": topDownAnnotatedJPEG,
                        "stitchedImg": stitchedJPEG,
                        "stitchedImgAnnotated": stitchedAnnotatedJPEG,
                    },
                )
            except Exception as e:  # noqa: BLE001
                memory.sioLogger.warning(f"SIO emit failed (algorithm_previews): {e}")

        # emit stitching previews just to the clients that are subscribed to them
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS) > 0:
            commonFeaturesJPEGsMap: dict[str, bytes | None] = {}

            # to prevent 'RuntimeError: dictionary changed size during iteration', keys are copied and used for atomic access to values
            with memory._stateLock:
                stitchingStages = list(memory.stitchingResultsHoldersByStitchingStages.items())
            for stitchingStage, stitchingResultsHolder in stitchingStages:
                if stitchingResultsHolder.wasActiveLastRound():
                    commonFeaturesJPEGsMap[stitchingStage.__str__()] = encodeJpegForSIO(
                        stitchingResultsHolder.commonFeaturesImg,
                        debugLabel="commonFeaturesImg",
                    )
            try:
                sio.emit(
                    event="stitching_previews",
                    room=SioRoomsStateTracker.SIO_ROOM_STITCHING_PREVIEWS,
                    data={
                        "commonFeaturesJPEGsMap": commonFeaturesJPEGsMap,
                    },
                )
            except Exception as e:  # noqa: BLE001
                memory.sioLogger.warning(f"SIO emit failed (stitching_previews): {e}")

        # emit map segmentation results just to the clients that are subscribed to them
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS) > 0:
            with memory._resultsLock:
                redStopLines = memory.algorithmResultsHolder.mapSegmentationRedStopLines
                yellowLines = memory.algorithmResultsHolder.mapSegmentationYellowLines
                roadComponents = memory.algorithmResultsHolder.mapSegmentationRoadComponents
            try:
                sio.emit(
                    event="map_segmentation_previews",
                    room=SioRoomsStateTracker.SIO_ROOM_MAP_SEGMENTATION_PREVIEWS,
                    data={
                        "redStopLines": encodeJpegForSIO(
                            redStopLines,
                            debugLabel="redStopLines",
                        ),
                        "yellowLines": encodeJpegForSIO(
                            yellowLines,
                            debugLabel="yellowLines",
                        ),
                        "roadComponents": encodeJpegForSIO(
                            roadComponents,
                            debugLabel="roadComponents",
                        ),
                    },
                )
            except Exception as e:  # noqa: BLE001
                memory.sioLogger.warning(f"SIO emit failed (map_segmentation_previews): {e}")

        # emit raw numpy map just to the clients that are subscribed to them
        if SioRoomsStateTracker.getClientsCountInRoom(SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY) > 0:
            with memory._resultsLock:
                roads = memory.algorithmResultsHolder.mapSegmentationRoadComponents
                redStops = memory.algorithmResultsHolder.mapSegmentationRedStopLines
                yellowLines = memory.algorithmResultsHolder.mapSegmentationYellowLines
            if roads is not None and redStops is not None and yellowLines is not None:
                mapWidthMeters = memory.algorithmConfig.mapWidthMeters
                mapHeightMeters = memory.algorithmConfig.mapHeightMeters

                [mapHeightPx, mapWidthPx] = roads.shape[:2]

                # take the average of density of the map in meters per pixel in both dimensions
                metersPerPixel = (mapWidthMeters / mapWidthPx + mapHeightMeters / mapHeightPx) / 2.0

                try:
                    sio.emit(
                        event="map_segmentation_numpy",
                        room=SioRoomsStateTracker.SIO_ROOM_MAP_NUMPY,
                        data={
                            # take the average of density of the map in meters per pixel in both dimensions
                            "metersPerPixel": metersPerPixel,
                            "widthPx": mapWidthPx,
                            "heightPx": mapHeightPx,
                            "roadsMask": roads.astype(np.uint8).tobytes(),
                            "stopLinesMask": redStops.astype(np.uint8).tobytes(),
                            "laneSeparatorLinesMask": yellowLines.astype(np.uint8).tobytes(),
                        },
                    )
                except Exception as e:  # noqa: BLE001
                    memory.sioLogger.warning(f"SIO emit failed (map_segmentation_numpy): {e}")

        # emit dedicated per-robot results for clients subscribed to them
        with memory._resultsLock:
            robotDetectionsSnapshot = list(memory.algorithmResultsHolder.robotDetections)
            fps = memory.algorithmResultsHolder.fps
        for det in robotDetectionsSnapshot:
            try:
                sio.emit(
                    event="single_robot_results",
                    room=genSingleRobotRoom(det.robot.id),
                    data={
                        "robot": det.robot.id,
                        "detection": det.toDTO(
                            # below: offset is needed since we are using a coordinate system that is 0-at-north,
                            # while the ROS consumers will use a coordinate system that is 0-at-east
                            include_robot_info=False,
                            azimuthOffsetDeg=-90,
                        ),
                        "fps": fps,
                    },
                )
            except Exception as e:  # noqa: BLE001
                memory.sioLogger.warning(f"SIO emit failed (robot): {e}")

        sioClientsCount = SioRoomsStateTracker.getTotalConnectedCount()
        memory.rosLogger.debug(f"> Emitting results to {sioClientsCount} client{'s' if sioClientsCount != 1 else ''}")

        memory.sioDataDirty.clear()

    memory.rosLogger.info("SIO emitter thread finished")


sioEmitterThread = threading.Thread(target=sioEmitterWorker, args=(memory,), name="SIO emitter")
sioEmitterThread.daemon = False
sioEmitterThread.start()

threads.append(sioEmitterThread)


def httpServerWorker(
    memory: InterThreadMemory,
    hostname: str,
    port: int,
    application: WSGIApplication,
    use_debugger: bool = False,
    use_evalex: bool = True,
    threaded: bool = False,
    processes: int = 1,
    request_handler: type[WSGIRequestHandler] | None = None,
    static_files: dict[str, str | tuple[str, str]] | None = None,
    passthrough_errors: bool = False,
):
    Tracing.threadName("httpServerWorker")

    memory.httpServerLogger.info("Initializing HTTP server...")

    if static_files:
        from werkzeug.middleware.shared_data import SharedDataMiddleware

        application = SharedDataMiddleware(application, static_files)

    if use_debugger:
        from werkzeug.debug import DebuggedApplication

        application = DebuggedApplication(application, evalex=use_evalex)
        # Allow the specified hostname to use the debugger, in addition to
        # localhost domains.
        application.trusted_hosts.append(hostname)

    if not is_running_from_reloader():
        fd = None
    else:
        fd = int(os.environ["WERKZEUG_SERVER_FD"])

    httpServer = make_server(
        hostname,
        port,
        application,
        threaded,
        processes,
        request_handler,
        passthrough_errors,
        fd=fd,
    )
    httpServer.socket.set_inheritable(True)
    os.environ["WERKZEUG_SERVER_FD"] = str(httpServer.fileno())

    httpServer.log_startup()

    memory.markHttpServerReady(httpServer)

    httpServer.serve_forever()

    memory.httpServerLogger.info("HTTP server thread finished")


httpServerThread = threading.Thread(
    target=httpServerWorker,
    kwargs={
        "memory": memory,
        "hostname": cast(str, httpConfig.get("host")),
        "port": cast(int, httpConfig.get("port")),
        "use_debugger": isDebug,
        "threaded": True,
        "application": app,
    },
    name="HTTP server",
)
httpServerThread.daemon = False
httpServerThread.start()

threads.append(httpServerThread)


def exitEverything():
    if not memory.isRunning():
        rootLogger.error("Exit requested, but the algorithm is already stopped, brutally quitting now!")
        exit(-1)
    else:
        rootLogger.info("> Stopping HTTP server & worker threads...")
        memory.shutdown()

        rootLogger.info("> Stopping threads before exit...")

        for thread in threads:
            if thread.is_alive():
                rootLogger.info(f"> Joining thread '{thread.name}'...")
                thread.join()

        if Tracing.isProfiling():
            rootLogger.info("> Shutting down Tracing utility...")
            Tracing.shutdown()

        rootLogger.info("> Exiting")

        exit(0)


signal.signal(signal.SIGINT, lambda _, __: exitEverything())


def keyboardThreadWorker(memory: InterThreadMemory):
    keycodeReload = keyboard.KeyCode.from_char("r")
    wasPressed = False
    with keyboard.Events() as events:
        while memory.isRunning():
            event = events.get(1)
            if event and event.key == keycodeReload:
                if not wasPressed:
                    print("💣💣💣 Resetting caches on user request 💣💣💣")
                    memory.invalidate_caches(reason="user input")
                    wasPressed = True
            else:
                wasPressed = False


keyboardThread = threading.Thread(target=keyboardThreadWorker, args=(memory,), name="Keyboard")
keyboardThread.daemon = False
keyboardThread.start()

threads.append(keyboardThread)

# threads sanity watchdog
while memory.isRunning():
    for thread in threads:
        if not thread.is_alive():
            rootLogger.fatal(f"Thread '{thread.name}' is dead, threads sanity watchdog killing everything!")

            exitEverything()

        memory._threadShutdownEvent.wait(1)

rootLogger.info("Main thread finished")
