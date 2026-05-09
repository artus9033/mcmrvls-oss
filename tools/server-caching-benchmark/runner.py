import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import socketio
import yaml


def get_progress_iter(iterable: list[tuple[str, str, dict[str, Any]]]):
    try:
        from tqdm import tqdm as tqdm_impl  # type: ignore
    except Exception:  # noqa: BLE001
        return iterable
    return tqdm_impl(iterable, total=len(iterable))


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = WORKSPACE_ROOT / "server"
CONFIG_PATH = SERVER_DIR / "config.yaml"
BACKUP_PATH = SERVER_DIR / "config.yaml.bkp"
OUTPUT_DIR = Path(__file__).resolve().parent / "out"
SIO_SERVER_URL = os.environ.get("SIO_SERVER_URL", "ws://localhost:6001")

PLAYBACK_DIRS = [
    "4cam-cache-busting-test-code-visibility-changing",
    "4cam-cache-busting-test-long-shaking-BL",
    "4cam-cache-busting-test-long-slow-rotating-BL",
    "4cam-cache-busting-test-slow-moving-TL",
    "4cam-cache-busting-test-slow-rotating-BL",
]

CACHE_TOGGLES_ORDER = [
    "stitchingHomographyCache",
    "topDownHomographyCache",
]


def resolve_program(program: str, env_var: str) -> str:
    exe: str | None = None
    env_path = os.environ.get(env_var)
    if env_path:
        exe = env_path
    else:
        exe = shutil.which(program)

    if not exe:
        raise FileNotFoundError(f"{program} not found. Set {env_var} or add to PATH.")

    if "aliased to" in exe:
        exe = exe.split("aliased to")[1].strip()

    return exe


def resolve_tracy_capture() -> str:
    return resolve_program("tracy-capture", "TRACY_CAPTURE_PATH")


def resolve_tracy_csvexport() -> str:
    return resolve_program("tracy-csvexport", "TRACY_CSVEXPORT_PATH")


def read_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def write_config(config: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, explicit_start=True)


PREALLOCATION_TOGGLES_ORDER = [
    "adaptiveBufferPreallocation",
    "homographyBufferPreallocation",
]


def build_cache_config(
    base_cache: dict[str, Any],
    enabled: bool,
    enabled_toggle: str | None,
) -> dict[str, Any]:
    cache_cfg = {
        "enabled": enabled,
        "stitchingHomographyCache": False,
        "topDownHomographyCache": False,
    }

    if enabled and enabled_toggle:
        cache_cfg[enabled_toggle] = True

    return cache_cfg


def build_all_on_cache_config(base_cache: dict[str, Any]) -> dict[str, Any]:
    cache_cfg = build_cache_config(
        base_cache=base_cache,
        enabled=True,
        enabled_toggle=None,
    )
    for toggle in CACHE_TOGGLES_ORDER:
        cache_cfg[toggle] = True
    return cache_cfg


def build_preallocation_config(
    adaptive: bool,
    homography: bool,
) -> dict[str, Any]:
    return {
        "adaptiveBufferPreallocation": adaptive,
        "homographyBufferPreallocation": homography,
    }


def build_preallocation_on_config() -> dict[str, Any]:
    return build_preallocation_config(adaptive=True, homography=True)


def build_preallocation_off_config() -> dict[str, Any]:
    return build_preallocation_config(adaptive=False, homography=False)


def start_tracy_capture(trace_path: Path) -> subprocess.Popen:
    tracy_capture = resolve_tracy_capture()
    cmd = [tracy_capture, "-o", str(trace_path), "-f"]

    address = os.environ.get("TRACY_CAPTURE_ADDRESS")
    port = os.environ.get("TRACY_CAPTURE_PORT")
    if address:
        cmd += ["-a", address]
    if port:
        cmd += ["-p", port]

    return subprocess.Popen(cmd, cwd=WORKSPACE_ROOT)


def stop_tracy_capture(proc: subprocess.Popen, timeout_seconds: int = 30) -> int:
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait(timeout=5)


def run_csvexport(csvexport_path: str, trace_path: Path, out_path: Path) -> bool:
    result = subprocess.run(
        [csvexport_path, str(trace_path)],
        capture_output=True,
        text=True,
        cwd=WORKSPACE_ROOT,
    )
    if result.returncode != 0:
        return False
    out_path.write_text(result.stdout)
    return True


class PositionCollector:
    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._data: list[dict[str, Any]] = []
        self._start_ts: float | None = None

    def start(self) -> None:
        self._start_ts = time.time()
        self._thread = threading.Thread(target=self._worker, name="PositionCollector")
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._data

    def _worker(self) -> None:
        client = socketio.SimpleClient()
        while not self._stop_event.is_set():
            try:
                print("Connecting to Socket.io server...")
                client.connect(url=self._server_url)
                print("Connected to Socket.io server!")
                break
            except Exception:
                time.sleep(1)

        if not client.connected:
            print("Failed to connect to Socket.io server!")
            return

        print("Subscribing to all detections...")

        client.emit("subscribe_all_detections")
        print("Subscribed to all detections!")

        isFirstMsg = True
        while not self._stop_event.is_set():
            try:
                [title, *args] = client.receive(timeout=0.5)
            except socketio.exceptions.TimeoutError:
                continue
            except Exception:
                break

            if title != "all_detections":
                continue

            if isFirstMsg:
                print(f"Received first Socket.io message: {title}")
                isFirstMsg = False

            dct = args[0]

            now = time.time()
            rel_time = 0.0 if self._start_ts is None else now - self._start_ts
            for det_type, detections in [
                ("topDown", dct.get("topDownDetections", [])),  # top down detections
                ("robot", dct.get("detections", [])),  # robot detections
            ]:
                for det_obj in detections:
                    if det_type == "robot":
                        self._data.append(
                            {
                                "time": rel_time,
                                "type": det_type,
                                "robotID": det_obj.get("robot"),
                                "x": det_obj.get("x"),
                                "y": det_obj.get("y"),
                                "theta": det_obj.get("azimuth"),
                            }
                        )
                    else:
                        self._data.append(
                            {
                                "time": rel_time,
                                "type": det_type,
                                **det_obj,
                            }
                        )

        try:
            client.disconnect()
        except Exception:
            pass


class CacheEventCollector:
    """Collects cache_event WebSocket telemetry (homography invalidations, buffer reallocations)."""

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._data: list[dict[str, Any]] = []
        self._start_ts: float | None = None

    def start(self) -> None:
        self._start_ts = time.time()
        self._thread = threading.Thread(target=self._worker, name="CacheEventCollector")
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self._data

    def _worker(self) -> None:
        client = socketio.SimpleClient()
        while not self._stop_event.is_set():
            try:
                print("Connecting CacheEventCollector to Socket.io server...")
                client.connect(url=self._server_url)
                print("CacheEventCollector connected!")
                break
            except Exception:
                time.sleep(1)

        if not client.connected:
            print("CacheEventCollector failed to connect!")
            return

        client.emit("subscribe_cache_events")
        print("CacheEventCollector subscribed to cache_events!")

        while not self._stop_event.is_set():
            try:
                [title, *args] = client.receive(timeout=0.5)
            except socketio.exceptions.TimeoutError:
                continue
            except Exception:
                break

            if title != "cache_event":
                continue

            dct = args[0] if args else {}
            now = time.time()
            rel_time = 0.0 if self._start_ts is None else now - self._start_ts
            self._data.append(
                {
                    "time": rel_time,
                    "type": dct.get("type"),
                    **{k: v for k, v in dct.items() if k != "type"},
                }
            )

        try:
            client.disconnect()
        except Exception:
            pass


def run_case(
    playback_dir: str,
    case_name: str,
    cache_cfg: dict[str, Any],
    preallocation_cfg: dict[str, Any],
    base_config: dict[str, Any],
    tracy_csvexport_path: str | None,
) -> None:
    run_dir = OUTPUT_DIR / playback_dir / case_name
    if run_dir.exists():
        print(f"Run directory already exists: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "trace.tracy"
    csv_path = run_dir / "trace.csv"
    summary_path = run_dir / "summary.json"
    positions_path = run_dir / "positions.json"
    cache_events_path = run_dir / "cache_events.json"

    config = dict(base_config)
    config["profiling"] = True
    config["playbackModeLoop"] = False
    config["caching"] = cache_cfg
    config["preallocation"] = preallocation_cfg
    write_config(config)

    tracy_proc = start_tracy_capture(trace_path)
    positions_collector = PositionCollector(server_url=SIO_SERVER_URL)
    cache_event_collector = CacheEventCollector(server_url=SIO_SERVER_URL)
    positions_collector.start()
    cache_event_collector.start()
    start_ts = time.time()

    server_cmd = ["just", "video", playback_dir]
    server_result: subprocess.CompletedProcess[str] | None = None
    tracy_exit: int | None = None
    positions: list[dict[str, Any]] = []
    interrupted = False
    try:
        server_result = subprocess.run(
            server_cmd,
            cwd=SERVER_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            capture_output=False,
        )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        positions = positions_collector.stop()
        cache_events = cache_event_collector.stop()
        positions_path.write_text(json.dumps(positions, indent=2))
        cache_events_path.write_text(json.dumps(cache_events, indent=2))
        tracy_exit = stop_tracy_capture(tracy_proc)
    duration_seconds = time.time() - start_ts

    csvexport_ok = False
    if tracy_csvexport_path:
        csvexport_ok = run_csvexport(tracy_csvexport_path, trace_path, csv_path)

    summary = {
        "playback_dir": playback_dir,
        "case_name": case_name,
        "cache_config": cache_cfg,
        "preallocation_config": preallocation_cfg,
        "server_exit_code": None if server_result is None else server_result.returncode,
        "tracy_exit_code": tracy_exit,
        "tracy_trace_path": str(trace_path),
        "csv_export_path": str(csv_path) if csvexport_ok else None,
        "positions_path": str(positions_path),
        "cache_events_path": str(cache_events_path),
        "duration_seconds": duration_seconds,
        "interrupted": interrupted,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    if interrupted:
        raise KeyboardInterrupt()
    if server_result is not None and server_result.returncode != 0:
        raise RuntimeError(
            f"Server run failed (exit code {server_result.returncode}) for {playback_dir}/{case_name}"
        )


def main() -> int:
    if not PLAYBACK_DIRS:
        print("No playback directories specified in PLAYBACK_DIRS.")
        return 1

    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}")
        return 1

    shutil.copy2(CONFIG_PATH, BACKUP_PATH)
    base_config = read_config()
    base_cache = base_config.get("caching", {})
    tracy_csvexport_path = resolve_tracy_csvexport()

    try:
        baseline_cache = build_cache_config(
            base_cache=base_cache,
            enabled=False,
            enabled_toggle=None,
        )

        cases: list[tuple[str, dict[str, Any]]] = [("baseline-none", baseline_cache)]
        for toggle in CACHE_TOGGLES_ORDER:
            cache_cfg = build_cache_config(
                base_cache=base_cache,
                enabled=True,
                enabled_toggle=toggle,
            )
            cases.append((f"only-{toggle}", cache_cfg))
        cases.append(("all-optimizations-on", build_all_on_cache_config(base_cache)))

        preallocation_variants: list[tuple[str, dict[str, Any]]] = [
            ("prealloc-on", build_preallocation_on_config()),
            ("prealloc-off", build_preallocation_off_config()),
        ]

        run_list: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
        for playback_dir in PLAYBACK_DIRS:
            for case_name, cache_cfg in cases:
                for prealloc_suffix, prealloc_cfg in preallocation_variants:
                    full_case_name = f"{case_name}-{prealloc_suffix}"
                    run_list.append((playback_dir, full_case_name, cache_cfg, prealloc_cfg))

        for playback_dir, case_name, cache_cfg, prealloc_cfg in get_progress_iter(run_list):
            print(f"Running {playback_dir}/{case_name}")
            run_case(
                playback_dir=playback_dir,
                case_name=case_name,
                cache_cfg=cache_cfg,
                preallocation_cfg=prealloc_cfg,
                base_config=base_config,
                tracy_csvexport_path=tracy_csvexport_path,
            )
    except KeyboardInterrupt:
        print("Interrupted by user, stopping benchmark run.")
        return 130
    finally:
        shutil.copy2(BACKUP_PATH, CONFIG_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
