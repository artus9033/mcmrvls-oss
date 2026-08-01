#!/usr/bin/env python3
"""
Static-placement precision evaluation.

Runs all configured solvers (homography + epipolar) with Kalman on and off,
collects poses, and writes comparison plots.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import socketio
import yaml

from ground_truth import (
    ScenarioGroundTruth,
    angular_error_deg,
    azimuth_gt_at_time,
    discover_scenario_dirs,
    load_scenario_ground_truth,
)
from plotting import (
    KALMAN_OFF_LABEL,
    KALMAN_ON_LABEL,
    compute_metrics,
    plot_scenario_solver_comparison,
    plot_summary_comparison,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = WORKSPACE_ROOT / "server"
VIDEO_ROOT = SERVER_DIR / "res" / "video"
CONFIG_PATH = SERVER_DIR / "config.yaml"
BACKUP_PATH = SERVER_DIR / "config.yaml.bkp"
OUTPUT_DIR = Path(__file__).resolve().parent / "out"
SIO_SERVER_URL = os.environ.get("SIO_SERVER_URL", "ws://localhost:6001")

SOLVER_HOMOGRAPHY = "homography"
SOLVER_EPIPOLAR = "epipolar"
ALL_SOLVERS = (SOLVER_HOMOGRAPHY, SOLVER_EPIPOLAR)


def kalman_variant_label(kalman_enabled: bool) -> str:
    return KALMAN_ON_LABEL if kalman_enabled else KALMAN_OFF_LABEL


def run_output_dir(scenario_name: str, solver: str, kalman_enabled: bool) -> Path:
    return OUTPUT_DIR / scenario_name / solver / kalman_variant_label(kalman_enabled)


def server_supports_epipolar() -> bool:
    return (SERVER_DIR / "algorithms" / "epipolar.py").exists()


def read_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def write_config(config: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, explicit_start=True)


def build_eval_config(base_config: dict[str, Any], solver: str, kalman_enabled: bool) -> dict[str, Any]:
    config = dict(base_config)
    config["profiling"] = False
    config["playbackModeLoop"] = False
    config["caching"] = {
        "enabled": False,
        "stitchingHomographyCache": False,
        "topDownHomographyCache": False,
        "mapSegmentationThrottle": True,
        "mapSegmentationIntervalSeconds": 3,
    }

    kalman_cfg = dict(config.get("robotKalmanFilter") or {})
    kalman_cfg["enabled"] = kalman_enabled
    config["robotKalmanFilter"] = kalman_cfg

    if server_supports_epipolar():
        config["useEpipolarGeometry"] = solver == SOLVER_EPIPOLAR

    return config


def normalize_robot_id(robot_ref: Any) -> int | None:
    if robot_ref is None:
        return None
    if isinstance(robot_ref, dict):
        rid = robot_ref.get("id")
        return int(rid) if rid is not None else None
    return int(robot_ref)


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
            self._thread.join(timeout=10)
        return self._data

    def _worker(self) -> None:
        client = socketio.SimpleClient()
        while not self._stop_event.is_set():
            try:
                client.connect(url=self._server_url)
                break
            except Exception:
                time.sleep(1)

        if not client.connected:
            return

        client.emit("subscribe_all_detections")
        while not self._stop_event.is_set():
            try:
                title, *args = client.receive(timeout=0.5)
            except socketio.exceptions.TimeoutError:
                continue
            except Exception:
                break

            if title != "all_detections":
                continue

            dct = args[0]
            now = time.time()
            rel_time = 0.0 if self._start_ts is None else now - self._start_ts
            for det_obj in dct.get("detections", []):
                self._data.append(
                    {
                        "time": rel_time,
                        "robotID": normalize_robot_id(det_obj.get("robot")),
                        "x": det_obj.get("x"),
                        "y": det_obj.get("y"),
                        "theta": det_obj.get("azimuth"),
                    }
                )

        try:
            client.disconnect()
        except Exception:
            pass


def infer_primary_robot_id(positions: list[dict[str, Any]], robot_id: int | None) -> int | None:
    if robot_id is not None:
        return robot_id
    if not positions:
        return None

    counts: dict[int, int] = {}
    for row in positions:
        rid = normalize_robot_id(row.get("robotID"))
        if rid is None:
            continue
        counts[rid] = counts.get(rid, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def positions_to_dataframe(
    positions: list[dict[str, Any]],
    scenario_gt: ScenarioGroundTruth,
    robot_id: int | None,
    rolling_window_s: float,
) -> pd.DataFrame:
    if not positions:
        raise ValueError("No robot detections were collected")

    primary_robot = infer_primary_robot_id(positions, robot_id)
    rows = [row for row in positions if normalize_robot_id(row.get("robotID")) == primary_robot]
    if not rows:
        raise ValueError(f"No detections found for robot {primary_robot}")

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    gt_x, gt_y = scenario_gt.position_norm
    duration_s = float(df["time"].max()) if len(df) else 0.0

    df["x_gt"] = gt_x
    df["y_gt"] = gt_y
    df["theta_gt"] = [
        azimuth_gt_at_time(float(t), duration_s, scenario_gt) for t in df["time"]
    ]
    df["pos_error"] = np.hypot(df["x"] - df["x_gt"], df["y"] - df["y_gt"])
    df["theta_error"] = [
        angular_error_deg(float(est), float(gt)) if gt is not None and pd.notna(est) else np.nan
        for est, gt in zip(df["theta"], df["theta_gt"])
    ]

    if len(df) > 1 and rolling_window_s > 0:
        dt = df["time"].diff().median()
        if pd.notna(dt) and dt > 0:
            window = max(int(round(rolling_window_s / dt)), 1)
            df["pos_rmse_roll"] = np.sqrt(df["pos_error"].pow(2).rolling(window, min_periods=1).mean())
            if scenario_gt.has_rotation:
                df["theta_rmse_roll"] = np.sqrt(
                    df["theta_error"].pow(2).rolling(window, min_periods=1).mean()
                )

    df.attrs["primary_robot_id"] = primary_robot
    return df


def run_variant(
    scenario_dir: Path,
    solver: str,
    kalman_enabled: bool,
    base_config: dict[str, Any],
    robot_id: int | None,
    force: bool,
) -> Path:
    scenario_name = scenario_dir.name
    run_dir = run_output_dir(scenario_name, solver, kalman_enabled)
    positions_path = run_dir / "positions.json"
    summary_path = run_dir / "summary.json"
    variant = kalman_variant_label(kalman_enabled)

    if positions_path.exists() and not force:
        print(f"Skipping existing run: {scenario_name}/{solver}/{variant}")
        return positions_path

    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_eval_config(base_config, solver=solver, kalman_enabled=kalman_enabled)
    write_config(config)

    collector = PositionCollector(server_url=SIO_SERVER_URL)
    collector.start()
    start_ts = time.time()

    try:
        result = subprocess.run(
            ["just", "video", scenario_name],
            cwd=SERVER_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        positions = collector.stop()

    positions_path.write_text(json.dumps(positions, indent=2))
    summary_path.write_text(
        json.dumps(
            {
                "scenario": scenario_name,
                "solver": solver,
                "kalman_enabled": kalman_enabled,
                "variant": variant,
                "server_exit_code": result.returncode,
                "duration_seconds": time.time() - start_ts,
                "detection_count": len(positions),
                "primary_robot_id": infer_primary_robot_id(positions, robot_id),
            },
            indent=2,
        )
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Server failed for {scenario_name}/{solver}/{variant} (exit {result.returncode})"
        )

    return positions_path


def load_variant_dataframe(
    scenario_dir: Path,
    solver: str,
    kalman_enabled: bool,
    robot_id: int | None,
    rolling_window_s: float,
) -> pd.DataFrame:
    scenario_gt = load_scenario_ground_truth(scenario_dir)
    positions_path = run_output_dir(scenario_dir.name, solver, kalman_enabled) / "positions.json"
    if not positions_path.exists():
        raise FileNotFoundError(f"Missing positions file: {positions_path}")

    positions = json.loads(positions_path.read_text())
    return positions_to_dataframe(positions, scenario_gt, robot_id, rolling_window_s)


def analyze_scenario_solver(
    scenario_dir: Path,
    solver: str,
    kalman_modes: list[bool],
    robot_id: int | None,
    rolling_window_s: float,
) -> list[dict[str, Any]]:
    scenario_gt = load_scenario_ground_truth(scenario_dir)
    dfs_by_variant: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict[str, Any]] = []

    for kalman_enabled in kalman_modes:
        variant = kalman_variant_label(kalman_enabled)
        df = load_variant_dataframe(scenario_dir, solver, kalman_enabled, robot_id, rolling_window_s)
        dfs_by_variant[variant] = df

        metrics = compute_metrics(df, scenario_gt)
        metrics.update(
            {
                "scenario": scenario_gt.name,
                "scenario_dir": scenario_dir.name,
                "solver": solver,
                "kalman_enabled": kalman_enabled,
                "variant": variant,
                "primary_robot_id": df.attrs.get("primary_robot_id"),
                "duration_s": float(df["time"].max()) if len(df) else 0.0,
            }
        )
        metrics_path = run_output_dir(scenario_dir.name, solver, kalman_enabled) / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        metrics_rows.append(metrics)

    if len(dfs_by_variant) > 1:
        plot_scenario_solver_comparison(
            dfs_by_variant=dfs_by_variant,
            scenario_gt=scenario_gt,
            solver=solver,
            out_dir=OUTPUT_DIR / "plots" / scenario_dir.name,
            rolling_window_s=rolling_window_s,
        )

    return metrics_rows


def available_solvers(requested: list[str]) -> list[str]:
    solvers: list[str] = []
    for solver in requested:
        if solver == SOLVER_EPIPOLAR and not server_supports_epipolar():
            print(
                "Warning: epipolar solver is not available on this server checkout "
                f"(missing {SERVER_DIR / 'algorithms' / 'epipolar.py'}). "
                "Use the feat/epipolar branch to enable it.",
                file=sys.stderr,
            )
            continue
        solvers.append(solver)
    return solvers


def default_kalman_modes(args: argparse.Namespace) -> list[bool]:
    if args.kalman_only:
        return [True]
    if args.raw_only:
        return [False]
    return [False, True]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run static eval for homography + epipolar, kalman on/off, and plot comparisons."
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        help="Scenario directory names (e.g. 4cam-static-eval-center). Defaults to all.",
    )
    parser.add_argument(
        "--solvers",
        nargs="*",
        choices=ALL_SOLVERS,
        default=list(ALL_SOLVERS),
        help="Solvers to evaluate (default: both).",
    )
    parser.add_argument("--robot-id", type=int, default=None, help="Robot ID to analyze.")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only run/analyze kalman-off (skip kalman-on).",
    )
    parser.add_argument(
        "--kalman-only",
        action="store_true",
        help="Only run/analyze kalman-on (skip kalman-off).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist.")
    parser.add_argument("--run-only", action="store_true", help="Collect data without plotting.")
    parser.add_argument("--plot-only", action="store_true", help="Plot from existing outputs.")
    parser.add_argument(
        "--rolling-window-s",
        type=float,
        default=1.0,
        help="Rolling window (seconds) for RMSE plots.",
    )
    return parser.parse_args()


def resolve_scenario_dirs(names: list[str] | None) -> list[Path]:
    all_dirs = discover_scenario_dirs(VIDEO_ROOT)
    if not names:
        return all_dirs

    by_name = {p.name: p for p in all_dirs}
    selected: list[Path] = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"Unknown scenario: {name}")
        selected.append(by_name[name])
    return selected


def main() -> int:
    args = parse_args()
    if args.raw_only and args.kalman_only:
        print("Cannot use --raw-only and --kalman-only together.", file=sys.stderr)
        return 1

    solvers = available_solvers(args.solvers)
    if not solvers:
        print("No solvers available to run.", file=sys.stderr)
        return 1

    kalman_modes = default_kalman_modes(args)
    if not CONFIG_PATH.exists():
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        return 1

    scenario_dirs = resolve_scenario_dirs(args.scenarios)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
        base_config = read_config()
        try:
            for scenario_dir in scenario_dirs:
                for solver in solvers:
                    for kalman_enabled in kalman_modes:
                        variant = kalman_variant_label(kalman_enabled)
                        print(f"Running {scenario_dir.name} / {solver} / {variant}")
                        run_variant(
                            scenario_dir=scenario_dir,
                            solver=solver,
                            kalman_enabled=kalman_enabled,
                            base_config=base_config,
                            robot_id=args.robot_id,
                            force=args.force,
                        )
        finally:
            shutil.copy2(BACKUP_PATH, CONFIG_PATH)

    if args.run_only:
        return 0

    metrics_rows: list[dict[str, Any]] = []
    for scenario_dir in scenario_dirs:
        for solver in solvers:
            print(f"Analyzing {scenario_dir.name} / {solver}")
            metrics_rows.extend(
                analyze_scenario_solver(
                    scenario_dir=scenario_dir,
                    solver=solver,
                    kalman_modes=kalman_modes,
                    robot_id=args.robot_id,
                    rolling_window_s=args.rolling_window_s,
                )
            )

    summary_df = pd.DataFrame(metrics_rows)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_summary_comparison(summary_df, OUTPUT_DIR / "plots")
    print(f"Wrote summary to {summary_path}")
    print(f"Plots saved under {OUTPUT_DIR / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
