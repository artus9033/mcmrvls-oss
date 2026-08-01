"""Plot estimated poses and RMSE vs ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ground_truth import ScenarioGroundTruth

KALMAN_OFF_LABEL = "kalman-off"
KALMAN_ON_LABEL = "kalman-on"

VARIANT_STYLES: dict[str, dict[str, str | float]] = {
    KALMAN_OFF_LABEL: {"color": "#1f77b4", "label": "raw (kalman off)"},
    KALMAN_ON_LABEL: {"color": "#ff7f0e", "label": "kalman on"},
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.grid": True,
            "savefig.bbox": "tight",
        }
    )


def compute_metrics(df: pd.DataFrame, scenario_gt: ScenarioGroundTruth) -> dict[str, float]:
    summary: dict[str, float] = {
        "position_rmse": float(np.sqrt(np.mean(np.square(df["pos_error"])))),
        "position_mae": float(np.mean(df["pos_error"])),
        "samples": int(len(df)),
    }
    if scenario_gt.has_rotation and df["theta_error"].notna().any():
        summary["azimuth_rmse_deg"] = float(np.sqrt(np.mean(np.square(df["theta_error"]))))
        summary["azimuth_mae_deg"] = float(np.mean(df["theta_error"]))
    return summary


def plot_scenario_solver_comparison(
    dfs_by_variant: dict[str, pd.DataFrame],
    scenario_gt: ScenarioGroundTruth,
    solver: str,
    out_dir: Path,
    rolling_window_s: float = 1.0,
) -> None:
    """Overlay kalman-on vs kalman-off (and GT) for one scenario and solver."""
    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_x, gt_y = scenario_gt.position_norm
    title_base = f"{scenario_gt.name} — {solver}"

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for variant, df in dfs_by_variant.items():
        style = VARIANT_STYLES.get(variant, {"color": "#333333", "label": variant})
        axes[0].plot(df["time"], df["x"], color=style["color"], label=style["label"], alpha=0.9)
        axes[1].plot(df["time"], df["y"], color=style["color"], label=style["label"], alpha=0.9)
        if scenario_gt.has_rotation and df["theta_gt"].notna().any():
            axes[2].plot(df["time"], df["theta"], color=style["color"], label=style["label"], alpha=0.9)

    axes[0].axhline(gt_x, color="#d62728", linestyle="--", label="ground truth")
    axes[1].axhline(gt_y, color="#d62728", linestyle="--", label="ground truth")
    if scenario_gt.has_rotation:
        first_df = next(iter(dfs_by_variant.values()))
        if first_df["theta_gt"].notna().any():
            axes[2].plot(
                first_df["time"],
                first_df["theta_gt"],
                color="#d62728",
                linestyle="--",
                label="ground truth",
            )

    axes[0].set_ylabel("x (norm)")
    axes[1].set_ylabel("y (norm)")
    axes[2].set_ylabel("azimuth (deg)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title(f"{title_base} — position & heading")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scenario_gt.name}_{solver}_pose_compare.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for variant, df in dfs_by_variant.items():
        style = VARIANT_STYLES.get(variant, {"color": "#333333", "label": variant})
        axes[0].plot(
            df["time"],
            df["pos_error"],
            color=style["color"],
            label=style["label"],
            alpha=0.75,
        )
        if "pos_rmse_roll" in df.columns:
            axes[0].plot(
                df["time"],
                df["pos_rmse_roll"],
                color=style["color"],
                linestyle="--",
                linewidth=2,
                alpha=0.9,
            )
        if scenario_gt.has_rotation and df["theta_error"].notna().any():
            axes[1].plot(
                df["time"],
                df["theta_error"],
                color=style["color"],
                label=style["label"],
                alpha=0.75,
            )
            if "theta_rmse_roll" in df.columns:
                axes[1].plot(
                    df["time"],
                    df["theta_rmse_roll"],
                    color=style["color"],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9,
                )

    axes[0].set_ylabel("position error (norm)")
    axes[1].set_ylabel("azimuth error (deg)")
    axes[1].set_xlabel("time (s)")
    roll_label = f"dashed = {rolling_window_s:g}s rolling RMSE"
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=2))
    labels.append(roll_label)
    axes[0].set_title(f"{title_base} — errors")
    axes[0].legend(handles, labels, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"{scenario_gt.name}_{solver}_rmse_compare.png")
    plt.close(fig)


def plot_summary_comparison(summary_df: pd.DataFrame, out_dir: Path) -> None:
    if summary_df.empty:
        return

    _configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summary_df.copy()
    summary_df["variant"] = summary_df["kalman_enabled"].map(
        {True: KALMAN_ON_LABEL, False: KALMAN_OFF_LABEL}
    )
    summary_df["series"] = summary_df["solver"] + " / " + summary_df["variant"]

    scenarios = sorted(summary_df["scenario"].unique())
    series_labels = sorted(summary_df["series"].unique())
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(series_labels), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, series in enumerate(series_labels):
        subset = summary_df[summary_df["series"] == series].set_index("scenario").reindex(scenarios)
        ax.bar(x + i * width, subset["position_rmse"], width=width, label=series)
    ax.set_xticks(x + width * (len(series_labels) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.set_ylabel("position RMSE (norm)")
    ax.set_title("Position RMSE by scenario, solver, and Kalman mode")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_position_rmse.png")
    plt.close(fig)

    if "azimuth_rmse_deg" not in summary_df.columns:
        return

    rotation_df = summary_df[summary_df["azimuth_rmse_deg"].notna()]
    if rotation_df.empty:
        return

    rotation_scenarios = sorted(rotation_df["scenario"].unique())
    rotation_series = sorted(rotation_df["series"].unique())
    x = np.arange(len(rotation_scenarios))
    width = 0.8 / max(len(rotation_series), 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, series in enumerate(rotation_series):
        subset = rotation_df[rotation_df["series"] == series].set_index("scenario").reindex(rotation_scenarios)
        ax.bar(x + i * width, subset["azimuth_rmse_deg"], width=width, label=series)
    ax.set_xticks(x + width * (len(rotation_series) - 1) / 2)
    ax.set_xticklabels(rotation_scenarios, rotation=35, ha="right")
    ax.set_ylabel("azimuth RMSE (deg)")
    ax.set_title("Azimuth RMSE by scenario, solver, and Kalman mode")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_azimuth_rmse.png")
    plt.close(fig)
