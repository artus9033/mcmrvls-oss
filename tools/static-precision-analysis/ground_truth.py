"""Ground-truth pose definitions for 4cam static evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

DEFAULT_MAP_WIDTH_CM = 300
DEFAULT_MAP_HEIGHT_CM = 300

SCENARIO_PREFIX = "4cam-static-eval-"

# Cardinal headings in the order they appear in the edited rotation videos (N, E, S, W).
ROTATION_AZIMUTHS_DEG = [0.0, 90.0, 180.0, 270.0]


@dataclass(frozen=True)
class ScenarioGroundTruth:
    name: str
    position_norm: tuple[float, float]
    has_rotation: bool
    rotation_azimuths_deg: tuple[float, ...] = field(default_factory=lambda: tuple(ROTATION_AZIMUTHS_DEG))
    corner: str | None = None


def _norm_from_cm(x_cm: float, y_cm: float, map_w: int, map_h: int) -> tuple[float, float]:
    return x_cm / map_w, y_cm / map_h


def _corner_position_norm(corner: str, dist_cm: float, map_w: int, map_h: int) -> tuple[float, float]:
    match corner:
        case "BL":
            return _norm_from_cm(dist_cm, map_h - dist_cm, map_w, map_h)
        case "BR":
            return _norm_from_cm(map_w - dist_cm, map_h - dist_cm, map_w, map_h)
        case "TL":
            return _norm_from_cm(dist_cm, dist_cm, map_w, map_h)
        case "TR":
            return _norm_from_cm(map_w - dist_cm, dist_cm, map_w, map_h)
        case _:
            raise ValueError(f"Unknown corner spec: {corner}")


def _half_position_norm(axis: str, map_w: int, map_h: int) -> tuple[float, float]:
    match axis:
        case "center":
            return _norm_from_cm(map_w / 2, map_h / 2, map_w, map_h)
        case "left":
            return _norm_from_cm(map_w / 4, map_h / 2, map_w, map_h)
        case "right":
            return _norm_from_cm(3 * map_w / 4, map_h / 2, map_w, map_h)
        case "top":
            return _norm_from_cm(map_w / 2, map_h / 4, map_w, map_h)
        case "bottom":
            return _norm_from_cm(map_w / 2, 3 * map_h / 4, map_w, map_h)
        case _:
            raise ValueError(f"Unknown half-axis scenario: {axis}")


def scenario_slug_from_dirname(dirname: str) -> str:
    if dirname.startswith(SCENARIO_PREFIX):
        return dirname[len(SCENARIO_PREFIX) :]
    return dirname


def load_scenario_ground_truth(
    scenario_dir: Path,
    map_width_cm: int = DEFAULT_MAP_WIDTH_CM,
    map_height_cm: int = DEFAULT_MAP_HEIGHT_CM,
) -> ScenarioGroundTruth:
    slug = scenario_slug_from_dirname(scenario_dir.name)

    corner_match = re.fullmatch(r"corner-(TL|TR|BL|BR)", slug)
    if corner_match:
        corner = corner_match.group(1)
        dist_path = scenario_dir / "dist-to-outer-corner.txt"
        if not dist_path.exists():
            raise FileNotFoundError(f"Missing {dist_path}")
        dist_cm = float(dist_path.read_text().strip())
        position = _corner_position_norm(corner, dist_cm, map_width_cm, map_height_cm)
        return ScenarioGroundTruth(
            name=slug,
            position_norm=position,
            has_rotation=False,
            corner=corner,
        )

    if slug in {"center", "left", "right", "top", "bottom"}:
        return ScenarioGroundTruth(
            name=slug,
            position_norm=_half_position_norm(slug, map_width_cm, map_height_cm),
            has_rotation=True,
        )

    raise ValueError(f"Unrecognized static eval scenario directory: {scenario_dir.name}")


def azimuth_gt_at_time(
    time_s: float,
    duration_s: float,
    scenario: ScenarioGroundTruth,
) -> float | None:
    if not scenario.has_rotation or duration_s <= 0:
        return None

    segment_count = len(scenario.rotation_azimuths_deg)
    ratio = min(max(time_s / duration_s, 0.0), 1.0 - 1e-9)
    segment_idx = min(int(ratio * segment_count), segment_count - 1)
    return scenario.rotation_azimuths_deg[segment_idx]


def angular_error_deg(estimated_deg: float, ground_truth_deg: float) -> float:
    delta = (estimated_deg - ground_truth_deg + 180.0) % 360.0 - 180.0
    return abs(delta)


def discover_scenario_dirs(video_root: Path) -> list[Path]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    dirs = sorted(
        p
        for p in video_root.iterdir()
        if p.is_dir() and p.name.startswith(SCENARIO_PREFIX)
    )
    if not dirs:
        raise FileNotFoundError(f"No {SCENARIO_PREFIX}* directories found in {video_root}")
    return dirs
