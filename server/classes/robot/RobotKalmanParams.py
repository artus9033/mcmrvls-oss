from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


def _as_bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    return bool(v)


def _as_int(v: Any, default: int) -> int:
    if v is None:
        return default
    return int(v)


def _as_float(v: Any, default: float) -> float:
    if v is None:
        return default
    return float(v)


@dataclass(frozen=True)
class RobotKalmanParams:
    """Per-robot Kalman / unicycle dynamics parameters (merged from global + robot overrides)."""

    enabled: bool
    grace_period_rounds: int
    max_speed_norm_per_sec: float
    max_yaw_rate_deg_per_sec: float
    process_noise_linear_velocity_std: float
    process_noise_angular_rate_std_deg: float
    position_sigma_min: float
    position_sigma_max: float
    azimuth_sigma_min_deg: float
    azimuth_sigma_max_deg: float
    position_sigma_from_area_coefficient: float
    azimuth_sigma_from_baseline_coefficient_deg: float

    @staticmethod
    def hard_defaults() -> RobotKalmanParams:
        return RobotKalmanParams(
            enabled=True,
            grace_period_rounds=8,
            max_speed_norm_per_sec=0.2,
            max_yaw_rate_deg_per_sec=200.0,
            process_noise_linear_velocity_std=0.03,
            process_noise_angular_rate_std_deg=45.0,
            position_sigma_min=0.001,
            position_sigma_max=0.06,
            azimuth_sigma_min_deg=2.0,
            azimuth_sigma_max_deg=30.0,
            position_sigma_from_area_coefficient=0.006,
            azimuth_sigma_from_baseline_coefficient_deg=300.0,
        )

    @staticmethod
    def from_config_dict(cfg: dict[str, Any] | None) -> RobotKalmanParams:
        base = RobotKalmanParams.hard_defaults()
        if not cfg:
            return base
        return base.merged_with(cfg)

    def merged_with(self, cfg: dict[str, Any] | None) -> RobotKalmanParams:
        if not cfg:
            return self
        d = self.__dict__.copy()
        key_map = {
            "enabled": ("enabled", _as_bool),
            "gracePeriodRounds": ("grace_period_rounds", _as_int),
            "maxSpeedNormPerSec": ("max_speed_norm_per_sec", _as_float),
            "maxYawRateDegPerSec": ("max_yaw_rate_deg_per_sec", _as_float),
            "processNoiseLinearVelocityStd": ("process_noise_linear_velocity_std", _as_float),
            "processNoiseAngularRateStdDeg": ("process_noise_angular_rate_std_deg", _as_float),
            "positionSigmaMin": ("position_sigma_min", _as_float),
            "positionSigmaMax": ("position_sigma_max", _as_float),
            "azimuthSigmaMinDeg": ("azimuth_sigma_min_deg", _as_float),
            "azimuthSigmaMaxDeg": ("azimuth_sigma_max_deg", _as_float),
            "positionSigmaFromAreaCoefficient": ("position_sigma_from_area_coefficient", _as_float),
            "azimuthSigmaFromBaselineCoefficientDeg": ("azimuth_sigma_from_baseline_coefficient_deg", _as_float),
        }
        for yaml_key, (field_name, caster) in key_map.items():
            if yaml_key in cfg and cfg[yaml_key] is not None:
                d[field_name] = caster(cfg[yaml_key], d[field_name])
        return RobotKalmanParams(**d)

    def with_replaced(self, **kwargs: Any) -> RobotKalmanParams:
        return replace(self, **kwargs)
