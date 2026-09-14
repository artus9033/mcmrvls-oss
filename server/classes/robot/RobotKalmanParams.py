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
        """
        Calibrated against the eight dyn-4cam-* trajectories on the orthorectified solver
        by calibration: R measured from
        raw-detection residuals vs ground truth, Q from the ground-truth motion itself.
        Keep in sync with the robotKalmanFilter block of config.example.yaml.
        """
        return RobotKalmanParams(
            enabled=True,
            grace_period_rounds=16,
            max_speed_norm_per_sec=0.4256,
            max_yaw_rate_deg_per_sec=289.0,
            process_noise_linear_velocity_std=0.00217,
            process_noise_angular_rate_std_deg=2.571,
            position_sigma_min=0.05608,
            position_sigma_max=0.4581,
            # Pinned together: azimuth noise does not follow the 1/baseline law on this
            # data, so this axis is a measured constant and the coefficient is inert.
            azimuth_sigma_min_deg=6.864,
            azimuth_sigma_max_deg=6.864,
            position_sigma_from_area_coefficient=0.003211,
            azimuth_sigma_from_baseline_coefficient_deg=0.08568,
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
