"""Deterministic quality gates for strict low-to-high IMU pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QualityDecision:
    """Serializable strict-training eligibility decision."""

    training_eligible: bool
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, float | int | str | None]


def evaluate_pair_quality(
    raw_metrics: dict[str, Any],
    transformed_metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> QualityDecision:
    """Apply blocking and warning thresholds with explicit boundary behavior."""

    blocking: list[str] = []
    warnings: list[str] = []

    if not bool(_required(raw_metrics, "calibration_available")):
        blocking.append("missing_calibration")
    if not bool(_required(raw_metrics, "translation_available")):
        blocking.append("missing_translation")

    quaternion_error = _finite_float(raw_metrics, "quaternion_norm_error")
    quaternion_max = _threshold(thresholds, "quaternion_norm_error_max", 1e-3)
    if quaternion_error > quaternion_max:
        blocking.append("invalid_quaternion")

    orthogonality = _finite_float(raw_metrics, "rotation_orthogonality_fro")
    orthogonality_max = _threshold(thresholds, "rotation_orthogonality_fro_max", 1e-5)
    if orthogonality >= orthogonality_max:
        blocking.append("invalid_rotation_orthogonality")

    determinant = _finite_float(raw_metrics, "rotation_determinant")
    determinant_min = _threshold(thresholds, "rotation_determinant_min", 0.999)
    if determinant <= determinant_min:
        blocking.append("invalid_rotation_determinant")

    if int(_required(raw_metrics, "decreasing_timestamps")) > 0:
        blocking.append("decreasing_timestamps")
    if not bool(_required(raw_metrics, "finite_six_axis")):
        blocking.append("nonfinite_six_axis")

    overlap = _finite_float(raw_metrics, "overlap_fraction")
    overlap_block = _threshold(thresholds, "overlap_block_below", 0.90)
    overlap_warn = _threshold(thresholds, "overlap_warn_below", 0.95)
    if overlap < overlap_block:
        blocking.append("insufficient_overlap")
    elif overlap < overlap_warn:
        warnings.append("overlap_below_preferred")

    target_period_s = _finite_float(raw_metrics, "target_period_s")
    if target_period_s <= 0.0:
        raise ValueError("target_period_s must be positive")
    match_dt_ms_p99 = _finite_float(raw_metrics, "match_dt_ms_p99")
    if match_dt_ms_p99 > target_period_s * 1e3:
        blocking.append("match_distance_exceeds_target_period")

    gap_periods = _finite_float(raw_metrics, "gap_periods_max")
    max_gap_periods = _threshold(thresholds, "max_interpolation_gap_periods", 3.0)
    if gap_periods > max_gap_periods:
        blocking.append("interpolation_gap_exceeded")

    correlation = _finite_float(transformed_metrics, "active_gyro_correlation")
    correlation_block = _threshold(
        thresholds,
        "active_gyro_correlation_block_below",
        0.70,
    )
    correlation_warn = _threshold(
        thresholds,
        "active_gyro_correlation_warn_below",
        0.90,
    )
    if correlation < correlation_block:
        blocking.append("low_active_gyro_correlation")
    elif correlation < correlation_warn:
        warnings.append("active_gyro_correlation_below_preferred")

    residual_lag_s = abs(_finite_float(transformed_metrics, "residual_lag_s"))
    maximum_lag_periods = _threshold(
        thresholds,
        "residual_lag_max_target_periods",
        1.0,
    )
    if residual_lag_s > maximum_lag_periods * target_period_s:
        blocking.append("residual_lag_exceeds_target_period")

    output_rows = int(_required(raw_metrics, "output_rows"))
    minimum_rows = int(thresholds.get("minimum_output_rows", 1))
    if output_rows < minimum_rows:
        blocking.append("insufficient_rows")

    raw_axis_correlation = _finite_float(raw_metrics, "raw_axis_correlation")
    transformed_axis_correlation = _finite_float(
        transformed_metrics,
        "transformed_axis_correlation",
    )
    if transformed_axis_correlation <= raw_axis_correlation:
        blocking.append("transform_correlation_not_improved")

    gravity_disagreement = _finite_float(
        transformed_metrics,
        "gravity_disagreement_mps2",
    )
    gravity_warn = _threshold(
        thresholds,
        "static_gravity_disagreement_warn_mps2",
        0.3,
    )
    if gravity_disagreement > gravity_warn:
        warnings.append("gravity_disagreement_above_preferred")

    outlier_fraction = _finite_float(
        transformed_metrics,
        "isolated_outlier_fraction",
    )
    outlier_warn = _threshold(thresholds, "isolated_outlier_fraction_warn", 0.001)
    if outlier_fraction > outlier_warn:
        warnings.append("isolated_outlier_fraction_above_preferred")
    if bool(raw_metrics.get("sensor_name_inconsistent", False)):
        warnings.append("upstream_sensor_name_inconsistency")

    metrics = {
        **{f"raw_{key}": _json_scalar(value) for key, value in raw_metrics.items()},
        **{
            f"transformed_{key}": _json_scalar(value)
            for key, value in transformed_metrics.items()
        },
    }
    return QualityDecision(
        training_eligible=not blocking,
        blocking_reasons=tuple(blocking),
        warnings=tuple(warnings),
        metrics=metrics,
    )


def _required(values: dict[str, Any], key: str) -> Any:
    if key not in values:
        raise ValueError(f"Missing required quality metric: {key}")
    return values[key]


def _finite_float(values: dict[str, Any], key: str) -> float:
    value = float(_required(values, key))
    if not np.isfinite(value):
        raise ValueError(f"Quality metric {key} must be finite")
    return value


def _threshold(values: dict[str, Any], key: str, default: float) -> float:
    value = float(values.get(key, default))
    if not np.isfinite(value):
        raise ValueError(f"Quality threshold {key} must be finite")
    return value


def _json_scalar(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)
