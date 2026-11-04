from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from virtual_imu.preprocessing.imu_quality import evaluate_pair_quality


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"calibration_available": False}, "missing_calibration"),
        ({"translation_available": False}, "missing_translation"),
        ({"quaternion_norm_error": 0.00101}, "invalid_quaternion"),
        ({"rotation_orthogonality_fro": 1e-5}, "invalid_rotation_orthogonality"),
        ({"rotation_determinant": 0.999}, "invalid_rotation_determinant"),
        ({"decreasing_timestamps": 1}, "decreasing_timestamps"),
        ({"finite_six_axis": False}, "nonfinite_six_axis"),
        ({"overlap_fraction": 0.8999}, "insufficient_overlap"),
        ({"match_dt_ms_p99": 10.01}, "match_distance_exceeds_target_period"),
        ({"gap_periods_max": 3.01}, "interpolation_gap_exceeded"),
        ({"output_rows": 99}, "insufficient_rows"),
    ],
)
def test_raw_metric_blocking_thresholds(updates: dict[str, object], reason: str) -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw.update(updates)

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert decision.training_eligible is False
    assert reason in decision.blocking_reasons


def test_transformed_correlation_below_point_seven_blocks() -> None:
    raw, transformed, thresholds = _passing_inputs()
    transformed["active_gyro_correlation"] = 0.6999
    transformed["transformed_axis_correlation"] = 0.6999

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert "low_active_gyro_correlation" in decision.blocking_reasons


def test_residual_lag_above_one_target_period_blocks() -> None:
    raw, transformed, thresholds = _passing_inputs()
    transformed["residual_lag_s"] = 0.01001

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert "residual_lag_exceeds_target_period" in decision.blocking_reasons


def test_transform_must_strictly_improve_physically_comparable_axes() -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw["raw_axis_correlation"] = 0.8
    transformed["transformed_axis_correlation"] = 0.8

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert "transform_correlation_not_improved" in decision.blocking_reasons


def test_exact_nonblocking_boundaries_are_deterministic() -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw.update(
        {
            "quaternion_norm_error": 0.001,
            "overlap_fraction": 0.90,
            "match_dt_ms_p99": 10.0,
            "gap_periods_max": 3.0,
            "output_rows": 100,
        }
    )
    transformed.update(
        {
            "active_gyro_correlation": 0.70,
            "transformed_axis_correlation": 0.70,
            "residual_lag_s": 0.01,
        }
    )

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert decision.training_eligible is True
    assert decision.blocking_reasons == ()
    assert "overlap_below_preferred" in decision.warnings
    assert "active_gyro_correlation_below_preferred" in decision.warnings


def test_warning_only_metrics_do_not_block() -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw.update({"overlap_fraction": 0.94, "sensor_name_inconsistent": True})
    transformed.update(
        {
            "active_gyro_correlation": 0.85,
            "transformed_axis_correlation": 0.85,
            "gravity_disagreement_mps2": 0.31,
            "isolated_outlier_fraction": 0.0011,
        }
    )

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert decision.training_eligible is True
    assert set(decision.warnings) == {
        "overlap_below_preferred",
        "active_gyro_correlation_below_preferred",
        "gravity_disagreement_above_preferred",
        "isolated_outlier_fraction_above_preferred",
        "upstream_sensor_name_inconsistency",
    }


def test_exact_warning_boundaries_do_not_warn() -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw["overlap_fraction"] = 0.95
    transformed.update(
        {
            "active_gyro_correlation": 0.90,
            "transformed_axis_correlation": 0.90,
            "gravity_disagreement_mps2": 0.3,
            "isolated_outlier_fraction": 0.001,
        }
    )

    decision = evaluate_pair_quality(raw, transformed, thresholds)

    assert decision.training_eligible is True
    assert decision.warnings == ()


def test_quality_decision_is_json_serializable_and_reason_order_is_stable() -> None:
    raw, transformed, thresholds = _passing_inputs()
    raw.update({"calibration_available": False, "finite_six_axis": False})

    decision = evaluate_pair_quality(raw, transformed, thresholds)
    encoded = json.dumps(asdict(decision), sort_keys=True)

    assert "missing_calibration" in encoded
    assert decision.blocking_reasons[:2] == ("missing_calibration", "nonfinite_six_axis")


def _passing_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    raw = {
        "calibration_available": True,
        "translation_available": True,
        "quaternion_norm_error": 0.0,
        "rotation_orthogonality_fro": 1e-10,
        "rotation_determinant": 1.0,
        "decreasing_timestamps": 0,
        "finite_six_axis": True,
        "overlap_fraction": 0.99,
        "match_dt_ms_p99": 2.0,
        "target_period_s": 0.01,
        "gap_periods_max": 1.0,
        "output_rows": 1000,
        "raw_axis_correlation": 0.5,
        "sensor_name_inconsistent": False,
    }
    transformed = {
        "active_gyro_correlation": 0.95,
        "transformed_axis_correlation": 0.95,
        "residual_lag_s": 0.0,
        "gravity_disagreement_mps2": 0.1,
        "isolated_outlier_fraction": 0.0,
    }
    thresholds = {
        "quaternion_norm_error_max": 0.001,
        "rotation_orthogonality_fro_max": 1e-5,
        "rotation_determinant_min": 0.999,
        "overlap_block_below": 0.90,
        "overlap_warn_below": 0.95,
        "active_gyro_correlation_block_below": 0.70,
        "active_gyro_correlation_warn_below": 0.90,
        "max_interpolation_gap_periods": 3.0,
        "residual_lag_max_target_periods": 1.0,
        "minimum_output_rows": 100,
        "static_gravity_disagreement_warn_mps2": 0.3,
        "isolated_outlier_fraction_warn": 0.001,
    }
    return raw, transformed, thresholds
