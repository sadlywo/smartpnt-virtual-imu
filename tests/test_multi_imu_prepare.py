from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from virtual_imu.io.multi_imu_calibration import RigidTransform
from virtual_imu.io.multi_imu_prepare import (
    STRICT_BASE_COLUMNS,
    STRICT_METADATA_COLUMNS,
    PreparationContext,
    _deterministic_sample_indices,
    prepare_pair,
    provenance_fingerprint,
    write_preparation_report,
)
from virtual_imu.preprocessing.rigid_imu import quaternion_wxyz_to_matrix
from virtual_imu.training.dataset import ensure_training_columns


def test_prepare_pair_writes_exact_strict_schema_and_provenance(tmp_path: Path) -> None:
    pair = _passing_pair()
    context = _context(tmp_path)

    record = prepare_pair(pair, context)

    assert record["training_eligible"] is True
    assert record["status"] == "prepared"
    strict_path = Path(record["strict_path"])
    diagnostic_path = Path(record["diagnostic_path"])
    assert strict_path.is_file()
    assert diagnostic_path.is_file()
    strict = pd.read_parquet(strict_path)
    assert strict.columns.tolist() == [*STRICT_BASE_COLUMNS, *STRICT_METADATA_COLUMNS]
    assert len(strict.columns) == 29
    assert strict["training_eligible"].all()
    np.testing.assert_allclose(
        strict[["gx_low", "gy_low", "gz_low"]],
        strict[["gx_high", "gy_high", "gz_high"]],
        atol=1e-8,
    )
    assert len(record["fingerprint"]) == 64
    assert Path(record["metrics_path"]).is_file()
    assert Path(record["record_path"]).is_file()


def test_missing_calibration_writes_diagnostics_but_no_strict_file(tmp_path: Path) -> None:
    pair = _passing_pair()
    pair.update(
        {
            "pair_id": "missing_calibration",
            "calibration_available": False,
            "translation_available": False,
            "low_transform": None,
            "high_transform": None,
        }
    )

    record = prepare_pair(pair, _context(tmp_path))

    assert record["training_eligible"] is False
    assert "missing_calibration" in record["blocking_reasons"]
    assert record["strict_path"] is None
    assert Path(record["diagnostic_path"]).is_file()


def test_low_correlation_pair_is_blocked_without_strict_file(tmp_path: Path) -> None:
    pair = _passing_pair()
    rng = np.random.default_rng(42)
    high = pair["high_frame"].copy()
    high.loc[:, ["gx", "gy", "gz", "ax", "ay", "az"]] = rng.normal(
        size=(len(high), 6)
    )
    pair.update(
        {
            "pair_id": "low_correlation",
            "high_frame": high,
            "low_transform": RigidTransform(np.eye(3), np.zeros(3), "low_identity"),
        }
    )

    record = prepare_pair(pair, _context(tmp_path))

    assert record["training_eligible"] is False
    assert "low_active_gyro_correlation" in record["blocking_reasons"]
    assert record["strict_path"] is None
    assert Path(record["diagnostic_path"]).is_file()


def test_diagnostic_sampling_is_exact_sorted_and_deterministic() -> None:
    first = _deterministic_sample_indices(500_000, 200_000, 20260711, "pair/sequence")
    second = _deterministic_sample_indices(500_000, 200_000, 20260711, "pair/sequence")

    assert len(first) == 200_000
    assert np.all(np.diff(first) > 0)
    np.testing.assert_array_equal(first, second)


def test_provenance_fingerprint_changes_with_source_content(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    source = tmp_path / "source.txt"
    calibration = tmp_path / "calibration.yaml"
    config.write_text("seed: 1\n", encoding="utf-8")
    source.write_text("first", encoding="utf-8")
    calibration.write_text("rotation: identity\n", encoding="utf-8")
    first = provenance_fingerprint(config, [source], [calibration])
    source.write_text("second", encoding="utf-8")

    second = provenance_fingerprint(config, [source], [calibration])

    assert len(first) == 64
    assert first != second


def test_training_loader_rejects_explicit_noneligible_rows() -> None:
    frame = _minimal_training_frame()
    frame["training_eligible"] = [True, False]

    with pytest.raises(ValueError, match="not eligible"):
        ensure_training_columns(frame)

    frame["training_eligible"] = [True, np.nan]
    with pytest.raises(ValueError, match="not eligible"):
        ensure_training_columns(frame)


def test_training_loader_accepts_eligible_and_legacy_rows() -> None:
    frame = _minimal_training_frame()
    eligible = frame.assign(training_eligible=True)

    assert ensure_training_columns(eligible)["training_eligible"].all()
    assert "training_eligible" not in ensure_training_columns(frame)


def test_preparation_report_contains_pair_quality_columns(tmp_path: Path) -> None:
    record = prepare_pair(_passing_pair(), _context(tmp_path))

    path = write_preparation_report([record], tmp_path / "strict" / "report.md")
    text = path.read_text(encoding="utf-8")

    assert "rotated_low_to_high" in text
    assert "P99 match (ms)" in text
    assert "Transformed corr" in text


def _passing_pair() -> dict[str, object]:
    rate_hz = 100.0
    timestamps = np.arange(400, dtype=float) / rate_hz
    body_gyro = np.column_stack(
        [
            np.sin(2 * np.pi * 0.7 * timestamps),
            0.7 * np.sin(2 * np.pi * 1.1 * timestamps + 0.3),
            0.4 * np.cos(2 * np.pi * 1.7 * timestamps),
        ]
    )
    body_accel = np.column_stack(
        [
            0.3 * np.sin(2 * np.pi * 0.5 * timestamps),
            0.2 * np.cos(2 * np.pi * 0.8 * timestamps),
            9.8 + 0.1 * np.sin(2 * np.pi * 1.3 * timestamps),
        ]
    )
    rotation = quaternion_wxyz_to_matrix([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    low_sensor_gyro = body_gyro @ rotation
    low_sensor_accel = body_accel @ rotation
    return {
        "dataset_id": "synthetic",
        "pair_id": "rotated_low_to_high",
        "sequence_id": "sequence_a",
        "split_group": "sequence_a",
        "source_kind": "synthetic",
        "source_file": "synthetic://sequence_a",
        "low_sensor": "low90",
        "high_sensor": "reference",
        "target_rate_hz": rate_hz,
        "low_frame": _six_axis_frame(timestamps, low_sensor_gyro, low_sensor_accel),
        "high_frame": _six_axis_frame(timestamps, body_gyro, body_accel),
        "low_transform": RigidTransform(rotation, np.zeros(3), "low_z90"),
        "high_transform": RigidTransform(np.eye(3), np.zeros(3), "high_body"),
        "calibration_id": "synthetic_calibration_v1",
        "calibration_available": True,
        "translation_available": True,
        "minimum_output_rows": 100,
    }


def _context(tmp_path: Path) -> PreparationContext:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 20260711\n", encoding="utf-8")
    return PreparationContext(
        strict_root=tmp_path / "strict",
        diagnostic_root=tmp_path / "diagnostics",
        config_path=config_path,
        config={
            "seed": 20260711,
            "diagnostics": {"max_rows_per_pair_sequence": 200_000},
            "lever_arm": {
                "gyro_lowpass_order": 4,
                "gyro_lowpass_hz": None,
                "angular_acceleration_window": 5,
                "angular_acceleration_polynomial_order": 2,
            },
            "alignment": {
                "max_interpolation_gap_periods": 3.0,
                "interpolation": "linear",
                "causal_anti_alias": {
                    "taps": 63,
                    "cutoff_fraction_of_target_nyquist": 0.8,
                },
                "max_residual_lag_s": 0.1,
                "residual_lag_max_target_periods": 1.0,
            },
            "quality_thresholds": {
                "rotation_orthogonality_fro_max": 1e-5,
                "rotation_determinant_min": 0.999,
                "overlap_block_below": 0.90,
                "overlap_warn_below": 0.95,
                "active_gyro_correlation_block_below": 0.70,
                "active_gyro_correlation_warn_below": 0.90,
                "static_gravity_disagreement_warn_mps2": 0.3,
                "isolated_outlier_fraction_warn": 0.001,
            },
        },
    )


def _six_axis_frame(
    timestamps: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "gx": gyro[:, 0],
            "gy": gyro[:, 1],
            "gz": gyro[:, 2],
            "ax": accel[:, 0],
            "ay": accel[:, 1],
            "az": accel[:, 2],
        }
    )


def _minimal_training_frame() -> pd.DataFrame:
    values = {
        "timestamp_s": [0.0, 0.01],
        "temperature_c": [25.0, 25.0],
        "sequence_id": ["a", "a"],
    }
    for column in (
        "gx_low",
        "gy_low",
        "gz_low",
        "ax_low",
        "ay_low",
        "az_low",
        "gx_high",
        "gy_high",
        "gz_high",
        "ax_high",
        "ay_high",
        "az_high",
    ):
        values[column] = [0.0, 0.0]
    return pd.DataFrame(values)
