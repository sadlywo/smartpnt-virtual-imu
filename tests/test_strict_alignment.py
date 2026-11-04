from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virtual_imu.preprocessing.strict_alignment import (
    align_strict_pair,
    causal_resample,
    clean_timestamps,
    residual_gyro_lag,
)


def test_clean_timestamps_removes_exact_duplicates_and_keeps_first() -> None:
    frame = _imu_frame(np.array([0.0, 0.01, 0.01, 0.02]), np.array([1.0, 2.0, 99.0, 3.0]))

    cleaned, stats = clean_timestamps(frame)

    assert cleaned["timestamp_s"].tolist() == [0.0, 0.01, 0.02]
    assert cleaned["gx"].tolist() == [1.0, 2.0, 3.0]
    assert stats == {"input_rows": 4, "duplicate_timestamps_removed": 1, "output_rows": 3}


def test_clean_timestamps_rejects_remaining_decrease() -> None:
    frame = _imu_frame(np.array([0.0, 0.02, 0.01]), np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="decreasing"):
        clean_timestamps(frame)


def test_causal_resample_refuses_long_gap() -> None:
    timestamps = np.array([0.0, 0.01, 0.02, 0.10, 0.11])
    frame = _imu_frame(timestamps, np.sin(timestamps))
    grid = np.arange(0.0, 0.111, 0.01)

    output, metrics = causal_resample(frame, grid, _alignment_config(taps=1))

    assert output.loc[output["timestamp_s"] == pytest.approx(0.05), "gap_flag"].all()
    assert output.loc[output["timestamp_s"] == pytest.approx(0.05), "gx"].isna().all()
    assert metrics["gap_rows"] >= 1


def test_causal_antialias_preserves_2hz_and_attenuates_80hz() -> None:
    source_rate = 200.0
    timestamps = np.arange(0.0, 8.0, 1.0 / source_rate)
    values = np.sin(2 * np.pi * 2 * timestamps) + 0.8 * np.sin(2 * np.pi * 80 * timestamps)
    frame = _imu_frame(timestamps, values)
    grid = np.arange(0.0, 8.0, 0.01)

    output, metrics = causal_resample(frame, grid, _alignment_config(taps=63))

    valid = output[(output["timestamp_s"] > 2.0) & output["gx"].notna()]
    truth = np.sin(2 * np.pi * 2 * valid["timestamp_s"].to_numpy())
    rmse = float(np.sqrt(np.mean((valid["gx"].to_numpy() - truth) ** 2)))
    amplitude_2hz = _tone_amplitude(valid["gx"].to_numpy(), 100.0, 2.0)
    aliased_20hz = _tone_amplitude(valid["gx"].to_numpy(), 100.0, 20.0)
    assert rmse < 0.08
    assert amplitude_2hz == pytest.approx(1.0, abs=0.08)
    assert aliased_20hz < 0.05
    assert metrics["anti_alias_applied"] is True
    assert metrics["algorithmic_latency_s"] > 0.0


def test_resample_cannot_depend_beyond_declared_latency() -> None:
    rng = np.random.default_rng(7)
    timestamps = np.arange(0.0, 4.0, 0.005)
    values = rng.normal(size=len(timestamps))
    frame = _imu_frame(timestamps, values)
    grid = np.arange(0.0, 4.0, 0.01)
    config = _alignment_config(taps=63)
    baseline, metrics = causal_resample(frame, grid, config)
    cutoff = 1.5
    latest_allowed = cutoff + metrics["algorithmic_latency_s"]
    modified = frame.copy()
    modified.loc[modified["timestamp_s"] > latest_allowed + 1e-12, "gx"] += 1e6

    changed, _ = causal_resample(modified, grid, config)

    mask = baseline["timestamp_s"] <= cutoff
    np.testing.assert_allclose(
        baseline.loc[mask, "gx"],
        changed.loc[mask, "gx"],
        equal_nan=True,
    )


def test_residual_gyro_lag_recovers_known_ten_milliseconds() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=500)
    low = _imu_frame(np.arange(500) / 100.0, values)
    high = _imu_frame(np.arange(500) / 100.0, np.r_[0.0, values[:-1]])

    lag = residual_gyro_lag(low, high, rate_hz=100.0, max_lag_s=0.1)

    assert lag["lag_samples"] == 1
    assert lag["lag_s"] == pytest.approx(0.01)
    assert lag["peak_correlation"] > 0.99


def test_align_strict_pair_reports_match_distance_and_blocks_large_lag() -> None:
    rng = np.random.default_rng(19)
    timestamps = np.arange(600) / 100.0
    values = rng.normal(size=600)
    low = _imu_frame(timestamps, values)
    high = _imu_frame(timestamps + 0.002, np.r_[0.0, 0.0, values[:-2]])
    config = {
        **_alignment_config(taps=1),
        "target_rate_hz": 100.0,
        "max_residual_lag_s": 0.1,
        "residual_lag_max_target_periods": 1.0,
    }

    aligned, metrics = align_strict_pair(low, high, config)

    assert {"match_dt_ms", "interpolation_flag", "gap_flag"}.issubset(aligned.columns)
    assert metrics["overlap_fraction"] > 0.99
    assert metrics["match_dt_ms_p99"] <= 2.1
    assert metrics["residual_lag"]["lag_samples"] == 2
    assert metrics["residual_lag_blocked"] is True
    assert "residual_lag_exceeds_target_period" in metrics["blocking_reasons"]


def _alignment_config(taps: int) -> dict[str, object]:
    return {
        "max_interpolation_gap_periods": 3.0,
        "interpolation": "linear",
        "causal_anti_alias": {
            "taps": taps,
            "cutoff_fraction_of_target_nyquist": 0.8,
        },
    }


def _imu_frame(timestamps: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    zeros = np.zeros(len(timestamps))
    return pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "gx": values,
            "gy": zeros,
            "gz": zeros,
            "ax": zeros,
            "ay": zeros,
            "az": zeros,
        }
    )


def _tone_amplitude(values: np.ndarray, rate_hz: float, frequency_hz: float) -> float:
    timestamps = np.arange(len(values)) / rate_hz
    basis = np.exp(-2j * np.pi * frequency_hz * timestamps)
    return float(2.0 * abs(np.dot(values, basis)) / len(values))
