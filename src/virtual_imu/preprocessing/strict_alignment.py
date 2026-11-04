"""Causal timestamp cleaning, anti-alias resampling, and strict IMU alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

IMU_COLUMNS = ["gx", "gy", "gz", "ax", "ay", "az"]


def clean_timestamps(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove exact timestamp duplicates and reject unresolved source-order decreases."""

    if "timestamp_s" not in frame:
        raise ValueError("frame is missing timestamp_s")
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    duplicate_mask = frame["timestamp_s"].duplicated(keep="first")
    cleaned = frame.loc[~duplicate_mask].copy().reset_index(drop=True)
    cleaned_timestamps = cleaned["timestamp_s"].to_numpy(dtype=float)
    if np.any(np.diff(cleaned_timestamps) < 0.0):
        raise ValueError("timestamps remain decreasing after exact duplicate removal")
    cleaned.attrs.update(frame.attrs)
    stats = {
        "input_rows": len(frame),
        "duplicate_timestamps_removed": int(duplicate_mask.sum()),
        "output_rows": len(cleaned),
    }
    return cleaned, stats


def causal_resample(
    frame: pd.DataFrame,
    grid_s: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Causally filter and resample one strictly increasing six-axis stream."""

    _require_columns(frame, ["timestamp_s", *IMU_COLUMNS])
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    grid = np.asarray(grid_s, dtype=float)
    if len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("At least two finite source timestamps are required")
    source_steps = np.diff(timestamps)
    if np.any(source_steps <= 0.0):
        raise ValueError("source timestamps must be strictly increasing")
    if grid.ndim != 1 or len(grid) == 0 or not np.isfinite(grid).all():
        raise ValueError("grid_s must be a non-empty finite vector")
    if len(grid) > 1 and np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid_s must be strictly increasing")

    source_period_s = float(np.median(source_steps))
    source_rate_hz = 1.0 / source_period_s
    target_period_s = source_period_s if len(grid) < 2 else float(np.median(np.diff(grid)))
    target_rate_hz = 1.0 / target_period_s
    values = frame[IMU_COLUMNS].to_numpy(dtype=float)
    filtered, group_delay_s, anti_alias_applied = _causal_antialias(
        values,
        source_rate_hz,
        target_rate_hz,
        config.get("causal_anti_alias", {}),
    )
    effective_timestamps = timestamps - group_delay_s
    interpolation = str(config.get("interpolation", "linear"))
    if interpolation not in {"linear", "latest"}:
        raise ValueError("interpolation must be 'linear' or 'latest'")
    interpolation_latency_s = source_period_s if interpolation == "linear" else 0.0
    algorithmic_latency_s = group_delay_s + interpolation_latency_s
    max_gap_periods = float(config.get("max_interpolation_gap_periods", 3.0))
    if not np.isfinite(max_gap_periods) or max_gap_periods <= 0.0:
        raise ValueError("max_interpolation_gap_periods must be finite and positive")

    output_values = np.full((len(grid), len(IMU_COLUMNS)), np.nan, dtype=float)
    interpolated = np.zeros(len(grid), dtype=bool)
    gap_flag = np.zeros(len(grid), dtype=bool)
    out_of_range = np.zeros(len(grid), dtype=bool)
    right_indices = np.searchsorted(effective_timestamps, grid, side="left")
    tolerance = max(1e-12, source_period_s * 1e-8)
    for index, timestamp in enumerate(grid):
        right = right_indices[index]
        if right < len(effective_timestamps) and abs(effective_timestamps[right] - timestamp) <= tolerance:
            output_values[index] = filtered[right]
            continue
        left = right - 1
        if left < 0 or right >= len(effective_timestamps):
            gap_flag[index] = True
            out_of_range[index] = True
            continue
        span = effective_timestamps[right] - effective_timestamps[left]
        if span > max_gap_periods * source_period_s + tolerance:
            gap_flag[index] = True
            continue
        right_wait = effective_timestamps[right] - timestamp
        if interpolation == "linear" and right_wait <= source_period_s + tolerance:
            weight = (timestamp - effective_timestamps[left]) / span
            output_values[index] = filtered[left] + weight * (filtered[right] - filtered[left])
            interpolated[index] = True
        else:
            output_values[index] = filtered[left]

    match_distance_s = _nearest_timestamp_distance(timestamps, grid)
    output = pd.DataFrame(output_values, columns=IMU_COLUMNS)
    output.insert(0, "timestamp_s", grid)
    output["match_dt_ms"] = match_distance_s * 1e3
    output["interpolation_flag"] = interpolated
    output["gap_flag"] = gap_flag
    output["out_of_range_flag"] = out_of_range
    metrics = {
        "source_rate_hz": source_rate_hz,
        "target_rate_hz": target_rate_hz,
        "source_period_s": source_period_s,
        "anti_alias_applied": anti_alias_applied,
        "fir_group_delay_s": group_delay_s,
        "interpolation_latency_s": interpolation_latency_s,
        "algorithmic_latency_s": algorithmic_latency_s,
        "interpolated_rows": int(interpolated.sum()),
        "gap_rows": int(gap_flag.sum()),
        "out_of_range_rows": int(out_of_range.sum()),
        "match_dt_ms_p50": float(np.quantile(output["match_dt_ms"], 0.50)),
        "match_dt_ms_p99": float(np.quantile(output["match_dt_ms"], 0.99)),
    }
    return output, metrics


def residual_gyro_lag(
    low: pd.DataFrame,
    high: pd.DataFrame,
    rate_hz: float,
    max_lag_s: float,
) -> dict[str, float]:
    """Estimate residual high-versus-low lag from gyroscope vector magnitudes."""

    _require_columns(low, ["gx", "gy", "gz"])
    _require_columns(high, ["gx", "gy", "gz"])
    if len(low) != len(high) or len(low) < 3:
        raise ValueError("low and high lag inputs must have equal length of at least three")
    rate = float(rate_hz)
    maximum = float(max_lag_s)
    if rate <= 0.0 or maximum < 0.0 or not np.isfinite([rate, maximum]).all():
        raise ValueError("rate_hz and max_lag_s must be finite and non-negative")
    low_norm = np.linalg.norm(low[["gx", "gy", "gz"]].to_numpy(dtype=float), axis=1)
    high_norm = np.linalg.norm(high[["gx", "gy", "gz"]].to_numpy(dtype=float), axis=1)
    valid = np.isfinite(low_norm) & np.isfinite(high_norm)
    if valid.sum() < 3:
        raise ValueError("At least three finite paired gyroscope rows are required")
    low_centered = low_norm[valid] - float(np.mean(low_norm[valid]))
    high_centered = high_norm[valid] - float(np.mean(high_norm[valid]))
    denominator = float(np.linalg.norm(low_centered) * np.linalg.norm(high_centered))
    if denominator == 0.0:
        return {"lag_samples": 0, "lag_s": 0.0, "peak_correlation": 0.0}
    correlations = signal.correlate(high_centered, low_centered, mode="full", method="fft")
    lags = signal.correlation_lags(len(high_centered), len(low_centered), mode="full")
    maximum_samples = int(round(maximum * rate))
    selected = np.abs(lags) <= maximum_samples
    selected_correlations = correlations[selected] / denominator
    selected_lags = lags[selected]
    peak_index = int(np.argmax(selected_correlations))
    lag_samples = int(selected_lags[peak_index])
    return {
        "lag_samples": lag_samples,
        "lag_s": float(lag_samples / rate),
        "peak_correlation": float(selected_correlations[peak_index]),
    }


def align_strict_pair(
    low: pd.DataFrame,
    high: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align a low/high pair on one grid without fitting a delay correction."""

    low_clean, low_cleanup = clean_timestamps(low)
    high_clean, high_cleanup = clean_timestamps(high)
    if len(low_clean) < 2 or len(high_clean) < 2:
        raise ValueError("Both streams require at least two rows after timestamp cleaning")
    low_times = low_clean["timestamp_s"].to_numpy(dtype=float)
    high_times = high_clean["timestamp_s"].to_numpy(dtype=float)
    overlap_start = max(float(low_times[0]), float(high_times[0]))
    overlap_end = min(float(low_times[-1]), float(high_times[-1]))
    overlap_duration = overlap_end - overlap_start
    if overlap_duration <= 0.0:
        raise ValueError("low and high streams have no positive timestamp overlap")
    low_duration = float(low_times[-1] - low_times[0])
    high_duration = float(high_times[-1] - high_times[0])
    overlap_fraction = overlap_duration / max(low_duration, high_duration)

    target_rate_hz = float(config["target_rate_hz"])
    if not np.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
        raise ValueError("target_rate_hz must be finite and positive")
    target_period_s = 1.0 / target_rate_hz
    grid_rows = int(np.floor(overlap_duration / target_period_s)) + 1
    grid = overlap_start + np.arange(grid_rows, dtype=float) * target_period_s
    grid = grid[grid <= overlap_end + target_period_s * 1e-8]

    low_resampled, low_metrics = causal_resample(low_clean, grid, config)
    high_resampled, high_metrics = causal_resample(high_clean, grid, config)
    in_effective_range = ~(
        low_resampled["out_of_range_flag"].to_numpy()
        | high_resampled["out_of_range_flag"].to_numpy()
    )
    grid = grid[in_effective_range]
    low_resampled = low_resampled.loc[in_effective_range].reset_index(drop=True)
    high_resampled = high_resampled.loc[in_effective_range].reset_index(drop=True)
    if len(grid) < 3:
        raise ValueError("Fewer than three aligned rows remain after causal edge trimming")
    aligned = pd.DataFrame({"timestamp_s": grid})
    for column in IMU_COLUMNS:
        aligned[f"{column}_low"] = low_resampled[column].to_numpy()
        aligned[f"{column}_high"] = high_resampled[column].to_numpy()
    aligned["low_match_dt_ms"] = low_resampled["match_dt_ms"].to_numpy()
    aligned["high_match_dt_ms"] = high_resampled["match_dt_ms"].to_numpy()
    aligned["match_dt_ms"] = np.maximum(
        aligned["low_match_dt_ms"],
        aligned["high_match_dt_ms"],
    )
    aligned["low_interpolated"] = low_resampled["interpolation_flag"].to_numpy()
    aligned["high_interpolated"] = high_resampled["interpolation_flag"].to_numpy()
    aligned["interpolation_flag"] = aligned["low_interpolated"] | aligned["high_interpolated"]
    aligned["low_gap"] = low_resampled["gap_flag"].to_numpy()
    aligned["high_gap"] = high_resampled["gap_flag"].to_numpy()
    aligned["gap_flag"] = aligned["low_gap"] | aligned["high_gap"]

    lag = residual_gyro_lag(
        _resampled_imu_view(low_resampled),
        _resampled_imu_view(high_resampled),
        rate_hz=target_rate_hz,
        max_lag_s=float(config.get("max_residual_lag_s", 0.5)),
    )
    maximum_lag_periods = float(config.get("residual_lag_max_target_periods", 1.0))
    lag_blocked = abs(lag["lag_s"]) > maximum_lag_periods * target_period_s
    blocking_reasons = ["residual_lag_exceeds_target_period"] if lag_blocked else []
    metrics = {
        "low_timestamp_cleanup": low_cleanup,
        "high_timestamp_cleanup": high_cleanup,
        "overlap_start_s": overlap_start,
        "overlap_end_s": overlap_end,
        "overlap_duration_s": overlap_duration,
        "overlap_fraction": overlap_fraction,
        "target_rate_hz": target_rate_hz,
        "rows": len(aligned),
        "low_resample": low_metrics,
        "high_resample": high_metrics,
        "match_dt_ms_p50": float(np.quantile(aligned["match_dt_ms"], 0.50)),
        "match_dt_ms_p99": float(np.quantile(aligned["match_dt_ms"], 0.99)),
        "residual_lag": lag,
        "residual_lag_blocked": lag_blocked,
        "blocking_reasons": blocking_reasons,
    }
    return aligned, metrics


def _causal_antialias(
    values: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
    config: Any,
) -> tuple[np.ndarray, float, bool]:
    if not isinstance(config, dict):
        raise ValueError("causal_anti_alias must be a mapping")
    taps = int(config.get("taps", 63))
    if taps <= 1 or target_rate_hz >= source_rate_hz * (1.0 - 1e-9):
        return values.copy(), 0.0, False
    if taps % 2 == 0:
        raise ValueError("causal anti-alias FIR taps must be odd")
    fraction = float(config.get("cutoff_fraction_of_target_nyquist", 0.8))
    if not 0.0 < fraction < 1.0:
        raise ValueError("anti-alias cutoff fraction must be between zero and one")
    cutoff_hz = fraction * target_rate_hz / 2.0
    coefficients = signal.firwin(taps, cutoff=cutoff_hz, fs=source_rate_hz)
    filtered = signal.lfilter(coefficients, [1.0], values, axis=0)
    group_delay_s = ((taps - 1) / 2.0) / source_rate_hz
    return filtered, group_delay_s, True


def _nearest_timestamp_distance(source: np.ndarray, grid: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, grid, side="left")
    right_clipped = np.clip(right, 0, len(source) - 1)
    left_clipped = np.clip(right - 1, 0, len(source) - 1)
    return np.minimum(
        np.abs(source[right_clipped] - grid),
        np.abs(source[left_clipped] - grid),
    )


def _resampled_imu_view(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["timestamp_s", *IMU_COLUMNS]]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
