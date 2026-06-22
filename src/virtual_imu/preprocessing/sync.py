"""Time synchronization and low/high IMU alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.preprocessing.resample import common_time_grid, resample_numeric

LOW_COLS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH_COLS = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]


def estimate_time_delay_s(
    low: pd.DataFrame,
    high: pd.DataFrame,
    rate_hz: float,
    max_lag_s: float = 0.5,
) -> float:
    """Estimate low-vs-high delay using cross-correlation of gyro norms."""
    start = max(float(low["timestamp_s"].min()), float(high["timestamp_s"].min()))
    end = min(float(low["timestamp_s"].max()), float(high["timestamp_s"].max()))
    grid = common_time_grid(start, end, rate_hz)
    low_r = resample_numeric(low, grid, ["gx_low", "gy_low", "gz_low"])
    high_r = resample_numeric(high, grid, ["gx_high", "gy_high", "gz_high"])
    a = np.linalg.norm(low_r[["gx_low", "gy_low", "gz_low"]].to_numpy(), axis=1)
    b = np.linalg.norm(high_r[["gx_high", "gy_high", "gz_high"]].to_numpy(), axis=1)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    max_lag = int(round(max_lag_s * rate_hz))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return 0.0
    best_lag = int(lags[mask][np.argmax(corr[mask])])
    return best_lag / rate_hz


def align_low_high(
    low: pd.DataFrame,
    high: pd.DataFrame,
    target_rate_hz: float,
    method: str = "linear",
    time_offset_s: float = 0.0,
    estimate_delay: bool = True,
    max_lag_s: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    """Resample low/high IMU to a common grid and return an alignment report."""
    low_shifted = low.copy()
    low_shifted["timestamp_s"] = low_shifted["timestamp_s"] - time_offset_s
    estimated_delay = (
        estimate_time_delay_s(low_shifted, high, target_rate_hz, max_lag_s=max_lag_s) if estimate_delay else 0.0
    )
    low_shifted["timestamp_s"] = low_shifted["timestamp_s"] - estimated_delay
    start = max(float(low_shifted["timestamp_s"].min()), float(high["timestamp_s"].min()))
    end = min(float(low_shifted["timestamp_s"].max()), float(high["timestamp_s"].max()))
    grid = common_time_grid(start, end, target_rate_hz)
    low_r = resample_numeric(low_shifted, grid, LOW_COLS + (["temperature_c"] if "temperature_c" in low_shifted else []), method)
    high_r = resample_numeric(high, grid, HIGH_COLS, method)
    out = pd.merge_asof(low_r.sort_values("timestamp_s"), high_r.sort_values("timestamp_s"), on="timestamp_s")
    out["sequence_id"] = "smartpnt_aligned"
    if "temperature_c" not in out:
        out["temperature_c"] = np.nan
    report = {
        "target_rate_hz": float(target_rate_hz),
        "configured_time_offset_s": float(time_offset_s),
        "estimated_delay_s": float(estimated_delay),
        "start_s": float(start),
        "end_s": float(end),
        "samples": int(len(out)),
        "method": method,
    }
    return out, report
