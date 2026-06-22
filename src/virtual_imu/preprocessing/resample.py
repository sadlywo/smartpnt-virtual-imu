"""IMU resampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def common_time_grid(start: float, end: float, rate_hz: float) -> np.ndarray:
    """Create an inclusive-ish uniform time grid."""
    if end <= start:
        raise ValueError("end time must be greater than start time")
    dt = 1.0 / rate_hz
    return np.arange(start, end + 0.5 * dt, dt)


def resample_numeric(
    df: pd.DataFrame,
    grid_s: np.ndarray,
    columns: list[str],
    method: str = "linear",
) -> pd.DataFrame:
    """Resample numeric columns onto a provided grid."""
    if "timestamp_s" not in df.columns:
        raise ValueError("DataFrame must contain timestamp_s")
    t = df["timestamp_s"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("Need at least two samples to resample")
    out = pd.DataFrame({"timestamp_s": grid_s})
    for col in columns:
        values = df[col].to_numpy(dtype=float)
        if method == "nearest":
            idx = np.searchsorted(t, grid_s, side="left")
            idx = np.clip(idx, 0, len(t) - 1)
            left = np.clip(idx - 1, 0, len(t) - 1)
            use_left = np.abs(grid_s - t[left]) < np.abs(grid_s - t[idx])
            out[col] = values[np.where(use_left, left, idx)]
        elif method in {"linear", "slerp"}:
            out[col] = np.interp(grid_s, t, values)
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
    return out
