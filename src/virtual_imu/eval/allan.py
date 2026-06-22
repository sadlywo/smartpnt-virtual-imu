"""Allan deviation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.signal_metrics import ACCEL_AXES, GYRO_AXES


def allan_deviation(series: np.ndarray, rate_hz: float, max_points: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Estimate overlapping Allan deviation for one uniformly sampled series."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return np.array([]), np.array([])
    max_m = max(2, n // 8)
    ms = np.unique(np.logspace(0, np.log10(max_m), num=max_points).astype(int))
    taus = []
    adev = []
    cumsum = np.cumsum(np.insert(y, 0, 0.0))
    for m in ms:
        if 2 * m >= n:
            continue
        avgs = (cumsum[m:] - cumsum[:-m]) / m
        diff = avgs[m:] - avgs[:-m]
        taus.append(m / rate_hz)
        adev.append(np.sqrt(0.5 * np.nanmean(diff**2)))
    return np.asarray(taus), np.asarray(adev)


def allan_summary(df: pd.DataFrame, pred_prefix: str, min_duration_s: float = 20.0) -> dict:
    """Compute Allan proxy metrics and insufficiency notes."""
    duration = float(df["timestamp_s"].max() - df["timestamp_s"].min()) if len(df) else 0.0
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    rate_hz = 1.0 / max(dt, 1e-9)
    result: dict = {"duration_s": duration, "status": "ok" if duration >= min_duration_s else "insufficient duration"}
    for sensor, axes in {"gyro": GYRO_AXES, "accel": ACCEL_AXES}.items():
        proxies = []
        for axis in axes:
            taus, adev = allan_deviation(df[f"{axis}_{pred_prefix}"].to_numpy(dtype=float), rate_hz)
            if len(adev) == 0:
                continue
            proxies.append(
                {
                    "axis": axis,
                    "min_allan_deviation": float(np.nanmin(adev)),
                    "tau_at_min_s": float(taus[int(np.nanargmin(adev))]),
                    "white_noise_proxy": float(adev[0] * np.sqrt(taus[0])),
                    "random_walk_proxy": float(adev[-1] / np.sqrt(max(taus[-1], 1e-12))),
                }
            )
        result[sensor] = proxies
    return result
