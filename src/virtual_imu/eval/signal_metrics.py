"""Signal-level metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

GYRO_AXES = ["gx", "gy", "gz"]
ACCEL_AXES = ["ax", "ay", "az"]


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.square(err))))


def _mae(err: np.ndarray) -> float:
    return float(np.nanmean(np.abs(err)))


def estimate_lag_s(a: np.ndarray, b: np.ndarray, dt: float, max_lag_s: float = 0.5) -> float:
    """Estimate lag from cross-correlation."""
    a = np.asarray(a, dtype=float) - np.nanmean(a)
    b = np.asarray(b, dtype=float) - np.nanmean(b)
    corr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(a) + 1, len(a))
    max_lag = int(max_lag_s / max(dt, 1e-9))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return 0.0
    return float(lags[mask][np.argmax(corr[mask])] * dt)


def signal_metrics(df: pd.DataFrame, pred_prefix: str, ref_prefix: str = "high") -> dict[str, float]:
    """Compute signal-level metrics for one prediction prefix."""
    metrics: dict[str, float] = {}
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    for sensor, axes in {"gyro": GYRO_AXES, "accel": ACCEL_AXES}.items():
        pred = df[[f"{axis}_{pred_prefix}" for axis in axes]].to_numpy(dtype=float)
        ref = df[[f"{axis}_{ref_prefix}" for axis in axes]].to_numpy(dtype=float)
        err = pred - ref
        for idx, axis in enumerate(axes):
            metrics[f"{sensor}_{axis}_rmse"] = _rmse(err[:, idx])
            metrics[f"{sensor}_{axis}_mae"] = _mae(err[:, idx])
            metrics[f"{sensor}_{axis}_bias_error"] = float(np.nanmean(err[:, idx]))
            metrics[f"{sensor}_{axis}_noise_std"] = float(np.nanstd(err[:, idx]))
            if np.nanstd(pred[:, idx]) > 1e-12 and np.nanstd(ref[:, idx]) > 1e-12:
                metrics[f"{sensor}_{axis}_correlation"] = float(np.corrcoef(pred[:, idx], ref[:, idx])[0, 1])
            else:
                metrics[f"{sensor}_{axis}_correlation"] = float("nan")
        pred_norm = np.linalg.norm(pred, axis=1)
        ref_norm = np.linalg.norm(ref, axis=1)
        err_norm = pred_norm - ref_norm
        metrics[f"{sensor}_norm_rmse"] = _rmse(err_norm)
        metrics[f"{sensor}_norm_mae"] = _mae(err_norm)
        metrics[f"{sensor}_time_lag_s"] = estimate_lag_s(pred_norm, ref_norm, dt)
        signal_power = np.nanmean(np.square(ref_norm))
        noise_power = np.nanmean(np.square(err_norm))
        metrics[f"{sensor}_snr_db"] = float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))
    return metrics


def snr_improvement(low_metrics: dict[str, float], virtual_metrics: dict[str, float]) -> dict[str, float]:
    """Compute SNR improvement between low and virtual predictions."""
    return {
        "gyro_snr_improvement_db": virtual_metrics.get("gyro_snr_db", np.nan) - low_metrics.get("gyro_snr_db", np.nan),
        "accel_snr_improvement_db": virtual_metrics.get("accel_snr_db", np.nan) - low_metrics.get("accel_snr_db", np.nan),
    }
