"""Simplified strapdown/dead-reckoning evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.signal_metrics import ACCEL_AXES, GYRO_AXES


def integrate_dead_reckoning(df: pd.DataFrame, prefix: str) -> dict[str, np.ndarray]:
    """Integrate IMU signals with a simplified orientation-free proxy."""
    t = df["timestamp_s"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    gyro = df[[f"{axis}_{prefix}" for axis in GYRO_AXES]].to_numpy(dtype=float)
    accel = df[[f"{axis}_{prefix}" for axis in ACCEL_AXES]].to_numpy(dtype=float)
    attitude = np.cumsum(gyro * dt[:, None], axis=0)
    accel_no_g = accel - np.array([0.0, 0.0, 9.80665])
    velocity = np.cumsum(accel_no_g * dt[:, None], axis=0)
    position = np.cumsum(velocity * dt[:, None], axis=0)
    return {"attitude": attitude, "velocity": velocity, "position": position}


def navigation_metrics(df: pd.DataFrame, pred_prefix: str, ref_prefix: str = "high") -> dict[str, float | str]:
    """Compare short-horizon dead-reckoning drift to high-grade proxy reference."""
    pred = integrate_dead_reckoning(df, pred_prefix)
    ref = integrate_dead_reckoning(df, ref_prefix)
    att_err = np.linalg.norm(pred["attitude"] - ref["attitude"], axis=1)
    vel_err = np.linalg.norm(pred["velocity"] - ref["velocity"], axis=1)
    pos_err = np.linalg.norm(pred["position"] - ref["position"], axis=1)
    return {
        "attitude_drift_rad_final": float(att_err[-1]) if len(att_err) else float("nan"),
        "velocity_drift_m_s_final": float(vel_err[-1]) if len(vel_err) else float("nan"),
        "position_drift_m_final": float(pos_err[-1]) if len(pos_err) else float("nan"),
        "attitude_drift_rad_rms": float(np.sqrt(np.nanmean(att_err**2))),
        "velocity_drift_m_s_rms": float(np.sqrt(np.nanmean(vel_err**2))),
        "position_drift_m_rms": float(np.sqrt(np.nanmean(pos_err**2))),
        "reference_note": "High-grade IMU integration is used as proxy reference when external ground truth is unavailable.",
    }
