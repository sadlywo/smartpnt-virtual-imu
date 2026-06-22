"""Simplified preintegration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.signal_metrics import ACCEL_AXES, GYRO_AXES


def _window_errors(pred: np.ndarray, ref: np.ndarray, dt: float, window: int) -> tuple[float, float, float]:
    rot_errors = []
    vel_errors = []
    pos_errors = []
    for start in range(0, max(len(pred) - window + 1, 1), window):
        end = min(start + window, len(pred))
        p = pred[start:end]
        r = ref[start:end]
        if len(p) < 2:
            continue
        gyro_p, accel_p = p[:, :3], p[:, 3:]
        gyro_r, accel_r = r[:, :3], r[:, 3:]
        rot_errors.append(float(np.linalg.norm(np.sum((gyro_p - gyro_r) * dt, axis=0))))
        dv = np.sum((accel_p - accel_r) * dt, axis=0)
        vel_errors.append(float(np.linalg.norm(dv)))
        pos = np.sum(np.cumsum((accel_p - accel_r) * dt, axis=0) * dt, axis=0)
        pos_errors.append(float(np.linalg.norm(pos)))
    return (
        float(np.nanmean(rot_errors)) if rot_errors else float("nan"),
        float(np.nanmean(vel_errors)) if vel_errors else float("nan"),
        float(np.nanmean(pos_errors)) if pos_errors else float("nan"),
    )


def preintegration_metrics(df: pd.DataFrame, pred_prefix: str, ref_prefix: str = "high", window_s: float = 2.0) -> dict[str, float]:
    """Compute simplified preintegration residuals."""
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    window = max(2, int(round(window_s / max(dt, 1e-9))))
    pred = df[[*(f"{axis}_{pred_prefix}" for axis in GYRO_AXES), *(f"{axis}_{pred_prefix}" for axis in ACCEL_AXES)]].to_numpy(float)
    ref = df[[*(f"{axis}_{ref_prefix}" for axis in GYRO_AXES), *(f"{axis}_{ref_prefix}" for axis in ACCEL_AXES)]].to_numpy(float)
    drot, dvel, dpos = _window_errors(pred, ref, dt, window)
    return {
        "imu_preintegration_residual": float(drot + dvel + dpos),
        "delta_rotation_error_rad": drot,
        "delta_velocity_error_m_s": dvel,
        "delta_position_proxy_error_m": dpos,
    }
