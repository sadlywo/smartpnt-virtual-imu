"""Synthetic 6-DOF motion profiles for paired IMU simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.utils.math_so3 import integrate_gyro, matrix_to_quat_wxyz

MOTION_LABELS = [
    "static",
    "constant_velocity",
    "sinusoidal_turn",
    "sharp_turn",
    "random_maneuver",
    "high_dynamic_vibration",
]


def _segment_labels(n: int) -> np.ndarray:
    labels = np.empty(n, dtype=object)
    edges = np.linspace(0, n, len(MOTION_LABELS) + 1, dtype=int)
    for idx, label in enumerate(MOTION_LABELS):
        labels[edges[idx] : edges[idx + 1]] = label
    return labels


def generate_motion_truth(
    duration_s: float,
    rate_hz: float,
    seed: int,
    sequence_id: str,
    motion_cfg: dict | None = None,
) -> pd.DataFrame:
    """Generate high-grade IMU signals and a simple orientation trace."""
    rng = np.random.default_rng(seed)
    motion_cfg = motion_cfg or {}
    n = max(2, int(round(duration_s * rate_hz)))
    t = np.arange(n, dtype=float) / rate_hz
    labels = _segment_labels(n)
    gyro = np.zeros((n, 3), dtype=float)
    accel = np.zeros((n, 3), dtype=float)

    max_turn = float(motion_cfg.get("max_turn_rate_rad_s", 1.1))
    vib_hz = float(motion_cfg.get("vibration_hz", 18.0))
    gravity = np.array([0.0, 0.0, 9.80665])

    for label in MOTION_LABELS:
        idx = np.where(labels == label)[0]
        if idx.size == 0:
            continue
        local_t = t[idx] - t[idx[0]]
        if label == "static":
            accel[idx] = gravity
        elif label == "constant_velocity":
            accel[idx] = gravity + np.column_stack(
                [
                    0.05 * np.sin(0.7 * local_t),
                    0.03 * np.cos(0.5 * local_t),
                    0.02 * np.sin(0.4 * local_t),
                ]
            )
            gyro[idx, 2] = 0.03
        elif label == "sinusoidal_turn":
            gyro[idx, 0] = 0.25 * np.sin(2.0 * np.pi * 0.25 * local_t)
            gyro[idx, 1] = 0.18 * np.cos(2.0 * np.pi * 0.18 * local_t)
            gyro[idx, 2] = 0.45 * np.sin(2.0 * np.pi * 0.12 * local_t)
            accel[idx] = gravity + np.column_stack(
                [0.5 * np.sin(local_t), 0.3 * np.cos(0.8 * local_t), 0.1 * np.sin(0.4 * local_t)]
            )
        elif label == "sharp_turn":
            pulse = np.exp(-((local_t - local_t.mean()) ** 2) / max(0.05, 0.08 * np.ptp(local_t)) ** 2)
            gyro[idx, 2] = max_turn * pulse
            gyro[idx, 0] = 0.15 * pulse
            accel[idx] = gravity + np.column_stack([1.8 * pulse, 0.9 * np.gradient(pulse), 0.2 * pulse])
        elif label == "random_maneuver":
            gyro_noise = rng.normal(0.0, 0.07, size=(idx.size, 3))
            accel_noise = rng.normal(0.0, 0.15, size=(idx.size, 3))
            gyro[idx] = np.cumsum(gyro_noise, axis=0) / np.sqrt(rate_hz)
            accel[idx] = gravity + np.cumsum(accel_noise, axis=0) / np.sqrt(rate_hz)
        elif label == "high_dynamic_vibration":
            vib = np.sin(2.0 * np.pi * vib_hz * local_t)
            gyro[idx] = np.column_stack(
                [0.2 * vib, 0.15 * np.sin(2.0 * np.pi * (vib_hz * 0.7) * local_t), 0.08 * vib]
            )
            accel[idx] = gravity + np.column_stack([0.5 * vib, 0.35 * np.cos(2.0 * np.pi * vib_hz * local_t), 0.25 * vib])

    rotations = integrate_gyro(gyro, 1.0 / rate_hz)
    quats = np.vstack([matrix_to_quat_wxyz(rot) for rot in rotations])
    temperature = 25.0 + 2.5 * np.sin(2.0 * np.pi * t / max(duration_s, 1.0)) + rng.normal(0.0, 0.05, n)

    return pd.DataFrame(
        {
            "timestamp_s": t,
            "gx_high": gyro[:, 0],
            "gy_high": gyro[:, 1],
            "gz_high": gyro[:, 2],
            "ax_high": accel[:, 0],
            "ay_high": accel[:, 1],
            "az_high": accel[:, 2],
            "qw": quats[:, 0],
            "qx": quats[:, 1],
            "qy": quats[:, 2],
            "qz": quats[:, 3],
            "temperature_c": temperature,
            "motion_label": labels,
            "sequence_id": sequence_id,
        }
    )
