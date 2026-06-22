"""Low-grade IMU degradation models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1_noise(rng: np.random.Generator, shape: tuple[int, int], alpha: float, std: float) -> np.ndarray:
    noise = np.zeros(shape, dtype=float)
    innovation_std = std * np.sqrt(max(1.0 - alpha * alpha, 1e-6))
    for idx in range(1, shape[0]):
        noise[idx] = alpha * noise[idx - 1] + rng.normal(0.0, innovation_std, shape[1])
    return noise


def _cross_axis_matrix(scale_error: np.ndarray, misalignment_deg: float) -> np.ndarray:
    angle = np.deg2rad(misalignment_deg)
    coupling = np.array([[0.0, angle, -0.5 * angle], [-0.4 * angle, 0.0, angle], [0.3 * angle, -0.6 * angle, 0.0]])
    return np.diag(1.0 + scale_error) + coupling


def _interp_delayed(values: np.ndarray, t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query_t, t, values[:, axis], left=values[0, axis], right=values[-1, axis]) for axis in range(values.shape[1])])


def apply_low_grade_errors(high: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Degrade high-grade gyro/accel signals with configurable low-grade IMU errors."""
    rng = np.random.default_rng(seed)
    t = high["timestamp_s"].to_numpy(dtype=float)
    gyro_high = high[["gx_high", "gy_high", "gz_high"]].to_numpy(dtype=float)
    accel_high = high[["ax_high", "ay_high", "az_high"]].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.01

    delay = float(cfg.get("time_delay_s", 0.0))
    jitter = rng.normal(0.0, float(cfg.get("jitter_std_s", 0.0)), size=len(t))
    query_t = np.clip(t - delay + jitter, t[0], t[-1])
    gyro = _interp_delayed(gyro_high, t, query_t)
    accel = _interp_delayed(accel_high, t, query_t)

    gyro_scale = np.asarray(cfg.get("gyro_scale_error", [0.0, 0.0, 0.0]), dtype=float)
    accel_scale = np.asarray(cfg.get("accel_scale_error", [0.0, 0.0, 0.0]), dtype=float)
    misalignment = float(cfg.get("misalignment_deg", 0.0))
    gyro = gyro @ _cross_axis_matrix(gyro_scale, misalignment).T
    accel = accel @ _cross_axis_matrix(accel_scale, misalignment).T

    gyro_bias = np.asarray(cfg.get("gyro_bias_rad_s", [0.0, 0.0, 0.0]), dtype=float)
    accel_bias = np.asarray(cfg.get("accel_bias_m_s2", [0.0, 0.0, 0.0]), dtype=float)
    rw_std = float(cfg.get("bias_random_walk_std", 0.0))
    bias_rw = np.cumsum(rng.normal(0.0, rw_std * np.sqrt(dt), size=(len(t), 6)), axis=0)
    gyro += gyro_bias + bias_rw[:, :3]
    accel += accel_bias + bias_rw[:, 3:]

    ar1_alpha = float(cfg.get("ar1_alpha", 0.0))
    ar1_std = float(cfg.get("ar1_std", 0.0))
    gyro += _ar1_noise(rng, gyro.shape, ar1_alpha, ar1_std)
    accel += _ar1_noise(rng, accel.shape, ar1_alpha, ar1_std * 9.0)
    gyro += rng.normal(0.0, float(cfg.get("gyro_noise_std", 0.0)), size=gyro.shape)
    accel += rng.normal(0.0, float(cfg.get("accel_noise_std", 0.0)), size=accel.shape)

    temp = high["temperature_c"].to_numpy(dtype=float)
    temp_delta = (temp - np.nanmean(temp))[:, None]
    temp_cfg = cfg.get("temperature_drift", {})
    gyro += float(temp_cfg.get("gyro_per_c", 0.0)) * temp_delta
    accel += float(temp_cfg.get("accel_per_c", 0.0)) * temp_delta

    labels = high["motion_label"].astype(str).to_numpy()
    vib_mask = (labels == "high_dynamic_vibration").astype(float)[:, None]
    gyro += vib_mask * 0.01 * np.abs(gyro_high)
    accel += vib_mask * 0.03 * np.square(np.tanh(accel_high / 9.80665))

    quant = cfg.get("quantization", {})
    gyro_q = float(quant.get("gyro", 0.0))
    accel_q = float(quant.get("accel", 0.0))
    if gyro_q > 0:
        gyro = np.round(gyro / gyro_q) * gyro_q
    if accel_q > 0:
        accel = np.round(accel / accel_q) * accel_q

    clip = cfg.get("clipping", {})
    gyro_clip = float(clip.get("gyro", np.inf))
    accel_clip = float(clip.get("accel", np.inf))
    gyro = np.clip(gyro, -gyro_clip, gyro_clip)
    accel = np.clip(accel, -accel_clip, accel_clip)

    paired = high.copy()
    paired["gx_low"] = gyro[:, 0]
    paired["gy_low"] = gyro[:, 1]
    paired["gz_low"] = gyro[:, 2]
    paired["ax_low"] = accel[:, 0]
    paired["ay_low"] = accel[:, 1]
    paired["az_low"] = accel[:, 2]
    ordered = [
        "timestamp_s",
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
        "temperature_c",
        "motion_label",
        "sequence_id",
    ]
    return paired[ordered]
