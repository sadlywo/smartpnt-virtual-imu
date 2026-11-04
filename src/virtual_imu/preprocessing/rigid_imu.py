"""Rigid-body rotation and causal IMU lever-arm compensation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from virtual_imu.io.multi_imu_calibration import (
    RigidTransform,
)
from virtual_imu.io.multi_imu_calibration import (
    quaternion_wxyz_to_matrix as _quaternion_wxyz_to_matrix,
)

GYRO_COLUMNS = ["gx", "gy", "gz"]
ACCEL_COLUMNS = ["ax", "ay", "az"]


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a near-unit quaternion using the shared calibration convention."""

    rotation = _quaternion_wxyz_to_matrix(quaternion)
    validate_rotation(rotation)
    return rotation


def validate_rotation(rotation: np.ndarray) -> None:
    """Reject matrices that are not proper three-dimensional rotations."""

    value = np.asarray(rotation, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    orthogonality_error = float(np.linalg.norm(value.T @ value - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(value))
    if orthogonality_error > 1e-5:
        raise ValueError(
            f"rotation orthogonality error exceeds 1e-5: {orthogonality_error}"
        )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError(f"rotation determinant must be one, got {determinant}")


def rotate_imu(frame: pd.DataFrame, transform: RigidTransform) -> pd.DataFrame:
    """Rotate gyroscope and accelerometer vectors into the common body frame."""

    _require_imu_columns(frame)
    validate_rotation(transform.rotation_body_from_sensor)
    result = frame.copy()
    rotation = transform.rotation_body_from_sensor
    result.loc[:, GYRO_COLUMNS] = frame[GYRO_COLUMNS].to_numpy(dtype=float) @ rotation.T
    result.loc[:, ACCEL_COLUMNS] = frame[ACCEL_COLUMNS].to_numpy(dtype=float) @ rotation.T
    result.attrs.update(frame.attrs)
    result.attrs.update(
        {
            "coordinate_frame": "body",
            "calibration_id": transform.calibration_id,
            "rotation_applied": True,
        }
    )
    return result


def causal_angular_kinematics(
    gyro: np.ndarray,
    rate_hz: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Causally estimate body angular velocity and angular acceleration."""

    values = np.asarray(gyro, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("gyro must be a finite array with shape (rows, 3)")
    rate = float(rate_hz)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_hz must be finite and positive")

    cutoff_value = config.get("gyro_lowpass_hz")
    delay_s = 0.0
    if cutoff_value is None:
        omega = values.copy()
    else:
        cutoff_hz = float(cutoff_value)
        order = int(config.get("gyro_lowpass_order", 4))
        if order <= 0 or not 0.0 < cutoff_hz < rate / 2.0:
            raise ValueError("gyro low-pass order/cutoff must be valid below Nyquist")
        sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=rate, output="sos")
        omega = signal.sosfilt(sos, values, axis=0)
        numerator, denominator = signal.sos2tf(sos)
        _, group_delay = signal.group_delay(
            (numerator, denominator),
            w=np.asarray([1e-6]),
        )
        delay_s = max(0.0, float(group_delay[0]) / rate)

    window = int(config.get("angular_acceleration_window", 9))
    polynomial_order = int(config.get("angular_acceleration_polynomial_order", 2))
    if window < 3 or window % 2 == 0 or polynomial_order < 1 or polynomial_order >= window:
        raise ValueError(
            "angular acceleration window must be odd and exceed polynomial order"
        )
    coefficients = signal.savgol_coeffs(
        window,
        polynomial_order,
        deriv=1,
        delta=1.0 / rate,
        pos=window - 1,
        use="dot",
    )
    if len(omega) == 0:
        alpha = omega.copy()
    else:
        padded = np.pad(omega, ((window - 1, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=window,
            axis=0,
        )
        alpha = np.einsum("ncw,w->nc", windows, coefficients)
    return omega, alpha, delay_s


def compensate_lever_arm(
    frame: pd.DataFrame,
    transform: RigidTransform,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Shift one sensor's acceleration to the body origin without target data."""

    result = rotate_imu(frame, transform)
    rate_hz = _sample_rate_hz(result, config)
    omega, alpha, delay_s = causal_angular_kinematics(
        result[GYRO_COLUMNS].to_numpy(dtype=float),
        rate_hz,
        config,
    )
    lever_arm = transform.position_body_to_sensor_m
    tangential = np.cross(alpha, lever_arm)
    centripetal = np.cross(omega, np.cross(omega, lever_arm))
    sensor_acceleration = result[ACCEL_COLUMNS].to_numpy(dtype=float)
    result.loc[:, ACCEL_COLUMNS] = sensor_acceleration - tangential - centripetal
    result.attrs.update(
        {
            "lever_arm_compensated": True,
            "lever_arm_m": lever_arm.tolist(),
            "kinematics_rate_hz": rate_hz,
            "causal_filter_delay_s": delay_s,
            "angular_acceleration_method": "causal_endpoint_savgol",
        }
    )
    return result


def _sample_rate_hz(frame: pd.DataFrame, config: dict[str, Any]) -> float:
    configured = config.get("rate_hz")
    if configured is not None:
        rate_hz = float(configured)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("configured rate_hz must be finite and positive")
        return rate_hz
    if "timestamp_s" not in frame.columns or len(frame) < 2:
        raise ValueError("At least two timestamps are required to infer IMU rate")
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    differences = np.diff(timestamps)
    if not np.isfinite(timestamps).all() or np.any(differences <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing before compensation")
    return float(1.0 / np.median(differences))


def _require_imu_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in [*GYRO_COLUMNS, *ACCEL_COLUMNS] if column not in frame]
    if missing:
        raise ValueError(f"IMU frame is missing columns: {missing}")
