from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from virtual_imu.io.multi_imu_calibration import RigidTransform
from virtual_imu.preprocessing.rigid_imu import (
    causal_angular_kinematics,
    compensate_lever_arm,
    quaternion_wxyz_to_matrix,
    rotate_imu,
    validate_rotation,
)


def test_validate_rotation_accepts_identity_and_known_quaternion() -> None:
    validate_rotation(np.eye(3))
    rotation = quaternion_wxyz_to_matrix([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    validate_rotation(rotation)

    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "rotation",
    [
        np.diag([1.0, 1.0, -1.0]),
        np.array([[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    ],
)
def test_validate_rotation_rejects_reflection_and_nonorthogonal(rotation: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_rotation(rotation)


def test_quaternion_allows_small_normalization_error_only() -> None:
    validate_rotation(quaternion_wxyz_to_matrix([1.0005, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="norm"):
        quaternion_wxyz_to_matrix([1.002, 0.0, 0.0, 0.0])


def test_rotate_imu_rotates_both_vector_triples() -> None:
    rotation = quaternion_wxyz_to_matrix([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    transform = RigidTransform(rotation, np.zeros(3), "known_z90")
    frame = _frame(
        gyro=np.array([[1.0, 0.0, 0.0]]),
        accel=np.array([[0.0, 1.0, 0.0]]),
        rate_hz=100.0,
    )

    rotated = rotate_imu(frame, transform)

    np.testing.assert_allclose(rotated[["gx", "gy", "gz"]], [[0.0, 1.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(rotated[["ax", "ay", "az"]], [[-1.0, 0.0, 0.0]], atol=1e-12)
    assert rotated.attrs["calibration_id"] == "known_z90"


def test_constant_omega_lever_arm_compensation_removes_centripetal_term() -> None:
    rows = 20
    omega = np.tile([0.0, 0.0, 2.0], (rows, 1))
    sensor_accel = np.tile([-4.0, 0.0, 0.0], (rows, 1))
    frame = _frame(omega, sensor_accel, rate_hz=100.0)
    transform = RigidTransform(np.eye(3), np.array([1.0, 0.0, 0.0]), "lever_x")

    corrected = compensate_lever_arm(frame, transform, _no_lowpass_config())

    np.testing.assert_allclose(corrected[["ax", "ay", "az"]], 0.0, atol=1e-8)


def test_linear_omega_lever_arm_compensation_removes_tangential_term() -> None:
    rate_hz = 100.0
    timestamps = np.arange(30) / rate_hz
    omega = np.column_stack([np.zeros(30), np.zeros(30), timestamps])
    alpha = np.tile([0.0, 0.0, 1.0], (30, 1))
    r = np.array([1.0, 0.0, 0.0])
    sensor_accel = np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
    frame = _frame(omega, sensor_accel, rate_hz=rate_hz)
    transform = RigidTransform(np.eye(3), r, "linear_omega")

    corrected = compensate_lever_arm(frame, transform, _no_lowpass_config())

    np.testing.assert_allclose(
        corrected.loc[8:, ["ax", "ay", "az"]],
        0.0,
        atol=1e-8,
    )


def test_causal_kinematics_has_no_response_before_gyro_step() -> None:
    gyro = np.zeros((200, 3))
    gyro[100:, 2] = 1.0
    config = {
        "gyro_lowpass_order": 4,
        "gyro_lowpass_hz": 20.0,
        "angular_acceleration_window": 9,
        "angular_acceleration_polynomial_order": 2,
    }

    omega, alpha, delay_s = causal_angular_kinematics(gyro, 200.0, config)

    np.testing.assert_array_equal(omega[:100], 0.0)
    np.testing.assert_allclose(alpha[:100], 0.0, atol=1e-12)
    assert delay_s > 0.0


def test_compensation_signature_and_result_do_not_accept_target_stream() -> None:
    assert list(inspect.signature(compensate_lever_arm).parameters) == [
        "frame",
        "transform",
        "config",
    ]
    frame = _frame(np.zeros((10, 3)), np.zeros((10, 3)), rate_hz=100.0)
    transform = RigidTransform(np.eye(3), np.zeros(3), "identity")

    result_a = compensate_lever_arm(frame, transform, _no_lowpass_config())
    result_b = compensate_lever_arm(frame.copy(), transform, _no_lowpass_config())

    pd.testing.assert_frame_equal(result_a, result_b)


def _frame(gyro: np.ndarray, accel: np.ndarray, rate_hz: float) -> pd.DataFrame:
    rows = len(gyro)
    return pd.DataFrame(
        {
            "timestamp_s": np.arange(rows, dtype=float) / rate_hz,
            "gx": gyro[:, 0],
            "gy": gyro[:, 1],
            "gz": gyro[:, 2],
            "ax": accel[:, 0],
            "ay": accel[:, 1],
            "az": accel[:, 2],
            "frame_id": ["imu"] * rows,
        }
    )


def _no_lowpass_config() -> dict[str, float | int | None]:
    return {
        "gyro_lowpass_order": 4,
        "gyro_lowpass_hz": None,
        "angular_acceleration_window": 5,
        "angular_acceleration_polynomial_order": 2,
    }
