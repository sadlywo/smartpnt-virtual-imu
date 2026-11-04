"""Geometric and leakage-boundary tests for independent-pose evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from virtual_imu.eval.downstream_attitude import (
    align_pose_ground_truth,
    evaluate_attitude_stream,
    evaluate_strapdown_stream,
    integrate_gyro_so3,
    integrate_strapdown,
    read_penn_pose,
)


def _truth(t: np.ndarray, rotations: Rotation) -> pd.DataFrame:
    matrices = rotations.as_matrix()
    result = pd.DataFrame({"timestamp_s": t})
    for i in range(3):
        for j in range(3):
            result[f"r{i}{j}"] = matrices[:, i, j]
    for name in ("px_m", "py_m", "pz_m"):
        result[name] = 0.0
    return result


def _stream(t: np.ndarray, omega: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp_s": t, "sequence_id": "example"})
    frame[["gx_low", "gy_low", "gz_low"]] = omega
    return frame


def test_noncommuting_body_rotations_are_composed_on_so3() -> None:
    t = np.array([0.0, 1.0, 2.0])
    gyro = np.array([[np.pi, 0, 0], [0, 0, 0], [0, np.pi, 0]])
    result = integrate_gyro_so3(t, gyro, np.eye(3))
    expected = Rotation.from_euler("x", 90, degrees=True) * Rotation.from_euler(
        "y", 90, degrees=True
    )
    np.testing.assert_allclose(result[-1], expected.as_matrix(), atol=1e-12)
    assert not np.allclose(result[-1], Rotation.from_rotvec([np.pi / 2, np.pi / 2, 0]).as_matrix())


def test_initial_attitude_is_used_once_then_gyro_propagates() -> None:
    t = np.arange(0, 1.01, 0.1)
    omega = np.tile([0, 0, 0.3], (len(t), 1))
    initial = Rotation.from_euler("xyz", [0.2, -0.4, 0.7])
    result = integrate_gyro_so3(t, omega, initial.as_matrix())
    expected = initial * Rotation.from_rotvec([0, 0, 0.3])
    np.testing.assert_allclose(result[-1], expected.as_matrix(), atol=1e-12)


def test_invalid_timestamps_and_nonrotations_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        integrate_gyro_so3(np.array([0, 0]), np.zeros((2, 3)), np.eye(3))
    with pytest.raises(ValueError, match="rotation"):
        integrate_gyro_so3(np.array([0, 1]), np.zeros((2, 3)), np.diag([1, 1, -1]))


def test_pose_parser_reads_actual_row_major_body_to_world_format(tmp_path) -> None:
    path = tmp_path / "pose.txt"
    path.write_text("0 1 0 0 4 0 1 0 5 0 0 1 6\n0.03 0 -1 0 7 1 0 0 8 0 0 1 9\n")
    parsed = read_penn_pose(path)
    np.testing.assert_allclose(parsed[["px_m", "py_m", "pz_m"]], [[4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(
        parsed.loc[1, [f"r{i}{j}" for i in range(3) for j in range(3)]]
        .to_numpy(dtype=float)
        .reshape(3, 3),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_ground_truth_slerp_never_crosses_pose_gaps() -> None:
    t = np.array([0.0, 0.05, 1.0, 1.05])
    truth = _truth(t, Rotation.from_rotvec(np.column_stack([np.zeros(4), np.zeros(4), t])))
    aligned = align_pose_ground_truth(np.array([0.025, 0.5, 1.025, 2.0]), truth, max_gap_s=0.1)
    assert aligned["gt_valid"].tolist() == [True, False, True, False]
    expected = Rotation.from_rotvec([0, 0, 0.025]).as_matrix()
    np.testing.assert_allclose(
        aligned.loc[0, [f"r{i}{j}" for i in range(3) for j in range(3)]]
        .to_numpy(dtype=float)
        .reshape(3, 3),
        expected,
        atol=1e-12,
    )


def test_fixed_windows_use_initial_truth_only_and_report_known_bias() -> None:
    t = np.arange(160) / 40
    stream = _stream(t, np.tile([0, 0, 0.1], (len(t), 1)))
    truth = _truth(t, Rotation.identity(len(t)))
    result = evaluate_attitude_stream(stream, truth, prefixes=["low"], window_samples=[80])
    row = result["windows"][0]
    assert row["count"] == 2
    assert row["horizon_s"] == pytest.approx(1.975)
    assert row["endpoint_rmse_deg"] == pytest.approx(np.degrees(0.1975))
    assert result["full_recordings"][0]["endpoint_deg"] == pytest.approx(np.degrees(0.3975))
    assert result["protocol"]["ground_truth_during_propagation"] is False


def test_physical_time_is_used_instead_of_prediction_availability() -> None:
    t = np.arange(80) / 40
    stream = _stream(t, np.tile([0, 0, 0.1], (len(t), 1)))
    stream["available_timestamp_s"] = t + 8
    stream["source_timestamp_s"] = t
    truth = _truth(
        t, Rotation.from_rotvec(np.column_stack([np.zeros(len(t)), np.zeros(len(t)), t * 0.1]))
    )
    result = evaluate_attitude_stream(stream, truth, prefixes=["low"], window_samples=[80])
    assert result["windows"][0]["endpoint_rmse_deg"] < 1e-8
    assert result["protocol"]["timestamp_column"] == "timestamp_s"


def test_explicit_source_clock_origin_maps_gt_without_fitting_lag() -> None:
    t = np.arange(80) / 40
    origin_s = 0.0049999
    stream = _stream(t, np.tile([0, 0, 0.1], (len(t), 1)))
    stream["source_timestamp_s"] = t + origin_s
    truth = _truth(
        t + origin_s,
        Rotation.from_rotvec(np.column_stack([np.zeros(len(t)), np.zeros(len(t)), t * 0.1])),
    )
    result = evaluate_attitude_stream(
        stream, truth, prefixes=["low"], window_samples=[80], gt_timestamp_origin_s=origin_s
    )
    assert result["support"]["common_valid_rows"] == 80
    assert result["windows"][0]["endpoint_rmse_deg"] < 1e-8
    assert result["protocol"]["gt_timestamp_origin_s"] == origin_s


def test_source_clock_origin_cannot_be_refitted_or_vary_within_recording() -> None:
    t = np.arange(80) / 40
    stream = _stream(t, np.zeros((len(t), 3)))
    stream["source_timestamp_s"] = t + 0.005
    truth = _truth(t + 0.005, Rotation.identity(len(t)))
    with pytest.raises(ValueError, match="source clock origin"):
        evaluate_attitude_stream(stream, truth, prefixes=["low"], gt_timestamp_origin_s=0.2)
    stream.loc[20, "source_timestamp_s"] += 0.001
    with pytest.raises(ValueError, match="constant"):
        evaluate_attitude_stream(stream, truth, prefixes=["low"])


def test_gap_or_recording_boundary_cannot_be_integrated_across() -> None:
    t = np.concatenate([np.arange(60) / 40, 5 + np.arange(60) / 40])
    stream = _stream(t, np.zeros((120, 3)))
    truth = _truth(t, Rotation.identity(len(t)))
    result = evaluate_attitude_stream(stream, truth, prefixes=["low"], window_samples=[80])
    assert result["windows"][0]["count"] == 0
    assert len(result["full_recordings"]) == 2


def test_all_methods_use_identical_finite_sample_support() -> None:
    t = np.arange(160) / 40
    stream = _stream(t, np.zeros((len(t), 3)))
    stream[["gx_virtual", "gy_virtual", "gz_virtual"]] = 0.0
    stream.loc[20, "gx_virtual"] = np.nan
    truth = _truth(t, Rotation.identity(len(t)))
    result = evaluate_attitude_stream(
        stream, truth, prefixes=["low", "virtual"], window_samples=[80]
    )
    assert [row["count"] for row in result["windows"]] == [1, 1]
    assert result["support"]["common_valid_rows"] == 159


def test_strapdown_rotates_specific_force_before_adding_world_gravity() -> None:
    t = np.arange(0, 2.01, 0.025)
    initial = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    gravity = np.array([0, 0, -9.80665])
    body_force = initial.T @ -gravity
    result = integrate_strapdown(
        t,
        np.zeros((len(t), 3)),
        np.tile(body_force, (len(t), 1)),
        initial,
        np.array([1, 2, 3]),
        np.zeros(3),
        gravity,
    )
    np.testing.assert_allclose(result["position_m"], np.tile([1, 2, 3], (len(t), 1)), atol=1e-12)
    np.testing.assert_allclose(result["velocity_mps"], 0, atol=1e-12)


def test_position_diagnostic_uses_gt_initial_velocity_once_and_real_position_error() -> None:
    truth_times = np.arange(0, 6.3, 0.025)
    truth = _truth(truth_times, Rotation.identity(len(truth_times)))
    truth["px_m"] = 2 * truth_times
    t = 0.2 + np.arange(240) / 40
    frame = _stream(t, np.zeros((len(t), 3)))
    frame[["ax_low", "ay_low", "az_low"]] = np.tile([0.1, 0, 9.80665], (len(t), 1))
    result = evaluate_strapdown_stream(frame, truth, prefixes=["low"], window_samples=[80])
    row = result["windows"][0]
    assert row["count"] == 3
    assert row["position_endpoint_rmse_m"] == pytest.approx(0.5 * 0.1 * 1.975**2)
    assert result["full_recordings"][0]["position_endpoint_m"] == pytest.approx(
        0.5 * 0.1 * 5.975**2
    )
    assert result["protocol"]["ground_truth_during_propagation"] is False
    assert (
        result["protocol"]["initial_velocity"] == "preceding_0.1s_optical_position_difference_once"
    )


def test_position_diagnostic_skips_window_without_preceding_gt_initial_velocity() -> None:
    t = np.arange(160) / 40
    truth = _truth(t, Rotation.identity(len(t)))
    frame = _stream(t, np.zeros((len(t), 3)))
    frame[["ax_low", "ay_low", "az_low"]] = np.tile([0, 0, 9.80665], (len(t), 1))
    result = evaluate_strapdown_stream(frame, truth, prefixes=["low"], window_samples=[80])
    assert result["windows"][0]["count"] == 1
    assert result["support"]["window_initializations_skipped"] == 1


def test_velocity_initializer_reads_no_optical_position_after_window_start() -> None:
    truth_times = np.arange(0, 2.3, 0.025)
    truth = _truth(truth_times, Rotation.identity(len(truth_times)))
    truth["px_m"] = 2 * truth_times
    truth.loc[truth["timestamp_s"] > 0.2, "px_m"] = 999.0
    t = 0.2 + np.arange(80) / 40
    frame = _stream(t, np.zeros((len(t), 3)))
    frame[["ax_low", "ay_low", "az_low"]] = np.tile([0, 0, 9.80665], (len(t), 1))
    result = evaluate_strapdown_stream(frame, truth, prefixes=["low"], window_samples=[80])
    np.testing.assert_allclose(
        result["window_details"][0]["initial_velocity_mps"], [2, 0, 0], atol=1e-10
    )


def test_velocity_initializer_cannot_bridge_gap_in_past_optical_support() -> None:
    truth_times = np.concatenate([np.array([0, 0.025]), np.arange(0.2, 2.3, 0.025)])
    truth = _truth(truth_times, Rotation.identity(len(truth_times)))
    t = 0.2 + np.arange(80) / 40
    frame = _stream(t, np.zeros((len(t), 3)))
    frame[["ax_low", "ay_low", "az_low"]] = np.tile([0, 0, 9.80665], (len(t), 1))
    result = evaluate_strapdown_stream(frame, truth, prefixes=["low"], window_samples=[80])
    assert result["windows"][0]["count"] == 0
    assert result["support"]["window_initializations_skipped"] == 1
