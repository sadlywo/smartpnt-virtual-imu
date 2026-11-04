"""Independent checks for the diagnostic clock and support handling."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from virtual_imu.preprocessing.fusion_alignment_audit import (
    bounded_interpolation,
    causal_supported_resample,
    fit_affine_clock,
    fit_time_shift,
    reference_pose_to_body,
)


def _matrix(values):
    flattened = np.asarray(values).ravel().tolist()
    return {"rows": 1, "cols": len(flattened), "data": flattened}


def test_pose_chain_matches_composed_homogeneous_transform():
    body_rotation = Rotation.from_euler("xyz", [13, -28, 67], degrees=True).as_matrix()
    local_rotation = Rotation.from_euler("xyz", [-8, 12, 33], degrees=True).as_matrix()
    body_translation = np.array([0.15, -0.1, 0.2])
    local_translation = np.array([-0.2, 0.11, 0.06])
    document = {
        "quaternion_sensor_bodyimu": _matrix(
            Rotation.from_matrix(body_rotation).as_quat()[[3, 0, 1, 2]]
        ),
        "quaternion_sensor_local": _matrix(
            Rotation.from_matrix(local_rotation).as_quat()[[3, 0, 1, 2]]
        ),
        "translation_sensor_bodyimu": _matrix(body_translation),
        "translation_sensor_local": _matrix(local_translation),
    }
    actual_rotation, actual_translation = reference_pose_to_body(document, "local")
    ref_body, ref_local = np.eye(4), np.eye(4)
    ref_body[:3, :3], ref_body[:3, 3] = body_rotation, body_translation
    ref_local[:3, :3], ref_local[:3, 3] = local_rotation, local_translation
    expected = np.linalg.inv(ref_body) @ ref_local
    np.testing.assert_allclose(actual_rotation, expected[:3, :3], atol=1e-12)
    np.testing.assert_allclose(actual_translation, expected[:3, 3], atol=1e-12)


def test_missing_time_is_not_compressed_or_interpolated():
    times = np.array([0.0, 0.01, 0.02, 0.1, 0.11])
    grid = np.arange(0, 0.111, 0.01)
    values, supported = bounded_interpolation(times, times[:, None], grid, 0.031)
    assert (~supported[3:10]).all()
    assert np.isnan(values[3:10]).all()
    assert values[10, 0] == pytest.approx(0.1)


def test_near_equal_rate_does_not_introduce_fir_delay():
    times = np.arange(1000) / 200.0145
    values = np.ones((len(times), 3))
    grid = np.arange(900) / 199.9954
    output, metrics = causal_supported_resample(times, values, grid, 200.0145, 199.9954, {})
    assert not metrics["anti_alias_applied"]
    assert metrics["group_delay_s"] == 0
    np.testing.assert_allclose(output, 1.0)


def test_fir_invalidates_warmup_and_post_gap_support():
    times = np.r_[np.arange(100) / 1000, 0.15 + np.arange(100) / 1000]
    values = np.ones((len(times), 1))
    grid = np.arange(0, 0.2, 0.001)
    output, metrics = causal_supported_resample(
        times, values, grid, 1000.0, 200.0, {"fir_taps": 63}
    )
    assert metrics["warmup_source_rows"] == 62
    assert np.isnan(output[grid < 0.031]).all()
    assert np.isnan(output[(grid > 0.13) & (grid < 0.175)]).all()
    np.testing.assert_allclose(output[(grid > 0.184) & (grid < 0.196)], 1.0, atol=1e-12)


def test_time_calibration_only_uses_fit_interval():
    times = np.arange(0, 20, 0.01)
    norm = 2 + np.sin(times * 2.4) + 0.5 * np.sin(times * 7.7)
    low = np.column_stack([norm, np.zeros((len(times), 2))])
    high = low.copy()
    shifts = np.arange(-0.2, 0.201, 0.01)
    fit_grid = np.arange(1.0, 7.0, 0.01)
    first = fit_time_shift(times - 0.12, low, times, high, fit_grid, shifts, 0.031, 0.031)
    high[times > 9] = 1000 - high[times > 9]
    second = fit_time_shift(times - 0.12, low, times, high, fit_grid, shifts, 0.031, 0.031)
    assert first[0] == pytest.approx(0.12)
    assert first == second


def test_affine_clock_recovers_known_drift_without_evaluation_targets():
    true_times = np.arange(0, 40, 0.005)
    norm = 2 + np.sin(1.2 * true_times + 0.07 * true_times**2) + 0.3 * np.sin(7.3 * true_times)
    values = np.column_stack([norm, np.zeros((len(true_times), 2))])
    low_times = (true_times - 0.181) / 1.0012
    config = {
        "diagnostic_rate_hz": 200,
        "fit_interval_s": [1, 25],
        "train_clock_blocks_s": [[1, 5], [6, 10], [11, 15], [16, 20], [21, 25]],
        "lag_search_s": [0.15, 0.23],
        "lag_search_step_s": 0.001,
    }
    result = fit_affine_clock(low_times, values, true_times, values, config, 0.016, 0.016)
    assert result["clock_scale"] == pytest.approx(1.0012, abs=4e-5)
    assert result["clock_offset_s"] == pytest.approx(0.181, abs=0.001)
    altered_targets = values.copy()
    altered_targets[true_times >= 29] = 1000
    repeated = fit_affine_clock(
        low_times, values, true_times, altered_targets, config, 0.016, 0.016
    )
    assert result == repeated
