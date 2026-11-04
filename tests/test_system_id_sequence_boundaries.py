from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.baselines import (
    LOW_COLUMNS,
    TARGET_COLUMNS,
    _baseline_predictions,
    fit_system_id_baseline_state,
    predict_system_id_baselines,
    predict_system_id_from_state,
)


def _fit_frame(sequence_id: str, offset: float) -> pd.DataFrame:
    timestamp = np.arange(16, dtype=float) * 0.01
    base = offset + np.arange(len(timestamp), dtype=float) ** 2
    low = np.column_stack([base + channel * 0.25 for channel in range(6)])
    previous = np.vstack([low[[0]], low[:-1]])
    high = 0.6 * low + 0.4 * previous
    frame = pd.DataFrame({"timestamp_s": timestamp, "sequence_id": sequence_id})
    frame[LOW_COLUMNS] = low
    frame[TARGET_COLUMNS] = high
    return frame


def _lag_state() -> dict[str, object]:
    fir_weights = np.zeros((12, 6), dtype=float)
    fir_weights[6:12] = np.eye(6)
    arx_weights = np.zeros((18, 6), dtype=float)
    arx_weights[6:12] = np.eye(6)
    arx_weights[12:18] = 2.0 * np.eye(6)
    return {
        "fir_taps": 2,
        "arx_taps": 2,
        "fir_weights": fir_weights.tolist(),
        "fir_bias": np.zeros(6).tolist(),
        "arx_weights": arx_weights.tolist(),
        "arx_bias": np.zeros(6).tolist(),
    }


def test_fit_resets_each_sequence_and_sorts_it_by_timestamp() -> None:
    first = _fit_frame("first", offset=0.0)
    second = _fit_frame("second", offset=1000.0)
    ordered = pd.concat([first, second], ignore_index=True)
    scrambled = ordered.sample(frac=1.0, random_state=17).reset_index(drop=True)

    expected = fit_system_id_baseline_state(ordered, fir_taps=3, alpha=1e-4)
    actual = fit_system_id_baseline_state(scrambled, fir_taps=3, alpha=1e-4)

    for key in ("fir_weights", "fir_bias", "arx_weights", "arx_bias"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-9)


def test_prediction_resets_fir_and_arx_and_preserves_input_row_alignment() -> None:
    low = np.asarray(
        [
            [3, 13, 23, 33, 43, 53],
            [20, 30, 40, 50, 60, 70],
            [1, 11, 21, 31, 41, 51],
            [10, 20, 30, 40, 50, 60],
            [2, 12, 22, 32, 42, 52],
            [30, 40, 50, 60, 70, 80],
        ],
        dtype=float,
    )
    target = pd.DataFrame(
        {
            "timestamp_s": [2.0, 1.0, 0.0, 0.0, 1.0, 2.0],
            "sequence_id": ["a", "b", "a", "b", "a", "b"],
        }
    )
    target[LOW_COLUMNS] = low

    predictions = predict_system_id_from_state(_lag_state(), target)

    expected_fir = low[[4, 3, 2, 3, 2, 1]]
    expected_arx = 2.0 * low - expected_fir
    expected_arx[[2, 3]] = low[[2, 3]]
    np.testing.assert_allclose(predictions["learned_fir_causal"], expected_fir)
    np.testing.assert_allclose(predictions["arx_linear"], expected_arx)


def test_prediction_without_sequence_id_keeps_legacy_row_order_history() -> None:
    low = np.arange(24, dtype=float).reshape(4, 6)
    target = pd.DataFrame(low, columns=LOW_COLUMNS)

    predictions = predict_system_id_from_state(_lag_state(), target)

    previous = np.vstack([low[[0]], low[:-1]])
    np.testing.assert_allclose(predictions["learned_fir_causal"], previous)
    np.testing.assert_allclose(predictions["arx_linear"], previous + 2.0 * (low - previous))


def test_baseline_evaluator_uses_sequence_safe_fir_and_arx_predictions() -> None:
    train = pd.concat(
        [_fit_frame("train-a", offset=0.0), _fit_frame("train-b", offset=1000.0)],
        ignore_index=True,
    ).sample(frac=1.0, random_state=11)
    test = pd.concat(
        [_fit_frame("test-a", offset=100.0), _fit_frame("test-b", offset=-100.0)],
        ignore_index=True,
    ).sample(frac=1.0, random_state=23)

    expected = predict_system_id_baselines(train, test, fir_taps=3, alpha=1e-2)
    actual = _baseline_predictions(
        test,
        train,
        moving_average_window=3,
        butterworth_cutoff_hz=10.0,
        savgol_window=5,
        fir_taps=3,
        neural_epochs=0,
        neural_window_size=8,
        neural_stride=4,
    )

    np.testing.assert_allclose(actual["learned_fir_causal"], expected["learned_fir_causal"])
    np.testing.assert_allclose(actual["arx_linear"], expected["arx_linear"])


def test_sequence_id_without_timestamp_reports_clear_schema_error() -> None:
    target = pd.DataFrame(np.arange(24, dtype=float).reshape(4, 6), columns=LOW_COLUMNS)
    target["sequence_id"] = ["a", "a", "b", "b"]
    train = _fit_frame("train", offset=0.0).drop(columns="timestamp_s")

    with pytest.raises(ValueError, match=r"sequence_id.*timestamp_s"):
        predict_system_id_from_state(_lag_state(), target)
    with pytest.raises(ValueError, match=r"sequence_id.*timestamp_s"):
        fit_system_id_baseline_state(train)
