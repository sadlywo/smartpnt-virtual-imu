from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.baselines import LOW_COLUMNS, TARGET_COLUMNS, VIRTUAL_COLUMNS
from virtual_imu.eval.penn_loso_comparison import (
    evaluate_penn_method,
    fit_penn_baselines,
    predict_penn_baselines,
    write_penn_comparison,
)


def _paired_frame(
    *,
    sequence_ids: tuple[str, ...] = ("train-a", "train-b"),
    samples_per_sequence: int = 48,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260712)
    chunks: list[pd.DataFrame] = []
    scales = np.array([0.8, 1.1, -0.6, 1.25, 0.7, 1.4])
    intercepts = np.array([0.2, -0.3, 0.5, 1.0, -0.8, 0.4])
    for sequence_index, sequence_id in enumerate(sequence_ids):
        low = rng.normal(size=(samples_per_sequence, 6))
        high = low * scales + intercepts
        chunk = pd.DataFrame(
            {
                "timestamp_s": np.arange(samples_per_sequence, dtype=float) * 0.01,
                "sequence_id": sequence_id,
            }
        )
        for channel in range(6):
            low_col = LOW_COLUMNS[channel]
            high_col = TARGET_COLUMNS[channel]
            chunk[low_col] = low[:, channel]
            chunk[high_col] = high[:, channel]
        chunk["source_order"] = sequence_index
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def _prediction_frame(samples: int = 320) -> pd.DataFrame:
    timestamp = np.arange(samples, dtype=float) * 0.01
    frame = pd.DataFrame({"timestamp_s": timestamp, "sequence_id": "bs"})
    phases = np.linspace(0.0, 1.0, 6)
    for index in range(6):
        low_col = LOW_COLUMNS[index]
        high_col = TARGET_COLUMNS[index]
        virtual_col = VIRTUAL_COLUMNS[index]
        high = np.sin((index + 1) * timestamp + phases[index]) + 0.1 * index
        frame[high_col] = high
        frame[low_col] = high + 0.2
        frame[virtual_col] = high + 0.05
    return frame


def test_fit_and_predict_four_train_fitted_baselines() -> None:
    train = _paired_frame()
    test = _paired_frame(sequence_ids=("bs",), samples_per_sequence=24)

    state = fit_penn_baselines(train, fir_taps=3, alpha=1e-10)
    predictions = predict_penn_baselines(state, test)

    assert set(state) == {"identity", "mean_bias", "channel_affine", "learned_fir_causal"}
    assert set(predictions) == set(state)
    assert all(values.shape == (len(test), 6) for values in predictions.values())
    np.testing.assert_allclose(predictions["identity"], test[LOW_COLUMNS])
    np.testing.assert_allclose(predictions["channel_affine"], test[TARGET_COLUMNS], atol=1e-8)


def test_predictions_and_fitted_state_ignore_test_targets() -> None:
    train = _paired_frame()
    test = _paired_frame(sequence_ids=("bs",), samples_per_sequence=24)
    mutated = test.copy()
    mutated[TARGET_COLUMNS] = mutated[TARGET_COLUMNS] * -1000.0 + 123.0

    state = fit_penn_baselines(train, fir_taps=2, alpha=1e-6)
    fitted_state = copy.deepcopy(state)
    original_predictions = predict_penn_baselines(state, test)
    mutated_predictions = predict_penn_baselines(state, mutated)

    assert json.dumps(state, sort_keys=True) == json.dumps(fitted_state, sort_keys=True)
    for method in state:
        np.testing.assert_allclose(original_predictions[method], mutated_predictions[method])


def test_channel_affine_has_only_independent_axis_coefficients() -> None:
    train = _paired_frame()
    changed = train.copy()
    changed[TARGET_COLUMNS[0]] += 7.0 * changed[LOW_COLUMNS[1]]

    state = fit_penn_baselines(train, fir_taps=2, alpha=1e-10)
    changed_state = fit_penn_baselines(changed, fir_taps=2, alpha=1e-10)
    affine = state["channel_affine"]

    assert np.asarray(affine["slopes"]).shape == (6,)
    assert np.asarray(affine["intercepts"]).shape == (6,)
    np.testing.assert_allclose(
        changed_state["channel_affine"]["slopes"][1:], affine["slopes"][1:]
    )
    np.testing.assert_allclose(
        changed_state["channel_affine"]["intercepts"][1:], affine["intercepts"][1:]
    )


def test_fir_history_resets_at_each_sequence_boundary() -> None:
    train = _paired_frame(samples_per_sequence=80)
    state = fit_penn_baselines(train, fir_taps=4, alpha=1e-6)
    first = _paired_frame(sequence_ids=("first",), samples_per_sequence=10)
    second = _paired_frame(sequence_ids=("second",), samples_per_sequence=10)
    second.loc[:, LOW_COLUMNS] += 50.0
    combined = pd.concat([first, second], ignore_index=True)

    combined_prediction = predict_penn_baselines(state, combined)["learned_fir_causal"]
    second_prediction = predict_penn_baselines(state, second)["learned_fir_causal"]

    np.testing.assert_allclose(combined_prediction[len(first) :], second_prediction)


def test_common_evaluation_includes_signal_spectral_and_preintegration_metrics() -> None:
    metrics = evaluate_penn_method(_prediction_frame())

    summary = metrics["summary"]
    assert summary["gyro_rmse"] == pytest.approx(0.05)
    assert summary["accel_mae"] == pytest.approx(0.05)
    assert summary["mean_axis_correlation"] == pytest.approx(1.0)
    assert summary["gyro_improvement_vs_identity_pct"] == pytest.approx(75.0)
    assert summary["accel_improvement_vs_identity_pct"] == pytest.approx(75.0)
    assert "gyro_snr_improvement_db" in metrics["signal"]["improvement_vs_identity"]
    assert "gyro_coherence" in metrics["spectral"]["method"]
    assert "imu_preintegration_residual" in metrics["preintegration"]["method"]


def test_write_penn_comparison_emits_per_method_and_common_outputs(tmp_path) -> None:
    frame = _prediction_frame()
    state = fit_penn_baselines(_paired_frame(), fir_taps=3, alpha=1e-6)
    test_df = frame.drop(columns=VIRTUAL_COLUMNS)
    predictions = predict_penn_baselines(state, test_df)

    outputs = write_penn_comparison(
        predictions=predictions,
        test_df=test_df,
        output_dir=tmp_path,
        baseline_state=state,
    )

    for method in predictions:
        assert (tmp_path / "predictions" / f"{method}.parquet").exists()
        assert (tmp_path / "metrics" / f"{method}.json").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison.json").exists()
    table = pd.read_csv(tmp_path / "comparison.csv")
    assert set(table["method"]) == set(predictions)
    assert {"parameter_count", "causal", "inference_class", "normalized_rmse_score"} <= set(table.columns)
    assert table["normalized_rmse_score"].is_monotonic_increasing
    assert table["rank"].tolist() == list(range(1, len(table) + 1))
    fir_row = table.loc[table["method"] == "learned_fir_causal"].iloc[0]
    assert fir_row["parameter_count"] == state["learned_fir_causal"]["parameter_count"]
    assert outputs["comparison_csv"] == tmp_path / "comparison.csv"
