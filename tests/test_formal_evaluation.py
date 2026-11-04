import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.eval import formal_evaluation as formal
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.models import make_model


def test_evaluation_rejects_changed_training_manifest_before_scoring(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"version": 2}')
    run = tmp_path / "run"
    run.mkdir()
    (run / "data_manifest.json").write_bytes(manifest.read_bytes())
    reference = {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (run / "provenance.json").write_text(
        json.dumps({"unified_dataset": {"inputs": {"dataset_manifest": reference}}})
    )
    config = {"data": {"dataset_manifest": str(manifest)}}
    formal.verify_run_provenance(run, config, config)
    manifest.write_text('{"version": 3}')
    with pytest.raises(ValueError, match=r"changed|differ"):
        formal.verify_run_provenance(run, config, config)


def _frame(n=160):
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.normal(size=(n, 6)), columns=formal.LOW)
    for low, high in zip(formal.LOW, formal.HIGH, strict=True):
        frame[high] = frame[low] + 0.3
    frame["timestamp_s"] = np.arange(n) / 40
    frame["sequence_id"] = "segment_a"
    frame["dataset_id"] = "alimu"
    frame["low_sensor"] = "BMX160"
    frame["high_sensor"] = "VN100"
    frame["pair_id"] = "BMX160_to_VN100"
    frame["original_sequence_id"] = "physical_a"
    return frame


def test_vector_error_detects_direction_reversal_even_when_norms_match():
    reference = np.tile([1, 0, 0, 0, 0, 9.8], (20, 1))
    prediction = reference.copy()
    prediction[:, 0] *= -1
    result = formal.physical_error_metrics(prediction, reference)
    assert result["gyro_vector_rmse_radps"] == pytest.approx(2)
    assert result["accel_vector_rmse_mps2"] == 0
    assert result["axis_rmse"][0] == 2


def test_integral_diagnostics_reset_at_each_contiguous_segment():
    first = _frame(80)
    second = _frame(80)
    second["sequence_id"] = "segment_b"
    for frame, error in ((first, 1.0), (second, -1.0)):
        frame[formal.HIGH] = 0.0
        frame[formal.LOW] = error
        for low, virtual in zip(formal.LOW, formal.VIRTUAL, strict=True):
            frame[virtual] = frame[low]
    report = formal.evaluate_prediction_frame(pd.concat([first, second], ignore_index=True))
    assert len(report["sequences"]) == 2
    assert report["pairs"][0]["samples"] == 160
    for sequence in report["sequences"]:
        assert sequence["diagnostics"]["delta_gyro_integral_error_rad"] == pytest.approx(
            np.sqrt(3) * 79 / 40
        )


def test_classical_baselines_never_fit_target_split_labels():
    train = _frame(120)
    target = _frame(80)
    state = formal.fit_pair_baselines(train, fir_taps=3)
    predicted = formal.predict_pair_baselines(state, target)
    target[formal.HIGH] = 999999.0
    changed = formal.predict_pair_baselines(state, target)
    assert set(predicted) >= {
        "identity",
        "bias_mean",
        "affine",
        "ridge_affine",
        "learned_fir_causal",
    }
    for name in predicted:
        np.testing.assert_array_equal(predicted[name], changed[name])
    np.testing.assert_allclose(predicted["bias_mean"], target[formal.LOW].to_numpy() + 0.3)


def test_virtual_generator_preserves_dataset_reference_and_sequence_identity(tmp_path):
    torch.manual_seed(8)
    config = {
        "type": "tcn",
        "in_channels": 6,
        "out_channels": 6,
        "hidden_channels": 8,
        "levels": 1,
        "kernel_size": 3,
        "dropout": 0,
    }
    model = make_model(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    path = tmp_path / "best.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "feature_columns": formal.LOW,
            "target_mode": "residual",
            "feature_stats": {"mean": [0] * 6, "std": [1] * 6},
            "target_stats": {"mean": [0] * 6, "std": [1] * 6},
            "config": {"data": {"window_size": 8, "stride": 4}},
        },
        path,
    )
    frame = _frame(20)
    result = VirtualIMUGenerator(path, device="cpu").run_frame(frame)
    for column in ("dataset_id", "high_sensor", "pair_id", "original_sequence_id"):
        assert result[column].tolist() == frame[column].tolist()
    np.testing.assert_allclose(result[formal.VIRTUAL].to_numpy(), frame[formal.LOW].to_numpy())
    formal_result, coverage = formal.predict_complete_windows(
        VirtualIMUGenerator(path, device="cpu"), frame, batch_size=3
    )
    assert len(formal_result) == 20
    assert coverage[0]["covered_samples"] == 20
    tail = _frame(22)
    formal_result, coverage = formal.predict_complete_windows(
        VirtualIMUGenerator(path, device="cpu"), tail
    )
    assert len(formal_result) == 20
    assert coverage[0]["uncovered_samples"] == 2
    np.testing.assert_allclose(
        formal_result[formal.VIRTUAL].to_numpy(), tail[formal.LOW].iloc[:20].to_numpy()
    )
