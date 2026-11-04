"""Formal endpoint-displacement evaluation, without touching the measured corpus."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.models import make_model
from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SMARTPNT_GT_POSITION_COLUMNS,
    SMARTPNT_GT_POSITION_DELTA_COLUMNS,
    SmartPNTGTWindowDataset,
)


def _formal():
    return importlib.import_module("virtual_imu.eval.formal_navigation")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment(original, name, offset, scale):
    t = np.arange(120, dtype=float) / 40
    frame = pd.DataFrame({
        "timestamp_s": offset + t,
        "sequence_id": name,
        "original_sequence_id": original,
        "dataset_id": "smartpnt_msf",
    })
    for axis, column in enumerate(SMARTPNT_GT_IMU_COLUMNS):
        frame[column] = scale + np.sin(t * (axis + 1)) + t * (axis + 1) / 10
    for axis, column in enumerate(SMARTPNT_GT_POSITION_COLUMNS):
        frame[column] = (axis + 1) * (scale * t + 0.2 * t**2)
    frame["gt_v_east_mps"] = scale + 0.4 * t
    frame["gt_v_north_mps"] = 2 * (scale + 0.4 * t)
    frame["gt_v_up_mps"] = 3 * (scale + 0.4 * t)
    frame["gt_heading_deg"] = scale * 30
    return frame


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    """Use real windows/models; replace only access to the immutable corpus."""
    formal = _formal()
    frame = pd.concat([
        _segment(original, name, offset, scale)
        for original, name, offset, scale in (
            ("Traj3", "train_a", 0, 1), ("Traj3", "train_b", 20, 2),
            ("Traj4", "val_a", 100, 3), ("Traj4", "val_b", 120, 4),
            ("Traj1", "test_a", 200, 5), ("Traj1", "test_b", 220, 6),
        )
    ], ignore_index=True)
    source = tmp_path / "frozen_inputs"
    source.mkdir()
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "dataset_id": "synthetic_formal_v2", "build_run": "runs/synthetic_build", "records": [],
    }), encoding="utf-8")
    split_manifest = source / "navigation_default.json"
    split_manifest.write_text(json.dumps({"splits": {
        "train": [{"original_sequence_id": "Traj3"}],
        "validation": [{"original_sequence_id": "Traj4"}],
        "test": [{"original_sequence_id": "Traj1"}],
    }}), encoding="utf-8")
    config = {
        "seed": 20261010,
        "data": {
            "dataset_manifest": str(manifest), "split_manifest": str(split_manifest),
            "label_mode": "position_delta", "rate_hz": 40.0,
            "window_size": 80, "stride": 40, "batch_size": 3, "num_workers": 0,
            "input_columns": list(SMARTPNT_GT_IMU_COLUMNS),
            "normalize_features": True, "normalize_targets": True,
            "train_sequences": ["Traj3"], "val_sequences": ["Traj4"],
            "test_sequences": ["Traj1"], "skipped_sequences": {"Traj2": "duplicate"},
        },
        "model": {"type": "tcn", "in_channels": 6, "out_channels": 3,
                  "hidden_channels": 4, "levels": 1, "kernel_size": 3, "dropout": 0,
                  "prediction_head": "last", "target_layout": "window"},
        "training": {"checkpoint_metric": "val_rmse", "velocity_integration_weight": 0.0,
                     "heading_integration_weight": 0.0},
    }
    train = SmartPNTGTWindowDataset(frame[frame.original_sequence_id == "Traj3"],
                                   window_size=80, stride=40, label_mode="position_delta")
    torch.manual_seed(7)
    model = make_model(config["model"])
    checkpoint = {
        "config": copy.deepcopy(config), "model_config": copy.deepcopy(config["model"]),
        "model_state_dict": model.state_dict(), "feature_stats": train.feature_stats,
        "target_stats": train.target_stats, "feature_columns": list(SMARTPNT_GT_IMU_COLUMNS),
        "target_columns": list(SMARTPNT_GT_POSITION_DELTA_COLUMNS), "label_mode": "position_delta",
        "target_layout": "window", "prediction_head": "last",
        "split_sequences": {"train": ["Traj3"], "val": ["Traj4"], "test": ["Traj1"]},
        "checkpoint_metric": "val_rmse", "best_epoch": 2, "best_metric": 1.25,
        "git_commit": "synthetic_commit",
    }
    provenance = {"git_commit": "synthetic_commit", "unified_dataset": {
        "dataset_id": "synthetic_formal_v2", "build_run": "runs/synthetic_build",
        "source_state_sha256": None,
        "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in (
            ("dataset_manifest", manifest), ("split_manifest", split_manifest),
        )},
    }}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def save_run(path=run_dir):
        path.mkdir(exist_ok=True)
        (path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        torch.save(checkpoint, path / "best.ckpt")
        (path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        shutil.copy2(manifest, path / "data_manifest.json")
        shutil.copy2(split_manifest, path / "split_manifest.json")

    calls = {"load": 0, "snapshot": 0, "model": []}

    def immutable_loader(data):
        calls["load"] += 1
        assert data == config["data"]
        for key, path in (("dataset_manifest", manifest), ("split_manifest", split_manifest)):
            if _sha256(path) != provenance["unified_dataset"]["inputs"][key]["sha256"]:
                raise ValueError("Frozen source hash mismatch")
        return frame.copy()

    def snapshot(data, destination):
        calls["snapshot"] += 1
        for key, name in (("dataset_manifest", "data_manifest.json"),
                          ("split_manifest", "split_manifest.json")):
            shutil.copy2(data[key], destination / name)
        return {"unified_dataset": copy.deepcopy(provenance["unified_dataset"])}

    def model_factory(model_config):
        result = make_model(model_config)

        def observe(module, args, output):
            calls["model"].append((module.training, torch.is_grad_enabled(), args[0].shape))

        result.register_forward_hook(observe)
        return result

    save_run()
    monkeypatch.setattr(formal, "unified_navigation_frame", immutable_loader)
    monkeypatch.setattr(formal, "snapshot_unified_training_inputs", snapshot)
    monkeypatch.setattr(formal, "make_model", model_factory)
    return {"run": run_dir, "frame": frame, "config": config, "checkpoint": checkpoint,
            "provenance": provenance, "save": save_run, "calls": calls, "source": source}


def test_displacement_metrics_measure_vector_direction_and_all_physical_axes():
    reference = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
    prediction = np.array([[-1, 2, 3], [3, 6, 3]], dtype=float)
    metrics = _formal().navigation_error_metrics(prediction, reference)
    assert metrics["windows"] == 2
    assert metrics["vector_rmse_m"] == pytest.approx(np.sqrt(12))
    assert metrics["component_rmse_m"] == pytest.approx(2)
    np.testing.assert_allclose(metrics["axis_rmse_m"], [2, np.sqrt(8), 0])
    np.testing.assert_allclose(metrics["axis_mae_m"], [2, 2, 0])
    np.testing.assert_allclose(metrics["axis_bias_m"], [0, 2, 0])
    assert metrics["horizontal_rmse_m"] == pytest.approx(np.sqrt(12))
    assert metrics["vertical_rmse_m"] == 0
    assert metrics["error_norm_median_m"] == pytest.approx(np.median([2, np.sqrt(20)]))
    assert metrics["error_norm_p95_m"] == pytest.approx(np.quantile([2, np.sqrt(20)], 0.95))


@pytest.mark.parametrize("prediction,reference", [
    (np.empty((0, 3)), np.empty((0, 3))),
    (np.zeros((2, 6)), np.zeros((2, 6))),
    (np.zeros((2, 3)), np.zeros((1, 3))),
    (np.full((1, 3), np.nan), np.zeros((1, 3))),
])
def test_displacement_metrics_reject_invalid_arrays(prediction, reference):
    with pytest.raises(ValueError):
        _formal().navigation_error_metrics(prediction, reference)


def test_baseline_predictions_need_only_low_six_axis_windows():
    formal = _formal()
    rng = np.random.default_rng(5)
    features = rng.normal(size=(12, 80, 6))
    targets = features.mean(axis=1)[:, :3] * 3 + [1, 2, 3]
    state = formal.fit_navigation_baselines(features, targets)
    predictions = formal.predict_navigation_baselines(state, features)
    assert state["fit_split"] == "train"
    assert state["fit_windows"] == 12
    assert state["alpha"] == 0.01
    assert set(predictions) == {"zero", "train_mean", "ridge"}
    np.testing.assert_array_equal(predictions["zero"], np.zeros((12, 3)))
    np.testing.assert_allclose(predictions["train_mean"], np.tile(targets.mean(axis=0), (12, 1)))
    assert np.mean((predictions["ridge"] - targets)**2) < 1e-5
    # Prediction has no label argument; unrelated held-out labels cannot enter the fit.
    changed_targets = targets + 100000
    assert changed_targets.mean() > 1000
    repeated = formal.predict_navigation_baselines(json.loads(json.dumps(state)), features)
    for method in predictions:
        np.testing.assert_array_equal(predictions[method], repeated[method])
    with pytest.raises(ValueError, match=r"six|6"):
        formal.predict_navigation_baselines(state, np.zeros((2, 80, 7)))


def test_evaluation_restores_real_checkpoint_and_preserves_contiguous_window_identity(frozen_run):
    formal = _formal()
    case = frozen_run
    report = formal.evaluate_navigation_run(case["run"], device="cpu")
    output = case["run"] / "evaluation"
    assert {"provenance.json", "validation_predictions.parquet", "test_predictions.parquet",
            "metrics.json", "navigation_metrics.csv", "baseline_train_state.json"} <= {
                path.name for path in output.iterdir()}
    assert report["endpoint_span_s"] == pytest.approx(1.975)
    assert "ENU" in report["scope"]
    assert "ATE" in report["scope"] and "RPE" in report["scope"]
    checkpoint = case["checkpoint"]
    expected_model = make_model(checkpoint["model_config"])
    expected_model.load_state_dict(checkpoint["model_state_dict"])
    expected_model.eval()
    for split, original in (("validation", "Traj4"), ("test", "Traj1")):
        predictions = pd.read_parquet(output / f"{split}_predictions.parquet")
        dataset = SmartPNTGTWindowDataset(
            case["frame"][case["frame"].original_sequence_id == original],
            window_size=80, stride=40, label_mode="position_delta",
            feature_stats=checkpoint["feature_stats"], target_stats=checkpoint["target_stats"],
        )
        x = torch.stack([dataset[i][0] for i in range(len(dataset))])
        with torch.no_grad():
            expected = expected_model(x)[:, :, -1].numpy()
        stats = checkpoint["target_stats"]
        expected = expected * np.asarray(stats["std"]) + stats["mean"]
        np.testing.assert_allclose(predictions[[f"pred_{c}" for c in SMARTPNT_GT_POSITION_DELTA_COLUMNS]],
                                   expected, atol=1e-6)
        assert len(predictions) == 4
        assert predictions.original_sequence_id.tolist() == [original] * 4
        assert predictions.groupby("sequence_id").size().tolist() == [2, 2]
        np.testing.assert_allclose(predictions.end_time_s - predictions.start_time_s, 1.975)
        assert predictions.start_index.tolist() == [0, 40, 0, 40]
        assert predictions.end_index.tolist() == [79, 119, 79, 119]
        for sequence in report["splits"][split]["sequences"]:
            assert sequence["original_sequence_id"] == original
            assert sequence["metrics"]["model"]["windows"] == 2
        result = report["splits"][split]
        assert result["windows"] == 4
        assert set(result["metrics"]) == {"model", "zero", "train_mean", "ridge"}
        truth = predictions[[f"true_{c}" for c in SMARTPNT_GT_POSITION_DELTA_COLUMNS]].to_numpy()
        assert result["metrics"]["model"] == formal.navigation_error_metrics(expected, truth)
    rows = pd.read_csv(output / "navigation_metrics.csv")
    assert len(rows) == 8
    assert set(rows.split) == {"validation", "test"}
    assert {"split", "method", "vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"} <= set(rows)
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["checkpoint"]["sha256"] == _sha256(case["run"] / "best.ckpt")
    assert provenance["training_provenance"] == case["provenance"]
    assert case["calls"]["load"] == 1
    assert case["calls"]["snapshot"] == 1
    assert all(not training and not gradients and shape[1:] == (6, 80)
               for training, gradients, shape in case["calls"]["model"])


@pytest.mark.parametrize("change,pattern", [
    (lambda c: c.update(label_mode="body_position_delta"), "position_delta"),
    (lambda c: c.update(target_layout="sequence"), "window"),
    (lambda c: c.update(prediction_head="mean"), "last"),
    (lambda c: c.update(target_columns=["delta_forward_m"] * 3), "target_columns"),
    (lambda c: c["feature_stats"].update(std=[0] * 6), "feature_stats"),
    (lambda c: c["target_stats"].update(mean=[1, 2]), "target_stats"),
    (lambda c: c["target_stats"]["mean"].__setitem__(0, 10000), "train|training"),
    (lambda c: c["feature_stats"]["mean"].__setitem__(0, 10000), "train|training"),
])
def test_evaluation_rejects_checkpoint_mode_or_normalization_mismatch(frozen_run, change, pattern):
    change(frozen_run["checkpoint"])
    frozen_run["save"]()
    with pytest.raises(ValueError, match=pattern):
        _formal().evaluate_navigation_run(frozen_run["run"], device="cpu")


@pytest.mark.parametrize("corruption", ["config", "snapshot", "frozen_input", "provenance"])
def test_evaluation_refuses_changed_config_or_frozen_data_snapshot(frozen_run, corruption):
    case = frozen_run
    if corruption == "config":
        config = copy.deepcopy(case["config"])
        config["seed"] += 1
        (case["run"] / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    elif corruption == "snapshot":
        (case["run"] / "data_manifest.json").write_text("{}", encoding="utf-8")
    elif corruption == "frozen_input":
        (case["source"] / "manifest.json").write_text("{}", encoding="utf-8")
    else:
        (case["run"] / "provenance.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _formal().evaluate_navigation_run(case["run"], device="cpu")
    assert not (case["run"] / "evaluation" / "test_predictions.parquet").exists()


@pytest.mark.parametrize("corruption", ["gap", "duplicate_time", "mixed", "nonfinite", "empty"])
def test_evaluation_rejects_invalid_or_mixed_segments(frozen_run, corruption):
    case = frozen_run
    frame = case["frame"]
    if corruption == "gap":
        frame.loc[frame.sequence_id.eq("val_a") & frame.index.to_series().ge(300), "timestamp_s"] += 1
    elif corruption == "duplicate_time":
        frame.loc[241, "timestamp_s"] = frame.loc[240, "timestamp_s"]
    elif corruption == "mixed":
        frame.loc[241, "original_sequence_id"] = "Traj1"
    elif corruption == "nonfinite":
        frame.loc[241, "ax_mps2"] = np.nan
    else:
        frame.drop(frame.index[frame.original_sequence_id.eq("Traj4")], inplace=True)
    with pytest.raises(ValueError):
        _formal().evaluate_navigation_run(case["run"], device="cpu")


def test_held_out_gt_changes_only_scores_not_model_or_baseline_predictions(frozen_run):
    formal = _formal()
    case = frozen_run
    formal.evaluate_navigation_run(case["run"], device="cpu")
    first_state = json.loads((case["run"] / "evaluation" / "baseline_train_state.json").read_text())
    first = pd.read_parquet(case["run"] / "evaluation" / "test_predictions.parquet")
    held_out = case["frame"].original_sequence_id.ne("Traj3")
    case["frame"].loc[held_out, SMARTPNT_GT_POSITION_COLUMNS] *= -1000
    case["frame"].loc[held_out, ["gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps", "gt_heading_deg"]] = 1e6
    second_run = case["run"].parent / "second_run"
    case["save"](second_run)
    formal.evaluate_navigation_run(second_run, device="cpu")
    second = pd.read_parquet(second_run / "evaluation" / "test_predictions.parquet")
    second_state = json.loads((second_run / "evaluation" / "baseline_train_state.json").read_text())
    assert first_state == second_state
    columns = [c for c in first if c.startswith(("pred_", "zero_", "train_mean_", "ridge_"))]
    pd.testing.assert_frame_equal(first[columns], second[columns])
    assert not np.allclose(first.true_delta_x_local_m, second.true_delta_x_local_m)


def test_existing_evaluation_is_never_silently_overwritten(frozen_run):
    output = frozen_run["run"] / "evaluation"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("previous report", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _formal().evaluate_navigation_run(frozen_run["run"], device="cpu")
    assert marker.read_text(encoding="utf-8") == "previous report"
