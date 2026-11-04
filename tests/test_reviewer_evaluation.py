"""Final-only evaluation, rotated splits, and validation-only baseline tuning."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.eval import reviewer_evaluation as reviewer
from virtual_imu.eval.formal_evaluation import HIGH, LOW
from virtual_imu.models import make_model
from virtual_imu.preprocessing.normalization import compute_standardization
from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SMARTPNT_GT_POSITION_COLUMNS,
    SmartPNTGTWindowDataset,
)


def _fixture(tmp_path, monkeypatch, task):
    root = tmp_path / "pipeline"
    root.mkdir()
    search = {
        "ridge_alpha": [0.0001, 1.0],
        "fir_taps": [1, 3],
        "arx_taps": [1, 2],
        "fit_split": "train",
        "select_split": "validation",
    }
    (root / "matrix_config.yaml").write_text(
        yaml.safe_dump({"classical_search": search}), encoding="utf-8"
    )
    rng = np.random.default_rng(15)
    frames = {}
    roles = {"train": "Traj1", "validation": "Traj4", "test": "Traj3"}
    for number, split in enumerate(("train", "validation", "test")):
        t = np.arange(360) / 40
        frame = pd.DataFrame(
            {
                "timestamp_s": t + number * 20,
                "sequence_id": f"{split}_segment",
                "original_sequence_id": roles[split],
                "split": split,
                "dataset_id": "toy",
                "low_sensor": "low",
                "high_sensor": "high",
            }
        )
        if task == "paired":
            values = rng.normal(size=(360, 6))
            frame[LOW] = values
            frame[HIGH] = values * np.linspace(0.8, 1.2, 6) + 0.1
        else:
            frame[SMARTPNT_GT_IMU_COLUMNS] = rng.normal(size=(360, 6))
            frame[SMARTPNT_GT_POSITION_COLUMNS] = np.column_stack([t, t**2, t * 0.1])
            for name in ("gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps", "gt_heading_deg"):
                frame[name] = 0.0
        frames[split] = frame
    data = {
        "window_size": 80,
        "stride": 40,
        "rate_hz": 40.0,
        "batch_size": 3,
        "load_splits": ["train", "validation"],
        "minimum_sequence_samples": 320,
    }
    for key in ("dataset_manifest", "split_manifest", "strict_manifest", "assignment_manifest"):
        path = root / f"{key}.json"
        path.write_text("{}", encoding="utf-8")
        data[key] = str(path)
    if task == "paired":
        data["feature_columns"] = LOW
        feature_stats = compute_standardization(frames["train"], LOW)
        residual = pd.DataFrame(
            frames["train"][HIGH].to_numpy() - frames["train"][LOW].to_numpy(), columns=HIGH
        )
        target_stats = compute_standardization(residual, HIGH)
        extra = {"target_mode": "residual", "feature_columns": LOW, "target_columns": HIGH}
    else:
        data.update(
            label_mode="position_delta",
            normalize_features=True,
            normalize_targets=True,
            input_columns=SMARTPNT_GT_IMU_COLUMNS,
            train_sequences=[roles["train"]],
            val_sequences=[roles["validation"]],
            test_sequences=[roles["test"]],
        )
        train = SmartPNTGTWindowDataset(frames["train"], 80, 40, label_mode="position_delta")
        feature_stats, target_stats = train.feature_stats, train.target_stats
        extra = {
            "label_mode": "position_delta",
            "target_layout": "window",
            "prediction_head": "last",
            "feature_columns": SMARTPNT_GT_IMU_COLUMNS,
            "target_columns": train.target_columns,
            "split_sequences": {
                "train": [roles["train"]],
                "val": [roles["validation"]],
                "test": [roles["test"]],
            },
        }
    model_config = {
        "type": "tcn",
        "in_channels": 6,
        "out_channels": 6 if task == "paired" else 3,
        "hidden_channels": 4,
        "levels": 1,
        "kernel_size": 3,
        "dropout": 0.0,
        **(
            {"target_mode": "residual"}
            if task == "paired"
            else {"target_layout": "window", "prediction_head": "last"}
        ),
    }
    config = {
        "seed": 1,
        "data": data,
        "model": model_config,
        "training": {
            "checkpoint_metric": "val_grouped_score" if task == "paired" else "val_vector_rmse"
        },
        "reviewer_protocol": {
            "protocol_id": f"toy_{task}",
            "candidate_id": "tcn",
            "model_family": "tcn",
            "phase": "final_tuned",
            "variant": "tuned",
            "selection_scope": "validation_only",
        },
    }
    model = make_model(model_config)
    checkpoint = {
        "config": copy.deepcopy(config),
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "feature_stats": feature_stats,
        "target_stats": target_stats,
        "checkpoint_metric": config["training"]["checkpoint_metric"],
        "best_epoch": 1,
        "best_metric": 1.0,
        **extra,
    }

    def save(name):
        run = root / "experiments" / name
        run.mkdir(parents=True)
        (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        checkpoint["config"] = copy.deepcopy(config)
        torch.save(checkpoint, run / "best.ckpt")
        (run / "provenance.json").write_text("{}", encoding="utf-8")
        return run

    monkeypatch.setattr(reviewer, "verify_run_provenance", lambda *args: None)
    monkeypatch.setattr(reviewer, "snapshot_unified_training_inputs", lambda *args: {})
    monkeypatch.setattr(
        reviewer,
        "load_training_split_manifest",
        lambda *args, **kwargs: SimpleNamespace(frames=frames),
    )

    def nav_loader(actual):
        assert "load_splits" not in actual
        return pd.concat(list(frames.values()), ignore_index=True)

    monkeypatch.setattr(reviewer, "unified_navigation_frame", nav_loader)
    return root, frames, config, checkpoint, save


@pytest.mark.parametrize("task", ["paired", "navigation"])
def test_final_evaluator_writes_actual_predictions_and_validation_tuned_baselines(
    tmp_path, monkeypatch, task
):
    root, frames, config, _checkpoint, save = _fixture(tmp_path, monkeypatch, task)
    run = save("first")
    report = reviewer.evaluate_reviewer_run(task, run, device="cpu")
    assert set(report["splits"]) == {"validation", "test"}
    assert report["reviewer_protocol"]["phase"] == "final_tuned"
    assert report["classical_cache"]["selection_split"] == "validation"
    state = json.loads((run / "evaluation/baseline_train_state.json").read_text(encoding="utf-8"))
    assert state["fit_split"] == "train"
    for split in ("validation", "test"):
        prediction = pd.read_parquet(run / "evaluation" / f"{split}_predictions.parquet")
        assert len(prediction) > 0
        assert prediction.sequence_id.str.startswith(split).all()
        if task == "paired":
            methods = report["splits"][split]["pairs"][0]["metrics"]
            assert set(methods) == {
                "model",
                "identity",
                "bias_mean",
                "affine",
                "ridge_affine",
                "learned_fir_causal",
                "arx_linear",
                "ridge_tuned",
                "fir_tuned",
                "arx_tuned",
            }
        else:
            assert set(report["splits"][split]["metrics"]) == {
                "model",
                "zero",
                "train_mean",
                "ridge",
                "ridge_tuned",
            }
            assert (
                prediction.original_sequence_id.unique().tolist()
                == config["data"]["val_sequences" if split == "validation" else "test_sequences"]
            )
    with pytest.raises(FileExistsError):
        reviewer.evaluate_reviewer_run(task, run, device="cpu")
    cache = list((root / "classical").rglob("*.json"))
    assert len(cache) == 1
    cached_bytes = cache[0].read_bytes()
    first_test_predictions = pd.read_parquet(run / "evaluation/test_predictions.parquet")
    if task == "paired":
        frames["test"][HIGH] += 10000
    else:
        frames["test"][SMARTPNT_GT_POSITION_COLUMNS] *= -100
    second_run = save("second")
    second = reviewer.evaluate_reviewer_run(task, second_run, device="cpu")
    assert second["classical_cache"]["reused"] is True
    assert cache[0].read_bytes() == cached_bytes
    assert len(list((root / "classical").rglob("*.json"))) == 1
    second_test_predictions = pd.read_parquet(second_run / "evaluation/test_predictions.parquet")
    model_columns = [
        column
        for column in first_test_predictions
        if column.startswith("model_") or column.startswith("pred_delta_")
    ]
    assert model_columns
    np.testing.assert_array_equal(
        first_test_predictions[model_columns], second_test_predictions[model_columns]
    )


def test_search_phase_is_rejected_before_data_loading_or_artifact_creation(tmp_path, monkeypatch):
    _, _, config, _, save = _fixture(tmp_path, monkeypatch, "paired")
    config["reviewer_protocol"].update(phase="search", variant="search")
    run = save("search")
    monkeypatch.setattr(
        reviewer,
        "load_training_split_manifest",
        lambda *a, **k: pytest.fail("search must not load test"),
    )
    with pytest.raises(ValueError, match="final"):
        reviewer.evaluate_reviewer_run("paired", run, device="cpu")
    assert not (run / "evaluation").exists()


def test_ridge_selection_accepts_only_train_and_validation_labels():
    rng = np.random.default_rng(6)
    train = rng.normal(size=(50, 4))
    validation = rng.normal(size=(30, 4))
    train_y = train[:, :3] * 2
    val_y = validation[:, :3] * 2
    selected = reviewer.select_ridge_train_validation(
        train, train_y, validation, val_y, [0.0001, 1000.0], task="navigation"
    )
    assert selected["alpha"] == 0.0001
    assert selected["fit_samples"] == 50
    assert selected["selection_samples"] == 30
    assert len(selected["history"]) == 2


def test_paired_contexts_share_identical_covered_rows_and_classical_cache(tmp_path, monkeypatch):
    root, _, config, _, save = _fixture(tmp_path, monkeypatch, "paired")
    counts = []
    for length in (80, 160, 320):
        config["data"]["window_size"] = length
        report = reviewer.evaluate_reviewer_run("paired", save(f"context{length}"), device="cpu")
        counts.append(report["splits"]["validation"]["samples"])
        assert report["classical_cache"]["reused"] is (length != 80)
    assert counts == [360, 360, 360]
    assert len(list((root / "classical").rglob("*.json"))) == 1


def test_classical_cache_state_tampering_is_rejected(tmp_path, monkeypatch):
    root, _, _, _, save = _fixture(tmp_path, monkeypatch, "navigation")
    reviewer.evaluate_reviewer_run("navigation", save("first"), device="cpu")
    path = next((root / "classical").rglob("*.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["states"]["ridge_tuned"]["alpha"] = 987
    path.write_text(json.dumps(payload), encoding="utf-8")
    run = save("second")
    with pytest.raises(ValueError, match="checksum"):
        reviewer.evaluate_reviewer_run("navigation", run, device="cpu")
    assert not (run / "evaluation").exists()
