from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.diffusion.data import PairedSource, sha256
from virtual_imu.diffusion.evaluation import comparison_table
from virtual_imu.diffusion.pipeline import run_experiment
from virtual_imu.diffusion.training import load_trained


def test_group_noise_removes_each_axis_bias_before_combining():
    high = np.ones((10, 6))
    low = high + np.array([1, -1, 0, 2, -2, 0])
    table = comparison_table(low, high, {"raw": low}).set_index("component")
    assert table.loc["gyro", "noise_std"] == 0
    np.testing.assert_allclose(table.loc["gyro", "bias"], np.sqrt(2 / 3))
    assert table.loc["gx", "bias"] == 1
    assert table.loc["gy", "bias"] == -1


def test_metric_units_and_negative_improvement_are_preserved():
    high = np.ones((10, 6))
    low = high + 0.2
    t = comparison_table(low, high, {"raw": low, "ddpm": high + 0.4})
    rows = t[t.method.eq("ddpm")]
    np.testing.assert_allclose(rows.rmse_improvement_pct, -100)
    assert rows.loc[rows.component.eq("gyro"), "unit"].iloc[0] == "rad/s"
    assert rows.loc[rows.component.eq("accel"), "unit"].iloc[0] == "m/s^2"


def test_full_pipeline_locks_selection_before_test_and_restores_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    records = []
    for role in ("train", "validation", "test"):
        n = 64
        rng = np.random.default_rng(8)
        f = pd.DataFrame(
            {
                "timestamp_s": np.arange(n) / 40,
                "sequence_id": role,
                "dataset_id": "alimu",
                "pair_id": "BMX160_to_VN100",
                "low_sensor": "BMX160",
                "high_sensor": "VN100",
            }
        )
        for i, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
            high = np.sin(np.arange(n) / 6 + i)
            f[f"{axis}_low"] = high + 0.2 + rng.normal(0, 0.03, n)
            f[f"{axis}_high"] = high
        path = tmp_path / f"{role}.parquet"
        f.to_parquet(path)
        records.append(
            {
                "dataset_id": "alimu",
                "pair_id": "BMX160_to_VN100",
                "task": "paired_signal",
                "training_eligible": True,
                "split": role,
                "physical_group_id": role,
                "path": path.name,
                "sha256": sha256(path),
                "rows": n,
                "sequence_id": role,
                "low_sensor": "BMX160",
                "high_sensor": "VN100",
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"records": records, "rate_hz": 40}))
    config = {
        "experiment_id": "tiny",
        "seed": 5,
        "data": {
            "manifest": "manifest.json",
            "dataset_id": "alimu",
            "pairs": ["BMX160_to_VN100"],
            "window_size": 16,
            "stride": 8,
            "eval_stride": 8,
            "batch_size": 4,
        },
        "model": {
            "condition_channels": 7,
            "d_model": 16,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
        },
        "diffusion": {"timesteps": 4, "prediction_type": "x0", "schedule": "cosine"},
        "training": {
            "device": "cpu",
            "amp": False,
            "epochs": 3,
            "lr": 0.001,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "min_delta": 1000000,
            "validation_windows": 2,
            "cpu_threads": 1,
        },
        "search": {"trials": [{}, {"training": {"lr": 0.002}}]},
        "evaluation": {"draws": 2},
        "output": {"runs_dir": "runs"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    original = PairedSource.load
    test_reads = []

    def guarded_load(self, role):
        if role == "test":
            assert (tmp_path / "runs/tiny/BMX160_to_VN100/selection.json").exists()
            test_reads.append(role)
        return original(self, role)

    monkeypatch.setattr(PairedSource, "load", guarded_load)
    import virtual_imu.diffusion.evaluation as evaluation

    original_sample = evaluation.sample_windows
    sample_draws = []

    def tracked_sample(model, dataset, device, batch_size, seed, draws=1, amp=False):
        sample_draws.append(draws)
        return original_sample(model, dataset, device, batch_size, seed, draws, amp)

    monkeypatch.setattr(evaluation, "sample_windows", tracked_sample)
    run = run_experiment(path, tmp_path, "tiny")
    assert sample_draws and set(sample_draws) == {2}
    assert len(sample_draws) == 6  # Four early-stop checks plus full validation and test.
    assert test_reads == ["test"]
    assert (run / "complete.json").exists()
    table = pd.read_csv(run / "comparison.csv")
    assert len(table) == 24
    selected = json.loads((run / "BMX160_to_VN100/selection.json").read_text(encoding="utf-8"))
    search = json.loads((run / "BMX160_to_VN100/search_results.json").read_text(encoding="utf-8"))
    assert all(trial["epochs_ran"] == 2 and trial["early_stopped"] for trial in search)
    for trial in search:
        history = pd.read_csv(run / "BMX160_to_VN100/trials" / trial["trial"] / "history.csv")
        np.testing.assert_allclose(
            trial["validation_score"], history.validation_score.min(), rtol=1e-12
        )
    model, checkpoint = load_trained(selected["checkpoint"], torch.device("cpu"))
    assert checkpoint["epoch"] == selected["best_epoch"]
    a = model.sample(torch.zeros(1, 7, 16), generator=torch.Generator().manual_seed(4))
    restored, _ = load_trained(selected["checkpoint"], torch.device("cpu"))
    b = restored.sample(torch.zeros(1, 7, 16), generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(a, b)
    run_experiment(path, tmp_path, "tiny", resume=True)
    assert test_reads == ["test"]
    manifest_path = tmp_path / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(original_manifest + "\n", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="manifest differs"):
        run_experiment(path, tmp_path, "tiny", resume=True)
    manifest_path.write_text(original_manifest, encoding="utf-8")
    # Resume cannot silently combine results from a changed configuration.
    altered = copy.deepcopy(config)
    altered["seed"] += 1
    path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="Resume config"):
        run_experiment(path, tmp_path, "tiny", resume=True)
