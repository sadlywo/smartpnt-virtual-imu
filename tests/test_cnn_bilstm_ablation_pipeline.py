"""Synthetic CPU regressions; no experiment data or scientific fits are used."""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.ablations import data, pipeline, training


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/ablations/cnn_bilstm_shared.yaml").read_text(encoding="utf-8"))


def frame(dataset="alimu", sequence="record-a", rows=440, shift=0.0):
    index = np.arange(rows, dtype=float)
    low = shift + index[:, None] * np.arange(1, 7)[None, :] / 100.0
    result = pd.DataFrame(low, columns=data.LOW)
    result[data.HIGH] = 2.0 * low
    result["dataset_id"] = dataset
    result["pair_id"] = "pair-a"
    result["sequence_id"] = sequence
    result["original_sequence_id"] = sequence
    result["physical_group_id"] = sequence
    result["timestamp_s"] = index / 40.0
    return result


def stub_model_import(monkeypatch, builder=None):
    """Keep this independent of the separately reviewed model implementation."""
    module = types.ModuleType("virtual_imu.ablations.models")
    module.build_ablation_model = builder or (lambda *args: (torch.nn.Identity(), {}))
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_all_windows_predict_identical_center40_keys_labels_and_physical_values():
    source = pd.concat([frame(), frame(sequence="record-b", rows=399, shift=20)], ignore_index=True)
    windows = data.CommonWindows(source)
    expected = np.concatenate([np.arange(140, 300), np.arange(580, 660)])
    np.testing.assert_array_equal(windows.target_indices, expected)
    assert not windows.target_frame.duplicated(data.KEYS).any()
    x_full, y_full = windows.arrays(320)
    support = windows.support_sha256
    for width in [40, 80, 160, 320]:
        x, y = windows.arrays(width)
        np.testing.assert_array_equal(y, y_full)
        np.testing.assert_array_equal(x, x_full[:, :, (320-width)//2:(320+width)//2])
        predicted = training.predict_normalized(
            torch.nn.Identity(), torch.from_numpy(x_full), window=width,
            max_window=320, block_size=40, batch_size=2, amp=False,
        )
        np.testing.assert_allclose(windows.physical(predicted), windows.target_frame[data.HIGH], rtol=1e-6, atol=1e-5)
        assert windows.support_sha256 == support


def test_training_loader_fits_stats_only_on_train_and_never_requests_test(monkeypatch, config, tmp_path):
    train_frame, validation_frame = frame(), frame(sequence="validation", shift=10000)
    loaded = []

    class Source:
        def __init__(self, *args):
            self.dataset = types.SimpleNamespace(manifest_sha256=config["data"]["manifest_sha256"])

        def load(self, role):
            loaded.append(role)
            assert role != "test", "Training requested TEST"
            return {"train": train_frame, "validation": validation_frame}[role]

    monkeypatch.setattr(training, "SharedSource", Source)
    monkeypatch.setattr(training, "_CACHE", {})
    job = {"root": str(tmp_path), "protocol": "pooled", "config": config}
    _, train, validation, _ = training.data_for_job(job, torch.device("cpu"))
    assert loaded == ["train", "validation"]
    assert validation.stats is train.stats
    np.testing.assert_allclose(train.stats["feature_mean"], train_frame[data.LOW].mean())
    np.testing.assert_allclose(train.stats["residual_mean"], train_frame[data.HIGH].to_numpy().mean(0) - train_frame[data.LOW].to_numpy().mean(0))
    expected = (validation_frame[data.LOW].to_numpy() - train.stats["feature_mean"]) / train.stats["feature_std"]
    np.testing.assert_allclose(validation.x, expected, rtol=1e-6)
    training.data_for_job(job, torch.device("cpu"))
    assert loaded == ["train", "validation"]


def test_balanced_sampling_is_dataset_balanced_and_uses_an_independent_paired_rng():
    windows = data.CommonWindows(pd.concat([
        frame(rows=800), frame(dataset="penncosyvio", sequence="penn", rows=360),
    ], ignore_index=True))
    masses = [windows.weights[windows.anchor_datasets == name].sum() for name in np.unique(windows.anchor_datasets)]
    np.testing.assert_allclose(masses, [1.0, 1.0])
    before = torch.random.get_rng_state().clone()
    first = data.balanced_order(windows, 20261010, 3)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(19)  # A different model's initialization/dropout must not affect sampling.
    np.testing.assert_array_equal(first, data.balanced_order(windows, 20261010, 3))
    assert not np.array_equal(first, data.balanced_order(windows, 20261010, 4))


def test_frozen_matrix_contains_all_138_unique_seed_blocked_fits(config, tmp_path):
    jobs = pipeline.jobs_for_config(tmp_path, config, tmp_path / "formal")
    assert len(jobs) == len({job["id"] for job in jobs}) == 138
    assert sum(job["variant"] in config["variants"] for job in jobs) == 120
    assert sum(job["variant"] in config["supplemental"] for job in jobs) == 18
    assert all(job["window"] == 320 for job in jobs if job["variant"] in config["supplemental"])
    assert [job["id"] for job in jobs] == [job["id"] for job in pipeline.jobs_for_config(tmp_path, config, tmp_path / "formal")]


def test_best_checkpoint_tracks_exact_validation_minimum_while_patience_uses_min_delta(monkeypatch, config, tmp_path):
    cfg = copy.deepcopy(config)
    cfg["training"].update(device="cpu", amp=False, epochs=9, early_stopping_patience=2, min_delta=1e-4)
    cfg["data"]["batch_size"] = 2
    train = data.CommonWindows(frame(rows=360))
    validation = data.CommonWindows(frame(sequence="val", rows=360), stats=train.stats)
    tx, ty = train.arrays(320)
    vx, vy = validation.arrays(320)
    tensors = tuple(torch.from_numpy(item) for item in [tx, ty, vx, vy, tx.copy()])
    monkeypatch.setattr(training, "configure", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(training, "data_for_job", lambda *args: (None, train, validation, tensors))
    scores = iter([5.0, 4.0, 3.99995, 4.1])
    monkeypatch.setattr(training, "grouped_score", lambda *args: next(scores))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            return inputs * self.scale

    stub_model_import(monkeypatch, lambda *args: (Toy(), {}))
    job = dict(id="toy", root=str(tmp_path), protocol="alimu_only", config=cfg,
               directory=str(tmp_path / "fit"), variant="cnn_bilstm", window=80, seed=20261010)
    result = training.fit_job(job)
    assert result["actual_epochs"] == 4
    assert result["best_epoch"] == 3
    assert result["best_val_score"] == 3.99995
    saved = torch.load(tmp_path / "fit/best.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 3 and saved["val_score"] == 3.99995
    monkeypatch.setattr(training, "data_for_job", lambda *args: pytest.fail("A verified complete fit was retrained"))
    assert training.fit_job(job) == result
    stale = copy.deepcopy(job)
    stale["window"] = 160
    with pytest.raises(ValueError, match="configuration mismatch"):
        training.fit_job(stale)


@pytest.mark.parametrize("defect", ["missing_checkpoint", "wrong_checkpoint_hash", "wrong_job_id", "stale_config"])
def test_test_payload_is_blocked_until_every_fit_identity_and_checkpoint_is_verified(monkeypatch, config, tmp_path, defect):
    cfg = copy.deepcopy(config)
    cfg.update(protocols=["alimu_only"], seeds=[20261010], windows=[40, 80], variants=["cnn_bilstm"], supplemental=[])
    cfg["training"].update(device="cpu", amp=False)
    jobs = pipeline.jobs_for_config(tmp_path, cfg, tmp_path)
    records = []
    for job in jobs:
        directory = Path(job["directory"])
        directory.mkdir(parents=True)
        checkpoint = directory / "best.pt"
        torch.save({"job": job, "model": {}, "stats": data.CommonWindows(frame()).stats,
                    "epoch": 1, "val_score": 1.0}, checkpoint)
        records.append(dict(job_id=job["id"], job_config_sha256=training.json_digest(job),
                            checkpoint_sha256=training.sha(checkpoint), best_epoch=1,
                            actual_epochs=1, best_val_score=1.0))
    bad = Path(jobs[-1]["directory"]) / "best.pt"
    if defect == "missing_checkpoint":
        bad.unlink()
    elif defect == "wrong_checkpoint_hash":
        records[-1]["checkpoint_sha256"] = "0" * 64
    elif defect == "wrong_job_id":
        records[-1]["job_id"] = "unplanned-job"
    else:
        records[-1]["job_config_sha256"] = "stale-config"
    support_dir = tmp_path / "supports/alimu_only"
    support_dir.mkdir(parents=True)
    (support_dir / "statistics.json").write_text(json.dumps(data.CommonWindows(frame()).stats), encoding="utf-8")
    loaded = []

    class Source:
        def __init__(self, *args):
            self.dataset = types.SimpleNamespace(manifest_sha256=cfg["data"]["manifest_sha256"])

        def load(self, role):
            loaded.append(role)
            if role == "test":
                raise RuntimeError("REGRESSION: TEST opened before all fit/checkpoint identities were verified")
            return frame(sequence=role)

    stub_model_import(monkeypatch)
    monkeypatch.setattr(pipeline, "configure", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(pipeline, "SharedSource", Source)
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        pipeline.evaluate(tmp_path, cfg, tmp_path, jobs, records)
    assert "test" not in loaded


def test_default_preflight_does_not_reuse_formal_fit_directories(monkeypatch, config, tmp_path):
    cfg = copy.deepcopy(config)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    seen = []
    monkeypatch.setattr(pipeline, "fit_job", lambda job: seen.append(copy.deepcopy(job)) or {})
    pipeline.run(tmp_path, config_path, mode="preflight")
    formal = pipeline.jobs_for_config(tmp_path, cfg, tmp_path / cfg["output"]["runs_dir"] / cfg["experiment_id"])
    assert len(seen) == 1
    assert seen[0]["directory"] not in {job["directory"] for job in formal}, "Two-epoch preflight fit.json would poison formal resume"
