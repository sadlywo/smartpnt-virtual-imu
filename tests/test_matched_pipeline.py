from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.diffusion.data import PairedSource, sha256
from virtual_imu.diffusion.matched import execute_jobs, run_matched_experiment


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression")
def test_amp_overflow_skips_step_and_recovers(tmp_path, monkeypatch):
    from virtual_imu.diffusion.data import PairedWindows
    from virtual_imu.diffusion.matched_training import fit_job

    class OverflowOnce(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def training_loss(self, clean, condition):
            self.calls += 1
            # Finite FP16 loss, but the first scaled backward overflows FP16.
            if self.calls == 1:
                return (self.weight.half() * 2).float()
            return self.weight.square()

    frame = pd.DataFrame({"sequence_id": ["one"] * 24})
    for i, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
        frame[f"{axis}_low"] = np.sin(np.arange(24) / 3 + i)
        frame[f"{axis}_high"] = frame[f"{axis}_low"] + 0.1
    windows = PairedWindows(frame, 16, 8)
    tensors = tuple(torch.stack([windows[i][j] for i in range(2)]).cuda() for j in (0, 1))
    monkeypatch.setattr(
        "virtual_imu.diffusion.matched_training.data_for_job",
        lambda *args: (windows, windows, tensors),
    )
    monkeypatch.setattr(
        "virtual_imu.diffusion.matched_training.build_matched_model", lambda cfg: OverflowOnce()
    )
    monkeypatch.setattr(
        "virtual_imu.diffusion.matched_training.physical_prediction",
        lambda *args: (windows.low, np.ones(len(frame), dtype=bool)),
    )
    cfg = dict(
        seed=51,
        method="no_diffusion",
        data=dict(batch_size=1),
        training=dict(
            device="cuda",
            amp=True,
            epochs=1,
            lr=0.01,
            grad_clip_norm=1.0,
            validation_windows=2,
            min_delta=0,
            early_stopping_patience=1,
        ),
    )
    result = fit_job(
        dict(
            root=str(tmp_path),
            directory=str(tmp_path / "fit"),
            config=cfg,
            pair_id="synthetic",
            candidate=0,
            context={},
        )
    )
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert torch.isfinite(checkpoint["model"]["weight"])
    assert checkpoint["model"]["weight"] < 1
    assert next(iter(checkpoint["optimizer"]["state"].values()))["step"] == 1
    assert checkpoint["scaler"]["scale"] == 32768
    assert pd.read_csv(tmp_path / "fit/history.csv").amp_skipped_steps.tolist() == [1]


def test_all_selections_locked_before_test_and_resume_pins_inputs(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    records = []
    for role in ("train", "validation", "test"):
        n = 48
        frame = pd.DataFrame(
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
            target = np.sin(np.arange(n) / 5 + i)
            frame[f"{axis}_high"] = target
            frame[f"{axis}_low"] = target + 0.1 + 0.01 * np.cos(np.arange(n))
        path = tmp_path / f"{role}.parquet"
        frame.to_parquet(path)
        records.append(
            dict(
                dataset_id="alimu",
                pair_id="BMX160_to_VN100",
                task="paired_signal",
                training_eligible=True,
                split=role,
                physical_group_id=role,
                path=path.name,
                sha256=sha256(path),
                rows=n,
                sequence_id=role,
                low_sensor="BMX160",
                high_sensor="VN100",
            )
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(records=records, rate_hz=40)), encoding="utf-8")
    config = dict(
        experiment_id="tiny_matched",
        seed=51,
        seeds=[51, 52],
        methods=["ddpm", "no_diffusion"],
        data=dict(
            manifest="manifest.json",
            dataset_id="alimu",
            pairs=["BMX160_to_VN100"],
            window_size=16,
            stride=8,
            eval_stride=8,
            batch_size=4,
        ),
        model=dict(
            condition_channels=7,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            max_length=16,
        ),
        diffusion=dict(timesteps=3, schedule="cosine", prediction_type="x0"),
        training=dict(
            device="cpu",
            amp=False,
            cpu_threads=1,
            epochs=2,
            lr=0.001,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            early_stopping_patience=1,
            min_delta=1000000.0,
            validation_windows=2,
        ),
        search=dict(trials=[{}]),
        evaluation=dict(draws=1, latency_warmup=0, latency_repeats=1),
        output=dict(runs_dir="runs"),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    original = PairedSource.load
    test_reads = []

    def guarded(self, role):
        if role == "test":
            lock = json.loads((tmp_path / "runs/tiny/selections.lock.json").read_text())
            assert len(lock["selections"]) == 2
            assert lock["test_used_for_selection"] is False
            test_reads.append(role)
        return original(self, role)

    monkeypatch.setattr(PairedSource, "load", guarded)
    run = run_matched_experiment(config_path, tmp_path, "tiny")
    assert test_reads == ["test"]
    metrics = pd.read_csv(run / "metrics_per_seed.csv")
    assert set(metrics.seed) == {51, 52}
    assert set(metrics.method) >= {"ddpm", "no_diffusion", "no_diffusion_matched"}
    assert metrics[metrics.component.eq("gyro")].samples.eq(48).all()
    delta = pd.read_csv(run / "matched_seed_deltas.csv")
    assert len(delta) == 4
    assert delta.initial_backbone_matches.all()
    run_matched_experiment(config_path, tmp_path, "tiny", resume=True)
    assert test_reads == ["test"]
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        run_matched_experiment(config_path, tmp_path, "tiny", resume=True)
    manifest.write_text(manifest.read_text().rstrip(), encoding="utf-8")
    complete = json.loads((run / "complete.json").read_text())
    assert any(p.endswith(".pt") for p in complete["artifacts_sha256"])
    assert any(p.endswith(".parquet") for p in complete["artifacts_sha256"])
    checkpoint = next(run.rglob("best.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        run_matched_experiment(config_path, tmp_path, "tiny", resume=True)


def test_truncated_summary_is_archived_and_restarted(tmp_path, monkeypatch):
    directory = tmp_path / "fit"
    directory.mkdir()
    (directory / "summary.json").write_text('{"checkpoint":', encoding="utf-8")
    job = dict(id="one", directory=str(directory), config={})
    called = []

    def fit(restarted):
        assert not directory.exists()
        called.append(restarted["id"])
        directory.mkdir()
        return dict(best_epoch=1, validation_score=0.5, seconds=1)

    monkeypatch.setattr("virtual_imu.diffusion.matched.fit_job", fit)
    execute_jobs([job], 1, tmp_path / "progress.json")
    assert called == ["one"]
    archives = list(tmp_path.glob("fit_interrupted_*"))
    assert len(archives) == 1
    assert (archives[0] / "summary.json").read_text() == '{"checkpoint":'
