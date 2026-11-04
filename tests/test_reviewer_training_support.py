"""Integration contracts for validation-only reviewer training and old models."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.io.unified_dataset import verify_unified_dataset
from virtual_imu.models import make_model
from virtual_imu.models.transformer import TransformerIMU
from virtual_imu.training import gt_trainer, unified_data
from virtual_imu.training.schedules import warmup_cosine_scheduler


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cosine_warmup_reaches_base_then_decays_to_same_absolute_minimum():
    parameters = [torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))]
    optimizer = torch.optim.SGD([{"params": parameters[:1], "lr": 1e-3}, {"params": parameters[1:], "lr": 2e-3}])
    scheduler = warmup_cosine_scheduler(optimizer, {"epochs": 8, "warmup_epochs": 2, "min_lr": 1e-5})
    rates = []
    for _ in range(8):
        rates.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    rates = np.asarray(rates)
    assert np.isfinite(rates).all()
    assert np.all(rates[0] < rates[1])
    np.testing.assert_allclose(rates.max(axis=0), [1e-3, 2e-3])
    assert np.all(np.diff(rates[2:], axis=0) <= 1e-12)
    np.testing.assert_allclose(rates[-1], [1e-5, 1e-5], atol=1e-12)


@pytest.mark.parametrize("settings", [{"epochs": 0}, {"epochs": 3, "warmup_epochs": 3}, {"epochs": 3, "min_lr": 0.1}])
def test_cosine_schedule_rejects_invalid_or_impossible_budget(settings):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    with pytest.raises(ValueError):
        warmup_cosine_scheduler(optimizer, settings)


def test_factory_legacy_transformer_restores_six_to_three_state_without_ff_override():
    torch.manual_seed(77)
    legacy = TransformerIMU(in_channels=6, out_channels=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=2048, norm_first=False, activation="relu")
    restored = make_model({"type": "transformer", "in_channels": 6, "out_channels": 3})
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert restored.encoder.layers[0].linear1.weight.shape == (2048, 64)
    assert not restored.encoder.layers[0].norm_first
    legacy.eval()
    restored.eval()
    inputs = torch.randn(2, 6, 16)
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), legacy(inputs))


def test_factory_applies_searched_transformer_ff_and_explicit_override():
    settings = {"type": "transformer", "in_channels": 6, "out_channels": 6, "d_model": 32, "nhead": 4, "num_layers": 1, "ff_multiplier": 4, "norm_first": True, "activation": "gelu"}
    model = make_model(settings)
    assert model.encoder.layers[0].linear1.out_features == 128
    assert model.encoder.layers[0].norm_first
    overridden = make_model({**settings, "dim_feedforward": 96})
    assert overridden.encoder.layers[0].linear1.out_features == 96


@pytest.mark.parametrize("kind", ["informer", "patchtst", "itransformer"])
def test_new_attention_factory_maps_searched_ff_and_preserves_dense_task_shape(kind):
    model = make_model({"type": kind, "in_channels": 6, "out_channels": 6, "window_size": 80, "d_model": 16, "nhead": 2, "num_layers": 1, "ff_multiplier": 4, "dropout": 0.0})
    assert any(isinstance(layer, torch.nn.Linear) and layer.in_features == 16 and layer.out_features == 64 for layer in model.modules())
    model.eval()
    with torch.no_grad():
        predictions = model(torch.randn(2, 6, 80))
    assert predictions.shape == (2, 6, 80)
    assert torch.isfinite(predictions).all()


def test_requested_paired_splits_do_not_open_or_require_a_test_payload(tmp_path, monkeypatch):
    split_records = {}
    strict_records = []
    for role in ("train", "validation", "test"):
        path = tmp_path / f"{role}.parquet"
        if role != "test":
            pd.DataFrame({"timestamp_s": [0.0, 0.025], "training_eligible": [True, True]}).to_parquet(path, index=False)
        entry = {"dataset_id": "fixture", "pair_id": "low_high", "sequence_id": role, "path": str(path), "strict_path": str(path), "fingerprint": f"fingerprint_{role}", "training_eligible": True, "group_value": role}
        split_records[role] = [entry]
        strict_records.append(entry)
    split_path, strict_path = tmp_path / "split.json", tmp_path / "strict.json"
    _write_json(split_path, {"split_kind": "leave_one_sequence_out", "splits": split_records})
    _write_json(strict_path, {"records": strict_records})
    actual_read = pd.read_parquet
    reads = []

    def read_selected(path, *args, **kwargs):
        reads.append(Path(path).name)
        assert Path(path).name != "test.parquet", "Held-out test payload was opened"
        return actual_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_selected)
    bundle = load_training_split_manifest(split_path, strict_path, split_names=["train", "validation"])
    assert set(bundle.frames) == {"train", "validation"}
    assert reads == ["train.parquet", "validation.parquet"]
    assert len(bundle.input_records) == 2


def _gt_frame(sequence: str, offset: float) -> pd.DataFrame:
    timestamps = np.arange(32, dtype=float) / 40
    frame = pd.DataFrame({"timestamp_s": timestamps, "sequence_id": sequence})
    for channel in ("gx_radps", "gy_radps", "gz_radps", "ax_mps2", "ay_mps2", "az_mps2"):
        frame[channel] = np.sin(timestamps) + offset
    for axis, scale in zip(("x", "y", "z"), (1.0, 2.0, 0.5), strict=True):
        frame[f"gt_{axis}_local_m"] = offset + timestamps * scale
    for axis, speed in zip(("east", "north", "up"), (1.0, 2.0, 0.5), strict=True):
        frame[f"gt_v_{axis}_mps"] = speed
    for angle in ("heading", "pitch", "roll"):
        frame[f"gt_{angle}_deg"] = 0.0
    return frame


@pytest.mark.parametrize("flags", [{"evaluate_test": False}, {"disable_test_evaluation": True}])
def test_gt_validation_only_training_never_constructs_test_loader_or_report(tmp_path, monkeypatch, flags):
    data = tmp_path / "aligned"
    data.mkdir()
    for sequence, offset in (("train_seq", 1.0), ("val_seq", 2.0)):
        _gt_frame(sequence, offset).to_parquet(data / f"{sequence}_imu_gt_aligned.parquet", index=False)
    actual_loader = gt_trainer.make_smartpnt_gt_split_dataloaders
    loader_roles = []

    def load_without_test(*args, **kwargs):
        assert set(kwargs["split_sequences"]) == {"train", "val"}
        loaders = actual_loader(*args, **kwargs)
        loader_roles.extend(loaders)
        return loaders

    monkeypatch.setattr(gt_trainer, "make_smartpnt_gt_split_dataloaders", load_without_test)
    config = {
        "seed": 9, "experiment_name": "reviewer_gt_train_only",
        "data": {"aligned_root": str(data), "label_mode": "position_delta", "window_size": 8, "stride": 8, "batch_size": 4, "train_sequences": ["train_seq"], "val_sequences": ["val_seq"], "test_sequences": ["missing_heldout_test"]},
        "model": {"type": "tcn", "hidden_channels": 4, "levels": 1, "kernel_size": 3, "dropout": 0.0},
        "training": {"epochs": 1, "lr": 0.001, "device": "cpu", "checkpoint_metric": "val_vector_rmse", "lr_scheduler": "cosine_warmup", "warmup_epochs": 0, **flags},
        "output": {"runs_dir": str(tmp_path / "runs")},
    }
    directory = gt_trainer.train_gt_supervised_model(config)
    assert set(loader_roles) == {"train", "val"}
    assert (directory / "best.ckpt").is_file()
    assert not (directory / "test_metrics.json").exists()
    history = pd.read_csv(directory / "metrics.csv")
    assert len(history) == 1 and np.isfinite(history.val_vector_rmse).all()


def _navigation_fixture(tmp_path: Path, monkeypatch):
    base_records, assigned_records = [], []
    canonical_roles = {"Traj1": "test", "Traj3": "train", "Traj4": "validation"}
    derived_roles = {"Traj1": "validation", "Traj3": "test", "Traj4": "train"}
    splits = {role: [] for role in ("train", "validation", "test")}
    for sequence, original in canonical_roles.items():
        path = tmp_path / f"{sequence}.parquet"
        frame = _gt_frame(sequence, 1.0)
        frame["split"] = original
        frame["training_eligible"] = True
        frame["has_signal_label"] = False
        for axis in ("gx", "gy", "gz", "ax", "ay", "az"):
            frame[f"{axis}_low"] = frame[f"{axis}_{'radps' if axis.startswith('g') else 'mps2'}"]
        frame.to_parquet(path, index=False)
        record = {"sequence_id": sequence, "original_sequence_id": sequence, "dataset_id": "smartpnt_msf", "task": "navigation", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "physical_group_id": f"smartpnt_msf/{sequence}", "split": original, "training_eligible": True, "rows": len(frame)}
        base_records.append(record)
        assigned = {**record, "split": derived_roles[sequence], "canonical_split": original}
        assigned_records.append(assigned)
        splits[derived_roles[sequence]].append({**assigned, "group_value": record["physical_group_id"]})
    manifest, assignment, split_path = (tmp_path / name for name in ("manifest.json", "assignment.json", "split.json"))
    snapshot = tmp_path / "config_snapshot.yaml"
    snapshot.write_text("rate_hz: 40\n", encoding="utf-8")
    _write_json(manifest, {"records": base_records, "rate_hz": 40, "config_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()})
    assignment_value = {"canonical_provenance": {"dataset_manifest": str(manifest), "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}, "records": assigned_records}
    _write_json(assignment, assignment_value)
    _write_json(split_path, {"split_kind": "leave_one_sequence_out", "splits": splits})
    monkeypatch.setattr(unified_data, "project_root_from_file", lambda _: tmp_path)
    config = {"dataset_manifest": str(manifest), "assignment_manifest": str(assignment), "split_manifest": str(split_path), "train_sequences": ["Traj4"], "val_sequences": ["Traj1"], "test_sequences": ["Traj3"], "load_splits": ["train", "validation"]}
    return config, assignment, assignment_value


def test_hash_only_verification_decodes_no_payload_and_full_default_still_validates(tmp_path, monkeypatch):
    config, _, _ = _navigation_fixture(tmp_path, monkeypatch)
    actual_read = pd.read_parquet
    reads = []

    def read_payload(path, *args, **kwargs):
        reads.append(Path(path).stem)
        return actual_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_payload)
    manifest = Path(config["dataset_manifest"])
    hash_check = verify_unified_dataset(manifest, tmp_path, validate_payloads=False)
    assert hash_check["checked_records"] == 3
    assert hash_check["checked_rows"] == 96
    assert hash_check["payloads_validated"] is False and reads == []
    full_check = verify_unified_dataset(manifest, tmp_path)
    assert full_check["payloads_validated"] is True
    assert set(reads) == {"Traj1", "Traj3", "Traj4"}


def test_derived_navigation_uses_new_roles_preserves_old_roles_and_skips_test(tmp_path, monkeypatch):
    config, _, _ = _navigation_fixture(tmp_path, monkeypatch)
    original_read = pd.read_parquet
    reads = []

    def read_selected(path, *args, **kwargs):
        reads.append(Path(path).stem)
        assert Path(path).stem != "Traj3", "Held-out navigation payload was opened"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_selected)
    frame = unified_data.unified_navigation_frame(config)
    assert reads == ["Traj4", "Traj1"]
    assert frame.set_index("sequence_id").split.to_dict() == {"Traj4": "train", "Traj1": "validation"}
    assert frame.set_index("sequence_id").canonical_split.to_dict() == {"Traj4": "validation", "Traj1": "test"}


@pytest.mark.parametrize("field", ["path", "sha256", "dataset_id", "physical_group_id", "original_sequence_id"])
def test_derived_navigation_rejects_mutated_physical_identity(tmp_path, monkeypatch, field):
    config, path, assignment = _navigation_fixture(tmp_path, monkeypatch)
    altered = copy.deepcopy(assignment)
    altered["records"][0][field] = "tampered"
    _write_json(path, altered)
    with pytest.raises(ValueError, match="immutable record identity"):
        unified_data.unified_navigation_frame(config)


def test_derived_navigation_rejects_unbound_canonical_manifest(tmp_path, monkeypatch):
    config, path, assignment = _navigation_fixture(tmp_path, monkeypatch)
    assignment["canonical_provenance"]["dataset_manifest_sha256"] = "0" * 64
    _write_json(path, assignment)
    with pytest.raises(ValueError, match="bind the original dataset"):
        unified_data.unified_navigation_frame(config)
