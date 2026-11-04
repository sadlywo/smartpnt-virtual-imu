"""Source-only normalization/calibration and completed-pipeline transfer gating."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.eval.formal_evaluation import HIGH, LOW, VIRTUAL
from virtual_imu.models import make_model


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_reviewer_transfer.py"
    spec = importlib.util.spec_from_file_location("reviewer_transfer", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def frame(dataset, split, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(320, 6)) + (20 if dataset == "penncosyvio" else 0)
    result = pd.DataFrame(values, columns=LOW)
    result[HIGH] = values + np.linspace(0.1, 0.6, 6)
    result["dataset_id"] = dataset
    result["low_sensor"], result["high_sensor"] = dataset + "low", dataset + "high"
    result["sequence_id"], result["original_sequence_id"] = dataset + split, split
    result["timestamp_s"] = np.arange(len(result)) / 40
    return result


def test_shared_source_calibration_selection_never_accepts_target_labels():
    transfer = module()
    train, validation = frame("alimu", "train", 1), frame("alimu", "validation", 2)
    search = {"ridge_alpha": [0.0001, 1.0], "fir_taps": [1], "arx_taps": [1]}
    state = transfer.fit_source_classical(train, validation, search)
    target = frame("penncosyvio", "test", 3)
    before = transfer.predict_source_classical(state, target)
    target[HIGH] = -9999
    after = transfer.predict_source_classical(state, target)
    assert state["fit_split"] == "source_train"
    assert state["selection_split"] == "source_validation"
    assert state["tuned"]["source_ridge_tuned"]["alpha"] == 0.0001
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])


def test_checkpoint_inference_preserves_source_stats_despite_target_label_shift(tmp_path):
    transfer = module()
    model_config = {
        "type": "tcn",
        "in_channels": 6,
        "out_channels": 6,
        "hidden_channels": 4,
        "levels": 1,
        "kernel_size": 3,
        "dropout": 0.0,
        "target_mode": "residual",
    }
    model = make_model(model_config)
    for parameter in model.parameters():
        parameter.data.zero_()
    path = tmp_path / "best.ckpt"
    stats = {"columns": LOW, "mean": [2.0] * 6, "std": [3.0] * 6}
    target_stats = {"columns": HIGH, "mean": [0.25] * 6, "std": [7.0] * 6}
    torch.save(
        {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "feature_stats": stats,
            "target_stats": target_stats,
            "feature_columns": LOW,
            "target_mode": "residual",
            "config": {"data": {"window_size": 80, "stride": 40}},
        },
        path,
    )
    generator = transfer.VirtualIMUGenerator(path, device="cpu")
    target = frame("penncosyvio", "test", 5)
    first, _ = transfer.predict_complete_windows(generator, target)
    target[HIGH] += 1234
    second, _ = transfer.predict_complete_windows(generator, target)
    np.testing.assert_allclose(first[VIRTUAL], first[LOW].to_numpy() + 0.25)
    np.testing.assert_array_equal(first[VIRTUAL], second[VIRTUAL])
    assert generator.feature_stats == stats
    assert generator.target_stats == target_stats


def test_full_gate_rejects_subset_before_loading_payloads(tmp_path):
    transfer = module()
    state = {
        "status": "complete",
        "smoke": False,
        "cells": {},
        "jobs": {},
        "final_seeds": [1],
        "final_variants": ["default", "tuned"],
    }
    with pytest.raises(ValueError, match=r"93|558"):
        transfer.verify_pipeline(state, tmp_path, allow_subset=False)


def test_immutable_record_verification_rejects_changed_prediction(tmp_path):
    transfer = module()
    path = tmp_path / "predictions.parquet"
    path.write_bytes(b"original")
    transfer.write_json_once(
        tmp_path / "artifact_manifest.json",
        {"files": {"predictions.parquet": hashlib.sha256(b"original").hexdigest()}},
    )
    transfer.verify_record(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        transfer.verify_record(tmp_path)


def test_direction_loader_binds_source_roles_to_the_trained_protocol(tmp_path, monkeypatch):
    transfer = module()

    def entry(name):
        return {
            "sequence_id": name,
            "path": name + ".parquet",
            "sha256": name + "hash",
            "row_filter": "all",
        }

    source = {"splits": {"train": [entry("train")], "validation": [entry("val")], "test": []}}
    target = {"splits": {"train": [], "validation": [], "test": [entry("test")]}}
    derived = {
        "canonical_provenance": {"dataset_manifest_sha256": "toy"},
        "splits": {
            "train": source["splits"]["train"],
            "validation": source["splits"]["validation"],
            "test": target["splits"]["test"],
        },
    }
    index = {"base_dataset_sha256": "toy", "protocols": {}}
    for name, payload in (
        ("alimu_only", source),
        ("penn_only", target),
        ("alimu_to_penn", derived),
    ):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        index["protocols"][name] = {"split_manifest": str(path), "strict_manifest": str(path)}
    frames = {
        "train": frame("alimu", "train", 1),
        "validation": frame("alimu", "validation", 2),
        "test": frame("penncosyvio", "test", 3),
    }
    monkeypatch.setattr(
        transfer, "load_training_split_manifest", lambda *a, **k: SimpleNamespace(frames=frames)
    )
    actual, metadata = transfer._load_direction(index, "alimu_to_penn", tmp_path)
    assert len(actual["test"]) == 320
    assert set(metadata["corresponding_in_domain_manifests"]) == {"alimu_only", "penn_only"}
    derived["splits"]["train"] = [entry("wrong_train")]
    Path(index["protocols"]["alimu_to_penn"]["split_manifest"]).write_text(
        json.dumps(derived), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trained in-domain protocol"):
        transfer._load_direction(index, "alimu_to_penn", tmp_path)


def test_explicit_subset_cli_runs_both_directions_and_resumes_without_overwrite(
    tmp_path, monkeypatch
):
    transfer = module()
    root = tmp_path / "pipeline"
    (root / "selections").mkdir(parents=True)
    state = {
        "status": "complete",
        "smoke": True,
        "dataset_manifest_sha256": "toy",
        "final_seeds": [1],
        "final_variants": ["default", "tuned"],
        "cells": {},
        "jobs": {},
        "contract": {
            "config": {
                "cpu_threads": 1,
                "classical_search": {"ridge_alpha": [0.0001], "fir_taps": [1], "arx_taps": [1]},
            }
        },
    }
    for source in ("alimu_only", "penn_only"):
        cell = source + "/tcn"
        selected = {"trial": {"parameters": {}}}
        selection_path = root / "selections" / (cell.replace("/", "__") + ".json")
        selection_path.write_text(json.dumps(selected), encoding="utf-8")
        state["cells"][cell] = {
            "selected": selected,
            "selection_sha256": transfer.sha(selection_path),
        }
        for variant in ("default", "tuned"):
            identifier = source + "_" + variant
            run = root / "experiments" / identifier
            run.mkdir(parents=True)
            model_cfg = {
                "type": "tcn",
                "in_channels": 6,
                "out_channels": 6,
                "target_mode": "residual",
                "hidden_channels": 4,
                "levels": 1,
                "kernel_size": 3,
                "dropout": 0.0,
            }
            config = {
                "seed": 1,
                "model": model_cfg,
                "data": {"feature_columns": LOW, "window_size": 80, "stride": 40, "batch_size": 4},
                "reviewer_protocol": {
                    "phase": "final_" + variant,
                    "protocol_id": source,
                    "model_family": "tcn",
                    "parameters": {},
                    "selection_scope": "validation_only",
                },
            }
            model = make_model(model_cfg)
            for parameter in model.parameters():
                parameter.data.zero_()
            torch.save(
                {
                    "config": config,
                    "model_config": model_cfg,
                    "model_state_dict": model.state_dict(),
                    "feature_columns": LOW,
                    "target_mode": "residual",
                    "feature_stats": {"columns": LOW, "mean": [2] * 6, "std": [3] * 6},
                    "target_stats": {"columns": HIGH, "mean": [0.25] * 6, "std": [7] * 6},
                },
                run / "best.ckpt",
            )
            (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            state["jobs"][identifier] = {
                "id": identifier,
                "cell": cell,
                "phase": "final_" + variant,
                "seed": 1,
                "status": "complete",
                "evaluation_status": "complete",
                "run_dir": str(run),
                "checkpoint_sha256": transfer.sha(run / "best.ckpt"),
                "run_config_sha256": transfer.sha(run / "config.yaml"),
            }
    (root / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"base_dataset_sha256": "toy"}), encoding="utf-8")
    monkeypatch.setattr(transfer, "_processing_sources", lambda *a, **k: {"test_source": "frozen"})
    monkeypatch.setattr(transfer, "verify_run_provenance", lambda *a: None)
    monkeypatch.setattr(
        transfer,
        "_reference",
        lambda *a: {"label": "target_domain_trained_reference_not_zero_shot"},
    )

    def load_direction(index, direction, project):
        source, target = transfer.DIRECTIONS[direction][2:]
        frames = {
            "train": frame(source, "train", 1),
            "validation": frame(source, "validation", 2),
            "test": frame(target, "test", 3),
        }
        return frames, {
            "physical_fingerprints": {
                role: transfer.frame_fingerprint(value) for role, value in frames.items()
            }
        }

    monkeypatch.setattr(transfer, "_load_direction", load_direction)
    out = tmp_path / "transfer"
    argv = [
        "--pipeline-root",
        str(root),
        "--protocol-index",
        str(index),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--allow-completed-subset",
    ]
    assert transfer.main(argv) == 0
    report = json.loads((out / "transfer_metrics.json").read_text(encoding="utf-8"))
    assert report["status"] == "debug_subset_complete" and report["records"] == 4
    assert {record["direction"] for record in report["source_model_records"]} == set(
        transfer.DIRECTIONS
    )
    for record in report["source_model_records"]:
        assert record["source_target_stats"]["mean"] == [0.25] * 6
        assert record["selection_uses_target_labels"] is False
    original = {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        transfer,
        "_load_direction",
        lambda *a: pytest.fail("completed resume must verify stored artifacts"),
    )
    assert transfer.main([*argv, "--resume"]) == 0
    assert original == {
        path.relative_to(out).as_posix(): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
