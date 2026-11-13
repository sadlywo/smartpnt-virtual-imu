"""Synthetic safety/identity tests for explicit reused controls and TEST gates."""

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

from virtual_imu.ablations.training import json_digest, sha, write_json
from virtual_imu.replacements import pipeline, reuse, training


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load(
        (root / "configs/ablations/cnn_lstm_module_replacement.yaml").read_text(encoding="utf-8")
    )


def test_matrix_has48_new18_explicit_reuses_and_seed_blocked_unique_jobs(config, tmp_path):
    pipeline.validate_config(config)
    a = pipeline.jobs_for_config(tmp_path, config, tmp_path / "run")
    b = pipeline.jobs_for_config(tmp_path, config, tmp_path / "run")
    assert a == b and len(a) == len({j["id"] for j in a}) == 66
    assert sum(j["reused"] for j in a) == 18
    assert sum(not j["reused"] for j in a) == 48
    for j in a:
        assert j["window"] == 320
        assert j["reused"] == (j["variant"] in config["reuse"]["variants"])
        assert (j.get("source_variant") is not None) == j["reused"]
    assert [j["id"] for j in a] != sorted(j["id"] for j in a)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_variant",
        "duplicate_seed",
        "wrong_window",
        "unknown_variant",
        "wrong_reference",
        "wrong_reuse",
    ],
)
def test_invalid_matrix_is_rejected_before_any_training(config, change):
    c = copy.deepcopy(config)
    if change == "duplicate_variant":
        c["variants"].append(c["variants"][0])
    if change == "duplicate_seed":
        c["seeds"].append(c["seeds"][0])
    if change == "wrong_window":
        c["windows"] = [160]
    if change == "unknown_variant":
        c["variants"].append("unregistered")
    if change == "wrong_reference":
        c["reference_variant"] = "cnn_bilstm32"
    if change == "wrong_reuse":
        c["reuse"]["variants"]["cnn_gru"] = "cnn_lstm_wide"
    with pytest.raises(ValueError):
        pipeline.validate_config(c)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "lr", 0.001),
        ("training", "optimizer", "Adam"),
        ("training", "epochs", 79),
        ("data", "stride", 80),
        ("data", "manifest_sha256", "bad"),
        ("model", "dropout", 0.1),
    ],
)
def test_import_rejects_recipe_or_data_change(config, section, key, value):
    old = copy.deepcopy(config)
    new = copy.deepcopy(config)
    new[section][key] = value
    with pytest.raises(ValueError):
        reuse.validate_reuse_compatibility(new, old)


def test_import_accepts_only_non_numerical_worker_count_change(config):
    old = copy.deepcopy(config)
    config["training"]["workers"] = 1
    reuse.validate_reuse_compatibility(config, old)


def test_incomplete_duplicate_or_hash_mismatch_never_unlocks_test(config, tmp_path):
    jobs = pipeline.jobs_for_config(tmp_path, config, tmp_path / "run")
    with pytest.raises(ValueError):
        pipeline.lock_selections(tmp_path, config, tmp_path / "run", jobs, [])
    records = [{"job_id": j["id"]} for j in jobs]
    records[-1]["job_id"] = records[0]["job_id"]
    with pytest.raises(ValueError):
        pipeline.lock_selections(tmp_path, config, tmp_path / "run", jobs, records)
    assert not (tmp_path / "run/selections.lock.json").exists()


def test_resume_checks_config_and_checkpoint_bytes_before_skipping(tmp_path):
    directory = tmp_path / "fit"
    directory.mkdir()
    job = {"id": "x", "directory": str(directory), "config": {}}
    (directory / "best.pt").write_bytes(b"checkpoint-a")
    record = {
        "job_id": "x",
        "job_config_sha256": json_digest(job),
        "checkpoint_sha256": sha(directory / "best.pt"),
    }
    write_json(directory / "fit.json", record)
    assert training.completed_job(job) == record
    (directory / "best.pt").write_bytes(b"checkpoint-b")
    with pytest.raises(ValueError):
        training.completed_job(job)
    (directory / "best.pt").write_bytes(b"checkpoint-a")
    job["other"] = "changed"
    with pytest.raises(ValueError):
        training.completed_job(job)


def test_borrowed_checkpoint_never_rewrites_original_training_identity(tmp_path):
    job = {
        "id": "new",
        "protocol": "pooled",
        "variant": "cnn_lstm64",
        "seed": 1,
        "window": 320,
        "reused": True,
    }
    original = dict(job, id="original", variant="cnn_lstm_wide")
    saved = {"job": original, "epoch": 5, "val_score": 0.2}
    record = {
        "job_id": "new",
        "best_epoch": 5,
        "best_val_score": 0.2,
        "job_config_sha256": json_digest(job),
        "reused": True,
    }
    receipt = {"source_job": original, "source_fit": {"best_epoch": 5, "best_val_score": 0.2}}
    reuse.check_checkpoint_identity(job, record, saved, receipt)
    assert saved["job"] == original
    altered = copy.deepcopy(saved)
    altered["job"] = job
    with pytest.raises(ValueError):
        reuse.check_checkpoint_identity(job, record, altered, receipt)


def test_checkpoint_epoch_and_minimum_validation_identity_are_required():
    job = {"id": "new", "reused": False}
    record = {
        "job_id": "new",
        "best_epoch": 5,
        "best_val_score": 0.2,
        "job_config_sha256": json_digest(job),
        "reused": False,
    }
    reuse.check_checkpoint_identity(job, record, {"job": job, "epoch": 5, "val_score": 0.2})
    for saved in [
        {"job": job, "epoch": 4, "val_score": 0.2},
        {"job": job, "epoch": 5, "val_score": 0.3},
    ]:
        with pytest.raises(ValueError):
            reuse.check_checkpoint_identity(job, record, saved)


def test_train_statistics_are_not_refit_on_validation_and_test_never_loaded(
    monkeypatch, config, tmp_path
):
    from virtual_imu.ablations import data

    def frame(sequence, shift):
        x = np.arange(360)[:, None] * np.arange(1, 7)[None, :] / 100.0 + shift
        f = pd.DataFrame(x, columns=data.LOW)
        f[data.HIGH] = x * 2
        for key, value in [
            ("dataset_id", "alimu"),
            ("pair_id", "p"),
            ("sequence_id", sequence),
            ("original_sequence_id", sequence),
            ("physical_group_id", sequence),
        ]:
            f[key] = value
        f["timestamp_s"] = np.arange(360) / 40
        return f

    frames = {"train": frame("a", 0), "validation": frame("b", 1000)}
    reads = []

    class Source:
        def __init__(self, *args):
            self.dataset = types.SimpleNamespace(manifest_sha256=config["data"]["manifest_sha256"])

        def load(self, role):
            reads.append(role)
            assert role != "test"
            return frames[role]

    monkeypatch.setattr(training, "SharedSource", Source)
    monkeypatch.setattr(training, "_CACHE", {})
    _, train, val, _ = training.data_for_job(
        {"root": str(tmp_path), "protocol": "alimu_only", "config": config}, torch.device("cpu")
    )
    assert reads == ["train", "validation"] and val.stats is train.stats
    np.testing.assert_allclose(train.stats["feature_mean"], frames["train"][data.LOW].mean())
    np.testing.assert_allclose(
        val.x,
        (frames["validation"][data.LOW].to_numpy() - train.stats["feature_mean"])
        / train.stats["feature_std"],
        rtol=1e-6,
    )


def test_lock_checks_actual_minimum_validation_epoch(tmp_path, monkeypatch):
    directory = tmp_path / "run/fits/a"
    directory.mkdir(parents=True)
    job = {"id": "a", "directory": str(directory), "reused": False}
    saved = {"job": job, "epoch": 2, "val_score": 0.2}
    torch.save(saved, directory / "best.pt")
    record = {
        "job_id": "a",
        "job_config_sha256": json_digest(job),
        "checkpoint_sha256": sha(directory / "best.pt"),
        "best_epoch": 2,
        "best_val_score": 0.2,
        "reused": False,
    }
    history = pd.DataFrame(
        {"epoch": [1, 2, 3], "val_score": [0.4, 0.2, 0.3], "train_loss": [0.5, 0.4, 0.3]}
    )
    history.to_csv(directory / "history.csv", index=False)
    monkeypatch.setattr(
        pipeline, "validate_scientific_checkpoint", lambda *args: None, raising=False
    )
    write_json(tmp_path / "run/analysis_plan.json", {})
    write_json(directory / "initialization.json", {})
    (tmp_path / "run/config.yaml").write_text("{}")
    pipeline.lock_selections(tmp_path, {}, tmp_path / "run", [job], [record])
    assert json.loads((tmp_path / "run/selections.lock.json").read_text())["test_loaded"] is False
    history.loc[0, "val_score"] = 0.1
    history.to_csv(directory / "history.csv", index=False)
    with pytest.raises(ValueError):
        pipeline.lock_selections(tmp_path, {}, tmp_path / "run", [job], [record])


@pytest.fixture
def scientific_gate(tmp_path, config, monkeypatch):
    from virtual_imu.ablations.data import support_digest

    root = tmp_path
    run = root / "run"
    directory = run / "fits/a"
    directory.mkdir(parents=True)
    job = {
        "id": "a",
        "root": str(root),
        "directory": str(directory),
        "config": config,
        "protocol": "alimu_only",
        "variant": "cnn_gru",
        "seed": 1,
        "window": 320,
        "reused": False,
    }
    model = torch.nn.Linear(1, 1)
    meta = {"init_state_sha256": "known-state"}
    module = types.ModuleType("virtual_imu.replacements.models")
    module.build_replacement_model = lambda *args: (torch.nn.Linear(1, 1), meta)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    frame = pd.DataFrame(
        {
            "dataset_id": ["a"],
            "pair_id": ["p"],
            "sequence_id": ["s"],
            "original_sequence_id": ["s"],
            "timestamp_s": [0.0],
        }
    )
    support = run / "supports/alimu_only"
    support.mkdir(parents=True)
    for role in ["train", "validation"]:
        frame.to_parquet(support / (role + ".parquet"), index=False)
    stats = {"feature_mean": [0.0]}
    write_json(support / "statistics.json", stats)
    write_json(directory / "stats.json", stats)
    saved = {
        "job": job,
        "model": model.state_dict(),
        "metadata": meta,
        "stats": stats,
        "train_support_sha256": support_digest(frame),
        "validation_support_sha256": support_digest(frame),
        "epoch": 2,
        "val_score": 0.2,
    }
    torch.save(saved, directory / "best.pt")
    write_json(directory / "initialization.json", meta)
    write_json(directory / "job.json", job)
    record = {
        "job_id": "a",
        "job_config_sha256": json_digest(job),
        "checkpoint_sha256": sha(directory / "best.pt"),
        "best_epoch": 2,
        "best_val_score": 0.2,
        "actual_epochs": 3,
        "reused": False,
    }
    pd.DataFrame(
        {"epoch": [1, 2, 3], "val_score": [0.4, 0.2, 0.3], "train_loss": [0.5, 0.4, 0.3]}
    ).to_csv(directory / "history.csv", index=False)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    write_json(run / "analysis_plan.json", {"source_sha256": {}})
    return root, run, job, record, saved


def test_checkpoint_lock_is_immutable_and_identical_reevaluation_is_idempotent(scientific_gate):
    root, run, job, record, saved = scientific_gate
    first = pipeline.lock_selections(root, job["config"], run, [job], [record])
    original = sha(run / "selections.lock.json")
    assert pipeline.lock_selections(root, job["config"], run, [job], [record]) == first
    assert sha(run / "selections.lock.json") == original
    saved["model"]["weight"].add_(1)
    torch.save(saved, Path(job["directory"]) / "best.pt")
    record["checkpoint_sha256"] = sha(Path(job["directory"]) / "best.pt")
    with pytest.raises(ValueError, match=r"[Ll]ock"):
        pipeline.lock_selections(root, job["config"], run, [job], [record])
    assert sha(run / "selections.lock.json") == original


@pytest.mark.parametrize(
    "damage",
    [
        "initialization",
        "metadata",
        "stats",
        "train_support_sha256",
        "validation_support_sha256",
        "nonfinite_state",
        "epochs",
    ],
)
def test_scientific_artifact_defects_fail_before_test_access(scientific_gate, damage, monkeypatch):
    root, run, job, record, saved = scientific_gate
    directory = Path(job["directory"])
    if damage == "initialization":
        write_json(directory / "initialization.json", {"init_state_sha256": "wrong"})
    elif damage == "epochs":
        record["actual_epochs"] = 80
    else:
        if damage == "metadata":
            saved["metadata"] = {"wrong": "state"}
        elif damage == "stats":
            saved["stats"] = {"wrong": "stats"}
        elif damage == "nonfinite_state":
            saved["model"]["weight"].fill_(float("nan"))
        else:
            saved[damage] = "wrong"
        torch.save(saved, directory / "best.pt")
        record["checkpoint_sha256"] = sha(directory / "best.pt")
    monkeypatch.setattr(pipeline, "freeze_analysis", lambda *args: None)
    monkeypatch.setattr(pipeline, "configure", lambda *args: torch.device("cpu"))

    class NeverLoad:
        def __init__(self, *args):
            raise AssertionError("Must reject before reading TEST or VALID")

    monkeypatch.setattr(pipeline, "SharedSource", NeverLoad)
    with pytest.raises(ValueError):
        pipeline.evaluate(root, job["config"], run, [job], [record])
    assert not (run / "test_access.json").exists()


def test_first_test_access_is_durable_and_cannot_be_reset(scientific_gate):
    root, run, job, record, _ = scientific_gate
    pipeline.lock_selections(root, job["config"], run, [job], [record])
    pipeline.record_test_access(run)
    first = sha(run / "test_access.json")
    pipeline.record_test_access(run)
    assert sha(run / "test_access.json") == first
    lock = json.loads((run / "selections.lock.json").read_text())
    lock["changed"] = True
    write_json(run / "selections.lock.json", lock)
    with pytest.raises(ValueError):
        pipeline.record_test_access(run)


@pytest.fixture
def import_fixture(tmp_path, config, monkeypatch):
    cfg = copy.deepcopy(config)
    cfg["reuse"]["run_dir"] = "old"
    manifest = tmp_path / cfg["data"]["manifest"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    cfg["data"]["manifest_sha256"] = sha(manifest)
    source = tmp_path / "old"
    source.mkdir()
    source_cfg = source / "config.yaml"
    source_cfg.write_text(yaml.safe_dump(cfg))
    (tmp_path / "dummy.py").write_text("original source")
    snapshot = source / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "dummy.py").write_text("original source")
    write_json(
        source / "provenance.json",
        {
            "config_sha256": sha(source_cfg),
            "manifest_sha256": sha(manifest),
            "source_sha256": {"dummy.py": sha(tmp_path / "dummy.py")},
        },
    )
    source_id = "alimu_only__cnn_lstm_wide__w320__s20261010"
    origin = source / "fits" / source_id
    origin.mkdir(parents=True)
    original = {
        "id": source_id,
        "protocol": "alimu_only",
        "seed": 20261010,
        "variant": "cnn_lstm_wide",
        "window": 320,
        "directory": str(origin),
    }
    model = torch.nn.Linear(1, 1)
    torch.save(
        {
            "job": original,
            "model": model.state_dict(),
            "stats": {},
            "metadata": {"init_state_sha256": "exact-init"},
            "epoch": 2,
            "val_score": 0.2,
        },
        origin / "best.pt",
    )
    write_json(origin / "job.json", original)
    write_json(origin / "initialization.json", {"init_state_sha256": "exact-init"})
    write_json(origin / "stats.json", {})
    pd.DataFrame({"epoch": [1, 2], "val_score": [0.3, 0.2], "train_loss": [0.5, 0.4]}).to_csv(
        origin / "history.csv", index=False
    )
    write_json(
        origin / "fit.json",
        {
            "job_id": source_id,
            "job_config_sha256": json_digest(original),
            "checkpoint_sha256": sha(origin / "best.pt"),
            "parameters": 2,
            "best_epoch": 2,
            "best_val_score": 0.2,
            "directory": str(origin),
        },
    )
    write_json(
        source / "artifact_manifest.json",
        {
            "files_sha256": {
                p.relative_to(source).as_posix(): sha(p)
                for p in [*origin.iterdir(), source_cfg, source / "provenance.json"]
            }
        },
    )
    write_json(source / "verification.json", {"status": "PASS"})
    write_json(
        source / "complete.json",
        {
            "status": "complete",
            "numerical_verification": "PASS",
            "verification_sha256": sha(source / "verification.json"),
            "artifact_manifest_sha256": sha(source / "artifact_manifest.json"),
        },
    )
    module = types.ModuleType("virtual_imu.replacements.models")
    module.build_replacement_model = lambda *args: (
        torch.nn.Linear(1, 1),
        {"init_state_sha256": "exact-init"},
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    job = {
        "id": "imported",
        "protocol": "alimu_only",
        "seed": 20261010,
        "variant": "cnn_lstm64",
        "source_variant": "cnn_lstm_wide",
        "window": 320,
        "reused": True,
        "root": str(tmp_path),
        "config": cfg,
        "directory": str(tmp_path / "new/fits/imported"),
    }
    return tmp_path, source, origin, original, job


def test_import_preserves_source_bytes_and_native_checkpoint_identity(import_fixture):
    from virtual_imu.replacements.reuse import reuse_control

    _, _source, origin, original, job = import_fixture
    before = {p.name: sha(p) for p in origin.iterdir()}
    result = reuse_control(job)
    destination = Path(job["directory"])
    assert (
        result["reused"]
        and result["fit_kind"] == "reused_control"
        and result["training_cost_origin"] == "prior_run"
    )
    assert result["job_config_sha256"] == json_digest(job)
    assert torch.load(destination / "best.pt")["job"] == original
    assert sha(destination / "best.pt") == before["best.pt"]
    assert before == {p.name: sha(p) for p in origin.iterdir()}
    assert json.loads((destination / "import.json").read_text())["training_performed"] is False


@pytest.mark.parametrize("corruption", ["checkpoint", "source_code", "config", "source_receipt"])
def test_import_rejects_corrupted_source_before_writing_a_fit(import_fixture, corruption):
    from virtual_imu.replacements.reuse import reuse_control

    root, source, origin, _, job = import_fixture
    if corruption == "checkpoint":
        (origin / "best.pt").write_bytes(b"corrupt")
    if corruption == "source_code":
        (root / "dummy.py").write_text("changed code")
    if corruption == "config":
        job["config"]["training"]["lr"] = 0.1
    if corruption == "source_receipt":
        (source / "verification.json").write_text("{}")
    with pytest.raises(ValueError):
        reuse_control(job)
    assert not (Path(job["directory"]) / "fit.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "destination_checkpoint",
        "history",
        "stats",
        "source_init",
        "source_fit",
        "source_job",
        "receipt_fields",
        "source_after_import",
        "unsealed_provenance",
    ],
)
def test_import_resume_requires_sealed_native_bytes_and_complete_evidence(import_fixture, damage):
    from virtual_imu.replacements.reuse import completed_job, reuse_control

    _, source, origin, _, job = import_fixture
    reuse_control(job)
    directory = Path(job["directory"])
    if damage == "destination_checkpoint":
        saved = torch.load(directory / "best.pt")
        saved["model"]["weight"].add_(1)
        torch.save(saved, directory / "best.pt")
        fit = json.loads((directory / "fit.json").read_text())
        fit["checkpoint_sha256"] = sha(directory / "best.pt")
        write_json(directory / "fit.json", fit)
    elif damage in ["history", "stats", "source_init", "source_fit", "source_job"]:
        name = {
            "history": "history.csv",
            "stats": "stats.json",
            "source_init": "source_initialization.json",
            "source_fit": "source_fit.json",
            "source_job": "source_job.json",
        }[damage]
        (directory / name).write_text("changed")
    elif damage == "receipt_fields":
        receipt = json.loads((directory / "import.json").read_text())
        receipt["source_sha256"] = {}
        write_json(directory / "import.json", receipt)
    elif damage == "source_after_import":
        (origin / "best.pt").write_bytes(b"changed source")
    else:
        prov = json.loads((source / "provenance.json").read_text())
        prov["extra"] = "not sealed"
        write_json(source / "provenance.json", prov)
    with pytest.raises(ValueError):
        completed_job(job)
