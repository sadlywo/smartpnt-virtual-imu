"""Explicit, hash-bound reuse of completed controls without rewriting history."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from virtual_imu.ablations.training import json_digest, sha, write_json

LOG = logging.getLogger(__name__)
CONTROL_MAP = {
    "cnn_lstm64": "cnn_lstm_wide",
    "cnn_bilstm32": "cnn_bilstm",
    "cnn_identity": "cnn_only",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_reuse_compatibility(config, original):
    for section in ["data", "model"]:
        if config[section] != original[section]:
            raise ValueError("Reused control differs in " + section)
    # Only process concurrency is allowed to differ. It does not enter the
    # sampler, model, optimizer, schedule or numerical prediction definition.
    keys = set(config["training"]) | set(original["training"])
    for key in keys - {"workers"}:
        if config["training"].get(key) != original["training"].get(key):
            raise ValueError("Reused training recipe differs: " + key)
    if not set(config["seeds"]).issubset(original["seeds"]):
        raise ValueError("Source does not contain requested seeds")
    if not set(config["protocols"]).issubset(original["protocols"]):
        raise ValueError("Source does not contain requested protocols")
    if 320 not in original["windows"] or config["windows"] != [320]:
        raise ValueError("Reused controls require original window320")


def check_checkpoint_identity(job, record, saved, receipt=None):
    if record["job_id"] != job["id"] or record["job_config_sha256"] != json_digest(job):
        raise ValueError("Fit identity differs from planned replacement job")
    if bool(record.get("reused", False)) != bool(job.get("reused", False)):
        raise ValueError("Fit reuse identity differs")
    expected_job = job
    if job.get("reused", False):
        if receipt is None:
            raise ValueError("Missing explicit import receipt")
        expected_job = receipt["source_job"]
        original = receipt["source_fit"]
        if record["best_epoch"] != original["best_epoch"] or not np.isclose(
            record["best_val_score"], original["best_val_score"], rtol=1e-12, atol=0
        ):
            raise ValueError("Imported selection differs from original fit")
    if saved["job"] != expected_job or saved["epoch"] != record["best_epoch"]:
        raise ValueError("Checkpoint native training identity differs")
    if not np.isclose(saved["val_score"], record["best_val_score"], rtol=1e-12, atol=0):
        raise ValueError("Checkpoint minimum VALID value differs")


def sealed_source(job):
    """Trust source metadata only after binding it to the sealed manifest."""
    cfg, root = job["config"], Path(job["root"])
    source = root / cfg["reuse"]["run_dir"]
    seal = read_json(source / "complete.json")
    if seal["status"] != "complete" or seal["numerical_verification"] != "PASS":
        raise ValueError("Source run is not completed and verified")
    for name, key in [
        ("verification.json", "verification_sha256"),
        ("artifact_manifest.json", "artifact_manifest_sha256"),
    ]:
        if sha(source / name) != seal[key]:
            raise ValueError("Source completion hash differs: " + name)
    artifacts = read_json(source / "artifact_manifest.json")["files_sha256"]
    for name in ["config.yaml", "provenance.json"]:
        if artifacts.get(name) != sha(source / name):
            raise ValueError("Unsealed source metadata: " + name)
    old_cfg = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    validate_reuse_compatibility(cfg, old_cfg)
    prov = read_json(source / "provenance.json")
    if (
        sha(source / "config.yaml") != prov["config_sha256"]
        or sha(root / cfg["data"]["manifest"]) != prov["manifest_sha256"]
    ):
        raise ValueError("Source config or data manifest differs")
    for relative, expected in prov["source_sha256"].items():
        if (
            sha(root / relative) != expected
            or sha(source / "source_snapshot" / relative) != expected
        ):
            raise ValueError("Original sealed source changed: " + relative)
    return source, artifacts


def validate_import(job, record):
    """Use identical checks at import, resume and TEST unlock."""
    source, artifacts = sealed_source(job)
    if CONTROL_MAP.get(job["variant"]) != job["source_variant"]:
        raise ValueError("Unrecognized borrowed control mapping")
    source_id = f"{job['protocol']}__{job['source_variant']}__w320__s{job['seed']}"
    origin, directory = source / "fits" / source_id, Path(job["directory"])
    receipt = read_json(directory / "import.json")
    if (
        receipt.get("source_run") != str(source)
        or receipt.get("source_job_id") != source_id
        or receipt.get("training_performed") is not False
        or receipt.get("checkpoint_native_job_preserved") is not True
    ):
        raise ValueError("Import receipt native identity missing or mismatched")
    names = ["best.pt", "fit.json", "job.json", "history.csv", "initialization.json", "stats.json"]
    expected_hashes = {}
    for name in names:
        path = origin / name
        expected = artifacts.get(path.relative_to(source).as_posix())
        if expected != sha(path):
            raise ValueError("Imported sealed source changed: " + str(path))
        expected_hashes[str(path)] = expected
    for name in [
        "complete.json",
        "verification.json",
        "artifact_manifest.json",
        "config.yaml",
        "provenance.json",
    ]:
        expected_hashes[str(source / name)] = sha(source / name)
    if receipt.get("source_sha256") != expected_hashes:
        raise ValueError("Import receipt source hashes incomplete or different")
    native_job, native_fit = read_json(origin / "job.json"), read_json(origin / "fit.json")
    if (
        receipt.get("source_job") != native_job
        or receipt.get("source_fit") != native_fit
        or read_json(directory / "source_job.json") != native_job
        or read_json(directory / "source_fit.json") != native_fit
    ):
        raise ValueError("Import native job/fit evidence changed")
    if (
        native_job["id"] != source_id
        or native_fit["job_id"] != source_id
        or native_job["protocol"] != job["protocol"]
        or native_job["variant"] != job["source_variant"]
        or native_job["seed"] != job["seed"]
        or native_job["window"] != 320
        or native_fit["job_config_sha256"] != json_digest(native_job)
    ):
        raise ValueError("Import source job mapping differs")
    for name, native_name in [
        ("best.pt", "best.pt"),
        ("history.csv", "history.csv"),
        ("stats.json", "stats.json"),
        ("source_initialization.json", "initialization.json"),
    ]:
        if sha(directory / name) != expected_hashes[str(origin / native_name)]:
            raise ValueError("Imported destination differs from native artifact: " + name)
    if record["checkpoint_sha256"] != native_fit["checkpoint_sha256"]:
        raise ValueError("Imported checkpoint digest differs from native fit")
    if record["checkpoint_sha256"] != expected_hashes[str(origin / "best.pt")]:
        raise ValueError("Imported checkpoint differs from sealed bytes")
    init, native_init = (
        read_json(directory / "initialization.json"),
        read_json(origin / "initialization.json"),
    )
    if (
        init["init_state_sha256"] != native_init["init_state_sha256"]
        or receipt["replacement_init_sha256"] != init["init_state_sha256"]
    ):
        raise ValueError("Imported initialization state differs")
    saved = torch.load(directory / "best.pt", map_location="cpu")
    if saved["metadata"] != native_init or saved["stats"] != read_json(directory / "stats.json"):
        raise ValueError("Imported native metadata/statistics differ")
    check_checkpoint_identity(job, record, saved, receipt)
    return receipt


def completed_job(job):
    directory = Path(job["directory"])
    complete = directory / "fit.json"
    if not complete.is_file():
        return None
    record = read_json(complete)
    if record["job_config_sha256"] != json_digest(job):
        raise ValueError("Resume configuration mismatch")
    if sha(directory / "best.pt") != record["checkpoint_sha256"]:
        raise ValueError("Resume checkpoint hash mismatch")
    if record.get("reused", False):
        validate_import(job, record)
    LOG.info("SKIPS completed and verified replacement fit %s", job["id"])
    return record


def reuse_control(job):
    """Copy native checkpoint plus source evidence and write a distinct import record."""
    from virtual_imu.replacements.models import build_replacement_model

    previous = completed_job(job)
    if previous is not None:
        return previous
    cfg, root = job["config"], Path(job["root"])
    if CONTROL_MAP.get(job["variant"]) != job["source_variant"]:
        raise ValueError("Unrecognized borrowed control mapping")
    source = root / cfg["reuse"]["run_dir"]
    source, artifacts = sealed_source(job)
    source_id = f"{job['protocol']}__{job['source_variant']}__w320__s{job['seed']}"
    directory, origin = Path(job["directory"]), source / "fits" / source_id
    files = ["best.pt", "fit.json", "job.json", "history.csv", "initialization.json", "stats.json"]
    hashes = {}
    for name in files:
        path = origin / name
        digest = sha(path)
        if artifacts.get(path.relative_to(source).as_posix()) != digest:
            raise ValueError("Original artifact hash mismatch: " + str(path))
        hashes[str(path)] = digest
    for name in [
        "complete.json",
        "verification.json",
        "artifact_manifest.json",
        "config.yaml",
        "provenance.json",
    ]:
        hashes[str(source / name)] = sha(source / name)
    original_fit, original_job = read_json(origin / "fit.json"), read_json(origin / "job.json")
    if original_fit["job_config_sha256"] != json_digest(original_job):
        raise ValueError("Original native job digest differs")
    if (
        original_fit["job_id"] != source_id
        or original_job["protocol"] != job["protocol"]
        or original_job["seed"] != job["seed"]
    ):
        raise ValueError("Original protocol or seed differs")
    model, meta = build_replacement_model(
        job["variant"], job["seed"], cfg["model"], cfg["replacement_models"]
    )
    original_init = read_json(origin / "initialization.json")
    if meta["init_state_sha256"] != original_init["init_state_sha256"]:
        raise ValueError("Replacement control initialization is not identical to source")
    saved = torch.load(origin / "best.pt", map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    expected_stats_path = directory.parent.parent / "supports" / job["protocol"] / "statistics.json"
    if expected_stats_path.is_file() and read_json(expected_stats_path) != saved["stats"]:
        raise ValueError("Reused control does not share current TRAIN-only statistics")
    record = {
        **original_fit,
        "job_id": job["id"],
        "variant": job["variant"],
        "directory": str(directory),
        "job_config_sha256": json_digest(job),
        "reused": True,
        "fit_kind": "reused_control",
        "source_job_id": source_id,
        "source_run": str(source),
        "training_cost_origin": "prior_run",
    }
    receipt = dict(
        source_run=str(source),
        source_job_id=source_id,
        source_job=original_job,
        source_fit=original_fit,
        source_sha256=hashes,
        training_performed=False,
        checkpoint_native_job_preserved=True,
        replacement_init_sha256=meta["init_state_sha256"],
    )
    check_checkpoint_identity(job, record, saved, receipt)
    if sha(origin / "best.pt") != original_fit["checkpoint_sha256"]:
        raise ValueError("Original checkpoint checksum differs")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "history.csv", "stats.json"]:
        shutil.copy2(origin / name, directory / name)
    shutil.copy2(origin / "initialization.json", directory / "source_initialization.json")
    write_json(directory / "source_fit.json", original_fit)
    write_json(directory / "source_job.json", original_job)
    write_json(directory / "initialization.json", meta)
    write_json(directory / "job.json", job)
    write_json(directory / "import.json", receipt)
    validate_import(job, record)
    write_json(directory / "fit.json", record)
    LOG.info(
        "READS sealed control %s; WRITES import %s; SKIPS retraining and TEST predictions",
        origin,
        directory,
    )
    return record
