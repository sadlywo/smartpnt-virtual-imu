"""Validate finished reviewer workflows and stream a portable multipart tar.gz.

No network, credentials, deletion, training or inference. The original curated
training corpus is referenced by its existing bundle fingerprint, not duplicated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tarfile
import traceback
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

PART_BYTES = 2 * 1024**3
INNER_PATH = "metadata/package_manifest.json"
FINAL_SEEDS = {20261010, 20261011, 20261012}
AUX_STAGES = {"aux1": {"transfer", "downstream", "latency_gpu", "latency_cpu"}, "aux2": {"accuracy_tables", "classical_latency"}}
DENIED_DIRS = {".git", ".venv", ".venv312", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ssh", ".aws", ".azure", ".codex", "node_modules", "auth", "credentials", "secrets"}
DENIED_FILES = {"auth.json", "auth.toml", ".env", ".netrc", "credentials", "credentials.json", "credentials.toml", "known_hosts", "authorized_keys", "id_rsa", "id_ed25519", ".bash_history", ".zsh_history", ".python_history", "history.sqlite", "history.json"}
SECRET_NAME = re.compile(r"(^|[._-])(passwords?|credentials?|secrets?|private[._-]?key|api[._-]?key|access[._-]?token)([._-]|$)", re.IGNORECASE)
DEFAULTS = {
    "core_workflow_log": "runs/reviewer_workflow_status.log",
    "aux1_console_log": "postprocessing_console.log", "aux2_console_log": "final_analysis_console.log",
    "dataset_manifest": "data/processed/unified_imu_v2/manifest.json",
    "dataset_manifest_sha256": "5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260",
    "training_bundle_sha256": "870745185b756dbcb34337be9fa6432af7c50959f035b1dad822e45ea1fab981",
    "compression_level": 6,
    "core_trees": ["src", "configs", "scripts", "tests", "docs"],
    "core_files": ["bundle_manifest.json", "SHA256SUMS", "pyproject.toml", "uv.lock", "README.md", "AGENTS.md"],
    "core_globs": ["runs/server_launch_*", "runs/server_bootstrap_*", "runs/reviewer_protocol_20261024", "runs/reviewer*preflight*", "runs/reviewer*.log", "runs/server_reviewer_deployment_20261024", "runs/hpo_research_20261024/penn_groundtruth_audit", "runs/hpo_research_20261024/attention_literature.json"],
    "gt_trees": ["data/raw/penncosyvio_ground_truth", "data/processed/penncosyvio_ground_truth"],
}


def portable(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or value.startswith("/") or any(ord(character) < 32 for character in value):
        raise ValueError(f"Unsafe portable path: {value!r}")
    if any(part in {"", ".", ".."} or part.endswith((".", " ")) or any(character in '<>"|?*' for character in part) or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".")[0]) for part in value.split("/")):
        raise ValueError(f"Unsafe portable path: {value!r}")
    return PurePosixPath(value)


def _denied(relative: Path) -> bool:
    return any(part.casefold() in DENIED_DIRS for part in relative.parts) or relative.name.casefold() in DENIED_FILES or relative.suffix.casefold() in {".pem", ".key", ".p12", ".pfx", ".kdbx"} or bool(SECRET_NAME.search(relative.name))


def _signature(value) -> tuple:
    # Windows Python stat()/fstat() disagree on ctime after an atomic replace:
    # one can expose creation time and the other metadata-change time. Preserve
    # stable device/inode/size/mtime plus full bytes SHA on that platform.
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns if os.name != "nt" else None


def _safe_source(root: Path, path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes its declared root: {path}")
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or getattr(root_info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Symlink/reparse root is forbidden: {root}")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symlink/reparse source is forbidden: {current}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Only regular source files are exportable: {path}")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite workflow metadata")


class SourceAudit:
    def __init__(self, roots: dict[str, Path]):
        self.roots = roots
        self.cache = {}

    def owner(self, path: Path) -> Path:
        absolute = Path(os.path.abspath(path))
        choices = [root for root in self.roots.values() if absolute.is_relative_to(root)]
        if not choices:
            raise ValueError(f"Artifact is outside the explicit source roots: {path}")
        return max(choices, key=lambda item: len(item.parts))

    def inspect(self, path: Path) -> dict:
        root = self.owner(path)
        path = _safe_source(root, path)
        before = path.stat()
        signature = _signature(before)
        previous = self.cache.get(str(path))
        if previous:
            if previous["signature"] != signature:
                raise ValueError(f"Source changed during validation: {path}")
            return previous
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            if _signature(os.fstat(stream.fileno())) != signature:
                raise ValueError("Source identity changed before opening")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            if _signature(os.fstat(stream.fileno())) != signature:
                raise ValueError("Source changed while hashing")
        if _signature(path.stat()) != signature:
            raise ValueError("Source identity changed after hashing")
        result = {"sha256": digest.hexdigest(), "size_bytes": before.st_size, "signature": signature}
        self.cache[str(path)] = result
        return result

    def check(self, path: Path, expected: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or self.inspect(path)["sha256"] != expected:
            raise ValueError(f"Source SHA256 differs: {path}")
        return self.inspect(path)

    def json(self, path: Path) -> dict:
        expected = self.inspect(path)["sha256"]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("JSON changed between hashing and reading")
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {path}")
        _finite(value)
        return value

    def assert_unchanged(self) -> None:
        for identity, expected in self.cache.items():
            path = _safe_source(self.owner(Path(identity)), Path(identity))
            if _signature(path.stat()) != expected["signature"]:
                raise ValueError(f"Source changed after validation/streaming: {path}")


def _walk(root: Path, start: Path, skipped: list | None = None):
    if not start.exists():
        return
    if start.is_file() or start.is_symlink():
        relative = start.relative_to(root)
        if _denied(relative):
            if skipped is not None:
                skipped.append({"source_path": str(start), "reason": "excluded credential/runtime/history file"})
            return
        yield _safe_source(root, start)
        return
    for directory, names, files in os.walk(start, topdown=True, followlinks=False):
        base = Path(directory)
        keep = []
        for name in sorted(names):
            path = base / name
            if _denied(path.relative_to(root)):
                if skipped is not None:
                    skipped.append({"source_path": str(path), "reason": "excluded credential/runtime/cache directory"})
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Symlink/reparse directory is forbidden: {path}")
            keep.append(name)
        names[:] = keep
        for name in sorted(files):
            path = base / name
            if _denied(path.relative_to(root)):
                if skipped is not None:
                    skipped.append({"source_path": str(path), "reason": "excluded credential/runtime/history file"})
                continue
            yield _safe_source(root, path)


def _file_map(root: Path, values: dict, audit: SourceAudit, *, absolute=False) -> None:
    if not isinstance(values, dict) or not values:
        raise ValueError("Required source/output inventory is empty")
    for name, expected in values.items():
        path = Path(name) if absolute else root.joinpath(*portable(name).parts)
        _safe_source(audit.owner(path), path)
        if _denied(path.relative_to(audit.owner(path))):
            raise ValueError("Inventoried credential/history/runtime file is forbidden")
        audit.check(path, expected)


def _terminal(path: Path, marker: str, audit: SourceAudit, *, workflow=False) -> None:
    audit.inspect(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if workflow:
        exits = [line for line in lines if line.startswith("WORKFLOW_EXIT=")]
        if marker not in lines or not exits or exits[-1] != "WORKFLOW_EXIT=0" or lines.index(marker) > max(index for index, line in enumerate(lines) if line.startswith("WORKFLOW_EXIT=")):
            raise ValueError("Core terminal success markers are missing or unsuccessful")
    elif not [line for line in lines if line.strip()] or [line for line in lines if line.strip()][-1] != marker:
        raise ValueError(f"Auxiliary terminal marker is missing: {marker}")


def _source_roster(project: Path, state: dict, audit: SourceAudit) -> None:
    current = {path.relative_to(project).as_posix(): audit.inspect(path)["sha256"] for path in _walk(project, project / "src/virtual_imu") if path.suffix == ".py"}
    frozen = {}
    for name, expected in state["contract"]["source_sha256"].items():
        parts = PurePosixPath(str(name).replace("\\", "/")).parts
        indices = [index for index in range(len(parts) - 1) if parts[index:index + 2] == ("src", "virtual_imu")]
        if not indices:
            continue
        if len(indices) != 1 or ".." in parts:
            raise ValueError("Ambiguous frozen source identity")
        relative = PurePosixPath(*parts[indices[0]:]).as_posix()
        if relative in frozen and frozen[relative] != expected:
            raise ValueError("Conflicting frozen source hashes")
        frozen[relative] = expected
    if not frozen or current != frozen:
        raise ValueError("Executing/frozen core source roster or hashes differ")


def _core_header(project: Path, pipeline: Path, settings: dict, audit: SourceAudit, expected: tuple[int, int]) -> dict:
    state = audit.json(pipeline / "pipeline.json")
    cells = state.get("cells", {})
    finals = [job for job in state.get("jobs", {}).values() if job.get("phase") in {"final_default", "final_tuned"}]
    required = {(cell, f"final_{variant}", seed) for cell in cells for variant in ("default", "tuned") for seed in FINAL_SEEDS}
    observed = [(job["cell"], job["phase"], job["seed"]) for job in finals]
    if state.get("status") != "complete" or state.get("smoke") or state.get("active_job") or len(cells) != expected[0] or len(finals) != expected[1] or len(set(observed)) != expected[1] or set(observed) != required or any("selected" not in cell for cell in cells.values()) or any(job.get("status") != "complete" or job.get("evaluation_status") != "complete" for job in finals):
        raise ValueError("Core is not the complete selected/final identity matrix")
    if state.get("counts", {}).get("final_evaluated") != expected[1] or set(state.get("final_seeds", [])) != FINAL_SEEDS or state.get("final_variants") != ["default", "tuned"]:
        raise ValueError("Core final coverage counters disagree")
    _terminal(project / settings["core_workflow_log"], "REVIEWER_CORE_PASSED", audit, workflow=True)
    _source_roster(project, state, audit)
    matrix = audit.json(pipeline / "pipeline.json")["contract"]["config"]
    roster = {f"{protocol['id']}/{candidate['id']}" for protocol in matrix["protocols"] for candidate in matrix["candidates"] if protocol["task"] in candidate["tasks"]}
    if roster != set(cells):
        raise ValueError("Declared model/protocol roster differs from completed cells")
    return state


def _find_aux_run(aux: Path, filename: str, project: Path, pipeline: Path, audit: SourceAudit) -> tuple[Path, dict]:
    candidates = []
    for path in _walk(aux, aux / "runs"):
        if path.name != filename:
            continue
        state = audit.json(path)
        contract = state.get("contract", {})
        if Path(contract.get("project", "")).resolve() == project and Path(contract.get("pipeline", "")).resolve() == pipeline and Path(contract.get("aux", "")).resolve() == aux:
            candidates.append((path.parent, state))
    if len(candidates) != 1:
        raise ValueError("Require one exact auxiliary run identity for this core pipeline")
    return candidates[0]


def _check_stage(name: str, directory: Path, pipeline: Path, core_sha: str, audit: SourceAudit, expected_finals: int) -> dict:
    if name in {"transfer", "downstream", "classical_latency"}:
        contract = audit.json(directory / "contract.json")
        if contract.get("pipeline_sha256") != core_sha or Path(contract.get("pipeline", contract.get("pipeline_path", ""))).resolve() != pipeline:
            raise ValueError("Auxiliary stage contract refers to a different core")
    if name == "transfer":
        report = audit.json(directory / "transfer_metrics.json")
        records = report.get("source_model_records", [])
        identities = [(row["direction"], row["model_family"], row["variant"], row["seed"]) for row in records]
        if report.get("status") != "complete" or contract.get("nonfinal_debug_subset") or not records or len(records) != report.get("records") or len(records) != contract.get("expected_records") or len(identities) != len(set(identities)) or (expected_finals == 558 and len(records) != 120):
            raise ValueError("Transfer stage is not a completed formal result")
    elif name == "downstream":
        report = audit.json(directory / "complete.json")
        if report.get("status") != "complete" or set(report.get("unit_manifests", {})) != {"penn_loso_af", "penn_loso_as", "penn_loso_bf", "penn_only", "pooled"}:
            raise ValueError("Downstream five-view completion is incomplete")
        for unit, expected in report["unit_manifests"].items():
            manifest_path = directory / "units" / unit / "artifact_manifest.json"
            audit.check(manifest_path, expected)
            _file_map(manifest_path.parent, audit.json(manifest_path)["files"], audit)
        manifest_path = directory / "overview" / "artifact_manifest.json"
        audit.check(manifest_path, report["overview_manifest_sha256"])
        _file_map(manifest_path.parent, audit.json(manifest_path)["files"], audit)
    elif name.startswith("latency_"):
        report = audit.json(directory / "latency.json")
        state = audit.json(directory / "latency_state.json")
        if report.get("status") != "complete" or state.get("status") != "complete" or report.get("source_pipeline_sha256") != core_sha or Path(report.get("source_pipeline", "")).resolve() != pipeline or report.get("entries") != expected_finals or any(group.get("status") != "complete" for group in state.get("groups", {}).values()):
            raise ValueError("Neural latency result is incomplete or belongs to another core")
        _file_map(directory, state["final_artifact_sha256"], audit)
    elif name == "accuracy_tables":
        report = audit.json(directory / "summary.json")
        if report.get("status") != "complete" or report.get("source_pipeline_sha256") != core_sha or report.get("runs") != expected_finals or Path(report.get("pipeline_root", "")).resolve() != pipeline:
            raise ValueError("Accuracy aggregation is not complete for this core")
    else:
        report = audit.json(directory / "latency.json")
        if report.get("status") != "complete" or report.get("final_run_mappings") != expected_finals:
            raise ValueError("Classical timing coverage is incomplete")
    if not name.startswith("latency_"):
        manifest = audit.json(directory / "artifact_manifest.json") if (directory / "artifact_manifest.json").is_file() else None
        if name != "downstream" and manifest is None:
            raise ValueError("Completed stage lacks its output inventory")
        if manifest:
            _file_map(directory, manifest["files"], audit)
            if manifest.get("source_artifacts"):
                _file_map(directory, manifest["source_artifacts"], audit, absolute=True)
    return {"status": "passed", "directory": str(directory), "terminal_report": str(directory / ("transfer_metrics.json" if name == "transfer" else "complete.json" if name == "downstream" else "summary.json" if name == "accuracy_tables" else "latency.json")), "count": len(records) if name == "transfer" else 5 if name == "downstream" else expected_finals}


def _aux_header(key: str, aux: Path, project: Path, pipeline: Path, core_sha: str, settings: dict, audit: SourceAudit, expected_finals: int) -> tuple[Path, dict]:
    run, state = _find_aux_run(aux, "postprocessing_state.json" if key == "aux1" else "final_analysis_state.json", project, pipeline, audit)
    manifest_path = aux / "aux_manifest.json"
    manifest = audit.json(manifest_path)
    if state.get("status") != "complete" or set(state.get("stages", {})) != AUX_STAGES[key] or state.get("core_pipeline_sha256") != core_sha or state["contract"].get("aux_manifest_sha256") != audit.inspect(manifest_path)["sha256"] or any(stage.get("status") != "complete" for stage in state["stages"].values()):
        raise ValueError("Auxiliary state is not a fully completed exact-source workflow")
    if Path(manifest.get("base_project", "")).resolve() != project or (project / manifest.get("base_pipeline", "")).resolve() != pipeline:
        raise ValueError("Auxiliary source manifest refers to a different core identity")
    _file_map(aux, manifest["files"], audit)
    _terminal(aux / settings[f"{key}_console_log"], "REVIEWER_AUXILIARY_PASSED" if key == "aux1" else "REVIEWER_FINAL_ANALYSIS_PASSED", audit)
    stage_audit = {}
    for name, stage in state["stages"].items():
        directory = run / name
        actual = {path.relative_to(directory).as_posix(): audit.inspect(path)["sha256"] for path in _walk(aux, directory)}
        if not actual or actual != stage["artifact_sha256"]:
            raise ValueError(f"Auxiliary stage output inventory differs: {key}/{name}")
        stage_audit[name] = _check_stage(name, directory, pipeline, core_sha, audit, expected_finals)
    state["export_validation"] = stage_audit
    return run, state


def _frozen_hpo_module(project: Path, audit: SourceAudit):
    path = project / "src/virtual_imu/training/reviewer_hpo.py"
    expected = audit.inspect(path)["sha256"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Frozen validator changed before compilation")
    module = types.ModuleType("reviewer_export_frozen_hpo")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def _finite_predictions(path: Path, task: str, audit: SourceAudit) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    audit.inspect(path)
    axes = ("gx", "gy", "gz", "ax", "ay", "az")
    if task == "paired":
        methods = ("model", "identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear", "ridge_tuned", "fir_tuned", "arx_tuned")
        numeric = ["timestamp_s", *[f"{axis}_{side}" for side in ("low", "high") for axis in axes], *[f"{method}_{axis}" for method in methods for axis in axes]]
        identity = ["sequence_id", "dataset_id", "low_sensor", "high_sensor"]
    else:
        axes = ("delta_x_local_m", "delta_y_local_m", "delta_z_local_m")
        numeric = ["start_index", "end_index", "start_time_s", "end_time_s", *[f"{method}_{axis}" for method in ("true", "pred", "zero", "train_mean", "ridge", "ridge_tuned") for axis in axes]]
        identity = ["sequence_id", "original_sequence_id"]
    count = 0
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        if set(numeric + identity) - set(parquet.schema_arrow.names):
            raise ValueError("Final prediction parquet lacks required physical fields")
        for batch in parquet.iter_batches(batch_size=8192, columns=numeric + identity):
            for index in range(len(numeric)):
                if not np.isfinite(np.asarray(batch.column(index).to_numpy(zero_copy_only=False), dtype=float)).all():
                    raise ValueError("Nonfinite final prediction/target values")
            if any(batch.column(index).null_count for index in range(len(numeric), batch.num_columns)):
                raise ValueError("Missing prediction physical identity")
            count += batch.num_rows
    if not count:
        raise ValueError("Empty final prediction parquet")
    return {"path": str(path), "sha256": audit.inspect(path)["sha256"], "rows": count, "total_columns": len(parquet.schema_arrow.names), "checked_numeric_columns": numeric, "checked_identity_columns": identity, "numeric_finite": True, "identity_nonnull": True}


def _validate_completed_core(project: Path, pipeline: Path, state: dict, audit: SourceAudit) -> dict:
    import torch
    import yaml

    validator = _frozen_hpo_module(project, audit)
    selections = []
    for cell in state["cells"]:
        validator._verify_selection(cell, state, pipeline)
        selections.append({"cell": cell, "status": "passed", "sha256": state["cells"][cell]["selection_sha256"], "path": str(pipeline / "selections" / f"{cell.replace('/', '__')}.json"), "criterion": "reconstructed from complete source validation promotion/confirmation ledger"})
    completed = finals = prediction_rows = 0
    jobs = []
    for job in state["jobs"].values():
        config_path = Path(job["config"])
        if not config_path.resolve().is_relative_to(pipeline / "configs"):
            raise ValueError("Frozen job config is outside the exact pipeline")
        audit.check(config_path, job["config_sha256"])
        if job["status"] != "complete":
            if job["status"] != "failed":
                raise ValueError("Unfinished fit evidence in a supposedly complete core")
            jobs.append({"job_id": job["id"], "phase": job["phase"], "status": "skipped_as_failed", "config_path": str(config_path), "config_sha256": job["config_sha256"], "error": job.get("error")})
            continue
        directory = Path(job["run_dir"]).resolve()
        if not directory.is_relative_to(pipeline / "experiments"):
            raise ValueError("Completed fit is outside the exact pipeline")
        _file_map(directory, job["artifact_sha256"], audit)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8")) != config:
            raise ValueError("Completed run config differs from its frozen config")
        checkpoint_path = directory / "best.ckpt"
        audit.check(checkpoint_path, job["checkpoint_sha256"])
        with checkpoint_path.open("rb") as stream:
            checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        metric, score, epoch = config["training"]["checkpoint_metric"], checkpoint.get("best_metric"), checkpoint.get("best_epoch")
        if checkpoint.get("config") != config or checkpoint.get("model_config") != config["model"] or config.get("seed") != job["seed"] or checkpoint.get("checkpoint_metric") != metric or not metric.startswith("val_") or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0 or type(epoch) is not int or not 1 <= epoch <= config["training"]["epochs"] or not math.isclose(score, job["score"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Completed checkpoint/config/validation identity differs")
        weights = checkpoint.get("model_state_dict")
        if not weights or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item() for value in weights.values()):
            raise ValueError("Nonfinite or missing completed model tensors")
        for key in ("feature_stats", "target_stats"):
            if key in checkpoint:
                _finite(checkpoint[key])
        with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(float(row["epoch"])) for row in history]
        values = [float(row[metric]) for row in history]
        if not values or not all(math.isfinite(value) for value in values) or epochs != sorted(set(epochs)) or epoch not in epochs or not math.isclose(min(values), score, rel_tol=1e-6, abs_tol=1e-9) or not math.isclose(values[epochs.index(epoch)], score, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Completed validation history is invalid")
        completed += 1
        job_audit = {"job_id": job["id"], "phase": job["phase"], "status": "passed", "config_path": str(config_path), "config_sha256": job["config_sha256"], "checkpoint": {"path": str(checkpoint_path), "sha256": job["checkpoint_sha256"], "model_tensors": len(weights), "tensor_elements": sum(value.numel() for value in weights.values()), "finite": True}, "validation": {"metric": metric, "score": score, "best_epoch": epoch, "history_epochs": len(epochs), "finite_and_selected_minimum": True}}
        jobs.append(job_audit)
        del checkpoint, weights
        if not job["phase"].startswith("final_"):
            continue
        validator._verify_evaluation(job, pipeline)
        _file_map(project, job["evaluation_artifact_sha256"], audit, absolute=True)
        report = audit.json(pipeline / "evaluations" / f"{job['id']}.json")
        if report["job_id"] != job["id"] or report["checkpoint_sha256"] != job["checkpoint_sha256"] or set(report["metrics"].get("splits", {})) != {"validation", "test"}:
            raise ValueError("Final evaluation identity/held-out split coverage differs")
        task = state["cells"][job["cell"]]["protocol"]["task"]
        predictions = {}
        for split in ("validation", "test"):
            predictions[split] = _finite_predictions(directory / "evaluation" / f"{split}_predictions.parquet", task, audit)
            prediction_rows += predictions[split]["rows"]
        with (directory / "evaluation" / f"{task}_metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or {row["split"] for row in rows} != {"validation", "test"}:
            raise ValueError("Incomplete final physical metric rows")
        text_columns = {"split", "method", "dataset_id", "low_sensor", "high_sensor"}
        if any(not math.isfinite(float(value)) for row in rows for key, value in row.items() if key not in text_columns):
            raise ValueError("Nonfinite final physical metrics")
        metrics_path = directory / "evaluation" / f"{task}_metrics.csv"
        job_audit["evaluation"] = {"status": "passed", "summary_path": str(pipeline / "evaluations" / f"{job['id']}.json"), "summary_sha256": job["evaluation_sha256"], "artifact_sha256": job["evaluation_artifact_sha256"], "predictions": predictions, "metrics": {"path": str(metrics_path), "sha256": audit.inspect(metrics_path)["sha256"], "rows": len(rows), "numeric_finite": True, "splits": ["validation", "test"]}}
        finals += 1
    return {"status": "passed", "cells": len(state["cells"]), "validated_completed_fits": completed, "validated_final_evaluations": finals, "finite_prediction_rows": prediction_rows, "failed_jobs_preserved": sum(job["status"] == "failed" for job in state["jobs"].values()), "selections": selections, "jobs": jobs}


def _validate_support(project: Path, settings: dict, audit: SourceAudit) -> dict:
    bundle = audit.json(project / "bundle_manifest.json")
    if bundle.get("dataset_manifest_sha256") != settings["dataset_manifest_sha256"]:
        raise ValueError("Original bundle refers to a different curated dataset")
    for entry in bundle["entries"]:
        audit.check(project.joinpath(*portable(entry["path"]).parts), entry["sha256"])
    audit.check(project / settings["dataset_manifest"], settings["dataset_manifest_sha256"])
    for tree in settings["gt_trees"]:
        manifest = audit.json(project / tree / "manifest.json")
        for record in manifest.get("records", []):
            for path_key, sha_key in (("path", "sha256"), ("processed_path", "processed_sha256")):
                if path_key in record and sha_key in record:
                    audit.check(project.joinpath(*portable(record[path_key]).parts), record[sha_key])
    return {"original_bundle_sha256": settings["training_bundle_sha256"], "dataset_manifest_sha256": settings["dataset_manifest_sha256"], "bundle_entries_verified": len(bundle["entries"]), "bulk_dataset_omitted": True}


def _collect(project: Path, pipeline: Path, aux1: Path, aux2: Path, settings: dict, audit: SourceAudit) -> tuple[dict, list]:
    selected, skipped = {}, []

    def add(root, start, prefix, role, metadata_only=False):
        for path in _walk(root, start, skipped):
            if metadata_only and path.suffix.casefold() not in {".json", ".yaml", ".yml", ".md", ".txt", ".csv"}:
                continue
            relative = path.relative_to(root).as_posix()
            name = f"{prefix}/{relative}"
            portable(name)
            record = audit.inspect(path)
            selected[name] = {"source_path": str(path), "size_bytes": record["size_bytes"], "sha256": record["sha256"], "role": role}

    add(project, pipeline, "server/core_project", "core_new_run_including_search_and_failure_evidence")
    for name in settings["core_trees"] + settings["core_files"]:
        add(project, project.joinpath(*portable(name).parts), "server/core_project", "frozen_source_or_environment_metadata")
    for pattern in settings["core_globs"]:
        if pattern.startswith("/") or ".." in PurePosixPath(pattern).parts or "\\" in pattern:
            raise ValueError("Unsafe project include pattern")
        for path in sorted(project.glob(pattern)):
            add(project, path, "server/core_project", "server_launch_preflight_bootstrap_protocol_or_support_evidence")
    for name in settings["gt_trees"]:
        add(project, project / name, "server/core_project", "independent_optical_GT_supplement")
    add(project, (project / settings["dataset_manifest"]).parent, "server/core_project", "curated_dataset_metadata_external_bulk_reference", metadata_only=True)
    for root, prefix in ((aux1, "server/aux1"), (aux2, "server/aux2")):
        add(root, root, prefix, "auxiliary_sources_dependencies_results_logs_and_state")
    folded = [name.casefold() for name in selected]
    if len(folded) != len(set(folded)):
        raise ValueError("Archive paths collide on case-insensitive filesystems")
    return dict(sorted(selected.items())), skipped


class PartWriter:
    """Split one continuous gzip byte stream without a whole-archive temporary."""

    def __init__(self, directory: Path, part_bytes: int):
        self.directory, self.part_bytes = directory, part_bytes
        directory.mkdir(parents=True)
        self.global_hash = hashlib.sha256()
        self.parts, self.total, self.current, self.current_size = [], 0, None, 0

    def write(self, data):
        view = memoryview(data)
        count = len(view)
        self.global_hash.update(view)
        self.total += count
        while view:
            if self.current is None:
                self.path = self.directory / f"reviewer_results.tar.gz.part{len(self.parts) + 1:06d}.partial"
                self.current = self.path.open("xb")
                self.part_hash, self.current_size = hashlib.sha256(), 0
            amount = min(len(view), self.part_bytes - self.current_size)
            self.current.write(view[:amount])
            self.part_hash.update(view[:amount])
            self.current_size += amount
            view = view[amount:]
            if self.current_size == self.part_bytes:
                self._finish_part()
        return count

    def flush(self):
        if self.current is not None:
            self.current.flush()

    def _finish_part(self):
        self.current.flush()
        os.fsync(self.current.fileno())
        self.current.close()
        destination = self.path.with_suffix("")
        self.path.rename(destination)
        digest = self.part_hash.hexdigest()
        if destination.stat().st_size != self.current_size or _sha(destination) != digest:
            raise ValueError("Written part failed disk SHA/size verification")
        self.parts.append({"path": "parts/" + destination.name, "sha256": digest, "size_bytes": self.current_size})
        print(f"WRITES: verified compressed part {destination.name} bytes={self.current_size}", flush=True)
        self.current = None

    def finish(self):
        if self.current is not None:
            self._finish_part()

    def abort(self):
        if self.current is not None:
            if not self.current.closed:
                self.current.flush()
                self.current.close()
            self.current = None


class CheckedReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.count += len(value)
        return value


def verify_export(out: Path, *, verify_sources=True) -> dict:
    manifest = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete" or manifest.get("archive", {}).get("format") != "tar.gz" or manifest["inner_manifest"]["path"] != INNER_PATH:
        raise ValueError("Export is not a completed recognized package")
    inner_path = out.joinpath(*portable(INNER_PATH).parts)
    if _sha(inner_path) != manifest["inner_manifest"]["sha256"]:
        raise ValueError("Inner manifest copy SHA differs")
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    if inner.get("schema_version") != 1 or inner.get("status") != "complete" or not inner.get("files") or set(inner["files"]) != set(inner.get("file_metadata", {})) or INNER_PATH in inner["files"] or not manifest.get("parts"):
        raise ValueError("Inner source inventory is incomplete")
    for name in inner["files"]:
        portable(name)
    global_hash, total, seen = hashlib.sha256(), 0, set()
    for part in manifest["parts"]:
        relative = portable(part["path"])
        if len(relative.parts) != 2 or relative.parts[0] != "parts" or part["path"].casefold() in seen:
            raise ValueError("Unsafe or repeated part identity")
        seen.add(part["path"].casefold())
        path = _safe_source(out, out.joinpath(*relative.parts))
        if path.stat().st_size != part["size_bytes"] or _sha(path) != part["sha256"]:
            raise ValueError("Export part SHA/size differs")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                global_hash.update(chunk)
                total += len(chunk)
    if total != manifest["archive"]["size_bytes"] or global_hash.hexdigest() != manifest["archive"]["sha256"]:
        raise ValueError("Complete compressed stream SHA/size differs")
    if verify_sources:
        for name, expected in inner["files"].items():
            if inner["file_metadata"][name]["source_path"] is None:
                if not name.startswith("metadata/processing/") and name != "metadata/completion_audit.json":
                    raise ValueError("Only generated processing provenance may lack a source path")
                continue
            source = Path(inner["file_metadata"][name]["source_path"])
            roots = [Path(root) for key, root in inner["source_roots"].items() if key in {"core_project", "aux1_project", "aux2_project"}]
            owner = next((root for root in roots if source.is_relative_to(root)), None)
            if owner is None or _safe_source(owner, source).stat().st_size != inner["file_metadata"][name]["size_bytes"] or _sha(source) != expected:
                raise ValueError(f"Export source changed: {source}")
    return manifest


def export_results(project_root: Path, pipeline_root: Path, aux1_root: Path, aux2_root: Path, out: Path, *, config: dict | None = None, resume=False, _expected_counts=(93, 558), _part_bytes=PART_BYTES, _config_path: Path | None = None) -> dict:
    settings = {**DEFAULTS, **(config or {})}
    for path in (project_root, pipeline_root, aux1_root, aux2_root):
        absolute = Path(os.path.abspath(path))
        for parent in (absolute, *absolute.parents):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError(f"Symlink/reparse source root/ancestor is forbidden: {parent}")
    project, pipeline, aux1, aux2, out = [path.resolve() for path in (project_root, pipeline_root, aux1_root, aux2_root, out)]
    roots = {"core_project": project, "aux1_project": aux1, "aux2_project": aux2}
    if not pipeline.is_relative_to(project / "runs") or any(out.is_relative_to(root) for root in roots.values()) or len(set(roots.values())) != 3:
        raise ValueError("Require an exact core run and independent source/output roots")
    if _part_bytes < 1 or not 0 <= settings["compression_level"] <= 9:
        raise ValueError("Invalid compression or private test part budget")
    print(f"READS: exact completed core {pipeline}; both auxiliary source/output inventories", flush=True)
    print(f"WRITES: streaming verified tar.gz parts and manifests under {out}", flush=True)
    print("SKIPS: network/authentication, deletion, environments/history, original bulk datasets, training and inference", flush=True)
    audit = SourceAudit(roots)
    state = _core_header(project, pipeline, settings, audit, _expected_counts)
    core_sha = audit.inspect(pipeline / "pipeline.json")["sha256"]
    aux1_run, aux1_state = _aux_header("aux1", aux1, project, pipeline, core_sha, settings, audit, _expected_counts[1])
    aux2_run, aux2_state = _aux_header("aux2", aux2, project, pipeline, core_sha, settings, audit, _expected_counts[1])
    if Path(aux2_state["contract"].get("preceding_state", "")).resolve() != aux1_run / "postprocessing_state.json" or aux2_state["contract"].get("preceding_manifest_sha256") != audit.inspect(aux1 / "aux_manifest.json")["sha256"]:
        raise ValueError("Aux2 preceding workflow identity differs from the verified aux1")
    if out.exists():
        if not resume:
            raise FileExistsError("Completed output exists; explicit --resume required")
        manifest = verify_export(out)
        if manifest.get("exporter_sha256") != _sha(Path(__file__)) or manifest.get("config") != settings:
            raise ValueError("Completed export implementation/config contract changed")
        return manifest
    if resume:
        attempts = list(out.parent.glob(out.name + ".partial.*"))
        raise RuntimeError(f"No completed export to resume; partial attempts are preserved: {len(attempts)}")
    validation = _validate_completed_core(project, pipeline, state, audit)
    support = _validate_support(project, settings, audit)
    files, skipped = _collect(project, pipeline, aux1, aux2, settings, audit)
    import yaml

    exporter_source = Path(__file__).resolve()
    source_bytes = exporter_source.read_bytes()
    exporter_sha = hashlib.sha256(source_bytes).hexdigest()
    snapshot = yaml.safe_dump(settings, sort_keys=True, allow_unicode=True).encode("utf-8")
    processing = {"exporter_source_path": str(exporter_source), "exporter_sha256": exporter_sha, "config_source_path": str(_config_path.resolve()) if _config_path else None, "config_source_sha256": _sha(_config_path) if _config_path else None, "effective_config_sha256": hashlib.sha256(snapshot).hexdigest(), "python": sys.version, "python_executable": sys.executable, "source_prefix_mappings": "The three experiment project roots only; generated processing members have source_path null."}
    generated = {"metadata/processing/export_reviewer_results.py": source_bytes, "metadata/processing/export_config.yaml": snapshot, "metadata/processing/provenance.json": _json_bytes(processing)}
    completion_audit = {"schema_version": 1, "status": "passed", "synthetic": _expected_counts != (93, 558), "counts": {"selected_cells": validation["cells"], "final_evaluated": validation["validated_final_evaluations"], "transfer_models": aux1_state["export_validation"]["transfer"]["count"], "downstream_views": aux1_state["export_validation"]["downstream"]["count"]}, "source_pipeline": str(pipeline / "pipeline.json"), "source_pipeline_sha256": core_sha, "checkpoint_checks": [{"job_id": job["job_id"], "phase": job["phase"], "source_path": job["checkpoint"]["path"], "sha256": job["checkpoint"]["sha256"], "finite": True, "tensor_count": job["checkpoint"]["model_tensors"], "tensor_elements": job["checkpoint"]["tensor_elements"]} for job in validation["jobs"] if job["status"] == "passed"], "prediction_checks": [{"job_id": job["job_id"], "split": split, "source_path": record["path"], "sha256": record["sha256"], "finite": record["numeric_finite"], "rows": record["rows"], "columns": record["total_columns"], "checked_numeric_columns": record["checked_numeric_columns"], "checked_identity_columns": record["checked_identity_columns"], "identity_nonnull": record["identity_nonnull"]} for job in validation["jobs"] for split, record in job.get("evaluation", {}).get("predictions", {}).items()], "selection_checks": validation["selections"], "fit_checks": validation["jobs"], "auxiliary_checks": {"aux1": aux1_state["export_validation"], "aux2": aux2_state["export_validation"]}, "marker_checks": [{"source_path": str(path), "sha256": audit.inspect(path)["sha256"], "status": "passed", "required_marker": marker} for path, marker in ((project / settings["core_workflow_log"], "REVIEWER_CORE_PASSED + final WORKFLOW_EXIT=0"), (aux1 / settings["aux1_console_log"], "REVIEWER_AUXILIARY_PASSED"), (aux2 / settings["aux2_console_log"], "REVIEWER_FINAL_ANALYSIS_PASSED"))], "source_roster_check": {"status": "passed", "scope": "Complete src/virtual_imu Python roster compared to pipeline contract.source_sha256; all bundle, selection, fit, evaluation, auxiliary source/output declared SHA inventories verified."}, "processing_source_sha256": exporter_sha}
    generated["metadata/completion_audit.json"] = _json_bytes(completion_audit)
    for name, data in generated.items():
        files[name] = {"source_path": None, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "role": "export_processing_source_or_effective_config_provenance"}
    files = dict(sorted(files.items()))
    source_roots = {**{key: str(value) for key, value in roots.items()}, "core_pipeline": str(pipeline), "aux1": str(aux1_run), "aux2": str(aux2_run)}
    mirrors = {"core_project": "server/core_project", "core_pipeline": "server/core_project/" + pipeline.relative_to(project).as_posix(), "aux1_project": "server/aux1", "aux1": "server/aux1/" + aux1_run.relative_to(aux1).as_posix(), "aux2_project": "server/aux2", "aux2": "server/aux2/" + aux2_run.relative_to(aux2).as_posix()}
    inner = {"schema_version": 1, "status": "complete", "synthetic": _expected_counts != (93, 558), "created_utc": datetime.now(UTC).isoformat(), "roots": mirrors, "source_roots": source_roots, "path_mappings": [{"source_prefix": str(root), "mirror_prefix": mirrors[key]} for key, root in roots.items()], "files": {name: entry["sha256"] for name, entry in files.items()}, "file_metadata": {name: {key: value for key, value in entry.items() if key != "sha256"} for name, entry in files.items()}, "validation": {"core": {key: value for key, value in validation.items() if key not in {"jobs", "selections"}}, "aux1_stages": sorted(aux1_state["stages"]), "aux2_stages": sorted(aux2_state["stages"]), "audit_path": "metadata/completion_audit.json"}, "external_inputs": support, "excluded": skipped, "scope": "All exact new core run evidence, frozen source/support/GT supplement, both complete auxiliary projects; original curated bulk data restored from the already verified original training bundle."}
    inner_data = _json_bytes(inner)
    attempt = out.with_name(out.name + ".partial." + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    writer = PartWriter(attempt / "parts", _part_bytes)
    try:
        (attempt / "metadata").mkdir()
        (attempt / INNER_PATH).write_bytes(inner_data)
        with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=settings["compression_level"], mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                info = tarfile.TarInfo(INNER_PATH)
                info.size, info.mode, info.mtime = len(inner_data), 0o644, 0
                archive.addfile(info, io.BytesIO(inner_data))
                for index, (name, entry) in enumerate(files.items()):
                    if name in generated:
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = entry["size_bytes"], 0o644, 0
                        archive.addfile(info, io.BytesIO(generated[name]))
                        continue
                    source = Path(entry["source_path"])
                    before = audit.inspect(source)
                    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    with os.fdopen(os.open(source, flags), "rb") as stream:
                        if _signature(os.fstat(stream.fileno())) != before["signature"]:
                            raise ValueError("Source changed before tar streaming")
                        reader = CheckedReader(stream)
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = entry["size_bytes"], 0o644, 0
                        archive.addfile(info, reader)
                        if reader.count != entry["size_bytes"] or reader.digest.hexdigest() != entry["sha256"] or stream.read(1) or _signature(os.fstat(stream.fileno())) != before["signature"]:
                            raise ValueError("Source content/identity changed during tar streaming")
                    if index and index % 500 == 0:
                        print(f"WRITES: streamed {index}/{len(files)} source files", flush=True)
        writer.finish()
        audit.assert_unchanged()
        if _sha(exporter_source) != exporter_sha or (_config_path and _sha(_config_path) != processing["config_source_sha256"]):
            raise ValueError("Export processing source/config changed during streaming")
        outer = {"schema_version": 1, "status": "complete", "archive": {"format": "tar.gz", "sha256": writer.global_hash.hexdigest(), "size_bytes": writer.total}, "inner_manifest": {"path": INNER_PATH, "sha256": hashlib.sha256(inner_data).hexdigest()}, "parts": writer.parts, "source_roots": source_roots, "files": len(files), "source_bytes": sum(entry["size_bytes"] for entry in files.values()), "config": settings, "exporter_sha256": exporter_sha, "processing_provenance": processing, "source_hash_verification": "Initial SHA plus streamed SHA equality and final unchanged source stat identities; each completed part was read back and SHA-verified."}
        _write_json(attempt / "export_manifest.json", outer)
        verify_export(attempt, verify_sources=False)
        attempt.rename(out)
        return outer
    except Exception as error:
        writer.abort()
        _write_json(attempt / "failure.json", {"status": "partial_failed_attempt_preserved", "error": repr(error), "traceback": traceback.format_exc(), "completed_parts": writer.parts})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "pipeline-root", "aux1-root", "aux2-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/reviewer_export.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    print(f"READS: export config {args.config}", flush=True)
    result = export_results(args.project_root, args.pipeline_root, args.aux1_root, args.aux2_root, args.out, config=yaml.safe_load(args.config.read_text(encoding="utf-8")), resume=args.resume, _config_path=args.config)
    print(f"WRITES: complete export {args.out} files={result['files']} parts={len(result['parts'])} compressed_bytes={result['archive']['size_bytes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
