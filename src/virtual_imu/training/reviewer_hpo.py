"""Validation-only random search with explicit restarted successive halving.

Search, seed confirmation, final refits and final evaluation have separate state.
Test evaluation is invoked only after every selected configuration is frozen.
"""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

LOW_FEATURES = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
PROTOCOL_TASKS = {
    "pooled": "paired", "alimu_only": "paired", "penn_only": "paired",
    "penn_loso_af": "paired", "penn_loso_as": "paired", "penn_loso_bf": "paired",
    "navigation_rotation_0": "navigation", "navigation_rotation_1": "navigation",
    "navigation_rotation_2": "navigation",
}
TrainBackend = Callable[[str, dict, Path], Path]
ValidationBackend = Callable[[Path, dict], dict]
EvaluationBackend = Callable[[str, Path, dict], dict]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _freeze_yaml(path: Path, value: dict) -> None:
    if path.is_file():
        if yaml.safe_load(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Frozen config changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _verify_artifact_hashes(directory: Path, hashes: dict) -> None:
    for name, expected in hashes.items():
        if _sha(directory / name) != expected:
            raise ValueError(f"Completed artifact checksum changed: {name}")


def _recoverable_training_error(error: Exception) -> bool:
    if isinstance(error, (MemoryError, FloatingPointError)):
        return True
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    oom_type = getattr(cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    # Older PyTorch releases expose CUDA OOM through RuntimeError instead of a
    # dedicated exception. Other RuntimeErrors are implementation failures.
    return isinstance(error, RuntimeError) and any(marker in str(error).lower() for marker in ("cuda out of memory", "cuda error: out of memory"))


def _release_resources() -> dict:
    result = {"gc_collected": gc.collect(), "cuda_cache_cleared": False}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                result["cuda_cache_cleared"] = True
        except (RuntimeError, MemoryError) as error:
            result["cleanup_error"] = str(error)
    return result


def _runtime_snapshot(torch, config: dict) -> dict:
    cuda = torch.backends.cuda
    device = str(config.get("device", "cuda"))
    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        chosen = torch.device(device)
        properties = torch.cuda.get_device_properties(chosen)
        gpu = {"name": properties.name, "total_memory_bytes": properties.total_memory, "compute_capability": list(torch.cuda.get_device_capability(chosen))}
    return {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "device": device, "gpu": gpu, "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "tf32_matmul": cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "sdpa": {name: bool(getattr(cuda, name)()) for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled") if hasattr(cuda, name)},
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _evaluation_inventory(run_dir: Path, report_path: Path) -> dict[str, str]:
    paths = {path.resolve() for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".json", ".csv"}}
    paths.add(report_path.resolve())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cache = report.get("metrics", {}).get("classical_cache", {})
    if cache.get("path"):
        cache_path = Path(cache["path"]).resolve()
        if _sha(cache_path) != cache["sha256"]:
            raise ValueError("Final evaluation classical cache checksum differs")
        paths.add(cache_path)
    return {str(path): _sha(path) for path in sorted(paths)}


def _verify_evaluation(job: dict, root: Path) -> None:
    report = root / "evaluations" / f"{job['id']}.json"
    if _sha(report) != job["evaluation_sha256"]:
        raise ValueError("Completed evaluation summary changed")
    if _evaluation_inventory(Path(job["run_dir"]), report) != job.get("evaluation_artifact_sha256"):
        raise ValueError("Completed evaluation artifact checksum or inventory changed")


def _verify_selection(cell_id: str, state: dict, root: Path) -> None:
    cell = state["cells"][cell_id]
    selected = cell["selected"]
    path = root / "selections" / f"{cell_id.replace('/', '__')}.json"
    if _sha(path) != cell.get("selection_sha256") or json.loads(path.read_text(encoding="utf-8")) != selected:
        raise ValueError("Frozen selection JSON checksum or selected state changed")
    config = state["contract"]["config"]
    search = config["search"]
    trials = {trial["id"]: trial for trial in cell["trials"]}
    survivors = list(trials)
    final_search = {}

    def find_job(trial_id: str, phase: str, seed: int, rung: int | None, epochs: int) -> dict:
        jobs = [job for job in state["jobs"].values() if job["cell"] == cell_id and job["trial_id"] == trial_id and job["phase"] == phase and job["seed"] == seed and job["rung"] == rung and job["epochs"] == epochs]
        if len(jobs) != 1 or jobs[0]["status"] not in {"complete", "failed"}:
            raise ValueError("Selection lacks the required completed validation jobs")
        job = jobs[0]
        if job["status"] == "failed" and not job.get("failure_recoverable", False):
            raise ValueError("Selection passed over a nonrecoverable failed job")
        return job

    for rung, epochs in enumerate(search["rung_epochs"]):
        jobs = [find_job(identifier, "search", config["seed"], rung, epochs) for identifier in survivors]
        successful = sorted((job for job in jobs if job["status"] == "complete"), key=lambda job: (job["score"], job["trial_id"]))
        if not successful:
            raise ValueError("Selection has no successful validation candidates")
        keep = search["rung_keep"][min(rung + 1, len(search["rung_keep"]) - 1)]
        survivors = [job["trial_id"] for job in successful[:keep]]
        if cell["rungs"][str(rung)] != {"epochs": epochs, "successful": len(successful), "promoted": survivors}:
            raise ValueError("Saved promotions disagree with validation ranks")
        final_search = {job["trial_id"]: job for job in successful}
    choices = []
    for identifier in survivors:
        confirmation = find_job(identifier, "confirmation", config["confirmation_seed"], None, search["rung_epochs"][-1])
        if confirmation["status"] == "complete":
            original = final_search[identifier]
            choices.append(((original["score"] + confirmation["score"]) / 2, identifier, [original["id"], confirmation["id"]]))
    if not choices:
        raise ValueError("Selection lacks any successful two-seed confirmation")
    score, identifier, selection_jobs = min(choices, key=lambda choice: (choice[0], choice[1]))
    if selected["trial"] != trials[identifier] or selected["selection_jobs"] != selection_jobs or not math.isclose(selected["mean_two_seed_validation_score"], score, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Selected winner disagrees with complete validation/confirmation ranks")


def _set_dotted(mapping: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = copy.deepcopy(value)


def _sample_domain(rng: random.Random, domain: Any) -> Any:
    if isinstance(domain, list):
        domain = {"choice": domain}
    if not isinstance(domain, dict) or len(domain) != 1:
        raise ValueError(f"A search domain requires exactly one sampling rule: {domain}")
    name, values = next(iter(domain.items()))
    if name == "choice":
        if not isinstance(values, list) or not values:
            raise ValueError("choice requires a nonempty list")
        return copy.deepcopy(rng.choice(values))
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} requires [lower, upper]")
    lower, upper = values
    if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float)) and math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise ValueError("Search bounds must be finite and increasing")
    if name == "uniform":
        return rng.uniform(lower, upper)
    if name == "loguniform" and lower > 0:
        return math.exp(rng.uniform(math.log(lower), math.log(upper)))
    if name == "int" and type(lower) is int and type(upper) is int:
        return rng.randint(lower, upper)
    raise ValueError(f"Unsupported or invalid search domain: {domain}")


def deterministic_candidates(config: dict, candidate: dict, task: str) -> list[dict]:
    """Draw the same unique family/task candidates in every training scope."""
    search = config["search"]
    count = int(search["trials"])
    if count < 1:
        raise ValueError("search.trials must be positive")
    domains = {**search.get("common_space", {}), **search.get(f"{task}_space", {}), **candidate.get("search_space", {})}
    protected = {"data.window_size", "data.batch_size", "training.lr", "training.weight_decay", "loss_profile"}
    if any(not (key in protected or key.startswith("model.")) for key in domains):
        raise ValueError("Search may change model parameters, context, batch, lr, weight decay and loss profile only")
    identity = f"{int(config['seed'])}/{candidate['id']}/{task}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest()[:8], "big"))
    default = {
        "training.lr": 0.001, "training.weight_decay": 1e-5,
        "data.batch_size": 128, "data.window_size": 80,
        "loss_profile": "composite" if task == "paired" else "huber_only",
    }
    for key in domains:
        if key.startswith("model."):
            value: Any = candidate["model"]
            for part in key.split(".")[1:]:
                if not isinstance(value, dict) or part not in value:
                    raise ValueError(f"Candidate template must freeze the default for {key}")
                value = value[part]
            default[key] = copy.deepcopy(value)
    seen = {_canonical(default)}
    trials = [{"id": "trial_000", "parameters": default, "is_default": True}]
    if count == 1:
        return trials
    for _ in range(max(1000, count * 1000)):
        parameters = {key: _sample_domain(rng, domain) for key, domain in sorted(domains.items())}
        encoded = _canonical(parameters)
        if encoded not in seen:
            seen.add(encoded)
            trials.append({"id": f"trial_{len(trials):03d}", "parameters": parameters, "is_default": False})
            if len(trials) == count:
                return trials
    raise ValueError("Search space does not contain enough distinct requested candidates")


def _navigation_roles(identifier: str) -> dict[str, list[str]]:
    trajectories = ("Traj1", "Traj3", "Traj4")
    index = int(identifier.rsplit("_", 1)[-1])
    return {"test": [trajectories[index]], "train": [trajectories[(index + 1) % 3]], "validation": [trajectories[(index + 2) % 3]]}


def build_trial_config(
    config: dict, source: dict, protocol: dict, candidate: dict, parameters: dict,
    *, seed: int, epochs: int, phase: str, output_root: Path,
) -> dict:
    """Apply sampled values, then enforce the shared scientific/data contract."""
    actual = copy.deepcopy(source)
    task, identifier = protocol["task"], protocol["id"]
    actual["seed"] = seed
    actual["experiment_name"] = output_root.name
    actual["model"] = copy.deepcopy(candidate["model"])
    for key, value in parameters.items():
        if key != "loss_profile":
            _set_dotted(actual, key, value)
    data, training, model = actual.setdefault("data", {}), actual.setdefault("training", {}), actual["model"]
    split_root = Path(config.get("protocol_split_root", "runs/reviewer_protocol_20261024/splits")) / identifier
    data.update(
        split_manifest=str(split_root / "split_manifest.json"),
        strict_manifest=str(split_root / "strict_manifest.json"),
        load_splits=["train", "validation"], rate_hz=40.0, stride=40,
    )
    model["in_channels"] = 6
    if task == "paired":
        data["feature_columns"] = list(LOW_FEATURES)
        data["minimum_sequence_samples"] = 320
        data.pop("static_context", None)
        model.update(target_mode="residual", out_channels=6)
        if int(data["window_size"]) not in {80, 160, 320}:
            raise ValueError("Paired context must be 80, 160 or 320 points")
        training["checkpoint_metric"] = "val_grouped_score"
        training["validation_score"] = {"method": "grouped_relative_rmse", "common_context_points": 320}
        if "window_size" in model:
            model["window_size"] = int(data["window_size"])
    else:
        roles = protocol.get("roles", _navigation_roles(identifier))
        data.update(
            assignment_manifest=str(split_root / "assignment_manifest.json"),
            window_size=80, train_sequences=roles["train"], val_sequences=roles["validation"],
            test_sequences=roles["test"], label_mode="position_delta",
        )
        model.update(out_channels=3, target_layout="window", prediction_head="last")
        # Window-dependent backbones must share the fixed navigation target span.
        if "window_size" in model:
            model["window_size"] = 80
        training["checkpoint_metric"] = "val_vector_rmse"
    profile = parameters.get("loss_profile", "composite")
    if profile not in {"huber_only", "mse_only", "composite"}:
        raise ValueError(f"Unknown loss profile: {profile}")
    training["loss"] = "mse" if profile == "mse_only" else "huber"
    if task == "paired":
        training.update(
            gyro_signal_weight=2.0 if profile == "composite" else 1.0,
            accel_signal_weight=1.0,
            derivative_weight=0.02 if profile == "composite" else 0.0,
            spectral_weight=0.02 if profile == "composite" else 0.0,
            preintegration_weight=0.005 if profile == "composite" else 0.0,
            allan_weight=0.0,
        )
    training.update(
        epochs=epochs, device=config.get("device", "cuda"), early_stopping_patience=15,
        evaluate_test=False, disable_test_evaluation=True,
        lr_scheduler="cosine_warmup", warmup_epochs=min(5, epochs // 4),
        min_lr=1e-6, grad_clip_norm=1.0,
    )
    actual.setdefault("output", {})["runs_dir"] = str(output_root)
    actual["reviewer_protocol"] = {
        "protocol_id": identifier, "candidate_id": candidate["id"], "phase": phase,
        "model_family": candidate["id"],
        "variant": phase.removeprefix("final_") if phase.startswith("final_") else "search",
        "loss_profile": profile, "parameters": copy.deepcopy(parameters),
        "selection_scope": "validation_only", "promotion": "restart_from_scratch_no_optimizer_resume",
        "historical_nine_channel_results_are_unmatched": True,
    }
    return actual


def read_validation_score(run_dir: Path, expected_config: dict) -> dict:
    """Verify a trained artifact and read only its saved validation criterion."""
    import torch

    print(f"READS: validation checkpoint and history {run_dir}", flush=True)
    frozen = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    if frozen != expected_config:
        raise ValueError("Run config differs from the frozen trial config")
    checkpoint_path = run_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != expected_config or checkpoint.get("model_config") != expected_config["model"]:
        raise ValueError("Checkpoint configuration differs from the frozen trial")
    score, epoch = checkpoint.get("best_metric"), checkpoint.get("best_epoch")
    if not isinstance(score, (float, int)) or not math.isfinite(score) or score < 0:
        raise ValueError("Checkpoint validation score must be finite and nonnegative")
    if type(epoch) is not int or not 1 <= epoch <= expected_config["training"]["epochs"]:
        raise ValueError("Invalid best validation checkpoint epoch")
    metric = expected_config["training"]["checkpoint_metric"]
    if checkpoint.get("checkpoint_metric") != metric:
        raise ValueError("Checkpoint selected a different validation criterion")
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict) or not weights or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Checkpoint model weights must be finite")
    with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history or any(metric not in row or not math.isfinite(float(row[metric])) for row in history):
        raise ValueError("Validation history lacks finite selection metrics")
    epochs = [int(float(row["epoch"])) for row in history]
    values = [float(row[metric]) for row in history]
    if epochs != sorted(set(epochs)) or epoch not in epochs:
        raise ValueError("Validation history epochs are inconsistent")
    if not math.isclose(min(values), score, rel_tol=1e-6, abs_tol=1e-9) or not math.isclose(values[epochs.index(epoch)], score, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("Checkpoint score/epoch do not match the minimum saved validation metric")
    for name in ("test_metrics.json", "test_predictions.parquet"):
        if (run_dir / name).exists():
            raise ValueError("Training unexpectedly emitted automatic test artifacts")
    return {"score": float(score), "best_epoch": epoch, "metric": metric}


def _default_train(task: str, config: dict, frozen: Path) -> Path:
    if task == "paired":
        from virtual_imu.training.trainer import train_model

        return train_model(config, config_path=frozen)
    from virtual_imu.training.gt_trainer import train_gt_supervised_model

    return train_gt_supervised_model(config, config_path=frozen)


def _default_evaluate(task: str, run_dir: Path, config: dict) -> dict:
    module = importlib.import_module("virtual_imu.eval.reviewer_evaluation")
    return module.evaluate_reviewer_run(task, run_dir, device=config["training"]["device"])


def _validate(config: dict) -> None:
    search = config["search"]
    budgets, keeps = search["rung_epochs"], search["rung_keep"]
    if len(budgets) != len(keeps) or not budgets or budgets != sorted(set(budgets)) or any(type(value) is not int or value < 1 for value in budgets + keeps):
        raise ValueError("Require positive increasing rung epochs and one positive keep count per rung")
    if keeps[0] != search["trials"] or keeps != sorted(keeps, reverse=True):
        raise ValueError("rung_keep must start at trials and decrease monotonically")
    seeds = config["seeds"]
    all_seeds = [config["seed"], config["confirmation_seed"], *seeds]
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in all_seeds) or len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Search, confirmation and final seeds must be distinct unsigned integers")
    if config.get("deterministic", True) is not True:
        raise ValueError("Reviewer HPO requires deterministic=true")
    default_seeds = config.get("comparison", {}).get("default_seeds", seeds)
    if default_seeds != seeds:
        raise ValueError("Default and tuned comparisons require the same final seeds")
    identifiers = [candidate["id"] for candidate in config["candidates"]]
    if len(set(identifiers)) != len(identifiers) or any(not re.fullmatch(r"[a-z][a-z0-9_-]*", name) for name in identifiers):
        raise ValueError("Candidate ids must be unique lowercase slugs")
    for candidate in config["candidates"]:
        if not candidate.get("model", {}).get("type") or not candidate.get("tasks"):
            raise ValueError("Candidates require a model type and declared tasks")
    protocols = config.get("protocols", [])
    if not protocols or len({protocol["id"] for protocol in protocols}) != len(protocols):
        raise ValueError("Require distinct declared protocols")
    if any(PROTOCOL_TASKS.get(protocol["id"]) != protocol["task"] for protocol in protocols):
        raise ValueError("Unknown protocol id or mismatched protocol task")


def _summary(root: Path, state: dict) -> None:
    rows = list(state["jobs"].values())
    fields = ["id", "cell", "phase", "trial_id", "rung", "seed", "epochs", "status", "score", "run_dir", "error", "evaluation_status"]
    with (root / "job_registry.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    search = state["contract"]["config"]["search"]
    search_jobs_per_cell = sum(search["rung_keep"]) + search["rung_keep"][-1]
    final_runs = len(state["cells"]) * len(state["final_seeds"]) * len(state["final_variants"])
    state["counts"] = {
        "cells": len(state["cells"]), "selected_cells": sum("selected" in cell for cell in state["cells"].values()),
        "training_jobs_materialized": len(rows), "training_jobs_complete": sum(row["status"] == "complete" for row in rows),
        "failed_training_jobs": sum(row["status"] == "failed" for row in rows),
        "final_evaluated": sum(row.get("evaluation_status") == "complete" for row in rows),
        "expected_final_runs": final_runs,
        "planned_search_training_jobs": len(state["cells"]) * search_jobs_per_cell,
        "planned_total_training_jobs": len(state["cells"]) * search_jobs_per_cell + final_runs,
    }
    _write_json(root / "pipeline.json", state)


def run_reviewer_hpo(
    config: dict, *, config_path: Path | None = None, root: Path | None = None,
    train_backend: TrainBackend | None = None, evaluate_backend: EvaluationBackend | None = None,
    validation_backend: ValidationBackend | None = None, plan_only: bool = False,
    smoke: bool = False, resume_root: Path | None = None,
    protocol_filter: list[str] | None = None, cell_filter: list[str] | None = None,
) -> Path:
    """Run all authorized phases; resume only untouched jobs or verified results."""
    _validate(config)
    project = (root or Path.cwd()).resolve()
    train = train_backend or _default_train
    validate = validation_backend or read_validation_score
    evaluate = evaluate_backend or _default_evaluate
    actual_config = copy.deepcopy(config)
    if smoke:
        actual_config["search"].update(trials=1, rung_epochs=[1], rung_keep=[1])
        actual_config["seeds"] = config["seeds"][:1]
    protocols = [protocol for protocol in actual_config["protocols"] if not protocol_filter or protocol["id"] in protocol_filter]
    if protocol_filter and set(protocol_filter) - {protocol["id"] for protocol in protocols}:
        raise ValueError("Unknown requested protocol filter")
    sources, source_contract = {}, {}
    for task in {protocol["task"] for protocol in protocols}:
        path = Path(actual_config["tasks"][task])
        path = path if path.is_absolute() else project / path
        print(f"READS: source {task} task config {path}", flush=True)
        sources[task] = yaml.safe_load(path.read_text(encoding="utf-8"))
        source_contract[str(path)] = _sha(path)
    source_files = sorted((project / "src" / "virtual_imu").rglob("*.py"))
    implementation_files = actual_config.get("implementation_files", [str(path) for path in source_files] or [str(Path(__file__).resolve())])
    for name in implementation_files:
        path = Path(name)
        path = path if path.is_absolute() else project / path
        source_contract[str(path)] = _sha(path)
    if config_path is not None:
        source_path = config_path if config_path.is_absolute() else project / config_path
        source_contract[str(source_path)] = _sha(source_path)
    cells = {}
    for protocol in protocols:
        for candidate in actual_config["candidates"]:
            identifier = f"{protocol['id']}/{candidate['id']}"
            if protocol["task"] in candidate["tasks"] and (not cell_filter or identifier in cell_filter):
                cells[identifier] = {"protocol": protocol, "candidate": candidate, "trials": deterministic_candidates(actual_config, candidate, protocol["task"])}
    if not cells or (cell_filter and set(cell_filter) - set(cells)):
        raise ValueError("No matching cells, or an unknown requested cell filter")
    contract = {"config": actual_config, "source_sha256": source_contract, "cells": list(cells), "smoke": smoke}
    if resume_root is not None:
        run_root = resume_root.resolve()
        state = json.loads((run_root / "pipeline.json").read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Resume contract differs in config, code, source YAML, cell selection or smoke scope")
        if any(job["status"] == "running" for job in state["jobs"].values()):
            raise RuntimeError("Interrupted partial optimization detected; automatic restart/resume is forbidden; preserve it and create an explicit replacement run")
        for cell_id, expected_cell in cells.items():
            saved = state["cells"][cell_id]
            if any(saved[key] != expected_cell[key] for key in ("protocol", "candidate", "trials")):
                raise ValueError("Frozen cell candidates differ from the deterministic search contract")
        for job in state["jobs"].values():
            if job["status"] != "complete":
                continue
            frozen = Path(job["config"])
            if _sha(frozen) != job["config_sha256"]:
                raise ValueError("Completed frozen config checksum changed")
            _verify_artifact_hashes(Path(job["run_dir"]), job["artifact_sha256"])
            evidence = validate(Path(job["run_dir"]), yaml.safe_load(frozen.read_text(encoding="utf-8")))
            if not math.isclose(float(evidence["score"]), job["score"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Completed validation score changed")
            if job.get("evaluation_status") == "complete":
                _verify_evaluation(job, run_root)
        for cell_id, cell in state["cells"].items():
            if "selected" in cell:
                _verify_selection(cell_id, state, run_root)
        print(f"READS: verified resume state {run_root}", flush=True)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        parent = Path(actual_config.get("runs_dir", "runs"))
        parent = parent if parent.is_absolute() else project / parent
        run_root = parent / f"reviewer_hpo_{'smoke_' if smoke else ''}{stamp}"
        run_root.mkdir(parents=True, exist_ok=False)
        variants = ["default", "tuned"] if actual_config.get("comparison", {}).get("default_refits", False) else ["tuned"]
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        state = {"schema_version": 1, "created_utc": _now(), "status": "planned", "smoke": smoke, "git_commit": commit, "dataset_manifest_sha256": actual_config.get("dataset_manifest_sha256"), "contract": contract, "source_config": str(config_path) if config_path else None, "final_seeds": actual_config["seeds"], "final_variants": variants, "cells": cells, "jobs": {}, "selection_uses_test": False, "promotion": "restart_from_scratch_no_optimizer_resume", "classical_search": actual_config.get("classical_search", {})}
        _freeze_yaml(run_root / "matrix_config.yaml", actual_config)
    print(f"WRITES: reviewer source-of-truth run {run_root} cells={len(cells)}", flush=True)
    search = actual_config["search"]

    def execute(cell_id: str, trial: dict, *, phase: str, seed: int, epochs: int, rung: int | None = None) -> dict | None:
        cell = state["cells"][cell_id]
        job_id = f"{cell_id.replace('/', '__')}__{phase}__{trial['id']}__r{rung if rung is not None else 'na'}__s{seed}"
        frozen = run_root / "configs" / f"{job_id}.yaml"
        actual = build_trial_config(actual_config, sources[cell["protocol"]["task"]], cell["protocol"], cell["candidate"], trial["parameters"], seed=seed, epochs=epochs, phase=phase, output_root=run_root / "experiments" / job_id)
        _freeze_yaml(frozen, actual)
        job = state["jobs"].setdefault(job_id, {"id": job_id, "cell": cell_id, "phase": phase, "trial_id": trial["id"], "rung": rung, "seed": seed, "epochs": epochs, "status": "planned", "config": str(frozen), "config_sha256": _sha(frozen)})
        if _sha(frozen) != job["config_sha256"]:
            raise ValueError("Frozen job config checksum changed")
        if job["status"] == "failed":
            if phase.startswith("final_") or not job.get("failure_recoverable", False):
                raise RuntimeError(f"Nonrecoverable/final training failed previously; explicit replacement required: {job_id}")
            return None
        if job["status"] == "complete":
            artifact = Path(job["run_dir"])
            _verify_artifact_hashes(artifact, job["artifact_sha256"])
            evidence = validate(artifact, actual)
            if not math.isclose(float(evidence["score"]), job["score"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Completed validation score changed")
            print(f"SKIPS: verified completed optimization {job_id}", flush=True)
            return job
        if plan_only:
            return job
        job.update(status="running", started_utc=_now())
        state["active_job"] = job_id
        _summary(run_root, state)
        failure_stage = "training"
        try:
            print(f"READS: frozen {phase} config {frozen} epochs={epochs} seed={seed}", flush=True)
            artifact = Path(train(cell["protocol"]["task"], actual, frozen)).resolve()
            failure_stage = "artifact_validation"
            evidence = validate(artifact, actual)
            score = float(evidence["score"])
            if not math.isfinite(score) or score < 0:
                raise ValueError("Trial backend returned a nonfinite or negative validation score")
            artifact_hashes = {
                name: _sha(artifact / name)
                for name in ("best.ckpt", "config.yaml", "metrics.csv", "provenance.json", "data_manifest.json", "split_manifest.json", "strict_manifest.json", "assignment_manifest.json")
                if (artifact / name).is_file()
            }
            job.update(status="complete", completed_utc=_now(), run_dir=str(artifact), score=score, validation=evidence, checkpoint_sha256=_sha(artifact / "best.ckpt"), run_config_sha256=_sha(artifact / "config.yaml"), artifact_sha256=artifact_hashes)
        except Exception as error:
            recoverable = failure_stage == "training" and _recoverable_training_error(error)
            job.update(status="failed", completed_utc=_now(), error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), failure_stage=failure_stage, failure_recoverable=recoverable)
            if recoverable:
                traceback.clear_frames(error.__traceback__)
                job["resource_cleanup"] = _release_resources()
            _summary(run_root, state)
            print(f"FAILED: {job_id} {type(error).__name__}: {error}", flush=True)
            if phase.startswith("final_") or not recoverable:
                raise
            return None
        finally:
            state.pop("active_job", None)
            _summary(run_root, state)
        return job

    # Freeze all initial trial configs even in a plan-only run, without data or CUDA.
    for cell_id, cell in state["cells"].items():
        for trial in cell["trials"]:
            if plan_only:
                execute(cell_id, trial, phase="search", seed=actual_config["seed"], epochs=search["rung_epochs"][0], rung=0)
    _summary(run_root, state)
    if plan_only:
        print("SKIPS: dataset reads, CUDA initialization, fitting, test evaluation (plan-only)", flush=True)
        return run_root
    try:
        if train_backend is None:
            inputs = {}
            expected_hash = actual_config.get("dataset_manifest_sha256")
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
                raise ValueError("Execution requires the exact dataset_manifest_sha256")
            for source in sources.values():
                path = Path(source["data"]["dataset_manifest"])
                path = path if path.is_absolute() else project / path
                print(f"READS: canonical dataset hash {path}", flush=True)
                if _sha(path) != expected_hash:
                    raise ValueError("Canonical dataset manifest hash differs from the frozen HPO contract")
                inputs[str(path)] = expected_hash
            split_parent = Path(actual_config.get("protocol_split_root", "runs/reviewer_protocol_20261024/splits"))
            split_parent = split_parent if split_parent.is_absolute() else project / split_parent
            for protocol in protocols:
                for name in ("split_manifest.json", "strict_manifest.json", "assignment_manifest.json"):
                    path = split_parent / protocol["id"] / name
                    print(f"READS: derived input metadata {path}", flush=True)
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("canonical_provenance", {}).get("dataset_manifest_sha256") != expected_hash:
                        raise ValueError("Derived protocol is not bound to the frozen canonical dataset")
                    inputs[str(path)] = _sha(path)
            if "input_artifact_sha256" in state and state["input_artifact_sha256"] != inputs:
                raise ValueError("Derived split or dataset metadata changed since the original launch")
            state["input_artifact_sha256"] = inputs
            _summary(run_root, state)
            import torch

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_num_threads(int(actual_config.get("cpu_threads", 4)))
            if str(actual_config.get("device", "cuda")).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was required; no silent CPU fallback")
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
            runtime = _runtime_snapshot(torch, actual_config)
            if "runtime" in state and state["runtime"] != runtime:
                raise RuntimeError("Effective training runtime changed since the original launch")
            state["runtime"] = runtime
            _summary(run_root, state)
        state["status"] = "searching"
        _summary(run_root, state)
        for cell_id, cell in state["cells"].items():
            if "selected" in cell:
                continue
            survivors = cell["trials"]
            last_results = []
            for rung, epochs in enumerate(search["rung_epochs"]):
                results = []
                for trial in survivors:
                    job = execute(cell_id, trial, phase="search", seed=actual_config["seed"], epochs=epochs, rung=rung)
                    if job is not None:
                        results.append((job["score"], trial["id"], trial, job))
                if not results:
                    raise RuntimeError(f"Every HPO candidate failed in {cell_id}, rung {rung}")
                results.sort(key=lambda value: (value[0], value[1]))
                last_results = results
                keep = search["rung_keep"][rung + 1] if rung + 1 < len(search["rung_keep"]) else search["rung_keep"][-1]
                survivors = [result[2] for result in results[:keep]]
                cell.setdefault("rungs", {})[str(rung)] = {"epochs": epochs, "successful": len(results), "promoted": [trial["id"] for trial in survivors]}
                _summary(run_root, state)
            confirmations = []
            last_by_id = {result[1]: result for result in last_results}
            for trial in survivors:
                job = execute(cell_id, trial, phase="confirmation", seed=actual_config["confirmation_seed"], epochs=search["rung_epochs"][-1])
                if job is not None:
                    original = last_by_id[trial["id"]][3]
                    confirmations.append(((original["score"] + job["score"]) / 2, trial["id"], trial, [original["id"], job["id"]]))
            if not confirmations:
                raise RuntimeError(f"Every confirmation failed in {cell_id}")
            confirmations.sort(key=lambda value: (value[0], value[1]))
            score, _, trial, jobs = confirmations[0]
            cell["selected"] = {"trial": trial, "mean_two_seed_validation_score": score, "selection_jobs": jobs, "selected_utc": _now()}
            selection_path = run_root / "selections" / f"{cell_id.replace('/', '__')}.json"
            _write_json(selection_path, cell["selected"])
            cell["selection_sha256"] = _sha(selection_path)
            _verify_selection(cell_id, state, run_root)
            _summary(run_root, state)
        state.update(status="search_complete", search_completed_utc=_now())
        _summary(run_root, state)
        print("WRITES: all validation-only selections frozen; final refits/evaluation may begin", flush=True)
        state["status"] = "finalizing"
        for cell_id, cell in state["cells"].items():
            final_specs = [(variant, seed) for variant in state["final_variants"] for seed in actual_config["seeds"]]
            for variant, seed in final_specs:
                trial = cell["trials"][0] if variant == "default" else cell["selected"]["trial"]
                job = execute(cell_id, trial, phase=f"final_{variant}", seed=seed, epochs=search["rung_epochs"][-1])
                if job is None:
                    raise RuntimeError("Final optimization did not return a verified result")
                evaluation_path = run_root / "evaluations" / f"{job['id']}.json"
                if job.get("evaluation_status") == "complete":
                    _verify_evaluation(job, run_root)
                    continue
                if job.get("evaluation_status") == "failed":
                    raise RuntimeError("Final evaluation failed previously; inspect failure before an explicit replacement")
                job["evaluation_status"] = "running"
                _summary(run_root, state)
                try:
                    actual = yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8"))
                    report = evaluate(cell["protocol"]["task"], Path(job["run_dir"]), actual)
                    _write_json(evaluation_path, {"job_id": job["id"], "variant": variant, "model_family": cell["candidate"]["id"], "run_dir": job["run_dir"], "checkpoint_sha256": job["checkpoint_sha256"], "selection_jobs": [] if variant == "default" else cell["selected"]["selection_jobs"], "metrics": report})
                    inventory = _evaluation_inventory(Path(job["run_dir"]), evaluation_path)
                    job.update(evaluation_status="complete", evaluation_sha256=_sha(evaluation_path), evaluation_artifact_sha256=inventory)
                except Exception as error:
                    job.update(evaluation_status="failed", evaluation_error=str(error), evaluation_traceback=traceback.format_exc())
                    raise
                finally:
                    _summary(run_root, state)
        _summary(run_root, state)
        if state["counts"]["final_evaluated"] != state["counts"]["expected_final_runs"]:
            raise RuntimeError("Final evaluation coverage is incomplete")
        state.update(status="complete", completed_utc=_now())
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), failed_utc=_now())
        raise
    finally:
        _summary(run_root, state)
    print(f"COMPLETED: reviewer HPO {run_root} status={state['status']}", flush=True)
    return run_root
