"""Retrospective architecture comparisons on the immutable formal IMU splits.

This runner deliberately leaves ``formal_pipeline`` unchanged.  Published-task
adaptations retain their declared fidelity; their results are not represented as
faithful reproductions of an original paper's task, losses or benchmark.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.eval.formal_evaluation import json_safe, verify_run_provenance
from virtual_imu.io.training_bundle import sha256_file
from virtual_imu.training.formal_pipeline import _evaluate, _train
from virtual_imu.utils.paths import git_commit, project_root_from_file

TASKS = {"paired", "navigation"}
IDENTITY = ["dataset_id", "low_sensor", "high_sensor"]
PRIMARY_METRICS = {
    "paired": ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"],
    "navigation": ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"],
}
BASELINES = {
    "paired": {
        "identity",
        "bias_mean",
        "affine",
        "ridge_affine",
        "learned_fir_causal",
        "arx_linear",
    },
    "navigation": {"zero", "train_mean", "ridge"},
}
SCOPE = (
    "Retrospective comparison after the original test results were inspected. "
    "All declared candidates are reported; checkpoints use validation only. "
    "Mean and sample SD describe optimization seeds, not independent recordings. "
    "Paper adaptations do not establish published SOTA reproduction."
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _yaml_text(value: dict) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, value: Any) -> None:
    """Replace state atomically so an interrupted write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def _finite_numbers(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite numeric value: {where}")


def _candidate_metadata(candidate: dict) -> None:
    if not isinstance(candidate, dict) or not re.fullmatch(
        r"[a-z][a-z0-9_-]*", str(candidate.get("id", ""))
    ):
        raise ValueError("Every candidate id must be a unique lowercase slug")
    if candidate["id"] == "classical":
        raise ValueError("candidate id 'classical' is reserved for baseline rows")
    for key in ("label", "fidelity"):
        if not isinstance(candidate.get(key), str) or not candidate[key].strip():
            raise ValueError(f"Candidate {candidate['id']} requires a nonempty {key}")


def validate_matrix(config: dict) -> None:
    """Validate the declared job matrix without opening any data or run files."""
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a mapping")
    _finite_numbers(config, "matrix")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be a nonempty list of distinct unsigned 32-bit integers")
    tasks = config.get("tasks")
    if (
        not isinstance(tasks, dict)
        or not tasks
        or set(tasks) - TASKS
        or any(not isinstance(path, str) or not path for path in tasks.values())
    ):
        raise ValueError("tasks must map paired and/or navigation to source YAML paths")
    if config.get("deterministic", True) is not True:
        raise ValueError("This fixed benchmark requires deterministic=true")
    if not re.fullmatch(r"cuda(?::\d+)?|cpu", str(config.get("device", "cuda"))):
        raise ValueError("device must be cuda, cuda:<index>, or an explicit cpu test environment")
    if type(config.get("cpu_threads", 4)) is not int or config.get("cpu_threads", 4) <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    if not re.fullmatch(r"[a-f0-9]{64}", str(config.get("data_manifest_sha256", ""))):
        raise ValueError("data_manifest_sha256 must bind the frozen dataset manifest")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    identifiers = []
    for candidate in candidates:
        _candidate_metadata(candidate)
        identifiers.append(candidate["id"])
        if not isinstance(candidate.get("model"), dict) or not candidate["model"].get("type"):
            raise ValueError(f"Candidate {candidate['id']} requires a model type")
        if (
            not isinstance(candidate.get("temporal_context"), str)
            or not candidate["temporal_context"]
        ):
            raise ValueError(f"Candidate {candidate['id']} requires temporal_context")
        references = candidate.get("references")
        if not isinstance(references, list) or any(
            not isinstance(reference, (str, dict)) or not reference for reference in references
        ):
            raise ValueError(f"Candidate {candidate['id']} references must be a list")
        selected = candidate.get("tasks")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(task, str) for task in selected)
            or len(selected) != len(set(selected))
            or set(selected) - set(tasks)
        ):
            raise ValueError(f"Candidate {candidate['id']} tasks must be unique declared tasks")
    previous = config.get("previous_candidate")
    if bool(config.get("previous_pipeline_root")) != bool(previous):
        raise ValueError("previous_pipeline_root and previous_candidate must be provided together")
    if previous:
        _candidate_metadata(previous)
        identifiers.append(previous["id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate ids, including the reused candidate, must be unique")


def _validate_source(task: str, source: dict) -> None:
    data, model, training = (
        source.get("data", {}),
        source.get("model", {}),
        source.get("training", {}),
    )
    if not data.get("dataset_manifest") or not data.get("split_manifest"):
        raise ValueError(f"{task} requires frozen dataset and split manifests")
    if (data.get("window_size"), data.get("stride"), float(data.get("rate_hz", 0))) != (
        80,
        40,
        40.0,
    ):
        raise ValueError(f"{task} must preserve the formal 80-point, stride-40, 40 Hz protocol")
    if type(training.get("epochs")) is not int or not 1 <= training["epochs"] <= 60:
        raise ValueError(f"{task} must preserve an explicit 1..60 epoch budget")
    if task == "paired":
        if (model.get("target_mode"), model.get("in_channels"), model.get("out_channels")) != (
            "residual",
            9,
            6,
        ):
            raise ValueError(
                "Paired base config must declare residual prediction, 9 inputs and 6 outputs"
            )
        if not data.get("strict_manifest") or len(data.get("feature_columns", [])) != 9:
            raise ValueError(
                "Paired base config requires its strict manifest and nine ordered features"
            )
        if training.get("checkpoint_metric") != "val_grouped_score":
            raise ValueError("Paired checkpoint selection must remain val_grouped_score")
    else:
        if (
            model.get("prediction_head"),
            model.get("target_layout"),
            model.get("in_channels"),
            model.get("out_channels"),
        ) != ("last", "window", 6, 3):
            raise ValueError(
                "Navigation base config must declare last/window prediction, 6 inputs and 3 outputs"
            )
        if (
            data.get("label_mode") != "position_delta"
            or training.get("checkpoint_metric") != "val_rmse"
        ):
            raise ValueError(
                "Navigation must preserve position_delta labels and val_rmse selection"
            )
    _finite_numbers(source, f"{task} source")


def build_experiment_config(
    source: dict, candidate: dict, task: str, seed: int, root: Path, *, device: str, smoke: bool
) -> dict:
    """Replace architecture settings while retaining every task/data/loss setting."""
    _validate_source(task, source)
    actual = copy.deepcopy(source)
    actual["seed"] = seed
    actual["experiment_name"] = (
        f"benchmark_{'smoke_' if smoke else ''}{candidate['id']}_{task}_seed{seed}"
    )
    actual.setdefault("output", {})["runs_dir"] = str(root / "experiments")
    actual["training"]["device"] = device
    if smoke:
        actual["training"]["epochs"] = 1
    protected = (
        {"target_mode": "residual", "in_channels": 9, "out_channels": 6}
        if task == "paired"
        else {
            "prediction_head": "last",
            "target_layout": "window",
            "in_channels": 6,
            "out_channels": 3,
        }
    )
    model = copy.deepcopy(candidate["model"])
    for key, expected in protected.items():
        if key in model and model[key] != expected:
            raise ValueError(f"Candidate {candidate['id']} cannot change {task} {key}={expected}")
    if task == "paired" and model.get("static_context"):
        raise ValueError("The fixed paired benchmark does not permit a static-context branch")
    actual["model"] = {**model, **protected}
    return actual


def _source_contract() -> dict:
    project = project_root_from_file(__file__)
    files = sorted((project / "src" / "virtual_imu").rglob("*.py"))
    cli = project / "scripts" / "run_benchmark_training.py"
    if cli.is_file():
        files.append(cli)
    fingerprints = {path.relative_to(project).as_posix(): sha256_file(path) for path in files}
    digest = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": fingerprints}


def _runtime(config: dict) -> dict:
    device = str(config.get("device", "cuda"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Benchmark requires requested CUDA; no silent CPU fallback")
        torch.cuda.get_device_properties(torch.device(device))
    torch.set_num_threads(config.get("cpu_threads", 4))
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "deterministic": True,
        "attention_backend": "math SDPA; flash/memory-efficient/cuDNN SDPA disabled",
        "cpu_threads": torch.get_num_threads(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _check_data_sources(sources: dict[str, dict], expected_sha256: str) -> None:
    for task, source in sources.items():
        for key in ("dataset_manifest", "split_manifest", "strict_manifest"):
            value = source["data"].get(key)
            if value and not Path(value).is_file():
                raise FileNotFoundError(f"{task} {key} is missing: {value}")
        path = Path(source["data"]["dataset_manifest"])
        if sha256_file(path) != expected_sha256:
            raise ValueError(
                f"{task} dataset manifest SHA256 differs from the fixed matrix: {path}"
            )


def _check_checkpoint(run_dir: Path, frozen: Path, expected_sha256: str) -> tuple[dict, dict]:
    if not frozen.is_file() or sha256_file(frozen) != expected_sha256:
        raise ValueError(f"Frozen experiment config SHA256 changed: {frozen}")
    expected = _load_yaml(frozen)
    if _load_yaml(run_dir / "config.yaml") != expected:
        raise ValueError(f"Run config differs from frozen experiment config: {run_dir}")
    checkpoint_path = run_dir / "best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("config") != expected or checkpoint.get("model_config") != expected["model"]:
        raise ValueError(f"Checkpoint config/model identity differs from frozen config: {run_dir}")
    weights = checkpoint.get("model_state_dict")
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
            for value in weights.values()
        )
    ):
        raise ValueError(f"Checkpoint weights are missing or nonfinite: {run_dir}")
    epoch, best = checkpoint.get("best_epoch"), checkpoint.get("best_metric")
    if type(epoch) is not int or not 1 <= epoch <= expected["training"]["epochs"]:
        raise ValueError(f"Invalid best checkpoint epoch: {run_dir}")
    if not isinstance(best, (int, float)) or not math.isfinite(best):
        raise ValueError(f"Nonfinite best checkpoint selection metric: {run_dir}")
    selected = expected["training"]["checkpoint_metric"]
    history = pd.read_csv(run_dir / "metrics.csv")
    needed = ["epoch", "train_loss", "val_loss", selected]
    if (
        history.empty
        or set(needed) - set(history)
        or not np.isfinite(history[needed].to_numpy(float)).all()
    ):
        raise ValueError(
            f"Training history lacks finite epochs/losses/selection metrics: {run_dir}"
        )
    if history.epoch.duplicated().any() or not np.all(np.diff(history.epoch.to_numpy()) > 0):
        raise ValueError(f"Training history has repeated or non-monotonic epochs: {run_dir}")
    best_rows = history[history.epoch == epoch]
    if (
        len(best_rows) != 1
        or not np.isclose(best_rows[selected].iloc[0], best, rtol=1e-6, atol=1e-9)
        or not np.isclose(history[selected].min(), best, rtol=1e-6, atol=1e-9)
    ):
        raise ValueError(
            f"Checkpoint best epoch/metric does not match validation history: {run_dir}"
        )
    verify_run_provenance(run_dir, expected, checkpoint["config"])
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    bundle = provenance.get("server_source_bundle")
    if bundle and sha256_file(run_dir / "source_bundle.json") != bundle.get("sha256"):
        raise ValueError(f"Training source bundle snapshot changed: {run_dir}")
    return expected, checkpoint


def _read_metric_frame(task: str, run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "evaluation" / f"{task}_metrics.csv")
    keys = ["split", "method", *(IDENTITY if task == "paired" else [])]
    metrics = PRIMARY_METRICS[task]
    if frame.empty or set([*keys, *metrics]) - set(frame):
        raise ValueError(f"Incomplete {task} evaluation metric columns: {run_dir}")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or duplicated evaluation identity: {run_dir}")
    if set(frame.split) != {"validation", "test"} or set(frame.method) != {
        "model",
        *BASELINES[task],
    }:
        raise ValueError(f"Evaluation must include every fixed split and baseline: {run_dir}")
    expected_groups = {
        (split, method)
        for split in ("validation", "test")
        for method in ("model", *BASELINES[task])
    }
    if set(frame.groupby(["split", "method"]).groups) != expected_groups:
        raise ValueError(f"A held-out split/method evaluation group is missing: {run_dir}")
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(float)).all() or (frame[metrics] < 0).any().any():
        raise ValueError(
            f"Evaluation metrics must be finite with nonnegative primary RMSE: {run_dir}"
        )
    expected_keys = None
    for _, group in frame.groupby(["split", "method"], sort=False):
        identities = set(map(tuple, group[IDENTITY].to_numpy())) if task == "paired" else {()}
        if expected_keys is None:
            expected_keys = identities
        if identities != expected_keys:
            raise ValueError(f"Evaluation methods/splits disagree on device mappings: {run_dir}")
    return frame


def verify_entry_artifacts(
    entry: dict, expected_data_sha256: str, *, evaluated: bool = True
) -> dict:
    """Validate identities and artifacts before reusing any completed run."""
    if (
        entry.get("reused")
        and sha256_file(Path(entry["source_pipeline"])) != entry["source_pipeline_sha256"]
    ):
        raise ValueError("Reused source pipeline SHA256 changed")
    run_dir, frozen = Path(entry["run_dir"]), Path(entry["config"])
    config, checkpoint = _check_checkpoint(run_dir, frozen, entry["config_sha256"])
    if config.get("seed") != entry["seed"]:
        raise ValueError(f"Run seed differs from benchmark job: {run_dir}")
    manifest = run_dir / "data_manifest.json"
    if sha256_file(manifest) != expected_data_sha256:
        raise ValueError(f"Run dataset manifest differs from the benchmark: {run_dir}")
    paths = [
        frozen,
        run_dir / "config.yaml",
        run_dir / "best.ckpt",
        run_dir / "metrics.csv",
        run_dir / "provenance.json",
        manifest,
    ]
    for name in ("source_bundle.json", "split_manifest.json", "strict_manifest.json"):
        if (run_dir / name).is_file():
            paths.append(run_dir / name)
    if evaluated:
        task = entry["task"]
        metric_frame = _read_metric_frame(task, run_dir)
        report_path = run_dir / "evaluation" / "metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _finite_numbers(report, "evaluation metrics")
        if set(report.get("splits", {})) != {"validation", "test"}:
            raise ValueError(f"Evaluation JSON does not bind both held-out splits: {run_dir}")
        if report.get("selection") != config["training"]["checkpoint_metric"]:
            raise ValueError(f"Evaluation checkpoint-selection identity differs: {run_dir}")
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        evaluation_provenance = json.loads(
            (run_dir / "evaluation" / "provenance.json").read_text(encoding="utf-8")
        )
        if evaluation_provenance.get("unified_dataset") != provenance["unified_dataset"]:
            raise ValueError(f"Evaluation input provenance differs from training: {run_dir}")
        baseline_state = json.loads(
            (run_dir / "evaluation" / "baseline_train_state.json").read_text(encoding="utf-8")
        )
        _finite_numbers(baseline_state, "train-fitted baseline state")
        baseline_states = baseline_state if isinstance(baseline_state, list) else [baseline_state]
        if not baseline_states or any(
            not isinstance(state, dict) or state.get("fit_split") != "train"
            for state in baseline_states
        ):
            raise ValueError(
                f"Classical baseline state does not declare fit_split=train: {run_dir}"
            )
        for key, value in provenance["unified_dataset"]["inputs"].items():
            name = "data_manifest.json" if key == "dataset_manifest" else f"{key}.json"
            snapshot = run_dir / "evaluation" / name
            if sha256_file(snapshot) != value["sha256"]:
                raise ValueError(f"Evaluation manifest snapshot differs: {snapshot}")
            paths.append(snapshot)
        paths.extend(
            [
                report_path,
                run_dir / "evaluation" / f"{task}_metrics.csv",
                run_dir / "evaluation" / "baseline_train_state.json",
                run_dir / "evaluation" / "provenance.json",
            ]
        )
        for split in ("validation", "test"):
            prediction_path = run_dir / "evaluation" / f"{split}_predictions.parquet"
            predictions = pd.read_parquet(prediction_path)
            columns = (
                [
                    f"{axis}_{kind}"
                    for kind in ("high", "virtual")
                    for axis in ("gx", "gy", "gz", "ax", "ay", "az")
                ]
                if task == "paired"
                else [
                    f"{method}_delta_{axis}_local_m"
                    for method in ("true", "pred", "zero", "train_mean", "ridge")
                    for axis in ("x", "y", "z")
                ]
            )
            if (
                predictions.empty
                or set(columns) - set(predictions)
                or not np.isfinite(predictions[columns].to_numpy(float)).all()
            ):
                raise ValueError(
                    f"Missing/nonfinite {split} prediction artifacts: {prediction_path}"
                )
            split_report = report["splits"][split]
            if split_report.get("samples" if task == "paired" else "windows") != len(predictions):
                raise ValueError(
                    f"Prediction count differs from evaluation JSON: {prediction_path}"
                )
            if task == "paired":
                reports = {
                    tuple(pair[key] for key in IDENTITY): pair["metrics"]
                    for pair in split_report["pairs"]
                }
            for row in metric_frame[metric_frame.split == split].to_dict("records"):
                reported_metrics = (
                    reports[tuple(row[key] for key in IDENTITY)][row["method"]]
                    if task == "paired"
                    else split_report["metrics"][row["method"]]
                )
                if any(
                    not np.isclose(row[metric], reported_metrics[metric], rtol=1e-10, atol=1e-12)
                    for metric in PRIMARY_METRICS[task]
                ):
                    raise ValueError(f"Evaluation CSV/JSON physical metrics disagree: {run_dir}")
            paths.append(prediction_path)
    previous = entry.get("artifacts", {}).get("sha256", {})
    fingerprints = {str(path): sha256_file(path) for path in paths}
    if previous and fingerprints != previous:
        raise ValueError(f"Completed run artifact SHA256 changed: {run_dir}")
    return {
        "verified_at": _now(),
        "best_epoch": checkpoint["best_epoch"],
        "best_metric": checkpoint["best_metric"],
        "sha256": fingerprints,
    }


def _previous_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    elif root.name in path.parts:
        index = path.parts.index(root.name)
        resolved = root.joinpath(*path.parts[index + 1 :]).resolve()
    else:
        resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Previous artifact path escapes its source-of-truth run: {value}")
    return resolved


def _import_previous(config: dict, sources: dict[str, dict]) -> list[dict]:
    root = Path(config["previous_pipeline_root"]).resolve()
    pipeline_path = root / "pipeline.json"
    previous = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if previous.get("status") != "complete" or previous.get("smoke") is not False:
        raise ValueError("Previous pipeline must be complete and must not be a smoke run")
    required = {(task, seed) for task in sources for seed in config["seeds"]}
    runs = previous.get("runs", [])
    if (
        len(runs) != len(required)
        or {(entry.get("task"), entry.get("seed")) for entry in runs} != required
    ):
        raise ValueError("Previous pipeline does not contain exactly the expected task/seed runs")
    candidate = config["previous_candidate"]
    imported = []
    for entry in runs:
        if entry.get("status") != "evaluated":
            raise ValueError("Every reused task/seed must already be evaluated")
        frozen = _previous_path(root, entry["config"])
        actual = _load_yaml(frozen)
        task = entry["task"]
        for section in ("data", "model", "training"):
            if actual.get(section) != sources[task].get(section):
                raise ValueError(f"Previous {task} {section} differs from the fixed base config")
        item = {
            "candidate_id": candidate["id"],
            "candidate_label": candidate["label"],
            "fidelity": candidate["fidelity"],
            "temporal_context": candidate.get("temporal_context", "causal_per_window"),
            "references": candidate.get("references", []),
            "task": task,
            "seed": entry["seed"],
            "run_dir": str(_previous_path(root, entry["run_dir"])),
            "config": str(frozen),
            "config_sha256": sha256_file(frozen),
            "status": "evaluated",
            "reused": True,
            "source_pipeline": str(pipeline_path),
            "source_pipeline_sha256": sha256_file(pipeline_path),
        }
        item["artifacts"] = verify_entry_artifacts(item, config["data_manifest_sha256"])
        imported.append(item)
    return imported


def aggregate_benchmark_results(root: Path, entries: list[dict]) -> dict:
    """Keep neural identities separate and deduplicate unchanged classical fits."""
    identities = [(entry["candidate_id"], entry["task"], entry["seed"]) for entry in entries]
    if (
        not entries
        or len(identities) != len(set(identities))
        or any(entry.get("status") != "evaluated" for entry in entries)
    ):
        raise ValueError("Aggregation requires unique evaluated candidate/task/seed entries")
    tables, summaries, baseline_sources = {}, {}, []
    for task in sorted({entry["task"] for entry in entries}):
        selected = [entry for entry in entries if entry["task"] == task]
        all_seeds = {entry["seed"] for entry in selected}
        for candidate in {entry["candidate_id"] for entry in selected}:
            if {
                entry["seed"] for entry in selected if entry["candidate_id"] == candidate
            } != all_seeds:
                raise ValueError(f"Unequal seed coverage for {task} candidate {candidate}")
        model_frames, references = [], {}
        groups = ["split", "method", *(IDENTITY if task == "paired" else [])]
        for entry in selected:
            frame = _read_metric_frame(task, Path(entry["run_dir"]))
            baseline = frame[frame.method != "model"].sort_values(groups).reset_index(drop=True)
            seed = entry["seed"]
            if seed in references:
                previous = references[seed][0]
                if set(baseline.columns) != set(previous.columns) or not baseline[groups].equals(
                    previous[groups]
                ):
                    raise ValueError(
                        f"Classical baseline identity mismatch across {task} candidates, seed={seed}"
                    )
                numeric = [column for column in baseline if column not in groups]
                if not np.allclose(
                    baseline[numeric].to_numpy(float),
                    previous[numeric].to_numpy(float),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Classical baseline metrics differ across {task} candidates, seed={seed}"
                    )
            else:
                references[seed] = (baseline, entry)
            model = frame[frame.method == "model"].copy()
            for key in (
                "candidate_id",
                "candidate_label",
                "fidelity",
                "temporal_context",
                "seed",
                "run_dir",
            ):
                model[key] = entry[key]
            model["method_category"] = "learned"
            model_frames.append(model)
        for seed, (baseline, entry) in references.items():
            baseline = baseline.copy()
            baseline["candidate_id"] = "classical"
            baseline["candidate_label"] = baseline.method
            baseline["fidelity"] = "unchanged_train_fitted_classical"
            baseline["temporal_context"] = "fixed_classical_protocol"
            baseline["method_category"] = "classical"
            baseline["seed"], baseline["run_dir"] = seed, entry["run_dir"]
            model_frames.append(baseline)
            baseline_sources.append(
                {
                    "task": task,
                    "seed": seed,
                    "reference_candidate": entry["candidate_id"],
                    "run_dir": entry["run_dir"],
                }
            )
        frame = pd.concat(model_frames, ignore_index=True)
        frame["task"] = task
        numeric = [
            column
            for column in frame.select_dtypes(include="number")
            if column not in {"seed", "windows"}
        ]
        summary_keys = [
            "task",
            "candidate_id",
            "candidate_label",
            "method_category",
            "fidelity",
            "temporal_context",
            *groups,
        ]
        summary = frame.groupby(summary_keys, dropna=False, sort=True)[numeric].agg(
            ["mean", "std", "count"]
        )
        summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
        tables[task], summaries[task] = frame, summary.reset_index()
    # Validate all tasks before writing a partial or misleading comparison table.
    root.mkdir(parents=True, exist_ok=True)
    for task, frame in tables.items():
        frame.to_csv(root / f"{task}_all_seeds.csv", index=False)
        summaries[task].to_csv(root / f"{task}_summary.csv", index=False)
    result = {
        "scope": SCOPE,
        "baseline_deduplication": "One unchanged reference per task/seed; every candidate's baseline metrics checked with rtol=1e-10, atol=1e-12. Counts represent seed slots, not independent classical fits.",
        "baseline_sources": baseline_sources,
        "tasks": {task: frame.to_dict("records") for task, frame in summaries.items()},
    }
    _write_json(root / "aggregate_metrics.json", result)
    lines = [
        "# Retrospective fixed-protocol comparison",
        "",
        SCOPE,
        "",
        "Classical baselines are included once per task/seed, after equality checks across candidates. Their zero SD is not evidence of independent replication.",
        "",
    ]
    for task, summary in summaries.items():
        for metric in PRIMARY_METRICS[task]:
            lines.extend(
                [
                    f"## {task}: {metric}",
                    "",
                    "| candidate / baseline | mapping | mean | sample SD | seed count | fidelity |",
                    "|---|---|---:|---:|---:|---|",
                ]
            )
            for row in summary[summary.split == "test"].to_dict("records"):
                mapping = (
                    f"{row['dataset_id']}: {row['low_sensor']} -> {row['high_sensor']}"
                    if task == "paired"
                    else "SmartPNT local ENU, 1.975 s"
                )
                sd = row[f"{metric}_std"]
                sd_text = f"{sd:.8g}" if pd.notna(sd) else "NA (one seed)"
                lines.append(
                    f"| {row['candidate_label']} | {mapping} | {row[f'{metric}_mean']:.8g} | {sd_text} | {row[f'{metric}_count']} | {row['fidelity']} |"
                )
            lines.append("")
    (root / "comparison_table.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def _create_jobs(config: dict, sources: dict[str, dict], root: Path, *, smoke: bool) -> list[dict]:
    jobs = []
    for candidate in config["candidates"]:
        for task in candidate["tasks"]:
            for seed in config["seeds"][:1] if smoke else config["seeds"]:
                actual = build_experiment_config(
                    sources[task],
                    candidate,
                    task,
                    seed,
                    root,
                    device=config.get("device", "cuda"),
                    smoke=smoke,
                )
                frozen = root / "configs" / f"{candidate['id']}_{task}_seed{seed}.yaml"
                frozen.write_text(_yaml_text(actual), encoding="utf-8")
                jobs.append(
                    {
                        "candidate_id": candidate["id"],
                        "candidate_label": candidate["label"],
                        "fidelity": candidate["fidelity"],
                        "temporal_context": candidate["temporal_context"],
                        "references": candidate["references"],
                        "task": task,
                        "seed": seed,
                        "run_dir": None,
                        "config": str(frozen),
                        "config_sha256": sha256_file(frozen),
                        "status": "planned",
                        "reused": False,
                    }
                )
    return jobs


def _resume_state(config: dict, sources: dict[str, dict], root: Path, *, smoke: bool) -> dict:
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    frozen_matrix = root / "matrix_config.yaml"
    if state.get("plan_only") or state.get("smoke") != smoke:
        raise ValueError("Cannot resume a plan-only run or mix smoke and full benchmark identities")
    if _load_yaml(frozen_matrix) != config or sha256_file(frozen_matrix) != state.get(
        "matrix_config_sha256"
    ):
        raise ValueError("Frozen matrix config changed; refusing to resume")
    if _source_contract() != json.loads(
        (root / "source_contract.json").read_text(encoding="utf-8")
    ):
        raise ValueError(
            "Training source code changed; refusing to resume under the old run identity"
        )
    for task, source in sources.items():
        path = root / "configs" / f"{task}_source.yaml"
        if (
            _load_yaml(path) != source
            or sha256_file(path) != state["source_configs"][task]["frozen_sha256"]
        ):
            raise ValueError(f"Frozen {task} base config changed; refusing to resume")
    expected = {
        (candidate["id"], task, seed)
        for candidate in config["candidates"]
        for task in candidate["tasks"]
        for seed in (config["seeds"][:1] if smoke else config["seeds"])
    }
    if config.get("previous_candidate") and not smoke:
        expected |= {
            (config["previous_candidate"]["id"], task, seed)
            for task in sources
            for seed in config["seeds"]
        }
    identities = [(entry["candidate_id"], entry["task"], entry["seed"]) for entry in state["runs"]]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Persisted candidate/task/seed matrix differs from the frozen contract")
    for entry in state["runs"]:
        if (
            not Path(entry["config"]).is_file()
            or sha256_file(Path(entry["config"])) != entry["config_sha256"]
        ):
            raise ValueError("Frozen experiment config SHA256 changed; refusing to resume")
        if not entry.get("reused"):
            candidate = next(
                candidate
                for candidate in config["candidates"]
                if candidate["id"] == entry["candidate_id"]
            )
            expected_config = build_experiment_config(
                sources[entry["task"]],
                candidate,
                entry["task"],
                entry["seed"],
                root,
                device=config.get("device", "cuda"),
                smoke=smoke,
            )
            if _load_yaml(Path(entry["config"])) != expected_config:
                raise ValueError(
                    "Frozen experiment config no longer matches its candidate/task/seed contract"
                )
        if entry["status"] == "evaluated":
            if smoke:
                raise ValueError("Smoke cannot reuse a formally evaluated entry")
            entry["artifacts"] = verify_entry_artifacts(entry, config["data_manifest_sha256"])
            print(
                f"SKIPS: verified evaluated candidate={entry['candidate_id']} task={entry['task']} seed={entry['seed']}",
                flush=True,
            )
        elif entry["status"] == "smoke_trained" and smoke:
            entry["artifacts"] = verify_entry_artifacts(
                entry, config["data_manifest_sha256"], evaluated=False
            )
        elif entry["status"] != "planned" or entry["run_dir"] is not None:
            raise RuntimeError(
                f"Partial run cannot be optimizer-resumed or silently retrained: candidate={entry['candidate_id']} task={entry['task']} seed={entry['seed']} status={entry['status']}. Preserve this run and request a separate replacement experiment explicitly."
            )
        elif list(
            (root / "experiments").glob(
                f"benchmark_*{entry['candidate_id']}_{entry['task']}_seed{entry['seed']}_*"
            )
        ):
            raise RuntimeError(
                "Unregistered partial artifacts exist for a planned job; refusing to retrain"
            )
    return state


def run_benchmark_pipeline(
    config: dict,
    *,
    config_path: Path | None = None,
    smoke: bool = False,
    resume_root: Path | None = None,
    plan_only: bool = False,
) -> Path:
    """Run or safely continue the predeclared retrospective benchmark matrix."""
    validate_matrix(config)
    if plan_only and resume_root is not None:
        raise ValueError("--plan-only and --resume-root cannot be combined")
    sources = {task: _load_yaml(Path(path)) for task, path in config["tasks"].items()}
    for task, source in sources.items():
        _validate_source(task, source)
    # Validate every architecture's protected task fields before creating any
    # run directory or reading a dataset, including in --plan-only mode.
    for candidate in config["candidates"]:
        for task in candidate["tasks"]:
            build_experiment_config(
                sources[task],
                candidate,
                task,
                config["seeds"][0],
                Path("."),
                device=config.get("device", "cuda"),
                smoke=smoke,
            )
    if not plan_only:
        _check_data_sources(sources, config["data_manifest_sha256"])
        runtime = _runtime(config)
    else:
        runtime = {
            "device": config.get("device", "cuda"),
            "not_checked": "plan-only does not initialize CUDA or read dataset/run artifacts",
        }
    if resume_root is not None:
        root = Path(resume_root).resolve()
        state = _resume_state(config, sources, root, smoke=smoke)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        kind = "plan" if plan_only else ("smoke" if smoke else "training")
        root = (Path(config.get("runs_dir", "runs")) / f"benchmark_{kind}_{stamp}").resolve()
        (root / "configs").mkdir(parents=True, exist_ok=False)
        (root / "experiments").mkdir()
        matrix_path = root / "matrix_config.yaml"
        matrix_path.write_text(_yaml_text(config), encoding="utf-8")
        source_configs = {}
        for task, source in sources.items():
            path = root / "configs" / f"{task}_source.yaml"
            path.write_text(_yaml_text(source), encoding="utf-8")
            source_configs[task] = {
                "path": config["tasks"][task],
                "source_sha256": sha256_file(Path(config["tasks"][task])),
                "frozen_sha256": sha256_file(path),
            }
        _write_json(root / "source_contract.json", _source_contract())
        state = {
            "status": "planned" if plan_only else "running",
            "created_at": _now(),
            "smoke": smoke,
            "plan_only": plan_only,
            "scope": SCOPE,
            "source_config": str(config_path) if config_path else None,
            "matrix_config_sha256": sha256_file(matrix_path),
            "source_configs": source_configs,
            "data_manifest_sha256": config["data_manifest_sha256"],
            "server_launch_run": os.environ.get("VIRTUAL_IMU_SERVER_LAUNCH_RUN"),
            "git_commit": git_commit(),
            "runtime": runtime,
            "runs": _create_jobs(config, sources, root, smoke=smoke),
        }
    _write_json(root / "pipeline.json", state)
    print(f"WRITES: benchmark source-of-truth {root}", flush=True)
    if plan_only:
        _write_json(
            root / "job_contract.json",
            {
                "scope": SCOPE,
                "expected_new_runs": len(state["runs"]),
                "previous_candidate_not_inspected": config.get("previous_candidate"),
                "runs": state["runs"],
            },
        )
        print(
            "SKIPS: dataset, previous results, CUDA initialization, training and evaluation (--plan-only)",
            flush=True,
        )
        return root
    active = None
    try:
        if (
            config.get("previous_candidate")
            and not smoke
            and not any(entry.get("reused") for entry in state["runs"])
        ):
            state["runs"] = [*_import_previous(config, sources), *state["runs"]]
            _write_json(root / "pipeline.json", state)
        for entry in state["runs"]:
            if entry["status"] in {"evaluated", "smoke_trained"}:
                continue
            active = entry
            state.update(
                status="running",
                active_candidate=entry["candidate_id"],
                active_task=entry["task"],
                active_seed=entry["seed"],
                updated_at=_now(),
            )
            entry.update(status="training", started_at=_now())
            _write_json(root / "pipeline.json", state)
            frozen = Path(entry["config"])
            actual = _load_yaml(frozen)
            print(
                f"READS: candidate={entry['candidate_id']} task={entry['task']} seed={entry['seed']} frozen_config={frozen}",
                flush=True,
            )
            run_dir = Path(_train(entry["task"], actual, frozen)).resolve()
            if not run_dir.is_relative_to(root / "experiments"):
                raise ValueError(
                    "Trainer returned a run outside this benchmark's experiment directory"
                )
            entry.update(run_dir=str(run_dir), status="trained")
            _write_json(root / "pipeline.json", state)
            entry["artifacts"] = verify_entry_artifacts(
                entry, config["data_manifest_sha256"], evaluated=False
            )
            if smoke:
                entry.update(status="smoke_trained", finished_at=_now())
                entry["smoke_evaluation_note"] = (
                    "Standalone validation/test evaluator skipped. Existing navigation trainer may write its built-in terminal test metrics; smoke artifacts are not benchmark results."
                )
            else:
                # The trained-only inventory grows after evaluation, so it is not
                # the completed-run inventory used to detect later tampering.
                entry.pop("artifacts")
                entry["status"] = "evaluating"
                _write_json(root / "pipeline.json", state)
                _evaluate(entry["task"], run_dir, config.get("device", "cuda"))
                entry["artifacts"] = verify_entry_artifacts(entry, config["data_manifest_sha256"])
                entry.update(status="evaluated", finished_at=_now())
            _write_json(root / "pipeline.json", state)
            print(
                f"COMPLETED: candidate={entry['candidate_id']} task={entry['task']} seed={entry['seed']} status={entry['status']} run={run_dir}",
                flush=True,
            )
            active = None
        if smoke:
            state["status"] = "smoke_complete"
            print(
                "SKIPS: smoke runs are excluded from formal comparison tables and test-driven candidate selection",
                flush=True,
            )
        else:
            aggregate_benchmark_results(root, state["runs"])
            state["status"] = "complete"
        state["finished_at"] = _now()
    except BaseException as error:
        if active is not None:
            active.update(
                failed_stage=active["status"],
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        state.update(
            status="failed", error_type=type(error).__name__, error=str(error), updated_at=_now()
        )
        _write_json(root / "pipeline.json", state)
        raise
    _write_json(root / "pipeline.json", state)
    print(f"BENCHMARK: {root} status={state['status']}", flush=True)
    return root
