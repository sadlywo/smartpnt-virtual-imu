"""Config-driven orchestration for the strict Penn default LOSO comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger

from virtual_imu.config import load_config, save_config
from virtual_imu.eval.penn_loso_comparison import (
    fit_penn_baselines,
    predict_penn_baselines,
    write_penn_comparison,
)
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.training.checkpointing import load_checkpoint
from virtual_imu.training.dataset import select_low_motion_source
from virtual_imu.training.trainer import train_model
from virtual_imu.utils.paths import git_commit, project_root_from_file, utc_now_iso

ANALYTIC_METHODS = (
    "identity",
    "mean_bias",
    "channel_affine",
    "learned_fir_causal",
)
NEURAL_METHODS = ("tcn", "bias_drift_tcn")
PENN_METHODS = (*ANALYTIC_METHODS, *NEURAL_METHODS)
LOW_FEATURE_COLUMNS = [
    "gx_low",
    "gy_low",
    "gz_low",
    "ax_low",
    "ay_low",
    "az_low",
]


def validate_penn_loso_config(config: dict[str, Any]) -> None:
    """Validate the comparison-level contract without touching input data."""
    if not isinstance(config, dict):
        raise ValueError("Penn LOSO config must be a mapping")
    if not str(config.get("experiment_name", "")).strip():
        raise ValueError("experiment_name must be configured")
    for key in ("split_manifest", "strict_manifest"):
        if not isinstance(config.get(key), (str, Path)) or not str(config[key]).strip():
            raise ValueError(f"{key} must be a non-empty path")

    models = config.get("models")
    if not isinstance(models, dict) or set(models) != set(NEURAL_METHODS):
        raise ValueError(f"models must contain exactly {list(NEURAL_METHODS)}")
    if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in models.values()):
        raise ValueError("models entries must be non-empty config paths")

    baselines = config.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("baselines must be configured")
    if int(baselines.get("fir_taps", 0)) < 1:
        raise ValueError("baselines.fir_taps must be at least 1")
    if float(baselines.get("ridge_alpha", -1.0)) < 0.0:
        raise ValueError("baselines.ridge_alpha must be non-negative")

    output = config.get("output")
    required_outputs = ("runs_dir", "latest_dir", "latest_run_file")
    if not isinstance(output, dict) or any(not str(output.get(key, "")).strip() for key in required_outputs):
        raise ValueError(f"output must configure {list(required_outputs)}")


def apply_smoke_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a one-epoch, CPU, small-network copy of a model config."""
    result = deepcopy(config)
    training = result.setdefault("training", {})
    training["epochs"] = 1
    training["early_stopping_patience"] = 1
    training["device"] = "cpu"

    model = result.setdefault("model", {})
    model["hidden_channels"] = min(int(model.get("hidden_channels", 8)), 8)
    model["levels"] = 1
    model["kernel_size"] = min(int(model.get("kernel_size", 3)), 3)
    model["dropout"] = 0.0

    data = result.setdefault("data", {})
    data["window_size"] = min(int(data.get("window_size", 16)), 16)
    data["stride"] = min(int(data.get("stride", 8)), 8)
    data["batch_size"] = min(int(data.get("batch_size", 8)), 8)
    static_context = data.get("static_context")
    if isinstance(static_context, dict) and static_context.get("enabled", False):
        static_context["window_size"] = min(int(static_context.get("window_size", 16)), 16)
    return result


def run_penn_loso_comparison(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    smoke: bool = False,
) -> Path:
    """Run all six Penn methods and return the source-of-truth comparison directory."""
    validate_penn_loso_config(config)
    comparison = deepcopy(config)
    project_root = project_root_from_file(__file__)
    source_config = Path(config_path).resolve() if config_path is not None else None
    config_dir = source_config.parent if source_config is not None else project_root
    split_path = _resolve_input_path(comparison["split_manifest"], config_dir, project_root)
    strict_path = _resolve_input_path(comparison["strict_manifest"], config_dir, project_root)
    model_paths = {
        method: _resolve_input_path(path, config_dir, project_root)
        for method, path in comparison["models"].items()
    }
    output = comparison["output"]
    runs_dir = _resolve_output_path(output["runs_dir"], project_root)
    latest_dir = _resolve_output_path(output["latest_dir"], project_root)
    latest_run_file = _resolve_output_path(output["latest_run_file"], project_root)
    run_dir = runs_dir / _run_name(str(comparison["experiment_name"]), smoke)
    run_dir.mkdir(parents=True, exist_ok=False)
    _log_write(run_dir, "comparison run root")

    if source_config is not None:
        _log_read(source_config, "comparison config")
    if smoke:
        _log_skip("full neural training budget", "--smoke selects one epoch and a small network")
    _log_skip("raw/processed data mutation", "all source Parquet inputs are read-only")

    configs_dir = run_dir / "configs"
    manifests_dir = run_dir / "manifests"
    comparison_snapshot = configs_dir / "comparison.yaml"
    save_config(comparison, comparison_snapshot)
    _log_write(comparison_snapshot, "comparison config snapshot")

    _log_read(split_path, "strict split manifest")
    _log_read(strict_path, "strict aggregate manifest")
    bundle = load_training_split_manifest(split_path, strict_path)
    train_df = bundle.frames["train"]
    test_df = bundle.frames["test"]
    split_snapshot = manifests_dir / "split_manifest.json"
    strict_snapshot = manifests_dir / "strict_manifest.json"
    _copy_snapshot(split_path, split_snapshot, "split manifest snapshot")
    _copy_snapshot(strict_path, strict_snapshot, "strict manifest snapshot")

    baseline_cfg = comparison["baselines"]
    logger.info(
        "READS kind=baseline_fit_frame split=train rows={} sequences={}",
        len(train_df),
        _sequence_ids(train_df),
    )
    baseline_state = fit_penn_baselines(
        train_df,
        fir_taps=int(baseline_cfg["fir_taps"]),
        alpha=float(baseline_cfg["ridge_alpha"]),
    )
    baseline_state_path = run_dir / "baselines" / "state.json"
    _write_json(baseline_state_path, baseline_state, "train-fitted baseline state")
    predictions: dict[str, Any] = predict_penn_baselines(baseline_state, test_df)

    model_provenance: dict[str, dict[str, Any]] = {}
    method_metadata: dict[str, dict[str, Any]] = {}
    for method in NEURAL_METHODS:
        model_path = model_paths[method]
        _log_read(model_path, f"{method} source config")
        model_config = load_config(model_path)
        _validate_model_config(method, model_config)
        effective = _effective_model_config(
            model_config,
            comparison,
            split_path,
            strict_path,
            run_dir / "training" / method,
            smoke,
        )
        effective_snapshot = configs_dir / f"{method}.yaml"
        save_config(effective, effective_snapshot)
        _log_write(effective_snapshot, f"{method} effective config snapshot")
        logger.info("WRITES kind=model_training_artifacts method={} root={}", method, effective["output"]["runs_dir"])
        training_run = train_model(effective, config_path=model_path)
        checkpoint_path = training_run / "best.ckpt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"{method} training did not produce best.ckpt: {checkpoint_path}")
        _log_read(checkpoint_path, f"{method} best checkpoint")
        generator = VirtualIMUGenerator(checkpoint_path, device="cpu" if smoke else "auto")
        predictions[method] = generator.run_frame(test_df)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        parameter_count = _parameter_count(checkpoint)
        method_metadata[method] = {
            "parameter_count": parameter_count,
            "causal": True,
            "inference_class": "online",
        }
        model_provenance[method] = {
            "source_config": str(model_path),
            "effective_config": str(effective_snapshot.resolve()),
            "training_run": str(training_run.resolve()),
            "best_checkpoint": str(checkpoint_path.resolve()),
            "best_epoch": int(checkpoint["best_epoch"]),
            "checkpoint_metric": str(checkpoint.get("checkpoint_metric", "unknown")),
            "best_metric": float(checkpoint.get("best_metric", float("nan"))),
            "parameter_count": parameter_count,
            "static_context_keys": sorted(str(key) for key in checkpoint.get("static_windows", {})),
            "static_context_metadata": checkpoint.get("static_window_metadata", {}),
            "static_context_sources": _static_context_sources(train_df, effective),
        }

    logger.info("READS kind=evaluation_frame split=test rows={} sequences={}", len(test_df), _sequence_ids(test_df))
    artifacts = write_penn_comparison(
        predictions,
        test_df,
        run_dir,
        method_metadata=method_metadata,
        baseline_state=baseline_state,
    )
    for path in _comparison_artifact_paths(artifacts):
        _log_write(path, "comparison artifact")

    split_provenance = {
        split: [str(record["sequence_id"]) for record in bundle.split_manifest["splits"][split]]
        for split in ("train", "validation", "test")
    }
    provenance = {
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(project_root),
        "smoke": bool(smoke),
        "run_dir": str(run_dir.resolve()),
        "comparison_config": {
            "source": str(source_config) if source_config is not None else None,
            "snapshot": str(comparison_snapshot.resolve()),
        },
        "manifests": {
            "split": _file_provenance(split_path, split_snapshot),
            "strict": _file_provenance(strict_path, strict_snapshot),
        },
        "split_provenance": split_provenance,
        "input_records": list(bundle.input_records),
        "baseline_fit": {
            "split": "train",
            "sequences": _sequence_ids(train_df),
            "row_count": int(len(train_df)),
            "state": str(baseline_state_path.resolve()),
            "fir_taps": int(baseline_cfg["fir_taps"]),
            "ridge_alpha": float(baseline_cfg["ridge_alpha"]),
        },
        "evaluation": {
            "split": "test",
            "sequences": _sequence_ids(test_df),
            "row_count": int(len(test_df)),
            "methods": list(PENN_METHODS),
        },
        "artifacts": _artifact_provenance(artifacts),
        "models": model_provenance,
    }
    provenance_path = run_dir / "provenance.json"
    _write_json(provenance_path, provenance, "comparison provenance")
    _update_latest(run_dir, latest_dir, latest_run_file, provenance_path, artifacts)
    logger.info("SKIPS kind=latest_snapshot_large_artifacts reason=predictions/checkpoints/training_runs remain under source run")
    return run_dir


def run_penn_loso_comparison_from_config(config_path: str | Path, *, smoke: bool = False) -> Path:
    """Load one comparison YAML and execute it."""
    path = Path(config_path).resolve()
    _log_read(path, "comparison config")
    return run_penn_loso_comparison(load_config(path), config_path=path, smoke=smoke)


def _validate_model_config(method: str, config: dict[str, Any]) -> None:
    data = config.get("data")
    model = config.get("model")
    training = config.get("training")
    if not isinstance(data, dict) or data.get("feature_columns") != LOW_FEATURE_COLUMNS:
        raise ValueError(f"{method} data.feature_columns must be the six ordered low-grade axes")
    if not isinstance(model, dict) or str(model.get("type", "")).lower() != method:
        raise ValueError(f"{method} model config must set model.type={method}")
    if str(model.get("target_mode", "")).lower() != "residual":
        raise ValueError(f"{method} model.target_mode must be residual")
    if not isinstance(training, dict) or int(training.get("epochs", 0)) < 1:
        raise ValueError(f"{method} training.epochs must be at least 1")


def _effective_model_config(
    model_config: dict[str, Any],
    comparison: dict[str, Any],
    split_path: Path,
    strict_path: Path,
    training_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    effective = deepcopy(model_config)
    effective["seed"] = int(comparison.get("seed", effective.get("seed", 0)))
    effective["data"]["split_manifest"] = str(split_path)
    effective["data"]["strict_manifest"] = str(strict_path)
    effective.setdefault("output", {})["runs_dir"] = str(training_root)
    return apply_smoke_overrides(effective) if smoke else effective


def _resolve_input_path(value: str | Path, config_dir: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    local = (config_dir / path).resolve()
    if local.exists():
        return local
    return (project_root / path).resolve()


def _resolve_output_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _run_name(experiment_name: str, smoke: bool) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_smoke" if smoke else ""
    return f"{experiment_name}_{stamp}{suffix}"


def _sequence_ids(frame: pd.DataFrame) -> list[str]:
    return [str(value) for value in frame["sequence_id"].drop_duplicates().tolist()]


def _parameter_count(checkpoint: dict[str, Any]) -> int:
    state = checkpoint.get("model_state_dict", {})
    return int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)))


def _static_context_sources(
    train_df: pd.DataFrame, model_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    static_context = model_config.get("data", {}).get("static_context", {})
    if not isinstance(static_context, dict) or not static_context.get("enabled", False):
        return {}
    if str(static_context.get("selection", "")).lower() != "auto_low_motion":
        return {}
    window_size = int(static_context["window_size"])
    group_column = "low_sensor" if "low_sensor" in train_df.columns else "sequence_id"
    sources: dict[str, dict[str, Any]] = {}
    for key, group in train_df.groupby(group_column, sort=False):
        source = select_low_motion_source(group, window_size)
        sources[str(key)] = {
            "sequence_id": str(source["sequence_id"].iloc[0]),
            "start_timestamp_s": float(source["timestamp_s"].iloc[0]),
            "window_size": window_size,
        }
    return sources


def _copy_snapshot(source: Path, destination: Path, kind: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _log_write(destination, kind)


def _write_json(path: Path, payload: Any, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    _log_write(path, kind)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_provenance(source: Path, snapshot: Path) -> dict[str, str]:
    return {
        "source": str(source.resolve()),
        "snapshot": str(snapshot.resolve()),
        "sha256": _sha256(source),
    }


def _comparison_artifact_paths(artifacts: dict[str, Any]) -> list[Path]:
    paths = [
        *artifacts["predictions"].values(),
        *artifacts["metrics"].values(),
        artifacts["comparison_csv"],
        artifacts["comparison_json"],
        artifacts["comparison_markdown"],
    ]
    return [Path(path) for path in paths]


def _artifact_provenance(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "predictions": {
            method: str(Path(path).resolve())
            for method, path in artifacts["predictions"].items()
        },
        "metrics": {
            method: str(Path(path).resolve()) for method, path in artifacts["metrics"].items()
        },
        "reports": {
            "csv": str(Path(artifacts["comparison_csv"]).resolve()),
            "json": str(Path(artifacts["comparison_json"]).resolve()),
            "markdown": str(Path(artifacts["comparison_markdown"]).resolve()),
        },
    }


def _update_latest(
    run_dir: Path,
    latest_dir: Path,
    latest_run_file: Path,
    provenance_path: Path,
    artifacts: dict[str, Any],
) -> None:
    if latest_dir.name.lower() != "latest":
        raise ValueError(f"output.latest_dir must end in 'latest': {latest_dir}")
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=False)
    report_paths = [
        Path(artifacts["comparison_csv"]),
        Path(artifacts["comparison_json"]),
        Path(artifacts["comparison_markdown"]),
        provenance_path,
    ]
    for source in report_paths:
        destination = latest_dir / source.name
        shutil.copy2(source, destination)
        _log_write(destination, "latest report snapshot")
    latest_run_file.parent.mkdir(parents=True, exist_ok=True)
    latest_run_file.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    _log_write(latest_run_file, "latest run pointer")


def _log_read(path: Path, kind: str) -> None:
    logger.info("READS kind={} path={}", kind, path)


def _log_write(path: Path, kind: str) -> None:
    logger.info("WRITES kind={} path={}", kind, path)


def _log_skip(kind: str, reason: str) -> None:
    logger.info("SKIPS kind={} reason={}", kind, reason)
