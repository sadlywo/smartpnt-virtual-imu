"""Resumable execution of the four-fold Penn experiment matrix."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.config import load_config, save_config
from virtual_imu.eval.baselines import VIRTUAL_COLUMNS, fit_system_id_baseline_state
from virtual_imu.eval.penn_loso_comparison import (
    PENN_BASELINE_METHODS,
    _prediction_frame,
    evaluate_penn_method,
    fit_penn_baselines,
    predict_penn_baselines,
)
from virtual_imu.eval.penn_matrix_aggregation import (
    aggregate_matrix_trials,
    trial_row,
    write_matrix_aggregation,
)
from virtual_imu.eval.penn_matrix_report import write_penn_matrix_report
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.io.training_split_manifest import (
    TrainingSplitBundle,
    load_training_split_manifest,
)
from virtual_imu.training.checkpointing import load_checkpoint
from virtual_imu.training.penn_matrix_config import (
    FOLD_ASSIGNMENTS,
    PennTrialSpec,
    build_effective_trial_config,
    build_trial_specs,
)
from virtual_imu.training.trainer import train_model
from virtual_imu.utils.paths import git_commit, project_root_from_file, utc_now_iso

_DRIFT_METHODS = {"drift_only", "bias_drift", "fir_bias_drift_residual"}


def run_penn_matrix(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    smoke: bool = False,
    resume_run: Path | None = None,
) -> Path:
    """Execute or safely resume the canonical 60-trial Penn matrix."""
    project_root = project_root_from_file(__file__)
    source_config = Path(config_path).resolve() if config_path is not None else None
    matrix_config = deepcopy(config)
    specs = build_trial_specs(matrix_config, project_root)
    if len(specs) != 60:
        raise ValueError("Penn matrix validation did not produce exactly 60 neural specs")

    base_path = _resolve_path(matrix_config["base_model_config"], project_root)
    strict_path = _resolve_path(matrix_config["strict_manifest"], project_root)
    base_config = load_config(base_path)
    bundles = _load_all_bundles(specs, strict_path)
    effective_configs = {
        spec.trial_id: _prepare_effective(
            matrix_config, base_config, spec, project_root, smoke
        )
        for spec in specs
    }
    output_paths = _output_paths(matrix_config, project_root)
    run_fingerprint_payload = _run_fingerprint_payload(
        matrix_config,
        base_config,
        source_config,
        base_path,
        strict_path,
        specs,
        smoke,
    )
    run_fingerprint = _json_sha256(run_fingerprint_payload)

    creating = resume_run is None
    if creating:
        run_dir = _create_run_dir(
            output_paths["runs_dir"], str(matrix_config["experiment_name"]), smoke
        )
        _snapshot_run_inputs(
            run_dir,
            matrix_config,
            source_config,
            base_path,
            strict_path,
            specs,
            run_fingerprint,
            run_fingerprint_payload,
        )
    else:
        run_dir = Path(resume_run).resolve()
        _validate_resume_run(run_dir, run_fingerprint)
        logger.info("READS kind=resume_run path={}", run_dir)

    effective_paths = _finalize_effective_configs(
        run_dir, specs, effective_configs, creating=creating
    )
    _write_pointer(output_paths["active_run_file"], run_dir, "active matrix run")
    analytic_rows, baseline_provenance = _run_baselines(
        run_dir, bundles, matrix_config, run_fingerprint
    )

    neural_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trial_artifacts: dict[str, dict[str, str]] = {}
    for spec in specs:
        bundle = bundles[spec.fold_id]
        effective = effective_configs[spec.trial_id]
        effective_path = effective_paths[spec.trial_id]
        fingerprints = _trial_fingerprints(
            spec, effective, effective_path, bundle
        )
        row, artifacts, skipped = _run_neural_trial(
            run_dir,
            spec,
            bundle,
            effective,
            effective_path,
            fingerprints,
            smoke,
        )
        neural_rows.append(row)
        trial_artifacts[spec.trial_id] = artifacts
        ledger_rows.append(
            {
                "trial_id": spec.trial_id,
                "fold_id": spec.fold_id,
                "method": spec.method,
                "seed": spec.seed,
                "status": "complete",
                "skipped": bool(skipped),
                "trial_json": artifacts["trial_json"],
            }
        )
        _write_ledger(run_dir / "trial_ledger.csv", ledger_rows)

    if len(ledger_rows) != 60:
        raise RuntimeError("Execution ledger must contain exactly 60 neural rows")
    neural = pd.DataFrame(neural_rows)
    analytic = pd.DataFrame(analytic_rows)
    all_trials = pd.concat([neural, analytic], ignore_index=True)
    if len(all_trials) != 76:
        raise RuntimeError("Aggregate trials table must contain exactly 76 metric rows")
    folds, overall = aggregate_matrix_trials(neural, analytic)

    provenance = {
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(project_root),
        "run_dir": str(run_dir),
        "smoke": bool(smoke),
        "run_fingerprint": run_fingerprint,
        "manifest_hashes": {
            fold_id: {
                "split": _sha256(bundle.split_manifest_path),
                "strict": _sha256(bundle.strict_manifest_path),
            }
            for fold_id, bundle in bundles.items()
        },
        "config_hashes": {
            "matrix": _json_sha256(matrix_config),
            "base": _sha256(base_path),
        },
        "split_provenance": {
            fold_id: _split_sequences(bundle) for fold_id, bundle in bundles.items()
        },
        "baseline_fit_sequences": {
            fold_id: item["fit_sequences"]
            for fold_id, item in baseline_provenance.items()
        },
        "baseline_artifacts": baseline_provenance,
        "trial_artifact_paths": trial_artifacts,
    }
    _write_json(run_dir / "provenance.json", provenance, "matrix provenance")
    aggregation = write_matrix_aggregation(all_trials, folds, overall, run_dir)
    _log_artifact_map(aggregation)
    report_path = write_penn_matrix_report(
        overall, folds, all_trials, provenance, run_dir / "report.md"
    )
    logger.info("WRITES kind=matrix_report path={}", report_path)
    _publish_latest(run_dir, output_paths, report_path)
    return run_dir


def _load_all_bundles(
    specs: tuple[PennTrialSpec, ...], strict_path: Path
) -> dict[str, TrainingSplitBundle]:
    bundles: dict[str, TrainingSplitBundle] = {}
    for spec in specs:
        if spec.fold_id in bundles:
            continue
        logger.info(
            "READS kind=strict_fold_bundle fold={} split={} strict={}",
            spec.fold_id,
            spec.manifest_path,
            strict_path,
        )
        bundles[spec.fold_id] = load_training_split_manifest(
            spec.manifest_path, strict_path
        )
    if set(bundles) != set(FOLD_ASSIGNMENTS):
        raise ValueError("Penn matrix requires all four strict fold bundles")
    return bundles


def _prepare_effective(
    matrix: dict[str, Any],
    base: dict[str, Any],
    spec: PennTrialSpec,
    project_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    effective = build_effective_trial_config(
        matrix, base, spec, project_root=project_root
    )
    if smoke:
        training = effective["training"]
        training["epochs"] = 1
        training["early_stopping_patience"] = 1
        training["device"] = "cpu"
        model = effective["model"]
        model["hidden_channels"] = 8
        model["levels"] = 1
        data = effective["data"]
        data["window_size"] = 16
        data["stride"] = min(int(data.get("stride", 16)), 16)
        static = data.get("static_context")
        if isinstance(static, dict) and static.get("enabled"):
            static["window_size"] = 16
    return effective


def _finalize_effective_configs(
    run_dir: Path,
    specs: tuple[PennTrialSpec, ...],
    effective_configs: Mapping[str, dict[str, Any]],
    *,
    creating: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        effective = effective_configs[spec.trial_id]
        effective.setdefault("output", {})["runs_dir"] = str(
            (run_dir / "trials" / spec.trial_id / "training").resolve()
        )
        effective["matrix_fold_id"] = spec.fold_id
        effective["matrix_method"] = spec.method
        effective["seed"] = spec.seed
        path = run_dir / "configs" / "effective" / f"{spec.trial_id}.yaml"
        paths[spec.trial_id] = path
        if creating:
            save_config(effective, path)
            logger.info("WRITES kind=effective_config_snapshot path={}", path)
            continue
        if not path.is_file():
            raise FileNotFoundError(
                f"Resume effective config snapshot is missing: {path}"
            )
        logger.info("READS kind=effective_config_snapshot path={}", path)
        if _json_sha256(load_config(path)) != _json_sha256(effective):
            raise ValueError(
                f"Resume effective config hash mismatch: {spec.trial_id}"
            )
    return paths


def _run_baselines(
    run_dir: Path,
    bundles: Mapping[str, TrainingSplitBundle],
    matrix: dict[str, Any],
    run_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: dict[str, dict[str, Any]] = {}
    for fold_id, bundle in bundles.items():
        fold_dir = run_dir / "baselines" / fold_id
        train = bundle.frames["train"]
        test = bundle.frames["test"]
        fit_sequences = _sequence_ids(train)
        baseline_fingerprint = {
            "run_fingerprint": run_fingerprint,
            "fold_id": fold_id,
            "fir": deepcopy(matrix["fir"]),
            "split_manifest_sha256": _sha256(bundle.split_manifest_path),
            "strict_manifest_sha256": _sha256(bundle.strict_manifest_path),
            "train_sequences": fit_sequences,
            "test_sequence": FOLD_ASSIGNMENTS[fold_id]["test"][0],
        }
        metadata_path = fold_dir / "baseline.json"
        if metadata_path.is_file():
            metadata = _read_json(metadata_path, "completed baseline metadata")
            if metadata.get("status") != "complete":
                raise ValueError(
                    f"Existing baseline metadata is not a valid completed fold: {fold_id}"
                )
            fold_rows, fold_provenance = _audit_completed_baseline(
                fold_dir, fold_id, metadata, baseline_fingerprint
            )
            rows.extend(fold_rows)
            provenance[fold_id] = fold_provenance
            logger.info(
                "SKIPS kind=baseline_fold fold={} reason=matching completed fingerprint",
                fold_id,
            )
            continue
        logger.info(
            "READS kind=baseline_fit_frame fold={} split=train sequences={} rows={}",
            fold_id,
            fit_sequences,
            len(train),
        )
        state = fit_penn_baselines(
            train,
            fir_taps=int(matrix["fir"]["taps"]),
            alpha=float(matrix["fir"]["ridge_alpha"]),
        )
        state_path = fold_dir / "state.json"
        _write_json(state_path, state, "train-only baseline state")
        predictions = predict_penn_baselines(state, test)
        artifacts: dict[str, dict[str, str]] = {}
        for method in PENN_BASELINE_METHODS:
            frame = _prediction_frame(test, predictions[method])
            _audit_prediction(frame, FOLD_ASSIGNMENTS[fold_id]["test"][0])
            prediction_path = write_parquet(
                frame, fold_dir / "predictions" / f"{method}.parquet"
            )
            logger.info("WRITES kind=baseline_prediction path={}", prediction_path)
            metrics = evaluate_penn_method(frame)
            metric_path = fold_dir / "metrics" / f"{method}.json"
            _write_json(metric_path, metrics, "baseline metrics")
            rows.append(trial_row(fold_id, method, None, metrics))
            artifacts[method] = {
                "prediction": str(prediction_path.resolve()),
                "metrics": str(metric_path.resolve()),
            }
        completed_artifacts = {
            "state": _hashed_artifact(state_path),
            "methods": {
                method: {
                    "prediction": _hashed_artifact(
                        Path(artifacts[method]["prediction"])
                    ),
                    "metrics": _hashed_artifact(Path(artifacts[method]["metrics"])),
                }
                for method in PENN_BASELINE_METHODS
            },
        }
        completed = {
            "status": "complete",
            "fold_id": fold_id,
            "fingerprint": baseline_fingerprint,
            "fit_split": "train",
            "fit_sequences": fit_sequences,
            "artifacts": completed_artifacts,
        }
        _write_json(metadata_path, completed, "completed baseline metadata")
        provenance[fold_id] = {
            "fit_split": "train",
            "fit_sequences": fit_sequences,
            "state": str(state_path.resolve()),
            "artifacts": artifacts,
            "metadata": str(metadata_path.resolve()),
        }
    if len(rows) != 16:
        raise RuntimeError("Analytic baseline matrix must contain exactly 16 rows")
    return rows, provenance


def _audit_completed_baseline(
    fold_dir: Path,
    fold_id: str,
    metadata: dict[str, Any],
    expected_fingerprint: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"Completed baseline fingerprint mismatch: {fold_id}")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Completed baseline artifacts are missing: {fold_id}")
    expected_state = fold_dir / "state.json"
    _audit_hashed_artifact(artifacts.get("state"), expected_state, fold_id)
    state = _read_json(expected_state, "completed baseline state")
    missing = [method for method in PENN_BASELINE_METHODS if method not in state]
    if missing:
        raise ValueError(
            f"Completed baseline state is missing methods for {fold_id}: {missing}"
        )

    method_artifacts = artifacts.get("methods")
    if not isinstance(method_artifacts, Mapping):
        raise ValueError(f"Completed baseline method artifacts are missing: {fold_id}")
    rows: list[dict[str, Any]] = []
    provenance_artifacts: dict[str, dict[str, str]] = {}
    for method in PENN_BASELINE_METHODS:
        recorded = method_artifacts.get(method)
        if not isinstance(recorded, Mapping):
            raise ValueError(
                f"Completed baseline artifacts are missing for {fold_id}/{method}"
            )
        prediction_path = fold_dir / "predictions" / f"{method}.parquet"
        metrics_path = fold_dir / "metrics" / f"{method}.json"
        _audit_hashed_artifact(
            recorded.get("prediction"), prediction_path, fold_id
        )
        _audit_hashed_artifact(recorded.get("metrics"), metrics_path, fold_id)
        logger.info("READS kind=completed_baseline_prediction path={}", prediction_path)
        prediction = pd.read_parquet(prediction_path)
        _audit_prediction(prediction, FOLD_ASSIGNMENTS[fold_id]["test"][0])
        metrics = _read_json(metrics_path, "completed baseline metrics")
        rows.append(trial_row(fold_id, method, None, metrics))
        provenance_artifacts[method] = {
            "prediction": str(prediction_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        }
    return rows, {
        "fit_split": "train",
        "fit_sequences": list(metadata.get("fit_sequences", [])),
        "state": str(expected_state.resolve()),
        "artifacts": provenance_artifacts,
        "metadata": str((fold_dir / "baseline.json").resolve()),
    }


def _run_neural_trial(
    run_dir: Path,
    spec: PennTrialSpec,
    bundle: TrainingSplitBundle,
    effective: dict[str, Any],
    effective_path: Path,
    fingerprints: dict[str, Any],
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, str], bool]:
    trial_dir = run_dir / "trials" / spec.trial_id
    trial_json = trial_dir / "trial.json"
    expected_artifacts = {
        "trial_json": str(trial_json.resolve()),
        "effective_config": str(effective_path.resolve()),
        "prediction": str((trial_dir / "prediction.parquet").resolve()),
        "metrics": str((trial_dir / "metrics.json").resolve()),
    }
    if trial_json.is_file():
        payload = _read_json(trial_json, "completed trial metadata")
        if payload.get("status") == "complete":
            if payload.get("fingerprints") != fingerprints:
                raise ValueError(
                    f"Completed trial fingerprint mismatch: {spec.trial_id}"
                )
            row = _audit_completed_trial(payload, spec, bundle)
            logger.info(
                "SKIPS kind=neural_training trial={} reason=matching completed fingerprint",
                spec.trial_id,
            )
            return row, _payload_artifacts(payload), True

    trial_dir.mkdir(parents=True, exist_ok=True)
    logger.info("READS kind=effective_trial_config path={}", effective_path)
    logger.info(
        "WRITES kind=training_artifacts trial={} root={}",
        spec.trial_id,
        effective["output"]["runs_dir"],
    )
    training_run = Path(train_model(effective, config_path=effective_path)).resolve()
    checkpoint_path = training_run / "best.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Neural trial checkpoint is missing: {checkpoint_path}"
        )
    logger.info("READS kind=best_checkpoint path={}", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    _audit_checkpoint(checkpoint, spec, bundle)
    generator = VirtualIMUGenerator(
        checkpoint_path, device="cpu" if smoke else "auto"
    )
    prediction = generator.run_frame(bundle.frames["test"])
    _audit_prediction(prediction, spec.test_sequence)
    prediction_path = write_parquet(prediction, trial_dir / "prediction.parquet")
    logger.info("WRITES kind=neural_prediction path={}", prediction_path)
    metrics = evaluate_penn_method(prediction)
    metric_path = trial_dir / "metrics.json"
    _write_json(metric_path, metrics, "neural trial metrics")
    artifacts = {
        **expected_artifacts,
        "checkpoint": str(checkpoint_path),
        "training_run": str(training_run),
    }
    payload = {
        "status": "complete",
        "trial_id": spec.trial_id,
        "fold_id": spec.fold_id,
        "method": spec.method,
        "seed": spec.seed,
        "split_sequences": _split_sequences(bundle),
        "fingerprints": fingerprints,
        "checkpoint": str(checkpoint_path),
        "prediction": str(prediction_path.resolve()),
        "metrics_path": str(metric_path.resolve()),
        "effective_config": str(effective_path.resolve()),
        "training_run": str(training_run),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "checkpoint_metric": str(checkpoint.get("checkpoint_metric", "unknown")),
        "best_metric": float(checkpoint.get("best_metric", np.nan)),
        "static_window_metadata": checkpoint.get("static_window_metadata", {}),
        "competitor_baseline": checkpoint.get("competitor_baseline", {}),
        "artifacts": artifacts,
        "metrics": metrics,
    }
    _write_json(trial_json, payload, "completed trial metadata")
    return trial_row(spec.fold_id, spec.method, spec.seed, metrics), artifacts, False


def _audit_completed_trial(
    payload: dict[str, Any], spec: PennTrialSpec, bundle: TrainingSplitBundle
) -> dict[str, Any]:
    checkpoint_path = Path(str(payload.get("checkpoint", "")))
    prediction_path = Path(str(payload.get("prediction", "")))
    metrics_path = Path(str(payload.get("metrics_path", "")))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Completed trial checkpoint is missing: {checkpoint_path}")
    if not prediction_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Completed trial prediction or metrics artifact is missing: {spec.trial_id}"
        )
    effective_path = Path(str(payload.get("effective_config", "")))
    if not effective_path.is_file():
        raise FileNotFoundError(
            f"Completed trial effective config is missing: {effective_path}"
        )
    effective_hash = _json_sha256(load_config(effective_path))
    recorded_hash = payload["fingerprints"].get("effective_config_sha256")
    if effective_hash != recorded_hash:
        raise ValueError(
            f"Completed trial effective config hash mismatch: {spec.trial_id}"
        )
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    _audit_checkpoint(checkpoint, spec, bundle)
    logger.info("READS kind=completed_prediction path={}", prediction_path)
    prediction = pd.read_parquet(prediction_path)
    _audit_prediction(prediction, spec.test_sequence)
    metrics = _read_json(metrics_path, "completed trial metrics")
    return trial_row(spec.fold_id, spec.method, spec.seed, metrics)


def _audit_prediction(frame: pd.DataFrame, test_sequence: str) -> None:
    if "sequence_id" not in frame.columns:
        raise ValueError("Prediction is missing held-out sequence identity")
    sequences = set(frame["sequence_id"].astype(str))
    if sequences != {test_sequence}:
        raise ValueError(
            f"Prediction must contain only held-out sequence {test_sequence}; got {sorted(sequences)}"
        )
    missing = [column for column in VIRTUAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction is missing six virtual columns: {missing}")
    values = frame[VIRTUAL_COLUMNS].to_numpy(dtype=float)
    if values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("Prediction virtual columns must contain six finite values per row")


def _audit_checkpoint(
    checkpoint: Mapping[str, Any], spec: PennTrialSpec, bundle: TrainingSplitBundle
) -> None:
    train_sequences = set(_sequence_ids(bundle.frames["train"]))
    if spec.method in _DRIFT_METHODS:
        metadata = checkpoint.get("static_window_metadata")
        if not isinstance(metadata, Mapping) or not metadata:
            raise ValueError(
                f"{spec.trial_id} static context metadata must identify a train split source"
            )
        sources = {
            str(value.get("sequence_id", ""))
            for value in metadata.values()
            if isinstance(value, Mapping)
        }
        if not sources or not sources <= train_sequences:
            raise ValueError(
                f"{spec.trial_id} static context source must belong to the train split"
            )
    if spec.method != "fir_bias_drift_residual":
        return
    if str(checkpoint.get("target_mode", "")) != "competitor_residual":
        raise ValueError("Hybrid checkpoint must target competitor_residual")
    competitor = checkpoint.get("competitor_baseline")
    if not isinstance(competitor, Mapping) or competitor.get("enabled") is not True:
        raise ValueError("Hybrid checkpoint competitor baseline must be enabled")
    if str(competitor.get("mode", "")) != "fir":
        raise ValueError("Hybrid checkpoint competitor baseline must use mode=fir")
    if not isinstance(competitor.get("state"), Mapping) or not competitor["state"]:
        raise ValueError("Hybrid checkpoint is missing serialized train-fitted FIR state")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Hybrid checkpoint is missing its effective config")
    training = checkpoint_config.get("training")
    model = checkpoint_config.get("model")
    configured = training.get("competitor_baseline") if isinstance(training, Mapping) else None
    if not isinstance(model, Mapping) or model.get("target_mode") != "competitor_residual":
        raise ValueError("Hybrid effective config must target competitor_residual")
    if not isinstance(configured, Mapping) or configured.get("enabled") is not True:
        raise ValueError("Hybrid effective config competitor baseline must be enabled")
    if configured.get("mode") != "fir":
        raise ValueError("Hybrid effective config competitor baseline must use mode=fir")
    expected_state = fit_system_id_baseline_state(
        bundle.frames["train"],
        fir_taps=int(configured.get("fir_taps", 9)),
        alpha=float(configured.get("ridge_alpha", 1e-2)),
    )
    if _json_sha256(competitor["state"]) != _json_sha256(expected_state):
        raise ValueError("Hybrid FIR state must be fitted from the fold train split")


def _trial_fingerprints(
    spec: PennTrialSpec,
    effective: dict[str, Any],
    effective_path: Path,
    bundle: TrainingSplitBundle,
) -> dict[str, Any]:
    return {
        "effective_config_sha256": _json_sha256(effective),
        "effective_config_path": str(effective_path.resolve()),
        "split_manifest_sha256": _sha256(bundle.split_manifest_path),
        "strict_manifest_sha256": _sha256(bundle.strict_manifest_path),
        "fold_id": spec.fold_id,
        "method": spec.method,
        "seed": spec.seed,
    }


def _run_fingerprint_payload(
    matrix: dict[str, Any],
    base: dict[str, Any],
    source_config: Path | None,
    base_path: Path,
    strict_path: Path,
    specs: tuple[PennTrialSpec, ...],
    smoke: bool,
) -> dict[str, Any]:
    split_paths = {
        spec.fold_id: spec.manifest_path.resolve()
        for spec in specs
        if spec.fold_id in FOLD_ASSIGNMENTS
    }
    source_files: dict[str, Any] = {
        "base_config": _source_file_fingerprint(base_path),
        "strict_manifest": _source_file_fingerprint(strict_path),
        "split_manifests": {
            fold_id: _source_file_fingerprint(path)
            for fold_id, path in split_paths.items()
        },
    }
    if source_config is not None:
        source_files["matrix_config"] = _source_file_fingerprint(source_config)
    return {
        "canonical_matrix_config": matrix,
        "fully_effective_base_config": base,
        "smoke": bool(smoke),
        "source_paths": {
            "matrix_config": str(source_config) if source_config is not None else None,
            "base_config": str(base_path.resolve()),
            "strict_manifest": str(strict_path.resolve()),
            "split_manifests": {
                fold_id: str(path) for fold_id, path in split_paths.items()
            },
        },
        "source_files": source_files,
    }


def _source_file_fingerprint(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    logger.info("READS kind=run_fingerprint_source path={}", resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "content": resolved.read_text(encoding="utf-8"),
    }


def _snapshot_run_inputs(
    run_dir: Path,
    matrix: dict[str, Any],
    source_config: Path | None,
    base_path: Path,
    strict_path: Path,
    specs: tuple[PennTrialSpec, ...],
    run_fingerprint: str,
    run_fingerprint_payload: dict[str, Any],
) -> None:
    configs = run_dir / "configs"
    manifests = run_dir / "manifests"
    save_config(matrix, configs / "matrix.yaml")
    logger.info("WRITES kind=matrix_config_snapshot path={}", configs / "matrix.yaml")
    _copy(base_path, configs / "base.yaml", "base config snapshot")
    if source_config is not None:
        _copy(source_config, configs / "matrix_source.yaml", "matrix YAML snapshot")
    _copy(strict_path, manifests / "strict_manifest.json", "strict manifest snapshot")
    seen: set[str] = set()
    for spec in specs:
        if spec.fold_id not in seen:
            _copy(
                spec.manifest_path,
                manifests / f"{spec.fold_id}.json",
                "split manifest snapshot",
            )
            seen.add(spec.fold_id)
    hashes = {
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": run_fingerprint_payload,
        "matrix_config_sha256": _json_sha256(matrix),
        "base_config_sha256": _sha256(base_path),
        "strict_manifest_sha256": _sha256(strict_path),
        "split_manifest_sha256": {
            spec.fold_id: _sha256(spec.manifest_path)
            for spec in specs
            if spec.fold_id in FOLD_ASSIGNMENTS
        },
    }
    _write_json(run_dir / "hashes.json", hashes, "run input hashes")


def _validate_resume_run(run_dir: Path, run_fingerprint: str) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run does not exist: {run_dir}")
    hashes = _read_json(run_dir / "hashes.json", "resume run hashes")
    if hashes.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Resume run fingerprint does not match current inputs")


def _publish_latest(
    run_dir: Path, output: Mapping[str, Path], report_path: Path
) -> None:
    latest_dir = output["latest_dir"]
    if latest_dir.name.lower() != "latest":
        raise ValueError(f"output.latest_dir must end in 'latest': {latest_dir}")
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True)
    for source in (
        run_dir / "trials.csv",
        run_dir / "fold_summary.csv",
        run_dir / "overall_summary.csv",
        run_dir / "provenance.json",
        report_path,
    ):
        destination = latest_dir / source.name
        shutil.copy2(source, destination)
        logger.info("WRITES kind=latest_lightweight_artifact path={}", destination)
    logger.info(
        "SKIPS kind=latest_large_artifacts reason=checkpoints and predictions remain in source run"
    )
    _write_pointer(output["latest_run_file"], run_dir, "latest successful run")


def _output_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    return {
        key: _resolve_path(config["output"][key], project_root)
        for key in (
            "runs_dir",
            "active_run_file",
            "latest_dir",
            "latest_run_file",
        )
    }


def _create_run_dir(runs_dir: Path, experiment: str, smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_smoke" if smoke else ""
    run_dir = (runs_dir / f"{experiment}_{stamp}{suffix}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("WRITES kind=matrix_run_root path={}", run_dir)
    return run_dir


def _split_sequences(bundle: TrainingSplitBundle) -> dict[str, list[str]]:
    return {
        split: [
            str(record["sequence_id"])
            for record in bundle.split_manifest["splits"][split]
        ]
        for split in ("train", "validation", "test")
    }


def _sequence_ids(frame: pd.DataFrame) -> list[str]:
    return [str(value) for value in frame["sequence_id"].drop_duplicates().tolist()]


def _payload_artifacts(payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Completed trial is missing artifact paths")
    return {str(key): str(value) for key, value in artifacts.items()}


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("WRITES kind=trial_execution_ledger path={} rows={}", path, len(rows))


def _write_pointer(path: Path, run_dir: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    logger.info("WRITES kind={} path={}", kind, path)


def _copy(source: Path, destination: Path, kind: str) -> None:
    logger.info("READS kind={} path={}", kind, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    logger.info("WRITES kind={} path={}", kind, destination)


def _write_json(path: Path, payload: Any, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    logger.info("WRITES kind={} path={}", kind, path)


def _read_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    logger.info("READS kind={} path={}", kind, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} must contain a JSON object")
    return payload


def _hashed_artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _audit_hashed_artifact(
    recorded: Any, expected_path: Path, fold_id: str
) -> None:
    if not isinstance(recorded, Mapping):
        raise ValueError(f"Completed baseline artifact metadata is missing: {fold_id}")
    if recorded.get("path") != str(expected_path.resolve()):
        raise ValueError(f"Completed baseline artifact path mismatch: {fold_id}")
    if not expected_path.is_file():
        raise FileNotFoundError(
            f"Completed baseline artifact is missing: {expected_path}"
        )
    if recorded.get("sha256") != _sha256(expected_path):
        raise ValueError(f"Completed baseline artifact hash mismatch: {fold_id}")


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _log_artifact_map(artifacts: Mapping[str, Mapping[str, Path]]) -> None:
    for group in artifacts.values():
        for path in group.values():
            logger.info("WRITES kind=matrix_aggregation path={}", path)
