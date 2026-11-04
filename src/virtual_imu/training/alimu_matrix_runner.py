"""Resumable execution of the strict ALiMU leave-sensor matrix."""

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
from virtual_imu.eval.alimu_matrix_aggregation import (
    ALIMU_ANALYTIC_METHODS,
    aggregate_alimu_trials,
    alimu_trial_row,
    write_alimu_matrix_aggregation,
)
from virtual_imu.eval.alimu_matrix_report import write_alimu_matrix_report
from virtual_imu.eval.baselines import VIRTUAL_COLUMNS, fit_system_id_baseline_state
from virtual_imu.eval.penn_loso_comparison import (
    _prediction_frame,
    evaluate_penn_method,
    fit_penn_baselines,
    predict_penn_baselines,
)
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.training.alimu_matrix_config import (
    ALIMU_HOLDOUT_SENSOR,
    ALIMU_TRAIN_SENSORS,
    ALiMUTrialSpec,
    build_alimu_effective_trial_config,
    build_alimu_trial_specs,
)
from virtual_imu.training.checkpointing import load_checkpoint
from virtual_imu.training.dataset import (
    TRAIN_GLOBAL_STATIC_CONTEXT_KEY,
    build_static_context_windows,
)
from virtual_imu.training.trainer import train_model
from virtual_imu.utils.paths import git_commit, project_root_from_file, utc_now_iso


def run_alimu_matrix(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    smoke: bool = False,
    resume_run: Path | None = None,
) -> Path:
    """Execute or safely resume the canonical 15-neural ALiMU matrix."""
    project_root = project_root_from_file(__file__).resolve()
    matrix = deepcopy(config)
    specs = build_alimu_trial_specs(matrix, project_root)
    if len(specs) != 15:
        raise ValueError("ALiMU matrix validation did not produce exactly 15 specs")

    base_path = _resolve_path(matrix["base_model_config"], project_root)
    base = load_config(base_path)
    source_config = Path(config_path).resolve() if config_path is not None else None
    split_paths = _split_paths(matrix, project_root)
    frames = _load_split_frames(split_paths)
    output = _output_paths(matrix, project_root)
    fingerprint_payload = _run_fingerprint_payload(
        matrix,
        base,
        base_path,
        source_config,
        split_paths,
        smoke,
    )
    run_fingerprint = _json_sha256(fingerprint_payload)

    creating = resume_run is None
    if creating:
        run_dir = _create_run_dir(
            output["runs_dir"], str(matrix["experiment_name"]), smoke
        )
        _snapshot_inputs(
            run_dir,
            matrix,
            base_path,
            source_config,
            split_paths,
            run_fingerprint,
            fingerprint_payload,
        )
    else:
        run_dir = Path(resume_run).resolve()
        _validate_resume_run(run_dir, run_fingerprint)
        logger.info("READS kind=alimu_resume_run path={}", run_dir)

    effective_configs = {
        spec.trial_id: _prepare_effective(
            matrix, base, spec, project_root, smoke, run_dir
        )
        for spec in specs
    }
    effective_paths = _finalize_effective_configs(
        run_dir, specs, effective_configs, creating=creating
    )
    _write_pointer(output["active_run_file"], run_dir, "active_alimu_matrix_run")

    analytic_rows, baseline_provenance = _run_baselines(
        run_dir, frames, matrix, run_fingerprint
    )
    neural_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    trial_artifacts: dict[str, dict[str, Any]] = {}
    for spec in specs:
        effective = effective_configs[spec.trial_id]
        effective_path = effective_paths[spec.trial_id]
        fingerprints = {
            "run_fingerprint": run_fingerprint,
            "effective_config_sha256": _json_sha256(effective),
            "effective_config_path": str(effective_path.resolve()),
            "method": spec.method,
            "seed": spec.seed,
        }
        row, artifacts, skipped = _run_neural_trial(
            run_dir,
            spec,
            frames["train"],
            frames["test"],
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
                "method": spec.method,
                "seed": spec.seed,
                "holdout_sensor": spec.holdout_sensor,
                "status": "complete",
                "skipped": bool(skipped),
                "trial_json": artifacts["trial_json"]["path"],
            }
        )
        _write_ledger(run_dir / "trial_ledger.csv", ledger_rows)

    if len(ledger_rows) != 15:
        raise RuntimeError("ALiMU trial ledger must contain exactly 15 rows")
    neural = pd.DataFrame(neural_rows)
    analytic = pd.DataFrame(analytic_rows)
    trials = pd.concat([neural, analytic], ignore_index=True)
    if len(trials) != 19:
        raise RuntimeError("ALiMU trials must contain exactly 15 neural and 4 analytic rows")
    overall = aggregate_alimu_trials(neural, analytic)
    if not creating:
        _audit_existing_final_artifacts(run_dir, run_fingerprint)
    aggregation = write_alimu_matrix_aggregation(trials, overall, run_dir)

    run_id = run_dir.name
    provenance: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "git_commit": git_commit(project_root),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "smoke": bool(smoke),
        "run_fingerprint": run_fingerprint,
        "config_snapshots": {
            "matrix": _hashed_artifact(run_dir / "configs/matrix.yaml"),
            "base": _hashed_artifact(run_dir / "configs/base.yaml"),
        },
        "split_manifest": _hashed_artifact(run_dir / "manifests/manifest.json"),
        "inputs": {
            split: _hashed_source(split_paths[split])
            for split in ("train", "val", "test")
        },
        "split_provenance": {
            "mode": "leave_sensor_out",
            "holdout_sensor": ALIMU_HOLDOUT_SENSOR,
            "train_sensors": sorted(ALIMU_TRAIN_SENSORS),
            "validation_sensors": sorted(ALIMU_TRAIN_SENSORS),
            "test_sensors": [ALIMU_HOLDOUT_SENSOR],
            "target_sensor": "VN100",
            "sample_rate_hz": 40,
            "train_sequences": _sequence_ids(frames["train"]),
            "validation_sequences": _sequence_ids(frames["val"]),
            "test_sequences": _sequence_ids(frames["test"]),
        },
        "baseline": baseline_provenance,
        "trials": trial_artifacts,
        "artifacts": {
            key: {
                **_hashed_artifact(group["csv"]),
                "run_id": run_id,
            }
            for key, group in aggregation.items()
        },
    }
    report_path = write_alimu_matrix_report(
        overall, trials, provenance, run_dir / "report.md"
    )
    provenance["artifacts"]["report"] = {
        **_hashed_artifact(report_path),
        "run_id": run_id,
    }
    _write_json(run_dir / "provenance.json", provenance, "alimu_matrix_provenance")
    _publish_latest(run_dir, output, report_path)
    return run_dir


def _split_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    split = config["split"]
    return {
        name: _resolve_path(split[f"{name}_path"], root)
        for name in ("train", "val", "test")
    } | {"manifest": _resolve_path(split["manifest_path"], root)}


def _load_split_frames(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        logger.info("READS kind=alimu_strict_parquet split={} path={}", split, paths[split])
        frames[split] = pd.read_parquet(paths[split])
    return frames


def _prepare_effective(
    matrix: dict[str, Any],
    base: dict[str, Any],
    spec: ALiMUTrialSpec,
    project_root: Path,
    smoke: bool,
    run_dir: Path,
) -> dict[str, Any]:
    effective = build_alimu_effective_trial_config(
        matrix, base, spec, project_root=project_root
    )
    if smoke:
        effective["training"]["epochs"] = 1
        effective["training"]["early_stopping_patience"] = 1
        effective["training"]["device"] = "cpu"
        effective["model"]["hidden_channels"] = 8
        effective["model"]["levels"] = 1
        effective["data"]["window_size"] = 16
        effective["data"]["stride"] = min(
            int(effective["data"].get("stride", 16)), 16
        )
        static = effective["data"].get("static_context")
        if isinstance(static, dict) and static.get("enabled"):
            static["window_size"] = 16
    effective.setdefault("output", {})["runs_dir"] = str(
        (run_dir / "trials" / spec.trial_id / "training").resolve()
    )
    effective["matrix_method"] = spec.method
    effective["matrix_holdout_sensor"] = spec.holdout_sensor
    effective["seed"] = spec.seed
    return effective


def _finalize_effective_configs(
    run_dir: Path,
    specs: tuple[ALiMUTrialSpec, ...],
    configs: Mapping[str, dict[str, Any]],
    *,
    creating: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for spec in specs:
        path = run_dir / "configs/effective" / f"{spec.trial_id}.yaml"
        paths[spec.trial_id] = path
        effective = configs[spec.trial_id]
        if creating:
            save_config(effective, path)
            logger.info("WRITES kind=alimu_effective_config path={}", path)
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Resume effective config is missing: {path}")
        logger.info("READS kind=alimu_effective_config path={}", path)
        if _json_sha256(load_config(path)) != _json_sha256(effective):
            raise ValueError(f"Resume effective config hash mismatch: {spec.trial_id}")
    return paths


def _run_baselines(
    run_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    matrix: Mapping[str, Any],
    run_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_dir = run_dir / "baselines"
    metadata_path = baseline_dir / "baseline.json"
    expected_fingerprint = {
        "run_fingerprint": run_fingerprint,
        "fit_split": "train",
        "fit_sensors": sorted(ALIMU_TRAIN_SENSORS),
        "heldout_sensor": ALIMU_HOLDOUT_SENSOR,
        "fir": deepcopy(matrix["fir"]),
    }
    if metadata_path.is_file():
        metadata = _read_json(metadata_path, "completed_alimu_baseline")
        rows, provenance = _audit_completed_baseline(
            baseline_dir, metadata, expected_fingerprint, frames["test"]
        )
        logger.info(
            "SKIPS kind=alimu_analytic_fit reason=matching_complete_cache path={}",
            metadata_path,
        )
        return rows, provenance

    train = frames["train"]
    test = frames["test"]
    logger.info(
        "READS kind=alimu_baseline_fit split=train sensors={} rows={}",
        sorted(set(train["low_sensor"].astype(str))),
        len(train),
    )
    state = fit_penn_baselines(
        train,
        fir_taps=int(matrix["fir"]["taps"]),
        alpha=float(matrix["fir"]["ridge_alpha"]),
    )
    state_path = baseline_dir / "state.json"
    _write_json(state_path, state, "alimu_train_only_baseline_state")
    predictions = predict_penn_baselines(state, test)
    rows: list[dict[str, Any]] = []
    method_artifacts: dict[str, Any] = {}
    for method in ALIMU_ANALYTIC_METHODS:
        prediction = _prediction_frame(test, predictions[method])
        _audit_prediction(prediction, test)
        prediction_path = write_parquet(
            prediction, baseline_dir / "predictions" / f"{method}.parquet"
        )
        logger.info("WRITES kind=alimu_baseline_prediction path={}", prediction_path)
        metrics = evaluate_penn_method(prediction)
        metrics_path = baseline_dir / "metrics" / f"{method}.json"
        _write_json(metrics_path, metrics, "alimu_baseline_metrics")
        rows.append(alimu_trial_row(ALIMU_HOLDOUT_SENSOR, method, None, metrics))
        method_artifacts[method] = {
            "prediction": _hashed_artifact(prediction_path),
            "metrics": _hashed_artifact(metrics_path),
        }
    metadata = {
        "status": "complete",
        "fingerprint": expected_fingerprint,
        "fit_split": "train",
        "fit_sensors": sorted(ALIMU_TRAIN_SENSORS),
        "fit_sequences": _sequence_ids(train),
        "heldout_sensor": ALIMU_HOLDOUT_SENSOR,
        "state_hash": _sha256(state_path),
        "prediction_hashes": {
            method: method_artifacts[method]["prediction"]["sha256"]
            for method in ALIMU_ANALYTIC_METHODS
        },
        "artifacts": {
            "state": _hashed_artifact(state_path),
            "methods": method_artifacts,
        },
    }
    _write_json(metadata_path, metadata, "completed_alimu_baseline")
    return rows, _baseline_provenance(metadata_path, metadata)


def _audit_completed_baseline(
    baseline_dir: Path,
    metadata: Mapping[str, Any],
    expected_fingerprint: Mapping[str, Any],
    test: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if metadata.get("status") != "complete":
        raise ValueError("Existing ALiMU baseline cache is not complete")
    if metadata.get("fingerprint") != expected_fingerprint:
        raise ValueError("Completed ALiMU baseline fingerprint mismatch")
    if metadata.get("fit_split") != "train":
        raise ValueError("Completed ALiMU baseline fit_split must be train")
    if set(metadata.get("fit_sensors", [])) != ALIMU_TRAIN_SENSORS:
        raise ValueError("Completed ALiMU baseline fit_sensors mismatch")
    if metadata.get("heldout_sensor") != ALIMU_HOLDOUT_SENSOR:
        raise ValueError("Completed ALiMU baseline heldout_sensor mismatch")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Completed ALiMU baseline artifacts are missing")
    _audit_hashed_artifact(artifacts.get("state"), baseline_dir / "state.json")
    if metadata.get("state_hash") != _sha256(baseline_dir / "state.json"):
        raise ValueError("Completed ALiMU baseline state_hash mismatch")
    methods = artifacts.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("Completed ALiMU baseline method artifacts are missing")
    rows: list[dict[str, Any]] = []
    for method in ALIMU_ANALYTIC_METHODS:
        recorded = methods.get(method)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"Completed ALiMU baseline artifact is missing: {method}")
        prediction_path = baseline_dir / "predictions" / f"{method}.parquet"
        metrics_path = baseline_dir / "metrics" / f"{method}.json"
        _audit_hashed_artifact(recorded.get("prediction"), prediction_path)
        _audit_hashed_artifact(recorded.get("metrics"), metrics_path)
        prediction_hashes = metadata.get("prediction_hashes")
        if not isinstance(prediction_hashes, Mapping) or prediction_hashes.get(method) != _sha256(prediction_path):
            raise ValueError("Completed ALiMU baseline prediction_hashes mismatch")
        logger.info("READS kind=completed_alimu_baseline_prediction path={}", prediction_path)
        prediction = pd.read_parquet(prediction_path)
        _audit_prediction(prediction, test)
        metrics = _read_json(metrics_path, "completed_alimu_baseline_metrics")
        rows.append(alimu_trial_row(ALIMU_HOLDOUT_SENSOR, method, None, metrics))
    return rows, _baseline_provenance(baseline_dir / "baseline.json", metadata)


def _baseline_provenance(
    metadata_path: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "fit_split": metadata["fit_split"],
        "fit_sensors": list(metadata["fit_sensors"]),
        "fit_sequences": list(metadata.get("fit_sequences", [])),
        "heldout_sensor": metadata["heldout_sensor"],
        "state_hash": metadata["state_hash"],
        "prediction_hashes": dict(metadata["prediction_hashes"]),
        "metadata": str(metadata_path.resolve()),
    }


def _run_neural_trial(
    run_dir: Path,
    spec: ALiMUTrialSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    effective: dict[str, Any],
    effective_path: Path,
    fingerprints: dict[str, Any],
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    trial_dir = run_dir / "trials" / spec.trial_id
    trial_json = trial_dir / "trial.json"
    if trial_json.is_file():
        payload = _read_json(trial_json, "completed_alimu_trial")
        if payload.get("status") == "complete":
            if payload.get("fingerprints") != fingerprints:
                raise ValueError(f"Completed trial fingerprint mismatch: {spec.trial_id}")
            row, artifacts = _audit_completed_trial(
                payload, spec, train, test, trial_dir, effective_path
            )
            logger.info(
                "SKIPS kind=alimu_neural_training trial={} reason=matching_complete_cache",
                spec.trial_id,
            )
            return row, artifacts, True

    trial_dir.mkdir(parents=True, exist_ok=True)
    logger.info("READS kind=alimu_effective_trial_config path={}", effective_path)
    logger.info(
        "WRITES kind=alimu_training_artifacts trial={} root={}",
        spec.trial_id,
        effective["output"]["runs_dir"],
    )
    training_run = Path(train_model(effective, config_path=effective_path)).resolve()
    checkpoint_path = training_run / "best.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Neural trial checkpoint is missing: {checkpoint_path}")
    logger.info("READS kind=alimu_best_checkpoint path={}", checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    _audit_checkpoint(checkpoint, spec, train)
    generator = VirtualIMUGenerator(checkpoint_path, device="cpu" if smoke else "auto")
    logger.info("READS kind=alimu_test_inference rows={}", len(test))
    prediction = generator.run_frame(test)
    _audit_prediction(prediction, test)
    prediction_path = write_parquet(prediction, trial_dir / "prediction.parquet")
    logger.info("WRITES kind=alimu_neural_prediction path={}", prediction_path)
    metrics = evaluate_penn_method(prediction)
    metrics_path = trial_dir / "metrics.json"
    _write_json(metrics_path, metrics, "alimu_neural_metrics")
    artifacts = {
        "checkpoint": _hashed_artifact(checkpoint_path),
        "prediction": _hashed_artifact(prediction_path),
        "metrics": _hashed_artifact(metrics_path),
        "effective_config": _hashed_artifact(effective_path),
    }
    payload = {
        "status": "complete",
        "trial_id": spec.trial_id,
        "method": spec.method,
        "seed": spec.seed,
        "holdout_sensor": spec.holdout_sensor,
        "fingerprints": fingerprints,
        "training_run": str(training_run),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "checkpoint_metric": str(checkpoint.get("checkpoint_metric", "unknown")),
        "best_metric": float(checkpoint.get("best_metric", float("nan"))),
        "static_window_metadata": checkpoint.get("static_window_metadata", {}),
        "competitor_baseline": checkpoint.get("competitor_baseline", {}),
        "artifacts": artifacts,
    }
    _write_json(trial_json, payload, "completed_alimu_trial")
    artifacts = {"trial_json": _hashed_artifact(trial_json), **artifacts}
    return (
        alimu_trial_row(ALIMU_HOLDOUT_SENSOR, spec.method, spec.seed, metrics),
        artifacts,
        False,
    )


def _audit_completed_trial(
    payload: Mapping[str, Any],
    spec: ALiMUTrialSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    trial_dir: Path,
    effective_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"Completed trial artifacts are missing: {spec.trial_id}")
    training_run = Path(str(payload.get("training_run", ""))).resolve()
    training_root = (trial_dir / "training").resolve()
    try:
        training_run.relative_to(training_root)
    except ValueError as exc:
        raise ValueError(
            f"Completed trial training run must stay under its trial root: {spec.trial_id}"
        ) from exc
    expected_paths = {
        "checkpoint": training_run / "best.ckpt",
        "prediction": trial_dir / "prediction.parquet",
        "metrics": trial_dir / "metrics.json",
        "effective_config": effective_path,
    }
    for name, expected_path in expected_paths.items():
        recorded = artifacts.get(name)
        if not isinstance(recorded, Mapping):
            raise ValueError(f"Completed trial {name} artifact is missing: {spec.trial_id}")
        try:
            _audit_hashed_artifact(recorded, expected_path)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f"Completed trial {name} {exc}: {spec.trial_id}") from exc
    prediction_path = expected_paths["prediction"]
    checkpoint_path = expected_paths["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    _audit_checkpoint(checkpoint, spec, train)
    logger.info("READS kind=completed_alimu_prediction path={}", prediction_path)
    prediction = pd.read_parquet(prediction_path)
    _audit_prediction(prediction, test)
    metrics_path = expected_paths["metrics"]
    metrics = _read_json(metrics_path, "completed_alimu_trial_metrics")
    result_artifacts = {
        str(name): dict(value)
        for name, value in artifacts.items()
        if isinstance(value, Mapping)
    }
    trial_path = trial_dir / "trial.json"
    result_artifacts["trial_json"] = _hashed_artifact(trial_path)
    return (
        alimu_trial_row(ALIMU_HOLDOUT_SENSOR, spec.method, spec.seed, metrics),
        result_artifacts,
    )


def _audit_prediction(prediction: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(prediction) != len(test):
        raise ValueError(
            f"Prediction row count must equal the MPU9250 test row count {len(test)}"
        )
    required_keys = ("timestamp_s", "sequence_id", "low_sensor")
    missing_keys = [column for column in required_keys if column not in prediction]
    if missing_keys:
        raise ValueError(f"Prediction is missing test identity columns: {missing_keys}")
    sensors = set(prediction["low_sensor"].astype(str))
    if sensors != {ALIMU_HOLDOUT_SENSOR}:
        raise ValueError(
            "Prediction low_sensor must contain only MPU9250; "
            f"got {sorted(sensors)}"
        )
    expected_sequences = test["sequence_id"].astype(str).tolist()
    actual_sequences = prediction["sequence_id"].astype(str).tolist()
    expected_sensors = test["low_sensor"].astype(str).tolist()
    actual_sensors = prediction["low_sensor"].astype(str).tolist()
    expected_time = test["timestamp_s"].to_numpy(dtype=float)
    actual_time = prediction["timestamp_s"].to_numpy(dtype=float)
    if (
        actual_sequences != expected_sequences
        or actual_sensors != expected_sensors
        or not np.array_equal(actual_time, expected_time)
    ):
        raise ValueError("Prediction row order and test key identity must be exact")
    missing_virtual = [column for column in VIRTUAL_COLUMNS if column not in prediction]
    if missing_virtual:
        raise ValueError(f"Prediction is missing six virtual columns: {missing_virtual}")
    values = prediction[VIRTUAL_COLUMNS].to_numpy(dtype=float)
    if values.shape != (len(test), 6) or not np.all(np.isfinite(values)):
        raise ValueError("Prediction must contain six finite virtual values per test row")


def _audit_checkpoint(
    checkpoint: Mapping[str, Any], spec: ALiMUTrialSpec, train: pd.DataFrame
) -> None:
    drift_methods = {"drift_only", "bias_drift", "fir_bias_drift_residual"}
    if spec.method in drift_methods:
        static_context = checkpoint.get("static_context")
        if (
            not isinstance(static_context, Mapping)
            or static_context.get("reference_scope") != "train_global"
        ):
            raise ValueError("Drift checkpoint static context must use train_global")
        static_windows = checkpoint.get("static_windows")
        if (
            not isinstance(static_windows, Mapping)
            or set(static_windows) != {TRAIN_GLOBAL_STATIC_CONTEXT_KEY}
        ):
            raise ValueError(
                "Drift checkpoint must store one static window at __train_global__"
            )
        metadata_all = checkpoint.get("static_window_metadata")
        metadata = (
            metadata_all.get(TRAIN_GLOBAL_STATIC_CONTEXT_KEY)
            if isinstance(metadata_all, Mapping)
            else None
        )
        if not isinstance(metadata, Mapping):
            raise ValueError("Drift checkpoint is missing train_global static metadata")
        required_metadata = {
            "reference_scope", "source_sensor", "sequence_id", "start_row",
            "start_timestamp_s", "end_timestamp_s", "score",
        }
        if any(key not in metadata for key in required_metadata):
            raise ValueError("Drift checkpoint static metadata is incomplete")
        if metadata.get("reference_scope", metadata.get("scope")) != "train_global":
            raise ValueError("Drift checkpoint static metadata must say train_global")
        source_sensor = str(
            metadata.get("source_sensor", metadata.get("low_sensor", ""))
        )
        if source_sensor not in ALIMU_TRAIN_SENSORS:
            raise ValueError(
                "Drift checkpoint static metadata source must be an ALiMU training sensor"
            )
        source_sequence = str(metadata.get("sequence_id", ""))
        train_source = train.loc[
            train["sequence_id"].astype(str).eq(source_sequence)
            & train["low_sensor"].astype(str).eq(source_sensor)
        ]
        if train_source.empty:
            raise ValueError(
                "Drift checkpoint static source sequence must belong only to train"
            )
        window = np.asarray(static_windows[TRAIN_GLOBAL_STATIC_CONTEXT_KEY], dtype=float)
        expected_size = int(static_context.get("window_size", 0))
        columns = list(static_context.get("columns", [
            "gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low", "temperature_c",
        ]))
        if window.shape != (expected_size, len(columns)) or not np.isfinite(window).all():
            raise ValueError("Drift checkpoint static window must be finite with the configured shape")
        feature_stats = checkpoint.get("feature_stats")
        if not isinstance(feature_stats, Mapping):
            raise ValueError("Drift checkpoint feature stats are missing for static window audit")
        expected_metadata: dict[str, dict] = {}
        expected_windows = build_static_context_windows(
            train, dict(static_context), dict(feature_stats), metadata_out=expected_metadata
        )
        expected_window = expected_windows[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]
        if not np.allclose(window, expected_window, rtol=1e-6, atol=1e-6):
            raise ValueError("Drift checkpoint static window does not match the train-derived context")

    if spec.method != "fir_bias_drift_residual":
        return
    if checkpoint.get("target_mode") != "competitor_residual":
        raise ValueError("Hybrid checkpoint must target competitor_residual")
    competitor = checkpoint.get("competitor_baseline")
    if not isinstance(competitor, Mapping) or competitor.get("enabled") is not True:
        raise ValueError("Hybrid checkpoint competitor baseline must be enabled")
    if competitor.get("mode") != "fir":
        raise ValueError("Hybrid checkpoint competitor baseline must use mode=fir")
    state = competitor.get("state")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Hybrid checkpoint is missing serialized train-fitted FIR state")
    checkpoint_config = checkpoint.get("config")
    training = (
        checkpoint_config.get("training")
        if isinstance(checkpoint_config, Mapping)
        else None
    )
    configured = (
        training.get("competitor_baseline") if isinstance(training, Mapping) else None
    )
    if not isinstance(configured, Mapping) or configured.get("mode") != "fir":
        raise ValueError("Hybrid effective config must use a train-fitted mode=fir state")
    if configured.get("fir_taps") != 9 or float(configured.get("ridge_alpha", -1.0)) != 0.01:
        raise ValueError("Hybrid checkpoint must use canonical FIR taps=9 and ridge alpha=0.01")
    expected_state = fit_system_id_baseline_state(
        train,
        fir_taps=9,
        alpha=0.01,
    )
    if _json_sha256(state) != _json_sha256(expected_state):
        raise ValueError("Hybrid FIR state must be fitted from the ALiMU train split")


def _run_fingerprint_payload(
    matrix: dict[str, Any],
    base: dict[str, Any],
    base_path: Path,
    source_config: Path | None,
    split_paths: Mapping[str, Path],
    smoke: bool,
) -> dict[str, Any]:
    source_files: dict[str, Any] = {
        "base_config": _text_source_fingerprint(base_path),
        "split_manifest": _text_source_fingerprint(split_paths["manifest"]),
        "parquet": {
            split: _hashed_source(split_paths[split])
            for split in ("train", "val", "test")
        },
    }
    if source_config is not None:
        source_files["matrix_config"] = _text_source_fingerprint(source_config)
    return {
        "canonical_matrix_config": matrix,
        "fully_effective_base_config": base,
        "smoke": bool(smoke),
        "source_files": source_files,
    }


def _snapshot_inputs(
    run_dir: Path,
    matrix: dict[str, Any],
    base_path: Path,
    source_config: Path | None,
    split_paths: Mapping[str, Path],
    run_fingerprint: str,
    fingerprint_payload: dict[str, Any],
) -> None:
    save_config(matrix, run_dir / "configs/matrix.yaml")
    logger.info("WRITES kind=alimu_matrix_config_snapshot path={}", run_dir / "configs/matrix.yaml")
    _copy(base_path, run_dir / "configs/base.yaml", "alimu_base_config_snapshot")
    if source_config is not None:
        _copy(
            source_config,
            run_dir / "configs/matrix_source.yaml",
            "alimu_matrix_source_snapshot",
        )
    _copy(
        split_paths["manifest"],
        run_dir / "manifests/manifest.json",
        "alimu_split_manifest_snapshot",
    )
    _write_json(
        run_dir / "hashes.json",
        {
            "run_fingerprint": run_fingerprint,
            "run_fingerprint_payload": fingerprint_payload,
            "snapshot_hashes": {
                "matrix": _sha256(run_dir / "configs/matrix.yaml"),
                "base": _sha256(run_dir / "configs/base.yaml"),
                "manifest": _sha256(run_dir / "manifests/manifest.json"),
            },
        },
        "alimu_run_input_hashes",
    )


def _validate_resume_run(run_dir: Path, run_fingerprint: str) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume run does not exist: {run_dir}")
    hashes = _read_json(run_dir / "hashes.json", "alimu_resume_input_hashes")
    if hashes.get("run_fingerprint") != run_fingerprint:
        raise ValueError("Resume run fingerprint does not match current inputs")
    snapshot_hashes = hashes.get("snapshot_hashes")
    if not isinstance(snapshot_hashes, Mapping):
        raise ValueError("Resume run snapshot hashes are missing")
    snapshot_paths = {
        "matrix": run_dir / "configs/matrix.yaml",
        "base": run_dir / "configs/base.yaml",
        "manifest": run_dir / "manifests/manifest.json",
    }
    for name, path in snapshot_paths.items():
        if not path.is_file() or snapshot_hashes.get(name) != _sha256(path):
            raise ValueError(f"Resume run {name} snapshot hash mismatch")


def _audit_existing_final_artifacts(run_dir: Path, run_fingerprint: str) -> None:
    """Reject completed aggregate/report tampering before a resume may overwrite it."""
    provenance_path = run_dir / "provenance.json"
    if not provenance_path.exists():
        return
    provenance = _read_json(provenance_path, "completed_alimu_provenance")
    if provenance.get("run_fingerprint") != run_fingerprint:
        raise ValueError("provenance run fingerprint mismatch")
    artifacts = provenance.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("provenance artifacts are missing")
    for name, path in {
        "trials": run_dir / "trials.csv",
        "overall_summary": run_dir / "overall_summary.csv",
        "report": run_dir / "report.md",
    }.items():
        recorded = artifacts.get(name)
        if not isinstance(recorded, Mapping) or recorded.get("sha256") != _sha256(path):
            label = "aggregate" if name != "report" else "report"
            raise ValueError(f"{label} artifact hash mismatch")


def _publish_latest(
    run_dir: Path, output: Mapping[str, Path], report_path: Path
) -> None:
    trusted_output_root = output["latest_run_file"].parent.resolve()
    latest_dir = output["latest_dir"].resolve()
    try:
        latest_dir.relative_to(trusted_output_root)
    except ValueError as error:
        raise ValueError(
            "output.latest_dir must resolve under the trusted ALiMU output root: "
            f"{trusted_output_root}"
        ) from error
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True)
    for source in (
        run_dir / "trials.csv",
        run_dir / "trials.json",
        run_dir / "trials.md",
        run_dir / "overall_summary.csv",
        run_dir / "overall_summary.json",
        run_dir / "overall_summary.md",
        run_dir / "provenance.json",
        report_path,
    ):
        destination = latest_dir / source.name
        shutil.copy2(source, destination)
        logger.info("WRITES kind=latest_alimu_matrix_artifact path={}", destination)
    logger.info(
        "SKIPS kind=latest_alimu_large_artifacts reason=checkpoints_and_predictions_stay_in_run"
    )
    _write_pointer(output["latest_run_file"], run_dir, "latest_alimu_matrix_run")


def _output_paths(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    output = {
        key: _resolve_path(config["output"][key], root)
        for key in ("runs_dir", "active_run_file", "latest_dir", "latest_run_file")
    }
    expected_names = {
        "active_run_file": "active_alimu_matrix_run.txt",
        "latest_dir": "latest_alimu_matrix",
        "latest_run_file": "latest_alimu_matrix_run.txt",
    }
    for key, expected in expected_names.items():
        if output[key].name != expected:
            configured = output[key]
            output[key] = configured.with_name(expected)
            logger.info(
                "SKIPS kind=configured_non_alimu_pointer key={} path={} "
                "reason=protect_generic_penn_artifact selected={}",
                key,
                configured,
                output[key],
            )
    return output


def _create_run_dir(runs_dir: Path, experiment: str, smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = "_smoke" if smoke else ""
    run_dir = (runs_dir / f"{experiment}_{stamp}{suffix}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("WRITES kind=alimu_matrix_run_root path={}", run_dir)
    return run_dir


def _sequence_ids(frame: pd.DataFrame) -> list[str]:
    return [str(value) for value in frame["sequence_id"].drop_duplicates().tolist()]


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("WRITES kind=alimu_trial_ledger path={} rows={}", path, len(rows))


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
        json.dumps(payload, indent=2, allow_nan=True, default=str), encoding="utf-8"
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


def _audit_hashed_artifact(recorded: Any, expected_path: Path) -> None:
    if not isinstance(recorded, Mapping):
        raise ValueError("Completed artifact metadata is missing")
    if recorded.get("path") != str(expected_path.resolve()):
        raise ValueError("Completed artifact path mismatch")
    if not expected_path.is_file():
        raise FileNotFoundError(f"Completed artifact is missing: {expected_path}")
    if recorded.get("sha256") != _sha256(expected_path):
        raise ValueError("Completed artifact hash mismatch")


def _hashed_artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _hashed_source(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    logger.info("READS kind=alimu_fingerprint_source path={}", resolved)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _text_source_fingerprint(path: Path) -> dict[str, str]:
    result = _hashed_source(path)
    result["content"] = path.read_text(encoding="utf-8")
    return result


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
