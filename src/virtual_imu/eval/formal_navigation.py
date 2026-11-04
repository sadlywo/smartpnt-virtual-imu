"""Frozen-checkpoint evaluation of SmartPNT local ENU endpoint displacement.

The 80-point, 40 Hz windows span 1.975 seconds between endpoints. Their
overlapping displacements are scored independently, never accumulated into a
trajectory or interpreted as ATE/RPE or SO(3) preintegration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from virtual_imu.eval.formal_evaluation import json_safe, verify_run_provenance, write_json
from virtual_imu.io.unified_dataset import file_sha256
from virtual_imu.models import make_model
from virtual_imu.preprocessing.normalization import apply_standardization, compute_standardization
from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SMARTPNT_GT_POSITION_COLUMNS,
    SMARTPNT_GT_POSITION_DELTA_COLUMNS,
    SmartPNTGTWindowDataset,
)
from virtual_imu.training.unified_data import (
    snapshot_unified_training_inputs,
    unified_navigation_frame,
)


def navigation_error_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    """Physical displacement errors, with vector and component RMSE kept distinct."""
    prediction = np.asarray(prediction, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Displacement prediction/reference must be matching N x 3 arrays")
    if not len(prediction) or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Displacement metrics require nonempty finite windows")
    error = prediction - reference
    norms = np.linalg.norm(error, axis=1)
    return {
        "windows": len(error),
        "vector_rmse_m": float(np.sqrt(np.mean(norms**2))),
        "component_rmse_m": float(np.sqrt(np.mean(error**2))),
        "axis_rmse_m": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "axis_mae_m": np.mean(np.abs(error), axis=0).tolist(),
        "axis_bias_m": np.mean(error, axis=0).tolist(),
        "horizontal_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
        "vertical_rmse_m": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "error_norm_median_m": float(np.median(norms)),
        "error_norm_p95_m": float(np.quantile(norms, 0.95)),
    }


def _window_summaries(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 3 or features.shape[2] != 6 or features.shape[1] < 2:
        raise ValueError("Baseline inputs must be N x T x 6 low six-axis IMU windows")
    if not len(features) or not np.isfinite(features).all():
        raise ValueError("Baseline inputs require nonempty finite low IMU windows")
    return np.concatenate([
        features.mean(axis=1), features.std(axis=1), features[:, -1] - features[:, 0],
    ], axis=1)


def fit_navigation_baselines(
    train_features: np.ndarray, train_targets: np.ndarray, *, alpha: float = 0.01,
) -> dict:
    """Fit constants and ridge on train windows only, using 18 low-IMU summaries.

    Features are physical N x T x 6 arrays. The summary standardization, constant
    target mean, and ridge coefficients all come exclusively from these inputs.
    The ridge intercept is unpenalized; alpha is fixed before held-out evaluation.
    """
    summary = _window_summaries(train_features)
    targets = np.asarray(train_targets, dtype=float)
    if targets.shape != (len(summary), 3) or not np.isfinite(targets).all():
        raise ValueError("Baseline train targets must be finite N x 3 ENU displacement")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    mean, std = summary.mean(axis=0), summary.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    design = np.column_stack([(summary - mean) / std, np.ones(len(summary))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {
        "fit_split": "train", "fit_windows": len(summary), "alpha": float(alpha),
        "feature_columns": list(SMARTPNT_GT_IMU_COLUMNS),
        "summary_columns": [f"{operation}_{column}" for operation in ("mean", "std", "end_minus_start")
                            for column in SMARTPNT_GT_IMU_COLUMNS],
        "target_columns": list(SMARTPNT_GT_POSITION_DELTA_COLUMNS),
        "summary_mean": mean.tolist(), "summary_std": std.tolist(),
        "train_mean_m": targets.mean(axis=0).tolist(),
        "ridge_coefficients": coefficients.tolist(),
        "intercept_penalized": False,
    }


def predict_navigation_baselines(state: dict, features: np.ndarray) -> dict[str, np.ndarray]:
    """Predict from low IMU only: this API accepts no held-out labels or states."""
    summary = _window_summaries(features)
    mean, std = np.asarray(state["summary_mean"]), np.asarray(state["summary_std"])
    coefficients = np.asarray(state["ridge_coefficients"])
    target_mean = np.asarray(state["train_mean_m"])
    if (mean.shape != (18,) or std.shape != (18,) or coefficients.shape != (19, 3)
            or target_mean.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or not np.isfinite(coefficients).all() or not np.isfinite(target_mean).all()):
        raise ValueError("Invalid train-fitted navigation baseline state")
    design = np.column_stack([(summary - mean) / std, np.ones(len(summary))])
    return {
        "zero": np.zeros((len(summary), 3)),
        "train_mean": np.broadcast_to(target_mean, (len(summary), 3)).copy(),
        "ridge": design @ coefficients,
    }


def _validate_stats(stats: dict | None, columns: list[str], enabled: bool, name: str) -> None:
    if not enabled:
        if stats is not None:
            raise ValueError(f"{name} exists although normalization is disabled")
        return
    if not isinstance(stats, dict) or stats.get("columns") != columns:
        raise ValueError(f"{name} must bind the checkpoint's ordered columns")
    mean = np.asarray(stats.get("mean"), dtype=float)
    std = np.asarray(stats.get("std"), dtype=float)
    if (mean.shape != (len(columns),) or std.shape != (len(columns),)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError(f"{name} must contain finite means and positive standard deviations")


def _validate_checkpoint(config: dict, checkpoint: dict) -> None:
    data, model, training = config["data"], config["model"], config.get("training", {})
    if data.get("label_mode") != "position_delta" or checkpoint.get("label_mode") != "position_delta":
        raise ValueError("Formal navigation evaluation supports only position_delta")
    if checkpoint.get("target_layout") != "window" or model.get("target_layout", "window") != "window":
        raise ValueError("Formal navigation evaluation requires target_layout=window")
    head = model.get("prediction_head", training.get("prediction_head", "last"))
    if checkpoint.get("prediction_head") != "last" or head != "last":
        raise ValueError("Formal navigation evaluation requires prediction_head=last")
    if checkpoint.get("target_columns") != SMARTPNT_GT_POSITION_DELTA_COLUMNS:
        raise ValueError("Checkpoint target_columns must be ordered local ENU displacement")
    if (checkpoint.get("feature_columns") != SMARTPNT_GT_IMU_COLUMNS
            or data.get("input_columns", SMARTPNT_GT_IMU_COLUMNS) != SMARTPNT_GT_IMU_COLUMNS):
        raise ValueError("Checkpoint feature_columns must contain only the low six-axis IMU")
    expected_model = {**model, "in_channels": 6, "out_channels": 3}
    if (checkpoint.get("model_config") != expected_model
            or int(model.get("in_channels", 6)) != 6 or int(model.get("out_channels", 3)) != 3):
        raise ValueError("Checkpoint model_config differs from the six-input, three-output run config")
    if (int(data.get("window_size", 0)) != 80 or int(data.get("stride", 0)) != 40
            or float(data.get("rate_hz", 0)) != 40.0):
        raise ValueError("Formal protocol requires 80-point windows, stride=40, and rate_hz=40")
    if int(data.get("batch_size", 128)) <= 0:
        raise ValueError("Evaluation batch_size must be positive")
    expected_splits = {"train": ["Traj3"], "val": ["Traj4"], "test": ["Traj1"]}
    for split, names in expected_splits.items():
        if data.get(f"{split}_sequences") != names:
            raise ValueError(f"Formal {split} split must contain original_sequence_id={names}")
    if checkpoint.get("split_sequences") != expected_splits:
        raise ValueError("Checkpoint split_sequences differ from the formal training splits")
    if checkpoint.get("checkpoint_metric") != training.get("checkpoint_metric", "val_rmse"):
        raise ValueError("Checkpoint selection metric differs from training config")
    if any(float(training.get(key, 0)) != 0 for key in (
        "velocity_integration_weight", "heading_integration_weight",
    )):
        raise ValueError("Formal position_delta evaluation requires zero auxiliary integration weights")
    if not data.get("dataset_manifest") or not data.get("split_manifest"):
        raise ValueError("Formal navigation evaluation requires frozen dataset and split manifests")
    _validate_stats(checkpoint.get("feature_stats"), SMARTPNT_GT_IMU_COLUMNS,
                    bool(data.get("normalize_features", True)), "feature_stats")
    _validate_stats(checkpoint.get("target_stats"), SMARTPNT_GT_POSITION_DELTA_COLUMNS,
                    bool(data.get("normalize_targets", True)), "target_stats")


def _validate_segments(frame: pd.DataFrame) -> None:
    required = ["sequence_id", "original_sequence_id", "timestamp_s",
                *SMARTPNT_GT_IMU_COLUMNS, *SMARTPNT_GT_POSITION_COLUMNS]
    missing = sorted(set(required) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Navigation evaluation requires nonempty measured segments; missing={missing}")
    if frame[["sequence_id", "original_sequence_id"]].isna().any().any():
        raise ValueError("Navigation segment identity cannot be missing")
    numeric = frame[["timestamp_s", *SMARTPNT_GT_IMU_COLUMNS, *SMARTPNT_GT_POSITION_COLUMNS]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Navigation windows require finite timestamps, low IMU, and position targets")
    if set(frame.original_sequence_id.astype(str)) != {"Traj1", "Traj3", "Traj4"}:
        raise ValueError("Frozen navigation frame must contain exactly the three formal original sequences")
    declared = {"Traj3": "train", "Traj4": "validation", "Traj1": "test"}
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        if not str(sequence_id) or group.original_sequence_id.nunique() != 1:
            raise ValueError("A contiguous sequence_id must belong to one original_sequence_id")
        for column in ("dataset_id", "low_sensor", "split"):
            if column in group and (group[column].isna().any() or group[column].nunique() != 1):
                raise ValueError(f"Mixed or missing {column} within navigation segment {sequence_id}")
        original = str(group.original_sequence_id.iloc[0])
        if "split" in group and str(group.split.iloc[0]) != declared[original]:
            raise ValueError("Segment split differs from its original-sequence assignment")
        timestamps = np.sort(group.timestamp_s.to_numpy(float))
        if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1 / 40, atol=1e-7, rtol=0):
            raise ValueError(f"Navigation segment has a gap, duplicate, or nonuniform timestamps: {sequence_id}")
        if len(group) < 80:
            logger.info("SKIPS navigation segment {}: {} samples cannot form an 80-point window", sequence_id, len(group))


def _physical_windows(dataset: SmartPNTGTWindowDataset) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Recover the same physical windows and identities as the checkpoint dataset."""
    groups = {str(name): group.sort_values("timestamp_s").reset_index(drop=True)
              for name, group in dataset.df.groupby("sequence_id", sort=False)}
    low = {name: group[SMARTPNT_GT_IMU_COLUMNS].to_numpy(np.float32) for name, group in groups.items()}
    positions = {name: group[SMARTPNT_GT_POSITION_COLUMNS].to_numpy(np.float32) for name, group in groups.items()}
    features, targets, rows = [], [], []
    for window in dataset.windows:
        name = window["sequence_id"]
        start, end = window["start_index"], window["end_index"]
        if (window["x"].shape != (80, 6) or window["y"].shape != (3,)
                or not np.isfinite(window["x"]).all() or not np.isfinite(window["y"]).all()
                or end - start != 79):
            raise ValueError("Invalid checkpoint dataset window shape or nonfinite values")
        group = groups[name]
        features.append(low[name][start:end + 1])
        targets.append(positions[name][end] - positions[name][start])
        rows.append({
            "sequence_id": name, "original_sequence_id": str(group.original_sequence_id.iloc[0]),
            "start_index": start, "end_index": end,
            "start_time_s": window["start_time_s"], "end_time_s": window["end_time_s"],
        })
    if not rows:
        raise ValueError("No finite contiguous navigation windows to evaluate")
    return np.stack(features), np.stack(targets), pd.DataFrame(rows)


def _verify_train_stats(dataset: SmartPNTGTWindowDataset, physical_targets: np.ndarray) -> None:
    if dataset.normalize_features:
        expected = compute_standardization(dataset.df, SMARTPNT_GT_IMU_COLUMNS)
        for key in ("mean", "std"):
            if not np.allclose(dataset.feature_stats[key], expected[key], atol=1e-8, rtol=1e-7):
                raise ValueError("Checkpoint feature_stats differ from frozen training-only statistics")
    if dataset.normalize_targets:
        expected_mean, expected_std = physical_targets.mean(axis=0), physical_targets.std(axis=0)
        expected_std = np.where(expected_std < 1e-8, 1.0, expected_std)
        for key, expected in (("mean", expected_mean), ("std", expected_std)):
            if not np.allclose(dataset.target_stats[key], expected, atol=1e-7, rtol=1e-7):
                raise ValueError("Checkpoint target_stats differ from frozen training-only statistics")


def _predict_checkpoint(model: torch.nn.Module, dataset: SmartPNTGTWindowDataset,
                        device: torch.device, batch_size: int) -> np.ndarray:
    batches = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    with torch.no_grad():
        for x, _, _ in loader:
            output = model(x.to(device))
            if not torch.is_tensor(output) or tuple(output.shape) != (len(x), 3, 80):
                raise ValueError("Checkpoint output must have shape batch x 3 x 80 before the last head")
            prediction = output[:, :, -1].detach().cpu().numpy()
            if dataset.normalize_targets:
                prediction = apply_standardization(prediction, dataset.target_stats, inverse=True)
            if not np.isfinite(prediction).all():
                raise ValueError("Checkpoint emitted nonfinite navigation displacement")
            batches.append(prediction)
    if not batches:
        raise ValueError("Checkpoint inference produced no navigation windows")
    return np.concatenate(batches)


def _score_windows(metadata: pd.DataFrame, predictions: dict[str, np.ndarray], targets: np.ndarray) -> dict:
    sequences = []
    for sequence_id, group in metadata.groupby("sequence_id", sort=False):
        indices = group.index.to_numpy()
        sequences.append({
            "sequence_id": str(sequence_id), "original_sequence_id": str(group.original_sequence_id.iloc[0]),
            "start_time_s": float(group.start_time_s.min()), "end_time_s": float(group.end_time_s.max()),
            "metrics": {name: navigation_error_metrics(values[indices], targets[indices])
                        for name, values in predictions.items()},
        })
    return {"windows": len(targets), "metrics": {
        name: navigation_error_metrics(values, targets) for name, values in predictions.items()
    }, "sequences": sequences}


def evaluate_navigation_run(run_dir: Path, *, device: str = "auto") -> dict:
    """Evaluate best.ckpt with its frozen config, splits and train-only statistics.

    All model and baseline inputs are the same six low-IMU channels. GT position
    endpoints are used for labels; GT velocity, attitude, body-frame transforms,
    and temperature are never model or baseline inputs. Validation/test labels
    cannot fit normalization, constants, ridge parameters, or alignment.
    """
    run_dir = Path(run_dir)
    evaluation_dir = run_dir / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError(f"Evaluation artifacts already exist: {evaluation_dir}")
    logger.info("READS navigation run config {} and selected checkpoint {}", run_dir / "config.yaml", run_dir / "best.ckpt")
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    logger.info("READS training provenance and manifest snapshots under {}", run_dir)
    verify_run_provenance(run_dir, config, checkpoint.get("config"))
    _validate_checkpoint(config, checkpoint)
    data = config["data"]
    frame = unified_navigation_frame(data)
    _validate_segments(frame)
    logger.info("SKIPS trajectory integration, ATE/RPE, SO(3), and GT velocity/attitude/temperature as prediction inputs")
    datasets, physical = {}, {}
    for split, original in (("train", "Traj3"), ("validation", "Traj4"), ("test", "Traj1")):
        dataset = SmartPNTGTWindowDataset(
            frame[frame.original_sequence_id.astype(str).eq(original)].reset_index(drop=True),
            window_size=80, stride=40, label_mode="position_delta",
            feature_stats=checkpoint.get("feature_stats"), target_stats=checkpoint.get("target_stats"),
            normalize_features=bool(data.get("normalize_features", True)),
            normalize_targets=bool(data.get("normalize_targets", True)),
        )
        datasets[split] = dataset
        physical[split] = _physical_windows(dataset)
    train_features, train_targets, train_meta = physical["train"]
    _verify_train_stats(datasets["train"], train_targets)
    baselines = fit_navigation_baselines(train_features, train_targets, alpha=0.01)
    baselines["original_sequence_ids"] = sorted(train_meta.original_sequence_id.unique().tolist())
    baselines["sequence_ids"] = sorted(train_meta.sequence_id.unique().tolist())
    resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model = make_model(checkpoint["model_config"]).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    report = {
        "training_run": run_dir.as_posix(), "selection": checkpoint["checkpoint_metric"],
        "best_epoch": checkpoint.get("best_epoch"), "best_metric": checkpoint.get("best_metric"),
        "seed": config.get("seed"), "axis_order": ["east", "north", "up"],
        "window_points": 80, "stride_points": 40, "sample_rate_hz": 40.0, "endpoint_span_s": 1.975,
        "scope": "Local ENU 1.975 s endpoint displacement errors on overlapping windows; no full-trajectory ATE/RPE or SO(3) claim. Six low-IMU inputs only; no body, attitude, initial velocity, or temperature inputs.",
        "provenance_file": "evaluation/provenance.json", "baseline_state_file": "evaluation/baseline_train_state.json",
        "splits": {},
    }
    tables = {}
    for split in ("validation", "test"):
        features, targets, metadata = physical[split]
        prediction = _predict_checkpoint(model, datasets[split], resolved_device, int(data.get("batch_size", 128)))
        methods = {"model": prediction, **predict_navigation_baselines(baselines, features)}
        report["splits"][split] = _score_windows(metadata, methods, targets)
        table = metadata.copy()
        table["split"] = split
        for axis, column in enumerate(SMARTPNT_GT_POSITION_DELTA_COLUMNS):
            table[f"true_{column}"] = targets[:, axis]
            for method, values in methods.items():
                table[f"{'pred' if method == 'model' else method}_{column}"] = values[:, axis]
        tables[split] = table
    logger.info("WRITES frozen navigation evaluation artifacts under {}", evaluation_dir)
    evaluation_dir.mkdir(parents=False, exist_ok=False)
    frozen_provenance = snapshot_unified_training_inputs(data, evaluation_dir)
    training_provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    if frozen_provenance.get("unified_dataset") != training_provenance["unified_dataset"]:
        raise ValueError("Evaluation input provenance differs from the frozen training snapshot")
    provenance = {
        **frozen_provenance, "training_provenance": training_provenance,
        "checkpoint": {"path": (run_dir / "best.ckpt").as_posix(), "sha256": file_sha256(run_dir / "best.ckpt"),
                       "git_commit": checkpoint.get("git_commit"), "best_epoch": checkpoint.get("best_epoch")},
        "config": {"path": (run_dir / "config.yaml").as_posix(), "sha256": file_sha256(run_dir / "config.yaml")},
        "seed": config.get("seed"), "device": str(resolved_device),
    }
    for name, value in (("provenance.json", provenance), ("baseline_train_state.json", baselines),
                        ("metrics.json", report)):
        logger.info("WRITES navigation evaluation {}", evaluation_dir / name)
        write_json(evaluation_dir / name, value)
    for split, table in tables.items():
        path = evaluation_dir / f"{split}_predictions.parquet"
        logger.info("WRITES {} navigation window predictions {}", len(table), path)
        table.to_parquet(path, index=False)
    rows = []
    for split, result in report["splits"].items():
        for method, metrics in result["metrics"].items():
            row = {"split": split, "method": method, **{key: value for key, value in metrics.items()
                                                        if key not in {"axis_rmse_m", "axis_mae_m", "axis_bias_m"}}}
            for metric in ("rmse", "mae", "bias"):
                row.update({f"{axis}_{metric}_m": value for axis, value in
                            zip(("east", "north", "up"), metrics[f"axis_{metric}_m"], strict=True)})
            rows.append(row)
    logger.info("WRITES navigation split comparison {}", evaluation_dir / "navigation_metrics.csv")
    pd.DataFrame(rows).to_csv(evaluation_dir / "navigation_metrics.csv", index=False)
    return json_safe(report)
