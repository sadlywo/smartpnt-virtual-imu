"""Physical, sequence-aware evaluation for the frozen paired IMU experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import correlate, correlation_lags

from virtual_imu.eval.baselines import fit_system_id_baseline_state, predict_system_id_from_state
from virtual_imu.eval.spectral_metrics import spectral_metrics
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.training_bundle import sha256_file
from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.preprocessing.normalization import apply_standardization
from virtual_imu.preprocessing.windowing import make_window_slices
from virtual_imu.training.unified_data import snapshot_unified_training_inputs

LOW = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH = [name.replace("_low", "_high") for name in LOW]
VIRTUAL = [name.replace("_low", "_virtual") for name in LOW]
IDENTITY = ["dataset_id", "low_sensor", "high_sensor"]


def json_safe(value):
    """Convert unavailable diagnostics to JSON null without nonstandard NaN literals."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def verify_run_provenance(run_dir: Path, config: dict, checkpoint_config: dict) -> None:
    """Refuse a changed config, corpus or split under an existing training-run identity."""
    if config != checkpoint_config:
        raise ValueError("Checkpoint and run config differ")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    inputs = provenance["unified_dataset"]["inputs"]
    names = {
        "dataset_manifest": "data_manifest.json",
        "split_manifest": "split_manifest.json",
        "strict_manifest": "strict_manifest.json",
        "assignment_manifest": "assignment_manifest.json",
    }
    for key, value in inputs.items():
        if config["data"].get(key) != value["path"]:
            raise ValueError(f"Training data path changed: {key}")
        for path in (Path(value["path"]), run_dir / names[key]):
            if not path.is_file() or sha256_file(path) != value["sha256"]:
                raise ValueError(f"Training manifest changed or differs: {path}")
    required = {key for key in names if config["data"].get(key)}
    if set(inputs) != required:
        raise ValueError("Training provenance does not bind every configured manifest")


def physical_error_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    """Report vector and per-axis errors, preserving gyro and accel physical units."""
    prediction, reference = np.asarray(prediction, float), np.asarray(reference, float)
    if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[1] != 6:
        raise ValueError("Prediction/reference must be matching N x 6 arrays")
    if not len(prediction) or not np.isfinite(prediction).all() or not np.isfinite(reference).all():
        raise ValueError("Physical metrics require nonempty finite measured samples")
    error = prediction - reference
    gyro, accel = np.linalg.norm(error[:, :3], axis=1), np.linalg.norm(error[:, 3:], axis=1)
    return {
        "samples": len(error),
        "gyro_vector_rmse_radps": float(np.sqrt(np.mean(gyro**2))),
        "accel_vector_rmse_mps2": float(np.sqrt(np.mean(accel**2))),
        "axis_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "axis_mae": np.mean(np.abs(error), axis=0).tolist(),
        "axis_mean_error": np.mean(error, axis=0).tolist(),
        "axis_error_std": np.std(error, axis=0).tolist(),
        "gyro_error_p95_radps": float(np.quantile(gyro, 0.95)),
        "accel_error_p95_mps2": float(np.quantile(accel, 0.95)),
    }


def sequence_diagnostics(frame: pd.DataFrame, rate_hz: float = 40.0) -> dict:
    """Reset integral/lag/spectral diagnostics for each real contiguous sequence."""
    dt = 1 / rate_hz
    timestamps = frame.timestamp_s.to_numpy(float)
    if len(frame) > 1 and not np.allclose(np.diff(timestamps), dt, atol=1e-6, rtol=1e-5):
        raise ValueError("Sequence diagnostic received a gap or nonuniform sampling")
    predicted, target = frame[VIRTUAL].to_numpy(float), frame[HIGH].to_numpy(float)
    error = predicted - target
    integrals = []
    window = 80
    for start in range(0, len(frame) - window + 1, window):
        part = error[start : start + window]
        integrals.append(
            [
                np.linalg.norm(np.trapz(part[:, :3], dx=dt, axis=0)),
                np.linalg.norm(np.trapz(part[:, 3:], dx=dt, axis=0)),
            ]
        )
    lag = []
    for axis in range(3):
        p, t = predicted[:, axis], target[:, axis]
        if len(p) < 16 or min(np.std(p), np.std(t)) < 1e-4:
            lag.append(None)
            continue
        correlations = correlate(p - p.mean(), t - t.mean(), mode="full", method="fft")
        lags = correlation_lags(len(p), len(t), mode="full")
        valid = np.abs(lags) <= round(0.5 * rate_hz)
        lag.append(float(lags[valid][np.argmax(correlations[valid])] / rate_hz))
    mean_integrals = np.mean(integrals, axis=0) if integrals else [None, None]
    return json_safe(
        {
            "integral_window_points": window,
            "integral_endpoint_span_s": (window - 1) * dt,
            "integral_windows": len(integrals),
            "delta_gyro_integral_error_rad": mean_integrals[0],
            "delta_accel_integral_error_mps": mean_integrals[1],
            "gyro_lag_s_by_axis": lag,
            "spectral": spectral_metrics(frame, "virtual")
            if len(frame) >= 80 and np.max(np.std(target, axis=0)) > 1e-4
            else {"status": "low excitation or short"},
            "note": "Additive signal-integral diagnostics; not SO(3) preintegration or navigation accuracy. Lag never fits a correction.",
        }
    )


def evaluate_prediction_frame(frame: pd.DataFrame) -> dict:
    """One physical sample per timestamp; temporal diagnostics never cross sequences."""
    if frame.duplicated(["sequence_id", "timestamp_s"]).any():
        raise ValueError("Overlapping predictions must be merged before physical evaluation")
    pairs, sequences = [], []
    for values, group in frame.groupby(IDENTITY, sort=True):
        metrics = {
            prefix: physical_error_metrics(group[columns].to_numpy(), group[HIGH].to_numpy())
            for prefix, columns in (("identity", LOW), ("model", VIRTUAL))
        }
        pairs.append(
            {**dict(zip(IDENTITY, values, strict=True)), "samples": len(group), "metrics": metrics}
        )
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s")
        sequences.append(
            {
                "sequence_id": sequence_id,
                **{k: group[k].iloc[0] for k in IDENTITY},
                "metrics": physical_error_metrics(
                    group[VIRTUAL].to_numpy(), group[HIGH].to_numpy()
                ),
                "diagnostics": sequence_diagnostics(group),
            }
        )
    return {"pairs": pairs, "sequences": sequences, "samples": len(frame)}


def fit_pair_baselines(train: pd.DataFrame, *, fir_taps: int = 9, alpha: float = 0.01) -> dict:
    """Fit only training samples for one specific device/reference mapping."""
    if len(train[IDENTITY].drop_duplicates()) != 1:
        raise ValueError("Classical baseline fits must be isolated by device/reference mapping")
    x, y = train[LOW].to_numpy(float), train[HIGH].to_numpy(float)
    design = np.column_stack([x, np.ones(len(x))])
    affine, *_ = np.linalg.lstsq(design, y, rcond=None)
    regularizer = np.eye(7) * alpha
    regularizer[-1, -1] = 0
    ridge = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return {
        "identity": {k: str(train[k].iloc[0]) for k in IDENTITY},
        "fit_split": "train",
        "fit_samples": len(train),
        "bias": np.mean(y - x, axis=0).tolist(),
        "affine": affine.tolist(),
        "ridge_affine": ridge.tolist(),
        "system_id": fit_system_id_baseline_state(train, fir_taps=fir_taps, alpha=alpha),
    }


def predict_pair_baselines(state: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    low = frame[LOW].to_numpy(float)
    design = np.column_stack([low, np.ones(len(low))])
    return {
        "identity": low,
        "bias_mean": low + np.asarray(state["bias"]),
        "affine": design @ np.asarray(state["affine"]),
        "ridge_affine": design @ np.asarray(state["ridge_affine"]),
        **predict_system_id_from_state(state["system_id"], frame),
    }


def predict_complete_windows(
    generator: VirtualIMUGenerator, frame: pd.DataFrame, *, batch_size: int = 128
):
    """Match validation coverage exactly; batch complete windows, without a padded tail."""
    if generator.target_mode != "residual" or generator.has_static_context:
        raise ValueError(
            "Formal paired inference requires residual prediction without static context"
        )
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    outputs, coverage = [], []
    for sequence_id, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        slices = make_window_slices(len(group), generator.window_size, generator.stride)
        counts = np.zeros(len(group), dtype=int)
        sums = np.zeros((len(group), 6), dtype=float)
        features = apply_standardization(
            group[generator.feature_columns].to_numpy(np.float32), generator.feature_stats
        ).astype(np.float32)
        for start in range(0, len(slices), batch_size):
            chunk = slices[start : start + batch_size]
            values = np.stack([features[slc].T for slc in chunk])
            with torch.no_grad():
                predicted = (
                    generator.model(torch.from_numpy(values).to(generator.device)).cpu().numpy()
                )
            for slc, result in zip(chunk, predicted, strict=True):
                physical = apply_standardization(
                    result.T.astype(float), generator.target_stats, inverse=True
                )
                physical += group.iloc[slc][LOW].to_numpy(float)
                if not np.isfinite(physical).all():
                    raise ValueError("Nonfinite formal prediction")
                sums[slc] += physical
                counts[slc] += 1
        covered = counts > 0
        coverage.append(
            {
                "sequence_id": str(sequence_id),
                "samples": len(group),
                "covered_samples": int(covered.sum()),
                "uncovered_samples": int((~covered).sum()),
                "windows": len(slices),
            }
        )
        if covered.any():
            out = group.loc[covered].copy()
            out[VIRTUAL] = sums[covered] / counts[covered, None]
            out["prediction_overlap_count"] = counts[covered]
            outputs.append(out)
    if not outputs:
        raise ValueError("Formal inference has no complete windows")
    return pd.concat(outputs, ignore_index=True), coverage


def evaluate_paired_run(run_dir: Path, *, device: str = "auto") -> dict:
    """Evaluate the selected checkpoint and fixed train-only baselines on held-out data."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    data = config["data"]
    generator = VirtualIMUGenerator(run_dir / "best.ckpt", device=device)
    verify_run_provenance(run_dir, config, generator.config)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=False)
    provenance = snapshot_unified_training_inputs(data, evaluation_dir)
    write_json(evaluation_dir / "provenance.json", provenance)
    bundle = load_training_split_manifest(
        Path(data["split_manifest"]), Path(data["strict_manifest"])
    )
    states = {}
    for identity, frame in bundle.frames["train"].groupby(IDENTITY, sort=True):
        states[identity] = fit_pair_baselines(frame)
    write_json(evaluation_dir / "baseline_train_state.json", list(states.values()))
    report = {
        "training_run": run_dir.as_posix(),
        "selection": config["training"]["checkpoint_metric"],
        "scope": "Fixed model, train-only baseline fits; no evaluation-fitted alignment.",
        "splits": {},
    }
    for split in ("validation", "test"):
        frame = bundle.frames[split]
        predictions, coverage = predict_complete_windows(
            generator, frame, batch_size=int(data.get("batch_size", 128))
        )
        predictions.to_parquet(evaluation_dir / f"{split}_predictions.parquet", index=False)
        evaluated = evaluate_prediction_frame(predictions)
        evaluated["coverage"] = coverage
        if set(map(tuple, frame[IDENTITY].drop_duplicates().to_numpy())) != set(
            map(tuple, predictions[IDENTITY].drop_duplicates().to_numpy())
        ):
            raise ValueError("A formal evaluation pair has no complete windows")
        for pair in evaluated["pairs"]:
            identity = tuple(pair[k] for k in IDENTITY)
            group = predictions
            for key, value in zip(IDENTITY, identity, strict=True):
                group = group[group[key] == value]
            methods = predict_pair_baselines(states[identity], group)
            for name, values in methods.items():
                pair["metrics"][name] = physical_error_metrics(values, group[HIGH].to_numpy())
            pair["improvement_percent"] = {
                baseline: {
                    metric: 100
                    * (1 - pair["metrics"]["model"][metric] / max(metrics[metric], 1e-12))
                    for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
                }
                for baseline, metrics in pair["metrics"].items()
                if baseline != "model"
            }
            pair["relative_score"] = float(
                np.mean(
                    [
                        pair["metrics"]["model"][metric]
                        / max(pair["metrics"]["identity"][metric], 1e-12)
                        for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
                    ]
                )
            )
            pair["near_zero_identity"] = {
                metric: pair["metrics"]["identity"][metric] <= 1e-12
                for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
            }
        datasets = sorted({pair["dataset_id"] for pair in evaluated["pairs"]})
        evaluated["grouped_relative_score"] = float(
            np.mean(
                [
                    np.mean(
                        [
                            pair["relative_score"]
                            for pair in evaluated["pairs"]
                            if pair["dataset_id"] == dataset
                        ]
                    )
                    for dataset in datasets
                ]
            )
        )
        report["splits"][split] = evaluated
    write_json(evaluation_dir / "metrics.json", report)
    rows = []
    for split, result in report["splits"].items():
        for pair in result["pairs"]:
            for method, metrics in pair["metrics"].items():
                rows.append(
                    {
                        "split": split,
                        **{k: pair[k] for k in IDENTITY},
                        "method": method,
                        "gyro_vector_rmse_radps": metrics["gyro_vector_rmse_radps"],
                        "accel_vector_rmse_mps2": metrics["accel_vector_rmse_mps2"],
                    }
                )
    pd.DataFrame(rows).to_csv(evaluation_dir / "paired_metrics.csv", index=False)
    return report
