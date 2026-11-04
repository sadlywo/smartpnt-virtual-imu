"""Lightweight supervised signal baselines for paired low/high IMU data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from virtual_imu.eval.baselines import (
    LOW_COLUMNS,
    TARGET_COLUMNS,
    fit_system_id_baseline_state,
    predict_system_id_from_state,
)
from virtual_imu.models.baseline_affine import AffineCalibration


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([x, np.ones(len(x))])
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return coef[:-1], coef[-1]


def _predict_system_id_by_sequence(state: dict[str, Any], test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    if "sequence_id" not in test_df.columns:
        return predict_system_id_from_state(state, test_df)
    predictions = {"learned_fir_causal": [], "arx_linear": []}
    for _, group in test_df.groupby("sequence_id", sort=False):
        group_predictions = predict_system_id_from_state(state, group)
        for name, values in group_predictions.items():
            predictions[name].append(values)
    return {name: np.vstack(chunks) for name, chunks in predictions.items()}


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse_6d": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        "gyro_rmse_3d": float(np.sqrt(np.mean(np.sum(err[:, :3] ** 2, axis=1)))),
        "accel_rmse_3d": float(np.sqrt(np.mean(np.sum(err[:, 3:] ** 2, axis=1)))),
        "mae_6d": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
    }


def _required_columns(df: pd.DataFrame, name: str) -> None:
    missing = [col for col in [*LOW_COLUMNS, *TARGET_COLUMNS] if col not in df.columns]
    if missing:
        raise ValueError(f"{name} split is missing columns: {missing}")


def evaluate_supervised_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    fir_taps: int = 9,
    ridge_alpha: float = 1e-2,
) -> dict[str, Any]:
    """Fit simple supervised low-to-high baselines on train and evaluate on test."""
    _required_columns(train_df, "train")
    _required_columns(test_df, "test")
    train_low = train_df[LOW_COLUMNS].to_numpy(dtype=float)
    train_high = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    test_low = test_df[LOW_COLUMNS].to_numpy(dtype=float)
    test_high = test_df[TARGET_COLUMNS].to_numpy(dtype=float)

    predictions: dict[str, np.ndarray] = {
        "identity": test_low,
        "bias_mean": test_low + np.mean(train_high - train_low, axis=0),
    }
    predictions["affine"] = AffineCalibration().fit(train_low, train_high).predict(test_low)
    ridge_weights, ridge_bias = _ridge_fit(train_low, train_high, alpha=ridge_alpha)
    predictions["ridge_affine"] = test_low @ ridge_weights + ridge_bias
    system_id_state = fit_system_id_baseline_state(train_df, fir_taps=fir_taps, alpha=ridge_alpha)
    predictions.update(_predict_system_id_by_sequence(system_id_state, test_df))

    baselines = {name: _metrics(pred, test_high) for name, pred in predictions.items()}
    leaderboard = [
        {"baseline": name, **metrics}
        for name, metrics in sorted(baselines.items(), key=lambda item: item[1]["rmse_6d"])
    ]
    return {
        "sample_count": int(len(test_df)),
        "train_sample_count": int(len(train_df)),
        "train_sequence_count": int(train_df["sequence_id"].nunique()) if "sequence_id" in train_df else 1,
        "test_sequence_count": int(test_df["sequence_id"].nunique()) if "sequence_id" in test_df else 1,
        "settings": {"fir_taps": int(fir_taps), "ridge_alpha": float(ridge_alpha)},
        "baselines": baselines,
        "leaderboard": leaderboard,
    }


def write_supervised_baseline_outputs(result: dict[str, Any], out_json: str | Path) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown summaries for supervised baseline results."""
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["baseline", "rmse_6d", "gyro_rmse_3d", "accel_rmse_3d", "mae_6d", "max_abs_error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["leaderboard"]:
            writer.writerow({key: row[key] for key in fieldnames})

    md_path = out_path.with_suffix(".md")
    lines = [
        "# Supervised Low-to-High IMU Baseline",
        "",
        f"- Train samples: {result['train_sample_count']}",
        f"- Test samples: {result['sample_count']}",
        f"- Train sequences: {result['train_sequence_count']}",
        f"- Test sequences: {result['test_sequence_count']}",
        f"- FIR taps: {result['settings']['fir_taps']}",
        f"- Ridge alpha: {result['settings']['ridge_alpha']}",
        "",
        "| rank | baseline | rmse 6d | gyro rmse 3d | accel rmse 3d | mae 6d |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(result["leaderboard"], start=1):
        lines.append(
            f"| {rank} | {row['baseline']} | {row['rmse_6d']:.6g} | "
            f"{row['gyro_rmse_3d']:.6g} | {row['accel_rmse_3d']:.6g} | {row['mae_6d']:.6g} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": out_path, "csv": csv_path, "markdown": md_path}
