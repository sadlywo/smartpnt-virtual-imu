"""Leakage-safe analytic baselines and common evaluation for Penn LOSO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.eval.baselines import (
    LOW_COLUMNS,
    TARGET_COLUMNS,
    VIRTUAL_COLUMNS,
    _lag_features,
    _ridge_fit,
    _ridge_predict,
)
from virtual_imu.eval.preintegration import preintegration_metrics
from virtual_imu.eval.signal_metrics import signal_metrics, snr_improvement
from virtual_imu.eval.spectral_metrics import spectral_metrics
from virtual_imu.io.parquet_store import write_parquet

PENN_BASELINE_METHODS = (
    "identity",
    "mean_bias",
    "channel_affine",
    "learned_fir_causal",
)

_BASELINE_METADATA: dict[str, dict[str, Any]] = {
    "identity": {"parameter_count": 0, "causal": True, "inference_class": "online"},
    "mean_bias": {"parameter_count": 6, "causal": True, "inference_class": "online"},
    "channel_affine": {"parameter_count": 12, "causal": True, "inference_class": "online"},
}


def _require_columns(df: pd.DataFrame, columns: list[str], split_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{split_name} data is missing columns: {missing}")
    values = df[columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{split_name} data contains non-finite IMU values")


def _sequence_slices(df: pd.DataFrame) -> list[slice]:
    """Return contiguous row slices, resetting whenever sequence identity changes."""
    if len(df) == 0:
        return []
    if "sequence_id" not in df.columns:
        return [slice(0, len(df))]
    sequence_ids = df["sequence_id"].astype(str).to_numpy()
    boundaries = np.flatnonzero(sequence_ids[1:] != sequence_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(df)]))
    return [slice(int(starts[index]), int(ends[index])) for index in range(len(starts))]


def _fit_independent_affine(
    low: np.ndarray, high: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    slopes = np.empty(low.shape[1], dtype=float)
    intercepts = np.empty(low.shape[1], dtype=float)
    for channel in range(low.shape[1]):
        weights, bias = _ridge_fit(
            low[:, [channel]], high[:, [channel]], alpha=alpha, intercept=True
        )
        slopes[channel] = weights[0, 0]
        intercepts[channel] = bias[0]
    return slopes, intercepts


def _sequence_lag_features(df: pd.DataFrame, taps: int) -> np.ndarray:
    low = df[LOW_COLUMNS].to_numpy(dtype=float)
    chunks = [_lag_features(low[rows], taps=taps) for rows in _sequence_slices(df)]
    return np.vstack(chunks)


def fit_penn_baselines(
    train_df: pd.DataFrame, fir_taps: int = 9, alpha: float = 1e-2
) -> dict[str, Any]:
    """Fit Penn analytic baselines using only the supplied training rows."""
    if fir_taps < 1:
        raise ValueError("fir_taps must be at least 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if train_df.empty:
        raise ValueError("train data must contain at least one row")
    _require_columns(train_df, [*LOW_COLUMNS, *TARGET_COLUMNS], "train")

    low = train_df[LOW_COLUMNS].to_numpy(dtype=float)
    high = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    slopes, intercepts = _fit_independent_affine(low, high, alpha)
    fir_weights, fir_bias = _ridge_fit(
        _sequence_lag_features(train_df, fir_taps), high, alpha=alpha, intercept=True
    )

    return {
        "identity": {"parameter_count": 0},
        "mean_bias": {
            "bias": np.mean(low - high, axis=0).tolist(),
            "parameter_count": 6,
        },
        "channel_affine": {
            "slopes": slopes.tolist(),
            "intercepts": intercepts.tolist(),
            "parameter_count": 12,
        },
        "learned_fir_causal": {
            "fir_taps": int(fir_taps),
            "alpha": float(alpha),
            "fir_weights": fir_weights.tolist(),
            "fir_bias": fir_bias.tolist(),
            "parameter_count": int(fir_weights.size + fir_bias.size),
        },
    }


def _predict_fir_by_sequence(fir_state: dict[str, Any], test_df: pd.DataFrame) -> np.ndarray:
    low = test_df[LOW_COLUMNS].to_numpy(dtype=float)
    prediction = np.empty_like(low)
    weights = np.asarray(fir_state["fir_weights"], dtype=float)
    bias = np.asarray(fir_state["fir_bias"], dtype=float)
    taps = int(fir_state["fir_taps"])
    for rows in _sequence_slices(test_df):
        prediction[rows] = _ridge_predict(_lag_features(low[rows], taps=taps), weights, bias)
    return prediction


def predict_penn_baselines(
    state: dict[str, Any], test_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Predict all Penn analytic baselines without reading test targets."""
    missing_methods = [method for method in PENN_BASELINE_METHODS if method not in state]
    if missing_methods:
        raise ValueError(f"Baseline state is missing methods: {missing_methods}")
    _require_columns(test_df, LOW_COLUMNS, "test")
    low = test_df[LOW_COLUMNS].to_numpy(dtype=float)
    affine = state["channel_affine"]

    return {
        "identity": low.copy(),
        "mean_bias": low - np.asarray(state["mean_bias"]["bias"], dtype=float),
        "channel_affine": (
            low * np.asarray(affine["slopes"], dtype=float)
            + np.asarray(affine["intercepts"], dtype=float)
        ),
        "learned_fir_causal": _predict_fir_by_sequence(
            state["learned_fir_causal"], test_df
        ),
    }


def _group_error_metrics(prediction_df: pd.DataFrame) -> dict[str, float]:
    prediction = prediction_df[VIRTUAL_COLUMNS].to_numpy(dtype=float)
    identity = prediction_df[LOW_COLUMNS].to_numpy(dtype=float)
    target = prediction_df[TARGET_COLUMNS].to_numpy(dtype=float)
    error = prediction - target
    identity_error = identity - target

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    def mae(values: np.ndarray) -> float:
        return float(np.mean(np.abs(values)))

    def improvement(reference: float, candidate: float) -> float:
        if reference <= 1e-12:
            return 0.0 if candidate <= 1e-12 else float("-inf")
        return float(100.0 * (reference - candidate) / reference)

    correlations = []
    for channel in range(6):
        pred_channel = prediction[:, channel]
        target_channel = target[:, channel]
        if np.std(pred_channel) > 1e-12 and np.std(target_channel) > 1e-12:
            correlations.append(float(np.corrcoef(pred_channel, target_channel)[0, 1]))

    gyro_rmse = rmse(error[:, :3])
    accel_rmse = rmse(error[:, 3:])
    identity_gyro_rmse = rmse(identity_error[:, :3])
    identity_accel_rmse = rmse(identity_error[:, 3:])
    return {
        "rmse_6d": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
        "mae_6d": mae(error),
        "gyro_rmse": gyro_rmse,
        "gyro_mae": mae(error[:, :3]),
        "accel_rmse": accel_rmse,
        "accel_mae": mae(error[:, 3:]),
        "mean_axis_correlation": float(np.mean(correlations)) if correlations else float("nan"),
        "gyro_identity_rmse": identity_gyro_rmse,
        "accel_identity_rmse": identity_accel_rmse,
        "gyro_improvement_vs_identity_pct": improvement(identity_gyro_rmse, gyro_rmse),
        "accel_improvement_vs_identity_pct": improvement(identity_accel_rmse, accel_rmse),
    }


def evaluate_penn_method(prediction_df: pd.DataFrame) -> dict[str, Any]:
    """Evaluate one method against high-grade Penn targets in physical units."""
    required = ["timestamp_s", *LOW_COLUMNS, *TARGET_COLUMNS, *VIRTUAL_COLUMNS]
    _require_columns(prediction_df, required[1:], "prediction")
    if "timestamp_s" not in prediction_df.columns:
        raise ValueError("prediction data is missing columns: ['timestamp_s']")
    if prediction_df.empty:
        raise ValueError("prediction data must contain at least one row")

    identity_signal = signal_metrics(prediction_df, "low")
    method_signal = signal_metrics(prediction_df, "virtual")
    summary = _group_error_metrics(prediction_df)
    improvement = {
        "gyro_rmse_pct": summary["gyro_improvement_vs_identity_pct"],
        "accel_rmse_pct": summary["accel_improvement_vs_identity_pct"],
        **snr_improvement(identity_signal, method_signal),
    }
    return {
        "sample_count": int(len(prediction_df)),
        "summary": summary,
        "signal": {
            "identity": identity_signal,
            "method": method_signal,
            "improvement_vs_identity": improvement,
        },
        "spectral": {
            "identity": spectral_metrics(prediction_df, "low"),
            "method": spectral_metrics(prediction_df, "virtual"),
        },
        "preintegration": {
            "identity": preintegration_metrics(prediction_df, "low"),
            "method": preintegration_metrics(prediction_df, "virtual"),
        },
    }


def _prediction_frame(test_df: pd.DataFrame, prediction: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(prediction, pd.DataFrame):
        frame = prediction.copy()
        if all(column in frame.columns for column in VIRTUAL_COLUMNS):
            return frame
        values = frame.to_numpy(dtype=float)
    else:
        values = np.asarray(prediction, dtype=float)
    if values.shape != (len(test_df), 6):
        raise ValueError(
            f"Prediction must have shape ({len(test_df)}, 6); received {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Prediction contains non-finite values")
    frame = test_df.copy()
    for channel, column in enumerate(VIRTUAL_COLUMNS):
        frame[column] = values[:, channel]
    return frame


def _default_metadata(
    method: str, baseline_state: dict[str, Any] | None
) -> dict[str, Any]:
    metadata = dict(_BASELINE_METADATA.get(method, {}))
    if method == "learned_fir_causal":
        metadata.update({"causal": True, "inference_class": "online"})
        metadata.setdefault("parameter_count", 0)
    if not metadata:
        metadata = {
            "parameter_count": 0,
            "causal": True,
            "inference_class": "online",
        }
    if baseline_state and method in baseline_state:
        metadata["parameter_count"] = int(
            baseline_state[method].get("parameter_count", metadata["parameter_count"])
        )
    return metadata


def write_penn_comparison(
    predictions: dict[str, np.ndarray | pd.DataFrame],
    test_df: pd.DataFrame,
    output_dir: str | Path,
    method_metadata: dict[str, dict[str, Any]] | None = None,
    baseline_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write per-method predictions/metrics and the common Penn comparison table."""
    if not predictions:
        raise ValueError("predictions must contain at least one method")
    _require_columns(test_df, [*LOW_COLUMNS, *TARGET_COLUMNS], "test")
    if "timestamp_s" not in test_df.columns:
        raise ValueError("test data is missing columns: ['timestamp_s']")

    root = Path(output_dir)
    prediction_paths: dict[str, Path] = {}
    metric_paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for method, prediction in predictions.items():
        frame = _prediction_frame(test_df, prediction)
        prediction_path = write_parquet(frame, root / "predictions" / f"{method}.parquet")
        logger.info("Wrote Penn prediction: {}", prediction_path)
        prediction_paths[method] = prediction_path

        metrics = evaluate_penn_method(frame)
        metadata = _default_metadata(method, baseline_state)
        metadata.update((method_metadata or {}).get(method, {}))
        payload = {"method": method, **metadata, **metrics}
        metric_path = root / "metrics" / f"{method}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
        logger.info("Wrote Penn metrics: {}", metric_path)
        metric_paths[method] = metric_path
        summary = metrics["summary"]
        gyro_reference = max(float(summary["gyro_identity_rmse"]), 1e-12)
        accel_reference = max(float(summary["accel_identity_rmse"]), 1e-12)
        normalized_rmse_score = 0.5 * (
            float(summary["gyro_rmse"]) / gyro_reference
            + float(summary["accel_rmse"]) / accel_reference
        )
        rows.append(
            {
                "method": method,
                **metadata,
                **summary,
                "normalized_rmse_score": normalized_rmse_score,
            }
        )

    table = pd.DataFrame(rows).sort_values(
        ["normalized_rmse_score", "method"], kind="stable"
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    comparison_csv = root / "comparison.csv"
    comparison_json = root / "comparison.json"
    comparison_markdown = root / "comparison.md"
    root.mkdir(parents=True, exist_ok=True)
    table.to_csv(comparison_csv, index=False)
    comparison_json.write_text(
        table.to_json(orient="records", indent=2), encoding="utf-8"
    )

    columns = [
        "rank",
        "method",
        "parameter_count",
        "causal",
        "inference_class",
        "normalized_rmse_score",
        "gyro_rmse",
        "accel_rmse",
        "gyro_improvement_vs_identity_pct",
        "accel_improvement_vs_identity_pct",
    ]
    lines = [
        "# Penn Default LOSO Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in table[columns].to_dict(orient="records"):
        values = [f"{value:.6g}" if isinstance(value, float) else str(value) for value in row.values()]
        lines.append("| " + " | ".join(values) + " |")
    comparison_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in (comparison_csv, comparison_json, comparison_markdown):
        logger.info("Wrote Penn comparison: {}", path)

    return {
        "predictions": prediction_paths,
        "metrics": metric_paths,
        "comparison_csv": comparison_csv,
        "comparison_json": comparison_json,
        "comparison_markdown": comparison_markdown,
    }
