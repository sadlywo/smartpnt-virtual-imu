"""Physical, coverage-aware validation for paired six-axis IMU training."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd

from virtual_imu.preprocessing.normalization import apply_standardization
from virtual_imu.preprocessing.windowing import make_window_slices
from virtual_imu.training.dataset import LOW_IMU_COLUMNS, TARGET_COLUMNS

BASELINE_EPSILON = 1e-12
PAIR_COLUMNS = ["dataset_id", "low_sensor", "high_sensor"]


def grouped_relative_rmse_validation(
    val_df: pd.DataFrame,
    pred_batches: Sequence[np.ndarray],
    target_stats: dict,
    *,
    window_size: int,
    stride: int,
    target_mode: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score predictions in the unshuffled order of ``IMUWindowDataset``.

    Invert target normalization and average overlapping physical predictions at
    each covered timestamp within its own sequence. LOW/HIGH come directly from
    the original SI dataframe, including for residual targets. Pool squared
    vector errors and sample counts within each device pair, then average the
    gyro/accel ratios, pairs within each dataset, and datasets equally. Physical
    summary RMSEs are descriptive and pooled over covered timestamps; they are
    never added across gyro/accel units. No temporal diagnostic joins sequences.
    """
    if target_mode not in {"direct", "residual"}:
        raise ValueError("grouped_relative_rmse validation supports only direct and residual target modes")
    required = ["sequence_id", "timestamp_s", *PAIR_COLUMNS, *LOW_IMU_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in val_df.columns]
    if missing:
        raise ValueError(f"grouped_relative_rmse validation is missing source columns: {missing}")
    for column in ["sequence_id", *PAIR_COLUMNS]:
        if val_df[column].isna().any():
            raise ValueError(f"Validation source column {column} contains missing identities")

    mean = np.asarray(target_stats.get("mean", []), dtype=np.float64)
    std = np.asarray(target_stats.get("std", []), dtype=np.float64)
    if (
        list(target_stats.get("columns", [])) != TARGET_COLUMNS
        or mean.shape != (6,)
        or std.shape != (6,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("target_stats must contain ordered six-axis target columns and finite mean/positive std")

    arrays = []
    for batch_index, batch in enumerate(pred_batches):
        array = np.asarray(batch)
        if array.ndim != 3 or array.shape[1:] != (6, window_size):
            raise ValueError(
                f"Prediction batch {batch_index} shape {array.shape} must be (batch, 6, {window_size})"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Prediction batch {batch_index} contains non-finite values")
        arrays.append(array)

    # Match dataset grouping, ordering, and slices exactly; never add a padded
    # final window or count the uncovered tail as an identity prediction.
    groups = [
        (sequence_id, group.sort_values("timestamp_s").reset_index(drop=True))
        for sequence_id, group in val_df.groupby("sequence_id", sort=False)
    ]
    layouts = [make_window_slices(len(group), window_size, stride) for _, group in groups]
    expected_windows = sum(len(slices) for slices in layouts)
    actual_windows = sum(len(array) for array in arrays)
    if actual_windows != expected_windows:
        raise ValueError(
            f"Validation prediction window count {actual_windows} does not match dataset window count {expected_windows}"
        )
    if expected_windows == 0:
        raise ValueError("No validation timestamps are covered by complete windows")
    predictions = iter(chain.from_iterable(arrays))
    pair_totals: dict[tuple[str, ...], dict[str, Any]] = {}
    observed_pairs: set[tuple[str, ...]] = set()
    sequence_details = []
    for (sequence_id, group), slices in zip(groups, layouts, strict=True):
        for column in PAIR_COLUMNS:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"Validation sequence {sequence_id!r} has inconsistent {column}")
        identity = {column: str(group[column].iloc[0]) for column in PAIR_COLUMNS}
        key = tuple(identity[column] for column in PAIR_COLUMNS)
        observed_pairs.add(key)
        times = group["timestamp_s"].to_numpy(dtype=np.float64)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError(f"Validation sequence {sequence_id!r} must have unique finite timestamps")
        low = group[LOW_IMU_COLUMNS].to_numpy(dtype=np.float64)
        high = group[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError(f"Validation sequence {sequence_id!r} contains non-finite SI LOW/HIGH values")
        prediction_sum = np.zeros((len(group), 6), dtype=np.float64)
        counts = np.zeros(len(group), dtype=np.int64)
        for slc in slices:
            normalized = np.asarray(next(predictions), dtype=np.float64).T
            physical = apply_standardization(normalized, target_stats, inverse=True)
            if target_mode == "residual":
                physical = physical + low[slc]
            if not np.isfinite(physical).all():
                raise ValueError(f"Validation sequence {sequence_id!r} has non-finite physical predictions")
            prediction_sum[slc] += physical
            counts[slc] += 1
        covered = counts > 0
        samples = int(covered.sum())
        sequence_details.append(
            {
                "sequence_id": str(sequence_id),
                **identity,
                "samples": len(group),
                "covered_samples": samples,
                "uncovered_samples": len(group) - samples,
                "window_count": len(slices),
                "max_overlap": int(counts.max()),
                "status": "scored" if samples else "no_complete_windows",
            }
        )
        if not samples:
            continue
        physical_prediction = prediction_sum[covered] / counts[covered, None]
        errors = physical_prediction - high[covered]
        baseline_errors = low[covered] - high[covered]
        totals = pair_totals.setdefault(
            key,
            {
                **identity,
                "samples": 0,
                "sequence_count": 0,
                "model_squared_error": np.zeros(2, dtype=np.float64),
                "identity_squared_error": np.zeros(2, dtype=np.float64),
            },
        )
        totals["samples"] += samples
        totals["sequence_count"] += 1
        for index, channels in enumerate((slice(0, 3), slice(3, 6))):
            totals["model_squared_error"][index] += np.sum(errors[:, channels] ** 2)
            totals["identity_squared_error"][index] += np.sum(baseline_errors[:, channels] ** 2)

    uncovered_pairs = observed_pairs - pair_totals.keys()
    if uncovered_pairs:
        raise ValueError(f"Validation pairs have no covered timestamps: {sorted(uncovered_pairs)}")

    pair_details = []
    dataset_pairs: dict[str, list[dict[str, Any]]] = {}
    for totals in pair_totals.values():
        rmse = np.sqrt(totals["model_squared_error"] / totals["samples"])
        baseline_rmse = np.sqrt(totals["identity_squared_error"] / totals["samples"])
        ratios = rmse / np.maximum(baseline_rmse, BASELINE_EPSILON)
        if not np.isfinite(ratios).all():
            raise ValueError("grouped_relative_rmse validation produced non-finite physical error ratios")
        pair = {
            **{column: totals[column] for column in PAIR_COLUMNS},
            "samples": totals["samples"],
            "sequence_count": totals["sequence_count"],
            "gyro_rmse_radps": float(rmse[0]),
            "accel_rmse_mps2": float(rmse[1]),
            "identity_gyro_rmse_radps": float(baseline_rmse[0]),
            "identity_accel_rmse_mps2": float(baseline_rmse[1]),
            "gyro_relative_rmse": float(ratios[0]),
            "accel_relative_rmse": float(ratios[1]),
            "score": float(np.mean(ratios)),
            "gyro_baseline_near_zero": bool(baseline_rmse[0] <= BASELINE_EPSILON),
            "accel_baseline_near_zero": bool(baseline_rmse[1] <= BASELINE_EPSILON),
        }
        pair_details.append(pair)
        dataset_pairs.setdefault(totals["dataset_id"], []).append(pair)
    dataset_details = [
        {
            "dataset_id": dataset_id,
            "pair_count": len(pairs),
            "samples": sum(pair["samples"] for pair in pairs),
            "score": float(np.mean([pair["score"] for pair in pairs])),
            "gyro_relative_rmse": float(np.mean([pair["gyro_relative_rmse"] for pair in pairs])),
            "accel_relative_rmse": float(np.mean([pair["accel_relative_rmse"] for pair in pairs])),
        }
        for dataset_id, pairs in dataset_pairs.items()
    ]
    score = float(np.mean([dataset["score"] for dataset in dataset_details]))
    covered_samples = sum(pair["samples"] for pair in pair_details)
    pooled_rmse = np.sqrt(
        sum(totals["model_squared_error"] for totals in pair_totals.values()) / covered_samples
    )
    pooled_baseline_rmse = np.sqrt(
        sum(totals["identity_squared_error"] for totals in pair_totals.values()) / covered_samples
    )
    metrics = {
        "val_score": score,
        "val_grouped_score": score,
        "val_gyro_rmse": float(pooled_rmse[0]),
        "val_accel_rmse": float(pooled_rmse[1]),
        "val_identity_gyro_rmse": float(pooled_baseline_rmse[0]),
        "val_identity_accel_rmse": float(pooled_baseline_rmse[1]),
        "val_relative_gyro_rmse": float(np.mean([dataset["gyro_relative_rmse"] for dataset in dataset_details])),
        "val_relative_accel_rmse": float(np.mean([dataset["accel_relative_rmse"] for dataset in dataset_details])),
        "val_covered_samples": float(covered_samples),
        "val_uncovered_samples": float(len(val_df) - covered_samples),
        "val_pair_count": float(len(pair_details)),
        "val_dataset_count": float(len(dataset_details)),
        "val_near_zero_baseline_pairs": float(sum(
            pair["gyro_baseline_near_zero"] or pair["accel_baseline_near_zero"] for pair in pair_details
        )),
    }
    details = {
        "method": "grouped_relative_rmse",
        "target_mode": target_mode,
        "baseline_epsilon": BASELINE_EPSILON,
        "aggregation": "vector RMSE over covered timestamps per pair; mean gyro/accel ratios; mean pairs per dataset; equal mean datasets",
        "overlap_policy": "arithmetic mean of physical predictions at each covered sequence timestamp",
        "physical_summary_aggregation": "pooled covered timestamps; gyro in rad/s and accel in m/s^2 reported separately",
        "metrics": metrics,
        "window_count": actual_windows,
        "pairs": pair_details,
        "datasets": dataset_details,
        "sequences": sequence_details,
    }
    return metrics, details
