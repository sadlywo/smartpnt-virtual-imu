"""Independent FusionPortable diagnosis; no legacy preparation outputs are modified.

Source timestamps are compared in physical seconds, missing intervals remain missing,
and temporal calibration is fitted on a named training interval only. Outputs remain
diagnostic because one short sequence cannot establish transferable calibration.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from loguru import logger
from scipy import signal

from virtual_imu.io.multi_imu_calibration import (
    RigidTransform,
    load_opencv_yaml,
    opencv_matrix,
    quaternion_wxyz_to_matrix,
)
from virtual_imu.preprocessing.rigid_imu import compensate_lever_arm
from virtual_imu.preprocessing.strict_alignment import align_strict_pair
from virtual_imu.utils.paths import utc_now_iso

AXES = ["gx", "gy", "gz", "ax", "ay", "az"]


def reference_pose_to_body(
    document: dict[str, Any], local_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compose parent-from-child poses using the official TF graph convention."""
    reference_from_body = quaternion_wxyz_to_matrix(
        opencv_matrix(document["quaternion_sensor_bodyimu"]).ravel()
    )
    reference_from_local = quaternion_wxyz_to_matrix(
        opencv_matrix(document[f"quaternion_sensor_{local_name}"]).ravel()
    )
    body_in_reference = opencv_matrix(document["translation_sensor_bodyimu"]).ravel()
    local_in_reference = opencv_matrix(document[f"translation_sensor_{local_name}"]).ravel()
    return (
        reference_from_body.T @ reference_from_local,
        reference_from_body.T @ (local_in_reference - body_in_reference),
    )


def bounded_interpolation(
    timestamps: np.ndarray, values: np.ndarray, grid: np.ndarray, maximum_gap_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate supported intervals, retaining clock gaps on the grid."""
    times = np.asarray(timestamps, dtype=float)
    observations = np.asarray(values, dtype=float)
    query = np.asarray(grid, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Interpolation requires at least two strictly increasing timestamps")
    right = np.searchsorted(times, query, side="left")
    clipped = np.clip(right, 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    exact = np.abs(times[clipped] - query) <= 1e-8
    supported = exact | (
        (right > 0) & (right < len(times)) & ((times[clipped] - times[left]) <= maximum_gap_s)
    )
    output = np.column_stack(
        [np.interp(query, times, observations[:, axis]) for axis in range(observations.shape[1])]
    )
    output[~supported] = np.nan
    return output, supported


def causal_supported_resample(
    timestamps: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    nominal_source_rate_hz: float,
    target_rate_hz: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Audit FIR with nominal clocks, a rate tolerance, and full support trimming.

    This only diagnoses a candidate. Packet timestamps, especially Ouster, may not
    represent a sufficiently uniform physical sample clock for this approximation.
    """
    times = np.asarray(timestamps, dtype=float)
    observations = np.asarray(values, dtype=float)
    nominal = float(nominal_source_rate_hz)
    relative_tolerance = float(config.get("equal_rate_relative_tolerance", 0.01))
    taps = int(config.get("fir_taps", 63))
    maximum_gap_s = float(config.get("max_interpolation_gap_periods", 3.0)) / nominal
    applied = target_rate_hz < nominal * (1.0 - relative_tolerance)
    delay_s = 0.0
    warmup = 0
    invalid_support = np.zeros(len(times), dtype=bool)
    filtered = observations.copy()
    if applied:
        if taps < 3 or taps % 2 == 0:
            raise ValueError("FIR taps must be an odd integer of at least three")
        coefficients = signal.firwin(
            taps, float(config.get("fir_cutoff_fraction", 0.8)) * target_rate_hz / 2, fs=nominal
        )
        filtered = signal.lfilter(coefficients, [1.0], observations, axis=0)
        delay_s = (taps - 1) / (2.0 * nominal)
        warmup = taps - 1
        invalid_support[:warmup] = True
        gaps = np.flatnonzero(np.diff(times) > maximum_gap_s) + 1
        for index in gaps:
            invalid_support[index : min(len(times), index + taps - 1)] = True
    else:
        gaps = np.flatnonzero(np.diff(times) > maximum_gap_s) + 1
    filtered[invalid_support] = np.nan
    output, clock_supported = bounded_interpolation(times - delay_s, filtered, grid, maximum_gap_s)
    finite = np.isfinite(output).all(axis=1)
    return output, {
        "anti_alias_applied": applied,
        "nominal_source_rate_hz": nominal,
        "target_rate_hz": target_rate_hz,
        "group_delay_s": delay_s,
        "warmup_source_rows": warmup,
        "source_gap_count": len(gaps),
        "unsupported_fir_source_rows": int(invalid_support.sum()),
        "output_rows": len(output),
        "clock_unsupported_output_rows": int((~clock_supported).sum()),
        "finite_output_rows": int(finite.sum()),
        "finite_output_fraction": float(finite.mean()),
    }


def paired_metrics(low: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(low).all(axis=1) & np.isfinite(high).all(axis=1)
    low_finite, high_finite = low[finite], high[finite]
    if len(low_finite) < 3:
        return {
            "grid_rows": len(low),
            "finite_rows": len(low_finite),
            "finite_fraction": float(finite.mean()),
            "correlation": [None] * low.shape[1],
            "rmse": [None] * low.shape[1],
        }
    correlations = []
    for axis in range(low.shape[1]):
        if min(np.std(low_finite[:, axis]), np.std(high_finite[:, axis])) <= 1e-10:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(low_finite[:, axis], high_finite[:, axis])[0, 1]))
    difference = low_finite - high_finite
    bias = np.mean(difference, axis=0)
    return {
        "grid_rows": len(low),
        "finite_rows": len(low_finite),
        "finite_fraction": float(finite.mean()),
        "correlation": correlations,
        "low_std": np.std(low_finite, axis=0).tolist(),
        "high_std": np.std(high_finite, axis=0).tolist(),
        "bias_low_minus_high": bias.tolist(),
        "rmse": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
        "bias_removed_rmse": np.sqrt(np.mean((difference - bias) ** 2, axis=0)).tolist(),
    }


def fit_time_shift(
    low_times: np.ndarray,
    low_gyro: np.ndarray,
    high_times: np.ndarray,
    high_gyro: np.ndarray,
    fit_grid: np.ndarray,
    shifts: np.ndarray,
    low_gap_s: float,
    high_gap_s: float,
) -> tuple[float, float]:
    """Fit an additive low timestamp correction using only the given fit grid."""
    high, _ = bounded_interpolation(high_times, high_gyro, fit_grid, high_gap_s)
    high_norm = np.linalg.norm(high, axis=1)
    scores = []
    for shift in shifts:
        low, _ = bounded_interpolation(low_times + shift, low_gyro, fit_grid, low_gap_s)
        low_norm = np.linalg.norm(low, axis=1)
        finite = np.isfinite(low_norm) & np.isfinite(high_norm)
        score = (
            float(np.corrcoef(low_norm[finite], high_norm[finite])[0, 1])
            if finite.sum() >= 3
            else -np.inf
        )
        scores.append(score if np.isfinite(score) else -np.inf)
    best = int(np.argmax(scores))
    return float(shifts[best]), float(scores[best])


def _clock_frame(frame: pd.DataFrame, column: str, epoch: float) -> tuple[np.ndarray, np.ndarray]:
    times = frame[column].to_numpy(dtype=float) - epoch
    keep = ~pd.Series(times).duplicated().to_numpy()
    times = times[keep]
    values = frame[AXES].to_numpy(dtype=float)[keep]
    if np.any(np.diff(times) < 0.0):
        raise ValueError(f"{column} timestamps decrease")
    return times, values


def _profile(path: Path, nominal_rate: float) -> dict[str, Any]:
    logger.info("READS full stream metadata and all numeric batches {}", path)
    parquet = pq.ParquetFile(path)
    steps, delays = [], []
    previous = None
    count = 0
    nonfinite = 0
    sums = np.zeros(6)
    squares = np.zeros(6)
    first, last = None, None
    for batch in parquet.iter_batches(
        batch_size=100_000, columns=["timestamp_s", "record_timestamp_s", *AXES]
    ):
        frame = batch.to_pandas()
        times = frame.timestamp_s.to_numpy(dtype=float)
        vectors = frame[AXES].to_numpy(dtype=float)
        differences = np.diff(times if previous is None else np.r_[previous, times])
        steps.append(differences)
        delays.append(frame.record_timestamp_s.to_numpy(dtype=float) - times)
        previous = float(times[-1])
        first = float(times[0]) if first is None else first
        last = float(times[-1])
        count += len(frame)
        nonfinite += int((~np.isfinite(vectors).all(axis=1)).sum())
        sums += np.nansum(vectors, axis=0)
        squares += np.nansum(vectors**2, axis=0)
    differences = np.concatenate(steps)
    positive = differences[differences > 0]
    means = sums / count
    sha = _sha256(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha,
        "bytes": path.stat().st_size,
        "rows": count,
        "start_s": first,
        "end_s": last,
        "duration_s": last - first,
        "duration_rate_hz": (count - 1) / (last - first),
        "median_instantaneous_rate_hz": 1.0 / np.median(positive),
        "nominal_rate_hz": nominal_rate,
        "duplicate_timestamps": int((differences == 0).sum()),
        "decreasing_timestamps": int((differences < 0).sum()),
        "step_ms_p01_p50_p99_max": (np.quantile(positive, [0.01, 0.5, 0.99, 1.0]) * 1e3).tolist(),
        "intervals_above_three_nominal_periods": int((differences > 3.0 / nominal_rate).sum()),
        "header_record_delta_s_p01_p50_p99": np.quantile(
            np.concatenate(delays), [0.01, 0.5, 0.99]
        ).tolist(),
        "nonfinite_source_rows": nonfinite,
        "mean_six_axis": means.tolist(),
        "std_six_axis": np.sqrt(np.maximum(0.0, squares / count - means**2)).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(interval: list[float], rate: float) -> np.ndarray:
    return np.arange(float(interval[0]), float(interval[1]), 1.0 / rate)


def finite_window_count(values: np.ndarray, rows: int, stride: int) -> int:
    """Count fixed-grid windows whose complete six-axis support is finite."""
    finite = np.isfinite(values).all(axis=1)
    if len(finite) < rows:
        return 0
    missing = np.r_[0, np.cumsum(~finite)]
    starts = np.arange(0, len(finite) - rows + 1, stride)
    return int(((missing[starts + rows] - missing[starts]) == 0).sum())


def fit_affine_clock(
    low_times: np.ndarray,
    low_gyro: np.ndarray,
    high_times: np.ndarray,
    high_gyro: np.ndarray,
    config: dict[str, Any],
    low_gap_s: float,
    high_gap_s: float,
) -> dict[str, Any]:
    """Fit clock drift from predeclared calibration blocks, never evaluation data."""
    rate = float(config["diagnostic_rate_hz"])
    shifts = np.arange(*config["lag_search_s"], float(config["lag_search_step_s"]))
    blocks = []
    for interval in config["train_clock_blocks_s"]:
        if interval[0] < config["fit_interval_s"][0] or interval[1] > config["fit_interval_s"][1]:
            raise ValueError("Affine clock blocks must lie wholly inside the calibration interval")
        shift, score = fit_time_shift(
            low_times,
            low_gyro,
            high_times,
            high_gyro,
            _grid(interval, rate),
            shifts,
            low_gap_s,
            high_gap_s,
        )
        blocks.append(
            {
                "interval_s": interval,
                "center_s": float(np.mean(interval)),
                "shift_s": shift,
                "norm_correlation": score,
            }
        )
    centers = np.array([block["center_s"] for block in blocks])
    observed_shifts = np.array([block["shift_s"] for block in blocks])
    slope, intercept = np.polyfit(centers, observed_shifts, 1)
    errors = observed_shifts - (intercept + slope * centers)
    return {
        "intercept_s": float(intercept),
        "slope_s_per_s": float(slope),
        "clock_scale": float(1.0 / (1.0 - slope)),
        "clock_offset_s": float(intercept / (1.0 - slope)),
        "equation": "corrected_low_relative_s = (low_relative_s + intercept_s) / (1 - slope_s_per_s)",
        "training_blocks": blocks,
        "training_block_residuals_s": errors.tolist(),
        "fit_interval_s": config["fit_interval_s"],
        "fit_uses_validation_or_test": False,
    }


def audit_affine_candidate(
    pair: dict[str, Any],
    low_frame: pd.DataFrame,
    high_frame: pd.DataFrame,
    epoch: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Freeze training-only affine parameters and assess the measured six-axis pair."""
    sensor = pair["sensor"]
    low_times, low_raw = _clock_frame(low_frame, "timestamp_s", epoch)
    high_times, high_values = _clock_frame(high_frame, "timestamp_s", epoch)
    low_rate = float(config["nominal_rates_hz"][sensor])
    low_gap = float(config["max_interpolation_gap_periods"]) / low_rate
    high_gap = float(config["max_interpolation_gap_periods"]) / 200.0
    low_gyro = low_raw[:, :3] @ rotation.T
    clock = fit_affine_clock(
        low_times, low_gyro, high_times, high_values[:, :3], config, low_gap, high_gap
    )
    corrected_times = low_times * clock["clock_scale"] + clock["clock_offset_s"]
    transform = RigidTransform(
        rotation, translation, f"fusionportable_20220209_correct_parent_child_{sensor}"
    )
    clean_source = pd.DataFrame(low_raw, columns=AXES)
    clean_source.insert(0, "timestamp_s", corrected_times)
    lever_config = {**config["lever_arm"], "rate_hz": low_rate}
    compensated = compensate_lever_arm(clean_source, transform, lever_config)
    low_values = compensated[AXES].to_numpy(dtype=float)
    intervals = {
        "train": config["fit_interval_s"],
        "validation": config["validation_interval_s"],
        "test": config["test_interval_s"],
    }
    residual_candidates = np.arange(
        *config["residual_search_s"], float(config["lag_search_step_s"])
    )
    evaluated = {}
    candidates = []
    blocked = []
    if sensor.startswith("ouster"):
        blocked.append("dynamic_timestamp_bursting_and_no_continuous_40Hz_FIR_support")
    rate = 200.0 if sensor.startswith("davis") else 100.0
    paired_arrays = {}
    for split, interval in intervals.items():
        grid = _grid(interval, rate)
        low, low_support = causal_supported_resample(
            corrected_times, low_values, grid, low_rate, rate, config
        )
        high, high_support = causal_supported_resample(
            high_times, high_values, grid, 200.0, rate, config
        )
        metrics = paired_metrics(low, high)
        residual, peak = fit_time_shift(
            corrected_times,
            low_gyro,
            high_times,
            high_values[:, :3],
            _grid(interval, 200.0),
            residual_candidates,
            low_gap,
            high_gap,
        )
        common_grid = _grid(interval, float(config["candidate_rate_hz"]))
        common_low, _ = causal_supported_resample(
            corrected_times,
            low_values,
            common_grid,
            low_rate,
            float(config["candidate_rate_hz"]),
            config,
        )
        common_high, _ = causal_supported_resample(
            high_times, high_values, common_grid, 200.0, float(config["candidate_rate_hz"]), config
        )
        common_metrics = paired_metrics(common_low, common_high)
        common_metrics["finite_2s_windows_stride_100ms"] = finite_window_count(
            np.column_stack([common_low, common_high]),
            round(float(config["candidate_window_s"]) * float(config["candidate_rate_hz"])),
            round(float(config["candidate_window_stride_s"]) * float(config["candidate_rate_hz"])),
        )
        evaluated[split] = {
            "native_rate_metrics": metrics,
            "common_40Hz_metrics": common_metrics,
            "low_support": low_support,
            "high_support": high_support,
            "diagnostic_residual_shift_s_not_applied": residual,
            "diagnostic_residual_norm_correlation": peak,
        }
        paired_arrays[split] = (grid, low, high)
        if split in {"validation", "test"}:
            gyro_r = metrics["correlation"][:3]
            accel_r = metrics["correlation"][3:]
            accel_rmse = metrics["rmse"][3:]
            if any(
                value is None or value < float(config["minimum_gyro_axis_correlation"])
                for value in gyro_r
            ):
                blocked.append(f"{split}_gyro_axis_correlation")
            if any(
                value is None or value < float(config["candidate_minimum_accel_axis_correlation"])
                for value in accel_r
            ):
                blocked.append(f"{split}_accel_axis_correlation")
            if any(
                value is None or value > float(config["candidate_maximum_accel_rmse_mps2"])
                for value in accel_rmse
            ):
                blocked.append(f"{split}_accel_rmse")
            if abs(residual) > float(config["common_grid_maximum_residual_lag_s"]):
                blocked.append(f"{split}_residual_lag_above_40Hz_period")
            if common_metrics["finite_2s_windows_stride_100ms"] == 0:
                blocked.append(f"{split}_no_supported_two_second_windows")
    eligible = not blocked
    if eligible and bool(config.get("export_candidates", False)):
        for split, (grid, low, high) in paired_arrays.items():
            candidate = pd.DataFrame({"timestamp_s": grid + epoch})
            for side, values in (("low", low), ("high", high)):
                for index, axis in enumerate(AXES):
                    candidate[f"{axis}_{side}"] = values[:, index]
            finite = np.isfinite(np.column_stack([low, high])).all(axis=1)
            candidate["training_eligible"] = finite
            candidate["gap_flag"] = ~finite
            candidate["original_sequence_id"] = "mcr_slow"
            candidate["sequence_id"] = f"mcr_slow_{split}"
            candidate["dataset_id"] = "fusionportable"
            candidate["pair_id"] = pair["id"]
            candidate["low_sensor"] = f"DAVIS_MPU6x50_{sensor}"
            candidate["high_sensor"] = "STIM300"
            candidate["source_kind"] = "within_sequence_temporal_measured_six_axis_candidate"
            candidate["split"] = split
            candidate["lever_arm_compensated"] = True
            candidate["calibration_id"] = transform.calibration_id
            candidate["alignment_method"] = (
                "official_parent_child_rotation_training_only_affine_clock_causal_supported_FIR"
            )
            candidate["quality_flags"] = np.where(
                finite,
                "within_sequence_only_time_calibration_requires_transfer_validation",
                "unsupported_source_or_FIR_gap",
            )
            candidate_path = output_root / "candidates" / pair["id"] / f"{split}.parquet"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate.to_parquet(candidate_path, index=False)
            logger.info(
                "WRITES candidate {} with {} supported / {} fixed-grid rows",
                candidate_path,
                int(finite.sum()),
                len(finite),
            )
            candidates.append(
                {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                    "dataset_id": "fusionportable",
                    "pair_id": pair["id"],
                    "sequence_id": f"mcr_slow_{split}",
                    "original_sequence_id": "mcr_slow",
                    "low_sensor": f"DAVIS_MPU6x50_{sensor}",
                    "high_sensor": "STIM300",
                    "rate_hz": rate,
                    "split": split,
                    "rows": len(candidate),
                    "finite_rows": int(finite.sum()),
                    "timestamp_start_s": float(candidate.timestamp_s.iloc[0]),
                    "timestamp_end_s": float(candidate.timestamp_s.iloc[-1]),
                    "source_low_path": str(
                        (Path(config["source_root"]) / "mcr_slow" / f"{sensor}.parquet").resolve()
                    ),
                    "source_high_path": str(
                        (Path(config["source_root"]) / "mcr_slow" / "stim300.parquet").resolve()
                    ),
                    "source_raw_bag": str(Path(config["dynamic_raw_bag"]).resolve()),
                    "clock_calibration": clock,
                    "rotation_body_from_local": rotation.tolist(),
                    "lever_arm_body_m": translation.tolist(),
                    "targets": "actual_measured_STIM300_six_axis",
                    "training_eligible": True,
                    "split_policy": "within_sequence_temporal; same physical intervals for both DAVIS pairs; 5 s guards",
                    "unsupported_rows": "retained_as_NaN_with_training_eligible_false; never_bridge_and_never_train_windows_crossing_them",
                    "claim_scope": "single_mcr_slow_trajectory_pilot_only; no_cross_trajectory_generalization",
                    "validation": evaluated[split],
                }
            )
    else:
        logger.info("SKIPS candidate exports for {}: {}", pair["id"], blocked)
    return {
        "clock": clock,
        "evaluations": evaluated,
        "candidate_training_eligible": eligible,
        "blocking_reasons": blocked,
        "candidate_records": candidates,
    }


def run_audit(config_path: Path) -> dict[str, Any]:
    logger.info("READS experiment config {}", config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    np.random.seed(int(config["seed"]))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "WRITES isolated audit artifacts {}; SKIPS old strict outputs and all raw data mutations",
        output_root,
    )
    snapshot_path = output_root / "config.yaml"
    snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_root, calibration_root = Path(config["source_root"]), Path(config["calibration_root"])
    profiles = {
        sequence: {
            sensor: _profile(source_root / sequence / f"{sensor}.parquet", nominal)
            for sensor, nominal in config["nominal_rates_hz"].items()
        }
        for sequence in ("mcr_slow", "static")
    }
    frames = {}
    for sensor in config["nominal_rates_hz"]:
        path = source_root / "mcr_slow" / f"{sensor}.parquet"
        logger.info("READS complete dynamic vectors {}", path)
        frames[sensor] = pd.read_parquet(path)
    high_frame = frames["stim300"]
    epoch = float(high_frame.timestamp_s.iloc[0])
    rate = float(config["diagnostic_rate_hz"])
    shifts = np.arange(*config["lag_search_s"], float(config["lag_search_step_s"]))
    intervals = {
        "calibration_train": config["fit_interval_s"],
        "validation": config["validation_interval_s"],
        "test": config["test_interval_s"],
    }
    pairs = []
    candidate_records = []
    for pair in config["pairs"]:
        sensor = pair["sensor"]
        logger.info("READS spatial calibration and assesses pair {}", pair["id"])
        document = load_opencv_yaml(calibration_root / pair["extrinsics_file"])
        imu_document = load_opencv_yaml(calibration_root / pair["imu_file"])
        rotation, translation = reference_pose_to_body(document, pair["local_name"])
        reference_from_body = quaternion_wxyz_to_matrix(
            opencv_matrix(document["quaternion_sensor_bodyimu"]).ravel()
        )
        reference_from_local = quaternion_wxyz_to_matrix(
            opencv_matrix(document[f"quaternion_sensor_{pair['local_name']}"]).ravel()
        )
        legacy_rotation = reference_from_body @ reference_from_local.T
        published_direct = float(imu_document["timeshift_sensor_bodyimu"])
        published_chain = float(document["timeshift_sensor_bodyimu"]) - float(
            document[f"timeshift_sensor_{pair['local_name']}"]
        )
        low_gap = float(config["max_interpolation_gap_periods"]) / float(
            config["nominal_rates_hz"][sensor]
        )
        high_gap = float(config["max_interpolation_gap_periods"]) / float(
            config["nominal_rates_hz"]["stim300"]
        )
        clocks = {}
        for column in ("timestamp_s", "record_timestamp_s"):
            low_times, low_values = _clock_frame(frames[sensor], column, epoch)
            high_times, high_values = _clock_frame(high_frame, column, epoch)
            for block in (slice(0, 3), slice(3, 6)):
                low_values[:, block] = low_values[:, block] @ rotation.T
            shift, peak = fit_time_shift(
                low_times,
                low_values[:, :3],
                high_times,
                high_values[:, :3],
                _grid(intervals["calibration_train"], rate),
                shifts,
                low_gap,
                high_gap,
            )
            evaluations = {}
            for split, interval in intervals.items():
                grid = _grid(interval, rate)
                high, _ = bounded_interpolation(high_times, high_values, grid, high_gap)
                low, _ = bounded_interpolation(low_times + shift, low_values, grid, low_gap)
                metrics = paired_metrics(low, high)
                residual_candidates = np.arange(
                    *config["residual_search_s"], float(config["lag_search_step_s"])
                )
                residual, residual_peak = fit_time_shift(
                    low_times + shift,
                    low_values[:, :3],
                    high_times,
                    high_values[:, :3],
                    grid,
                    residual_candidates,
                    low_gap,
                    high_gap,
                )
                metrics.update(
                    {
                        "diagnostic_residual_additive_shift_s": residual,
                        "diagnostic_residual_norm_correlation": residual_peak,
                        "parameters_refitted_for_evaluation": False,
                    }
                )
                evaluations[split] = metrics
            grid = _grid(intervals["calibration_train"], rate)
            high, _ = bounded_interpolation(high_times, high_values, grid, high_gap)
            alternatives = {}
            for label, alternative in {
                "zero": 0.0,
                "direct_published": published_direct,
                "direct_published_sign_reversed": -published_direct,
                "camera_chain": published_chain,
                "camera_chain_sign_reversed": -published_chain,
            }.items():
                low, _ = bounded_interpolation(low_times + alternative, low_values, grid, low_gap)
                alternatives[label] = {"shift_s": alternative, **paired_metrics(low, high)}
            clocks[column] = {
                "fitted_additive_low_timestamp_shift_s": shift,
                "fit_norm_correlation": peak,
                "fit_interval_s": intervals["calibration_train"],
                "evaluations": evaluations,
                "published_shift_diagnostics_train_only": alternatives,
            }
        legacy_aligned, legacy_metrics = align_strict_pair(
            frames[sensor],
            high_frame,
            {
                "target_rate_hz": 100 if sensor.startswith("ouster") else 200,
                "max_interpolation_gap_periods": 3,
                "causal_anti_alias": {"taps": 63, "cutoff_fraction_of_target_nyquist": 0.8},
            },
        )
        legacy_finite = np.isfinite(
            legacy_aligned[
                [f"{axis}_{side}" for side in ("low", "high") for axis in AXES]
            ].to_numpy()
        ).all(axis=1)
        low_times, raw_values = _clock_frame(frames[sensor], "timestamp_s", epoch)
        high_times, high_values = _clock_frame(high_frame, "timestamp_s", epoch)
        train_grid = _grid(intervals["calibration_train"], rate)
        high, _ = bounded_interpolation(high_times, high_values, train_grid, high_gap)
        rotation_candidates = {}
        selected_shift = clocks["timestamp_s"]["fitted_additive_low_timestamp_shift_s"]
        for label, alternative_rotation in {
            "legacy": legacy_rotation,
            "legacy_inverse": legacy_rotation.T,
            "official_parent_child_chain": rotation,
            "identity": np.eye(3),
            "global_gyro_sign_reversed": rotation,
        }.items():
            rotated = raw_values.copy()
            rotated[:, :3] = rotated[:, :3] @ alternative_rotation.T
            rotated[:, 3:] = rotated[:, 3:] @ alternative_rotation.T
            if label == "global_gyro_sign_reversed":
                rotated[:, :3] *= -1.0
            low, _ = bounded_interpolation(low_times + selected_shift, rotated, train_grid, low_gap)
            rotation_candidates[label] = paired_metrics(low, high)
        resampling = {}
        rotated = raw_values.copy()
        rotated[:, :3] = rotated[:, :3] @ rotation.T
        rotated[:, 3:] = rotated[:, 3:] @ rotation.T
        for target_rate in (200.0, float(config["candidate_rate_hz"])):
            grid = _grid([1.0, 51.0], target_rate)
            low, low_support = causal_supported_resample(
                low_times + selected_shift,
                rotated,
                grid,
                float(config["nominal_rates_hz"][sensor]),
                target_rate,
                config,
            )
            high, high_support = causal_supported_resample(
                high_times, high_values, grid, 200.0, target_rate, config
            )
            resampling[str(target_rate)] = {
                "low_support": low_support,
                "high_support": high_support,
                "paired": paired_metrics(low, high),
            }
        affine_candidate = audit_affine_candidate(
            pair, frames[sensor], high_frame, epoch, rotation, translation, config, output_root
        )
        candidate_records.extend(affine_candidate["candidate_records"])
        candidate_eligible = affine_candidate["candidate_training_eligible"]
        pairs.append(
            {
                "pair_id": pair["id"],
                "sensor": sensor,
                "formal_training_eligible": candidate_eligible,
                "recommended_state": "within_sequence_temporal_measured_six_axis_candidate"
                if candidate_eligible
                else (
                    "diagnostic_only_failed_frozen_parameter_or_continuous_window_gate"
                    if not sensor.startswith("ouster")
                    else "excluded_dynamic_timestamp_bursting"
                ),
                "reason": "Training-only temporal correction is assessed on disjoint physical intervals; calibration transfer to independent recordings is not established. Static has no independent motion excitation.",
                "rotation_body_from_local": rotation.tolist(),
                "position_body_to_local_m": translation.tolist(),
                "rotation_source": "official_parent_child_pose_chain_verified_from_upstream_TFGraph",
                "temporal_source": "training_interval_gyro_norm_correlation_only",
                "clock_comparisons": clocks,
                "rotation_sign_diagnostics_train_only": rotation_candidates,
                "resampling_diagnostics": resampling,
                "affine_clock_lever_arm_candidate": affine_candidate,
                "legacy_alignment": {
                    "rows": len(legacy_aligned),
                    "nonfinite_rows": int((~legacy_finite).sum()),
                    "low_resample": legacy_metrics["low_resample"],
                    "high_resample": legacy_metrics["high_resample"],
                },
                "static_recommendation": "independent_sensor_noise_and_bias_analysis_only; correlation_and_lag_not_identifiable_from_stationary_noise",
            }
        )
        logger.info(
            "WRITES metrics for {}: constant header correction {:.3f} s; affine candidate eligible {}",
            pair["id"],
            selected_shift,
            candidate_eligible,
        )
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    raw_sources = {}
    for key in ("dynamic_raw_bag", "static_raw_bag"):
        path = Path(config[key])
        raw_sources[key] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256_source": "existing acquisition manifest; not rehashed during numeric audit",
            "acquisition_manifest": str(
                Path("data/raw/fusionportable_pilot/_manifests/pilot_manifest.json").resolve()
            ),
        }
    calibration_sources = {
        path.name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in calibration_root.glob("*.yaml")
    }
    result = {
        "experiment_id": config["experiment_id"],
        "created_utc": utc_now_iso(),
        "git_commit": git_commit,
        "worktree_state": "dirty_existing_user_work; audit_does_not_modify_shared_legacy_modules",
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "config_snapshot": str(snapshot_path.resolve()),
        "data_manifest_path": str((output_root / "data_manifest.json").resolve()),
        "metrics_path": str((output_root / "metrics.json").resolve()),
        "report_path": str((output_root / "report.md").resolve()),
        "candidate_manifest_path": str((output_root / "candidate_manifest.json").resolve()),
        "intervals_s_relative_to_stim300_start": intervals,
        "guard_between_intervals_s": 5.0,
        "profiles": profiles,
        "pairs": pairs,
        "upstream_sources": config["upstream_sources"],
        "limitations": [
            "No rotation, axis sign, gain, or evaluation shift is optimized using validation/test targets.",
            "Residual lag search on evaluation intervals is a diagnostic only and never applied.",
            "A failed constant-shift audit preceded the affine model audit; the affine drift hypothesis is established from training subintervals, with no fitted validation/test parameters. Evaluation is not a preregistered blind experiment.",
            "Sequence-specific time correction does not establish cross-sequence or deployment clock synchronization.",
            "FIR support analysis uses nominal sensor rates; irregular packet clocks remain a source limitation.",
            "Constant-clock diagnostic acceleration is before lever-arm correction; affine candidate acceleration uses published lever arm and only each low sensor's gyro. Targets remain measured STIM300.",
            "Static inter-sensor independent noise cannot certify spatial/time alignment.",
        ],
    }
    manifest = {
        "experiment_id": config["experiment_id"],
        "git_commit": git_commit,
        "config_snapshot": result["config_snapshot"],
        "metrics_path": result["metrics_path"],
        "raw_sources": raw_sources,
        "calibration_sources": calibration_sources,
        "profiles": profiles,
    }
    (output_root / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "candidate_manifest.json").write_text(
        json.dumps(candidate_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        key: result[key]
        for key in (
            "experiment_id",
            "git_commit",
            "config_snapshot",
            "data_manifest_path",
            "metrics_path",
            "report_path",
            "candidate_manifest_path",
        )
    }
    summary["formal_training_eligible"] = any(p["formal_training_eligible"] for p in pairs)
    summary["pairs"] = [
        {
            "pair_id": p["pair_id"],
            "formal_training_eligible": p["formal_training_eligible"],
            "recommended_state": p["recommended_state"],
            "header_shift_s": p["clock_comparisons"]["timestamp_s"][
                "fitted_additive_low_timestamp_shift_s"
            ],
            "constant_shift_validation": p["clock_comparisons"]["timestamp_s"]["evaluations"][
                "validation"
            ],
            "constant_shift_test": p["clock_comparisons"]["timestamp_s"]["evaluations"]["test"],
            "affine_clock": p["affine_clock_lever_arm_candidate"]["clock"],
            "affine_validation": p["affine_clock_lever_arm_candidate"]["evaluations"]["validation"],
            "affine_test": p["affine_clock_lever_arm_candidate"]["evaluations"]["test"],
            "blocking_reasons": p["affine_clock_lever_arm_candidate"]["blocking_reasons"],
        }
        for p in pairs
    ]
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# FusionPortable independent alignment audit",
        "",
        f"Run: `{config['experiment_id']}`. Git commit: `{git_commit}` (pre-existing dirty worktree).",
        "",
        "Official spatial chain is `R_body_local = R_ref_body.T @ R_ref_local`, `p_body_local = R_ref_body.T @ (p_ref_local - p_ref_body)`. The local legacy parser applies the opposite parent/child convention.",
        "",
        "Timestamp correction is fitted only on 1-25 s. Validation is 30-40 s and test 45-51 s, with 5 s guards. Validation/test diagnostic residual shifts are not applied.",
        "",
        "| Pair | Affine intercept (ms), slope (ppm) | Validation gyro axis r | Test gyro axis r | Test residual (ms) | Formal state |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pair in summary["pairs"]:
        v = pair["affine_validation"]["native_rate_metrics"]
        t = pair["affine_test"]["native_rate_metrics"]
        clock = pair["affine_clock"]
        lines.append(
            f"| {pair['pair_id']} | {clock['intercept_s'] * 1e3:.1f}, {clock['slope_s_per_s'] * 1e6:.1f} | {v['correlation'][:3]} | {t['correlation'][:3]} | {pair['affine_test']['diagnostic_residual_shift_s_not_applied'] * 1e3:.1f} | {pair['recommended_state']} |"
        )
    lines += (
        [
            "",
            "Only candidates passing frozen-parameter six-axis, lag and continuous 2 s window gates are exported, strictly for within-sequence temporal evaluation. Ouster dynamic packet times and static independent noise are excluded from paired supervision.",
            "",
            "Source duration checked across full streams: dynamic approximately 51.7 s; static approximately 12505 s. The bounded 120 s dynamic extraction is naturally identical because the recording is shorter than the bound.",
            "",
            "Numeric diagnostics and causal support counts, alternative published time-shift signs, clock comparisons, source SHA256 hashes, and calibration provenance are in `metrics.json` / `data_manifest.json`. Candidate rows retain unsupported timestamps as NaNs with training_eligible=false so windows cannot bridge them.",
            "",
            "## Limitations",
            "",
        ]
        + [f"- {item}" for item in result["limitations"]]
        + ["", "## Official references", ""]
        + [f"- {url}" for url in result["upstream_sources"]]
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "WRITES summary.json, metrics.json, data_manifest.json, candidate_manifest.json, report.md; SKIPS legacy strict output modifications"
    )
    return summary
