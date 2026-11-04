"""Orchestration and provenance for strict multi-dataset IMU pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.io.multi_imu_calibration import (
    FUSION_CALIBRATION_FILES,
    RigidTransform,
    load_fusion_body_transforms,
    load_penn_body_transforms,
)
from virtual_imu.io.multi_imu_sources import (
    extract_rosbag_imu_topics,
    read_penn_vector_file,
    read_penn_vi_imu,
)
from virtual_imu.io.multi_imu_splits import fusion_temporal_manifest, penn_loso_manifests
from virtual_imu.io.pilot_download import sha256_file
from virtual_imu.preprocessing.imu_quality import evaluate_pair_quality
from virtual_imu.preprocessing.rigid_imu import compensate_lever_arm
from virtual_imu.preprocessing.strict_alignment import (
    align_strict_pair,
    causal_resample,
    clean_timestamps,
)
from virtual_imu.utils.paths import utc_now_iso

STRICT_BASE_COLUMNS = [
    "timestamp_s",
    "gx_low",
    "gy_low",
    "gz_low",
    "ax_low",
    "ay_low",
    "az_low",
    "gx_high",
    "gy_high",
    "gz_high",
    "ax_high",
    "ay_high",
    "az_high",
    "temperature_c",
    "motion_label",
    "sequence_id",
    "source_kind",
    "low_sensor",
    "high_sensor",
    "source_file",
]
STRICT_METADATA_COLUMNS = [
    "dataset_id",
    "pair_id",
    "calibration_id",
    "split_group",
    "alignment_method",
    "lever_arm_compensated",
    "match_dt_ms",
    "quality_flags",
    "training_eligible",
]

SIX_AXIS_COLUMNS = ["gx", "gy", "gz", "ax", "ay", "az"]


@dataclass(frozen=True)
class PreparationContext:
    """Resolved output roots and immutable settings for pair preparation."""

    strict_root: Path
    diagnostic_root: Path
    config_path: Path
    config: dict[str, Any]
    dry_run: bool = False


def prepare_pair(
    pair_config: dict[str, Any],
    context: PreparationContext,
) -> dict[str, Any]:
    """Prepare one sequence-level pair and always emit diagnostic provenance."""

    dataset_id = _safe_segment(pair_config["dataset_id"], "dataset_id")
    pair_id = _safe_segment(pair_config["pair_id"], "pair_id")
    sequence_id = _safe_segment(pair_config["sequence_id"], "sequence_id")
    strict_path = context.strict_root / dataset_id / pair_id / f"{sequence_id}.parquet"
    diagnostic_dir = context.diagnostic_root / dataset_id / pair_id
    diagnostic_path = diagnostic_dir / f"{sequence_id}.parquet"
    metrics_path = diagnostic_dir / f"{sequence_id}.metrics.json"
    record_path = diagnostic_dir / f"{sequence_id}.manifest.json"
    fingerprint = _pair_fingerprint(pair_config, context)

    cached = _verified_cached_record(record_path, fingerprint)
    if cached is not None:
        logger.info("SKIPS verified prepared pair {}/{}/{}", dataset_id, pair_id, sequence_id)
        return cached
    if context.dry_run:
        return {
            "dataset_id": dataset_id,
            "pair_id": pair_id,
            "sequence_id": sequence_id,
            "status": "dry_run",
            "fingerprint": fingerprint,
            "strict_path": str(strict_path),
            "diagnostic_path": str(diagnostic_path),
        }

    low_source = _load_pair_frame(pair_config, "low")
    high_source = _load_pair_frame(pair_config, "high")
    low_clean, low_cleanup = clean_timestamps(low_source)
    high_clean, high_cleanup = clean_timestamps(high_source)
    alignment_config = {
        **dict(context.config.get("alignment", {})),
        **dict(pair_config.get("alignment", {})),
        "target_rate_hz": float(pair_config["target_rate_hz"]),
    }
    raw_aligned, raw_alignment = align_strict_pair(low_clean, high_clean, alignment_config)

    low_transform = pair_config.get("low_transform")
    high_transform = pair_config.get("high_transform")
    calibration_available = bool(pair_config.get("calibration_available", False))
    translation_available = bool(pair_config.get("translation_available", False))
    transforms_valid = _transforms_are_usable(low_transform, high_transform)
    lever_arm_compensated = calibration_available and translation_available and transforms_valid
    if lever_arm_compensated:
        lever_config = {
            **dict(context.config.get("lever_arm", {})),
            **dict(pair_config.get("lever_arm", {})),
        }
        low_transformed = compensate_lever_arm(low_clean, low_transform, lever_config)
        high_transformed = compensate_lever_arm(high_clean, high_transform, lever_config)
    else:
        logger.warning(
            "BLOCKS calibrated transform for {}/{}: calibration={}, translation={}, rotations={}",
            pair_id,
            sequence_id,
            calibration_available,
            translation_available,
            transforms_valid,
        )
        low_transformed = low_clean.copy()
        high_transformed = high_clean.copy()
    aligned, transformed_alignment = align_strict_pair(
        low_transformed,
        high_transformed,
        alignment_config,
    )

    rotation_orthogonality, rotation_determinant = _rotation_metrics(
        low_transform,
        high_transform,
    )
    raw_axis_correlation = _mean_axis_correlation(raw_aligned, SIX_AXIS_COLUMNS)
    transformed_axis_correlation = _mean_axis_correlation(aligned, SIX_AXIS_COLUMNS)
    active_gyro_correlation = _mean_axis_correlation(aligned, ["gx", "gy", "gz"])
    finite_six_axis = bool(
        np.isfinite(
            aligned[
                [f"{column}_{side}" for side in ("low", "high") for column in SIX_AXIS_COLUMNS]
            ].to_numpy(dtype=float)
        ).all()
    )
    target_period_s = 1.0 / float(pair_config["target_rate_hz"])
    raw_metrics = {
        "calibration_available": calibration_available,
        "translation_available": translation_available,
        "quaternion_norm_error": float(pair_config.get("quaternion_norm_error", 0.0)),
        "rotation_orthogonality_fro": rotation_orthogonality,
        "rotation_determinant": rotation_determinant,
        "decreasing_timestamps": int(
            pair_config.get("decreasing_timestamps", 0)
        ),
        "finite_six_axis": finite_six_axis,
        "overlap_fraction": float(transformed_alignment["overlap_fraction"]),
        "match_dt_ms_p99": float(transformed_alignment["match_dt_ms_p99"]),
        "target_period_s": target_period_s,
        "gap_periods_max": max(_gap_periods(low_clean), _gap_periods(high_clean)),
        "output_rows": len(aligned),
        "raw_axis_correlation": raw_axis_correlation,
        "sensor_name_inconsistent": bool(pair_config.get("sensor_name_inconsistent", False)),
    }
    transformed_metrics = {
        "active_gyro_correlation": active_gyro_correlation,
        "transformed_axis_correlation": transformed_axis_correlation,
        "residual_lag_s": float(transformed_alignment["residual_lag"]["lag_s"]),
        "gravity_disagreement_mps2": _gravity_disagreement(aligned),
        "isolated_outlier_fraction": float(pair_config.get("isolated_outlier_fraction", 0.0)),
    }
    quality_thresholds = dict(context.config.get("quality_thresholds", {}))
    quality_thresholds.setdefault(
        "max_interpolation_gap_periods",
        float(alignment_config.get("max_interpolation_gap_periods", 3.0)),
    )
    quality_thresholds["minimum_output_rows"] = int(
        pair_config.get(
            "minimum_output_rows",
            quality_thresholds.get("minimum_output_rows", 1),
        )
    )
    decision = evaluate_pair_quality(raw_metrics, transformed_metrics, quality_thresholds)
    candidate = _strict_candidate(
        aligned,
        pair_config,
        decision.training_eligible,
        decision.blocking_reasons,
        decision.warnings,
        lever_arm_compensated,
        alignment_config,
    )
    diagnostic = _diagnostic_frame(candidate, aligned)
    cap = int(
        context.config.get("diagnostics", {}).get(
            "max_rows_per_pair_sequence",
            200_000,
        )
    )
    seed = int(context.config.get("seed", 0))
    indices = _deterministic_sample_indices(
        len(diagnostic),
        cap,
        seed,
        f"{dataset_id}/{pair_id}/{sequence_id}",
    )
    sampled_diagnostic = diagnostic.iloc[indices].reset_index(drop=True)
    logger.info("WRITES sampled diagnostic pair {}", diagnostic_path)
    _write_parquet_atomic(sampled_diagnostic, diagnostic_path)

    written_strict_path: str | None = None
    if decision.training_eligible:
        logger.info("WRITES strict training pair {}", strict_path)
        _write_parquet_atomic(candidate[[*STRICT_BASE_COLUMNS, *STRICT_METADATA_COLUMNS]], strict_path)
        written_strict_path = str(strict_path)
    elif strict_path.is_file():
        logger.warning("BLOCKS pair and removes stale generated strict output {}", strict_path)
        strict_path.unlink()

    metrics_payload = {
        "created_utc": utc_now_iso(),
        "fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "pair_id": pair_id,
        "sequence_id": sequence_id,
        "source_rows": {"low": len(low_source), "high": len(high_source)},
        "timestamp_cleanup": {"low": low_cleanup, "high": high_cleanup},
        "raw_alignment": raw_alignment,
        "transformed_alignment": transformed_alignment,
        "raw_metrics": raw_metrics,
        "transformed_metrics": transformed_metrics,
        "quality": asdict(decision),
        "source_preparation": pair_config.get("source_preparation"),
        "diagnostic_full_rows": len(diagnostic),
        "diagnostic_sample_rows": len(sampled_diagnostic),
    }
    logger.info("WRITES full pair metrics {}", metrics_path)
    _write_json_atomic(metrics_path, metrics_payload)
    record = {
        "created_utc": utc_now_iso(),
        "fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "pair_id": pair_id,
        "sequence_id": sequence_id,
        "status": "prepared" if decision.training_eligible else "blocked",
        "training_eligible": decision.training_eligible,
        "blocking_reasons": list(decision.blocking_reasons),
        "warnings": list(decision.warnings),
        "rows": len(candidate),
        "start_s": float(candidate["timestamp_s"].iloc[0]),
        "end_s": float(candidate["timestamp_s"].iloc[-1]),
        "split_group": str(pair_config.get("split_group", sequence_id)),
        "source_kind": str(pair_config.get("source_kind", "unknown")),
        "report_metrics": {
            "low_rate_hz": transformed_alignment["low_resample"]["source_rate_hz"],
            "high_rate_hz": transformed_alignment["high_resample"]["source_rate_hz"],
            "output_rate_hz": transformed_alignment["target_rate_hz"],
            "duration_s": transformed_alignment["overlap_duration_s"],
            "overlap_fraction": transformed_alignment["overlap_fraction"],
            "residual_lag_s": transformed_metrics["residual_lag_s"],
            "match_dt_ms_p99": raw_metrics["match_dt_ms_p99"],
            "raw_axis_correlation": raw_axis_correlation,
            "transformed_axis_correlation": transformed_axis_correlation,
            "gravity_disagreement_mps2": transformed_metrics["gravity_disagreement_mps2"],
        },
        "strict_path": written_strict_path,
        "diagnostic_path": str(diagnostic_path),
        "metrics_path": str(metrics_path),
        "record_path": str(record_path),
        "calibration_id": str(pair_config.get("calibration_id", "unavailable")),
        "source_file": str(pair_config.get("source_file", "unknown")),
    }
    logger.info("WRITES pair manifest record {}", record_path)
    _write_json_atomic(record_path, record)
    return record


def prepare_multi_dataset(
    config: dict[str, Any],
    datasets: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prepare all explicitly resolved pair configs and write source-of-truth manifests."""

    strict_root = Path(str(config.get("strict_output_root", "data/processed/multi_imu_strict")))
    diagnostic_root = Path(
        str(config.get("diagnostic_output_root", "data/processed/multi_imu_diagnostics"))
    )
    config_path = Path(str(config.get("_config_path", "configs/multi_dataset_strict_prepare.yaml")))
    context = PreparationContext(strict_root, diagnostic_root, config_path, config, dry_run=dry_run)
    selected = None if datasets is None else set(datasets)
    pair_configs = config.get("preparation_pairs", [])
    if not isinstance(pair_configs, list):
        raise ValueError("preparation_pairs must be a list")
    if not pair_configs and dry_run:
        planned_datasets = datasets or ["penncosyvio", "fusionportable", "navinst"]
        records = [
            {"dataset_id": dataset, "status": "dry_run", "training_eligible": False}
            for dataset in planned_datasets
        ]
        return {
            "created_utc": utc_now_iso(),
            "config_path": str(config_path),
            "dry_run": True,
            "record_count": len(records),
            "eligible_count": 0,
            "records": records,
        }
    if not pair_configs:
        pair_configs = resolve_real_pair_configs(config, datasets=datasets)
    records = []
    for pair_config in pair_configs:
        if not isinstance(pair_config, dict):
            raise ValueError("Each preparation pair must be a mapping")
        if selected is not None and str(pair_config.get("dataset_id")) not in selected:
            logger.info("SKIPS unselected dataset pair {}", pair_config.get("pair_id"))
            continue
        records.append(prepare_pair(pair_config, context))
    records = _discover_pair_records(diagnostic_root, records)
    summary = {
        "created_utc": utc_now_iso(),
        "config_path": str(config_path),
        "dry_run": dry_run,
        "record_count": len(records),
        "eligible_count": sum(bool(record.get("training_eligible")) for record in records),
        "records": records,
    }
    if not dry_run:
        strict_manifest = strict_root / "manifest.json"
        diagnostic_manifest = diagnostic_root / "manifest.json"
        report_path = write_preparation_report(records, strict_root / "report.md")
        split_paths = _write_available_splits(summary, strict_root / "splits", config)
        summary["strict_manifest"] = str(strict_manifest)
        summary["diagnostic_manifest"] = str(diagnostic_manifest)
        summary["report_path"] = str(report_path)
        summary["split_manifests"] = [str(path) for path in split_paths]
        logger.info("WRITES strict source-of-truth manifest {}", strict_manifest)
        _write_json_atomic(strict_manifest, summary)
        logger.info("WRITES diagnostic source-of-truth manifest {}", diagnostic_manifest)
        _write_json_atomic(diagnostic_manifest, summary)
    return summary


def resolve_real_pair_configs(
    config: dict[str, Any],
    datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve downloaded Penn, Fusion, and NavINST sources into pair configs."""

    selected = set(datasets or ("penncosyvio", "fusionportable", "navinst"))
    unknown = selected - {"penncosyvio", "fusionportable", "navinst"}
    if unknown:
        raise ValueError(f"Unknown strict dataset IDs: {sorted(unknown)}")
    pairs: list[dict[str, Any]] = []
    if "penncosyvio" in selected:
        pairs.extend(_resolve_penn_pairs(config))
    if "fusionportable" in selected:
        pairs.extend(_resolve_fusion_pairs(config))
    if "navinst" in selected:
        pairs.extend(_resolve_navinst_pairs(config))
    return pairs


def _resolve_penn_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = _mapping(config, "penncosyvio")
    calibration_path = Path(
        str(section.get("acquired_path", section["calibration_definition"]))
    )
    transforms = load_penn_body_transforms(calibration_path)
    maximum_duration = _optional_duration(config.get("max_duration_s"))
    sequences = section.get("bounded_sequences" if maximum_duration else "sequences", [])
    pairs = []
    for sequence_value in sequences:
        sequence = _safe_segment(sequence_value, "Penn sequence")
        tango_dir = Path(str(section["tango_root"])) / sequence
        vi_dir = Path(str(section["visensor_root"])) / sequence
        source_paths = [
            tango_dir / "accelerometer.txt",
            tango_dir / "gyroscope.txt",
            vi_dir / "imu.txt",
        ]
        accel = read_penn_vector_file(source_paths[0], ["ax", "ay", "az"])
        gyro = read_penn_vector_file(source_paths[1], ["gx", "gy", "gz"])
        high = read_penn_vi_imu(source_paths[2])
        bounded = _bounded_common_frames(
            {"accel": accel, "gyro": gyro, "high": high},
            maximum_duration,
        )
        low, assembly_metrics = _assemble_penn_low(
            bounded["accel"],
            bounded["gyro"],
            dict(config.get("alignment", {})),
        )
        pair_definition = _mapping(section, "pair")
        pairs.append(
            {
                "dataset_id": "penncosyvio",
                "pair_id": str(pair_definition["id"]),
                "sequence_id": sequence,
                "split_group": sequence,
                "source_kind": "trajectory",
                "source_file": ";".join(str(path) for path in source_paths),
                "source_paths": source_paths,
                "calibration_paths": [calibration_path],
                "source_preparation": {"penn_accel_to_gyro": assembly_metrics},
                "low_sensor": str(pair_definition["low_sensor"]),
                "high_sensor": str(pair_definition["high_sensor"]),
                "target_rate_hz": float(pair_definition["target_rate_hz"]),
                "low_frame": low,
                "high_frame": bounded["high"],
                "low_transform": transforms["tango_top_imu"],
                "high_transform": transforms["vi_sensor_imu"],
                "calibration_id": "penncosyvio_published_body_extrinsics_v1",
                "calibration_available": calibration_path.is_file(),
                "translation_available": True,
                "minimum_output_rows": _minimum_output_rows(config),
            }
        )
    return pairs


def _resolve_fusion_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = _mapping(config, "fusionportable")
    calibration_root = Path(str(section["calibration_root"]))
    calibration_paths = [calibration_root / name for name in FUSION_CALIBRATION_FILES]
    transforms = load_fusion_body_transforms(calibration_root)
    maximum_duration = _optional_duration(config.get("max_duration_s"))
    variant = str(config.get("output_variant", "full"))
    cache_root = Path(str(section["source_cache_root"])) / variant
    topic_map = {
        "/os_cloud_node/imu/data_raw": ("ouster_icm20948", "ouster_imu"),
        "/stereo/davis_left/imu/data_raw": ("davis_left", "davis_left_imu"),
        "/stereo/davis_right/imu/data_raw": ("davis_right", "davis_right_imu"),
    }
    sequence_sources = (
        ("mcr_slow", "dynamic", Path(str(section["dynamic_bag"]))),
        ("static", "static", Path(str(section["static_bag"]))),
    )
    pairs: list[dict[str, Any]] = []
    for sequence_id, source_kind, bag_path in sequence_sources:
        specs: dict[str, dict[str, Any]] = {
            str(section["body_topic"]): {"id": "stim300"}
        }
        for pair_definition in section["pairs"]:
            low_topic = str(pair_definition["low_topic"])
            specs[low_topic] = {"id": topic_map[low_topic][0]}
        if maximum_duration is not None:
            for spec in specs.values():
                spec["max_duration_s"] = maximum_duration
        extraction = extract_rosbag_imu_topics(
            bag_path,
            specs,
            cache_root / sequence_id,
            chunk_rows=int(section.get("chunk_rows", 100_000)),
        )
        high_path = Path(str(extraction["topics"]["stim300"]["path"]))
        for definition in section["pairs"]:
            low_topic = str(definition["low_topic"])
            output_id, transform_key = topic_map[low_topic]
            pairs.append(
                {
                    "dataset_id": "fusionportable",
                    "pair_id": str(definition["id"]),
                    "sequence_id": sequence_id,
                    "split_group": sequence_id,
                    "source_kind": source_kind,
                    "source_file": str(bag_path),
                    "source_paths": [bag_path],
                    "calibration_paths": calibration_paths,
                    "source_preparation": extraction,
                    "source_fingerprint": extraction["fingerprint"],
                    "low_sensor": str(definition["low_sensor"]),
                    "high_sensor": str(definition["high_sensor"]),
                    "target_rate_hz": float(definition["target_rate_hz"]),
                    "low_path": extraction["topics"][output_id]["path"],
                    "high_path": high_path,
                    "low_transform": transforms[transform_key],
                    "high_transform": transforms["stim300"],
                    "calibration_id": f"fusionportable_20220209_{transform_key}_to_stim300",
                    "calibration_available": all(path.is_file() for path in calibration_paths),
                    "translation_available": True,
                    "sensor_name_inconsistent": bool(definition.get("upstream_sensor_names")),
                    "minimum_output_rows": _minimum_output_rows(config),
                }
            )
    return pairs


def _resolve_navinst_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    section = _mapping(config, "navinst")
    bag_path = Path(str(section["imu_bag"]))
    maximum_duration = _optional_duration(config.get("max_duration_s"))
    variant = str(config.get("output_variant", "full"))
    specs: dict[str, dict[str, Any]] = {str(section["high_topic"]): {"id": "kvh1750"}}
    for definition in section["pairs"]:
        spec = {"id": f"{definition['id']}_low"}
        if definition.get("accel_scale") is not None:
            spec["accel_scale"] = float(definition["accel_scale"])
        specs[str(definition["low_topic"])] = spec
    if maximum_duration is not None:
        for spec in specs.values():
            spec["max_duration_s"] = maximum_duration
    extraction = extract_rosbag_imu_topics(
        bag_path,
        specs,
        Path(str(section["source_cache_root"])) / variant / "urban03",
        chunk_rows=int(section.get("chunk_rows", 100_000)),
    )
    high_path = extraction["topics"]["kvh1750"]["path"]
    expected_calibration = Path(str(section["expected_path"]))
    pairs = []
    for definition in section["pairs"]:
        output_id = f"{definition['id']}_low"
        pairs.append(
            {
                "dataset_id": "navinst",
                "pair_id": str(definition["id"]),
                "sequence_id": "urban03",
                "split_group": "urban03",
                "source_kind": "dynamic_diagnostic",
                "source_file": str(bag_path),
                "source_paths": [bag_path],
                "source_preparation": extraction,
                "source_fingerprint": extraction["fingerprint"],
                "low_sensor": str(definition["low_sensor"]),
                "high_sensor": str(section["high_sensor"]),
                "target_rate_hz": float(definition["target_rate_hz"]),
                "low_path": extraction["topics"][output_id]["path"],
                "high_path": high_path,
                "low_transform": None,
                "high_transform": None,
                "calibration_id": "navinst_missing_setup_transformation",
                "calibration_available": False,
                "translation_available": False,
                "minimum_output_rows": _minimum_output_rows(config),
                "missing_calibration_path": str(expected_calibration),
            }
        )
    return pairs


def _assemble_penn_low(
    accel: pd.DataFrame,
    gyro: pd.DataFrame,
    alignment_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    accel_clean, accel_cleanup = clean_timestamps(accel)
    gyro_clean, gyro_cleanup = clean_timestamps(gyro)
    accel_six = pd.DataFrame(
        {
            "timestamp_s": accel_clean["timestamp_s"],
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "ax": accel_clean["ax"],
            "ay": accel_clean["ay"],
            "az": accel_clean["az"],
        }
    )
    grid = gyro_clean["timestamp_s"].to_numpy(dtype=float)
    resampled, metrics = causal_resample(accel_six, grid, alignment_config)
    low = gyro_clean[["timestamp_s", "gx", "gy", "gz"]].copy()
    low[["ax", "ay", "az"]] = resampled[["ax", "ay", "az"]].to_numpy()
    valid = ~resampled["gap_flag"].to_numpy() & np.isfinite(
        low[SIX_AXIS_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)
    low = low.loc[valid].reset_index(drop=True)
    return low, {
        "accel_timestamp_cleanup": accel_cleanup,
        "gyro_timestamp_cleanup": gyro_cleanup,
        "resample": metrics,
        "output_rows": len(low),
    }


def _bounded_common_frames(
    frames: dict[str, pd.DataFrame],
    maximum_duration_s: float | None,
) -> dict[str, pd.DataFrame]:
    if maximum_duration_s is None:
        return {key: value.copy() for key, value in frames.items()}
    start_s = max(float(frame["timestamp_s"].iloc[0]) for frame in frames.values())
    available_end = min(float(frame["timestamp_s"].iloc[-1]) for frame in frames.values())
    end_s = min(available_end, start_s + maximum_duration_s)
    if end_s <= start_s:
        raise ValueError("Source streams have no overlap for bounded preparation")
    return {
        key: frame[
            frame["timestamp_s"].between(start_s, end_s, inclusive="both")
        ].reset_index(drop=True)
        for key, frame in frames.items()
    }


def _minimum_output_rows(config: dict[str, Any]) -> int:
    return int(config.get("quality_thresholds", {}).get("minimum_output_rows", 100))


def _optional_duration(value: Any) -> float | None:
    if value is None:
        return None
    duration = float(value)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("max_duration_s must be finite and positive")
    return duration


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    result = values.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"Config section {key} must be a mapping")
    return result


def _discover_pair_records(
    diagnostic_root: Path,
    current_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    if diagnostic_root.is_dir():
        for path in sorted(diagnostic_root.glob("*/*/*.manifest.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("WARNS unreadable pair manifest during aggregate rebuild {}", path)
                continue
            if not isinstance(record, dict):
                continue
            key = _record_key(record)
            if key is not None:
                merged[key] = record
    for record in current_records:
        key = _record_key(record)
        if key is not None:
            merged[key] = record
    return [merged[key] for key in sorted(merged)]


def _record_key(record: dict[str, Any]) -> tuple[str, str, str] | None:
    values = tuple(str(record.get(key, "")) for key in ("dataset_id", "pair_id", "sequence_id"))
    return values if all(values) else None


def _write_available_splits(
    strict_manifest: dict[str, Any],
    output_root: Path,
    config: dict[str, Any],
) -> list[Path]:
    paths: list[Path] = []
    try:
        paths.extend(penn_loso_manifests(strict_manifest, output_root))
    except ValueError as exc:
        logger.warning("WARNS Penn LOSO manifest not written: {}", exc)
    guard_s = float(config.get("splits", {}).get("fusion", {}).get("guard_s", 10.0))
    try:
        paths.append(fusion_temporal_manifest(strict_manifest, output_root, guard_s))
    except ValueError as exc:
        logger.warning("WARNS Fusion temporal manifest not written: {}", exc)
    return paths


def write_preparation_report(records: list[dict[str, Any]], path: Path) -> Path:
    """Write a compact Markdown table from immutable pair manifest records."""

    headers = [
        "Dataset",
        "Pair",
        "Sequence",
        "Rates low/high/out (Hz)",
        "Rows",
        "Duration (s)",
        "Overlap",
        "Lag (ms)",
        "P99 match (ms)",
        "Raw corr",
        "Transformed corr",
        "Gravity diff",
        "Warnings",
        "Blocks",
        "Strict path",
    ]
    lines = [
        "# Strict Multi-Dataset IMU Preparation Report",
        "",
        f"Generated: {utc_now_iso()}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for record in records:
        metrics = record.get("report_metrics", {})
        rates = "/".join(
            _format_metric(metrics.get(key), 2)
            for key in ("low_rate_hz", "high_rate_hz", "output_rate_hz")
        )
        row = [
            str(record.get("dataset_id", "")),
            str(record.get("pair_id", "")),
            str(record.get("sequence_id", "")),
            rates,
            str(record.get("rows", "")),
            _format_metric(metrics.get("duration_s"), 2),
            _format_metric(metrics.get("overlap_fraction"), 4),
            _format_metric(_scaled(metrics.get("residual_lag_s"), 1e3), 3),
            _format_metric(metrics.get("match_dt_ms_p99"), 3),
            _format_metric(metrics.get("raw_axis_correlation"), 4),
            _format_metric(metrics.get("transformed_axis_correlation"), 4),
            _format_metric(metrics.get("gravity_disagreement_mps2"), 4),
            ", ".join(str(value) for value in record.get("warnings", [])) or "-",
            ", ".join(str(value) for value in record.get("blocking_reasons", [])) or "-",
            str(record.get("strict_path") or "-"),
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    logger.info("WRITES preparation report {}", path)
    _write_text_atomic(path, "\n".join(lines) + "\n")
    return path


def provenance_fingerprint(
    config_path: Path,
    source_paths: list[Path],
    calibration_paths: list[Path],
) -> str:
    """Hash config, source, and calibration contents with deterministic path labels."""

    digest = hashlib.sha256()
    for category, paths in (
        ("config", [config_path]),
        ("source", source_paths),
        ("calibration", calibration_paths),
    ):
        for path in sorted((item.resolve() for item in paths), key=str):
            if not path.is_file():
                raise FileNotFoundError(f"Fingerprint input is missing: {path}")
            digest.update(category.encode("ascii"))
            digest.update(str(path).encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _deterministic_sample_indices(
    total_rows: int,
    maximum_rows: int,
    seed: int,
    identity: str,
) -> np.ndarray:
    if total_rows < 0 or maximum_rows <= 0:
        raise ValueError("total_rows must be non-negative and maximum_rows positive")
    if total_rows <= maximum_rows:
        return np.arange(total_rows, dtype=np.int64)
    identity_seed = int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8],
        byteorder="little",
    )
    generator = np.random.default_rng(np.uint64(seed) ^ np.uint64(identity_seed))
    return np.sort(generator.choice(total_rows, size=maximum_rows, replace=False))


def _load_pair_frame(pair_config: dict[str, Any], side: str) -> pd.DataFrame:
    frame_value = pair_config.get(f"{side}_frame")
    if isinstance(frame_value, pd.DataFrame):
        logger.info("READS in-memory {} pair frame", side)
        return frame_value.copy()
    path_value = pair_config.get(f"{side}_path")
    if path_value is None:
        raise ValueError(f"Pair config requires {side}_frame or {side}_path")
    path = Path(str(path_value))
    logger.info("READS {} pair stream {}", side, path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _pair_fingerprint(pair_config: dict[str, Any], context: PreparationContext) -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for relative in (
        "io/multi_imu_prepare.py",
        "io/multi_imu_calibration.py",
        "io/multi_imu_sources.py",
        "preprocessing/rigid_imu.py",
        "preprocessing/strict_alignment.py",
        "preprocessing/imu_quality.py",
    ):
        code_path = package_root / relative
        digest.update(relative.encode("ascii"))
        digest.update(sha256_file(code_path).encode("ascii"))
    if context.config_path.is_file():
        digest.update(provenance_fingerprint(context.config_path, [], []).encode("ascii"))
    scalars = {
        key: value
        for key, value in pair_config.items()
        if key
        not in {
            "low_frame",
            "high_frame",
            "low_transform",
            "high_transform",
            "source_paths",
            "calibration_paths",
            "source_preparation",
        }
    }
    digest.update(json.dumps(_json_ready(scalars), sort_keys=True).encode("utf-8"))
    for side in ("low", "high"):
        frame = pair_config.get(f"{side}_frame")
        if isinstance(frame, pd.DataFrame):
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        path = pair_config.get(f"{side}_path")
        if path is not None:
            digest.update(sha256_file(Path(str(path))).encode("ascii"))
        transform = pair_config.get(f"{side}_transform")
        if isinstance(transform, RigidTransform):
            digest.update(transform.rotation_body_from_sensor.tobytes())
            digest.update(transform.position_body_to_sensor_m.tobytes())
            digest.update(transform.calibration_id.encode("utf-8"))
    for category, paths in (
        ("source", pair_config.get("source_paths", [])),
        ("calibration", pair_config.get("calibration_paths", [])),
    ):
        for value in sorted(paths, key=str):
            path = Path(str(value))
            if not path.is_file():
                raise FileNotFoundError(f"Pair fingerprint input is missing: {path}")
            stat = path.stat()
            digest.update(category.encode("ascii"))
            digest.update(str(path.resolve()).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            if stat.st_size <= 100 * 1024 * 1024 or category == "calibration":
                digest.update(sha256_file(path).encode("ascii"))
            else:
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _verified_cached_record(record_path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if record.get("fingerprint") != fingerprint:
        return None
    declared = [record.get("diagnostic_path"), record.get("metrics_path"), record.get("record_path")]
    if record.get("training_eligible"):
        declared.append(record.get("strict_path"))
    if not all(value and Path(str(value)).is_file() for value in declared):
        return None
    return {**record, "status": "verified_skip"}


def _transforms_are_usable(low: Any, high: Any) -> bool:
    return isinstance(low, RigidTransform) and isinstance(high, RigidTransform) and all(
        _proper_rotation(transform.rotation_body_from_sensor) for transform in (low, high)
    )


def _proper_rotation(rotation: np.ndarray) -> bool:
    return bool(
        rotation.shape == (3, 3)
        and np.isfinite(rotation).all()
        and np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro") < 1e-5
        and np.linalg.det(rotation) > 0.999
    )


def _rotation_metrics(low: Any, high: Any) -> tuple[float, float]:
    transforms = [value for value in (low, high) if isinstance(value, RigidTransform)]
    if len(transforms) != 2:
        return 1.0, 0.0
    orthogonality = max(
        float(
            np.linalg.norm(
                transform.rotation_body_from_sensor.T @ transform.rotation_body_from_sensor
                - np.eye(3),
                ord="fro",
            )
        )
        for transform in transforms
    )
    determinant = min(
        float(np.linalg.det(transform.rotation_body_from_sensor)) for transform in transforms
    )
    return orthogonality, determinant


def _mean_axis_correlation(aligned: pd.DataFrame, columns: list[str]) -> float:
    correlations = []
    for column in columns:
        low = aligned[f"{column}_low"].to_numpy(dtype=float)
        high = aligned[f"{column}_high"].to_numpy(dtype=float)
        valid = np.isfinite(low) & np.isfinite(high)
        if valid.sum() < 3 or np.std(low[valid]) <= 1e-10 or np.std(high[valid]) <= 1e-10:
            continue
        correlations.append(float(np.corrcoef(low[valid], high[valid])[0, 1]))
    return 0.0 if not correlations else float(np.mean(correlations))


def _gap_periods(frame: pd.DataFrame) -> float:
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    steps = np.diff(timestamps)
    return 0.0 if len(steps) == 0 else float(np.max(steps) / np.median(steps))


def _gravity_disagreement(aligned: pd.DataFrame) -> float:
    low = aligned[["ax_low", "ay_low", "az_low"]].to_numpy(dtype=float)
    high = aligned[["ax_high", "ay_high", "az_high"]].to_numpy(dtype=float)
    difference = np.abs(np.linalg.norm(low, axis=1) - np.linalg.norm(high, axis=1))
    finite = difference[np.isfinite(difference)]
    return float(np.median(finite)) if len(finite) else 1e9


def _strict_candidate(
    aligned: pd.DataFrame,
    pair_config: dict[str, Any],
    training_eligible: bool,
    blocking_reasons: tuple[str, ...],
    warnings: tuple[str, ...],
    lever_arm_compensated: bool,
    alignment_config: dict[str, Any],
) -> pd.DataFrame:
    result = aligned[["timestamp_s"]].copy()
    for side in ("low", "high"):
        for column in SIX_AXIS_COLUMNS:
            result[f"{column}_{side}"] = aligned[f"{column}_{side}"].to_numpy()
    result["temperature_c"] = np.nan
    result["motion_label"] = pair_config.get("motion_label")
    result["sequence_id"] = str(pair_config["sequence_id"])
    result["source_kind"] = str(pair_config.get("source_kind", "unknown"))
    result["low_sensor"] = str(pair_config["low_sensor"])
    result["high_sensor"] = str(pair_config["high_sensor"])
    result["source_file"] = str(pair_config.get("source_file", "unknown"))
    result["dataset_id"] = str(pair_config["dataset_id"])
    result["pair_id"] = str(pair_config["pair_id"])
    result["calibration_id"] = str(pair_config.get("calibration_id", "unavailable"))
    result["split_group"] = str(pair_config.get("split_group", pair_config["sequence_id"]))
    interpolation = str(alignment_config.get("interpolation", "linear"))
    result["alignment_method"] = f"causal_fir_{interpolation}"
    result["lever_arm_compensated"] = lever_arm_compensated
    result["match_dt_ms"] = aligned["match_dt_ms"].to_numpy()
    result["quality_flags"] = json.dumps(
        {"blocking": list(blocking_reasons), "warnings": list(warnings)},
        sort_keys=True,
    )
    result["training_eligible"] = training_eligible
    return result[[*STRICT_BASE_COLUMNS, *STRICT_METADATA_COLUMNS]]


def _diagnostic_frame(candidate: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    result = candidate.copy()
    for column in (
        "low_match_dt_ms",
        "high_match_dt_ms",
        "low_interpolated",
        "high_interpolated",
        "interpolation_flag",
        "low_gap",
        "high_gap",
        "gap_flag",
        "out_of_range_flag",
    ):
        if column == "out_of_range_flag":
            result[column] = False
        else:
            result[column] = aligned[column].to_numpy()
    return result


def _safe_segment(value: Any, label: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or path.drive or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a single safe path segment: {value}")
    return path.name


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _scaled(value: Any, factor: float) -> float | None:
    return None if value is None else float(value) * factor


def _format_metric(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)
