"""Build a portable, traceable corpus with separate measured and navigation labels."""
# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from virtual_imu.preprocessing.smartpnt_native import native_smartpnt_frame
from virtual_imu.preprocessing.unified_signals import regularize_segments
from virtual_imu.utils.paths import git_commit, project_root_from_file

LOW = [f"{axis}_low" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]
HIGH = [column.replace("_low", "_high") for column in LOW]
SMART_INPUT = [f"{axis}_radps" for axis in ("gx", "gy", "gz")] + [
    f"{axis}_mps2" for axis in ("ax", "ay", "az")
]
POSITION = [f"gt_{axis}_local_m" for axis in ("x", "y", "z")]
VELOCITY = [f"gt_v_{axis}_mps" for axis in ("east", "north", "up")]
ATTITUDE = [f"gt_{angle}_deg" for angle in ("heading", "pitch", "roll")]
SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def columns_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns].reset_index(drop=True), index=False)
    return hashlib.sha256(values.to_numpy().tobytes()).hexdigest()


def measurement_digest(frame: pd.DataFrame) -> str:
    """Hash measured/time/label content independent of sequence/source metadata."""
    columns = [
        column
        for column in frame
        if column == "timestamp_s"
        or column in LOW + HIGH + SMART_INPUT + POSITION + VELOCITY + ATTITUDE
        or column.startswith("gt_")
    ]
    if not columns or "timestamp_s" not in columns:
        raise ValueError("Cannot deduplicate without timestamps and measurement columns")
    selected = frame[sorted(columns)].reset_index(drop=True)
    digest = hashlib.sha256(json.dumps(list(selected.columns)).encode())
    digest.update(pd.util.hash_pandas_object(selected, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("WRITES {}", path)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def _source_path(root: Path, value: str) -> Path:
    # Configs made on Windows are also readable after an archive is moved to Linux.
    return root / Path(str(value).replace("\\", "/"))


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sources(config: dict[str, Any], root: Path, hashes: dict[str, str]):
    for dataset, definition in config["datasets"].items():
        if dataset == "alimu":
            split_by_sequence = {}
            for split, value in definition.get("split_sources", {}).items():
                split_path = _source_path(root, value)
                logger.info("READS ALiMU split identities {}", split_path)
                hashes[_relative(split_path, root)] = file_sha256(split_path)
                ids = pd.read_parquet(split_path, columns=["sequence_id"]).sequence_id.unique()
                for identity in ids:
                    if str(identity) in split_by_sequence:
                        raise ValueError(
                            f"ALiMU source sequence occurs in multiple splits: {identity}"
                        )
                    split_by_sequence[str(identity)] = split
            path = _source_path(root, definition["source"])
            logger.info("READS ALiMU real paired source {}", path)
            frames = {
                str(identity): frame
                for identity, frame in pd.read_parquet(path).groupby("sequence_id", sort=True)
            }
            recordings = {}
            lookup = {}
            for identity, frame in frames.items():
                match = re.search(r"real_sensor_data_((?:dyn|stat)_\d+)", identity)
                if not match:
                    raise ValueError(f"ALiMU recording identity unavailable: {identity}")
                recordings[identity] = match.group(1)
                lookup[(str(frame.low_sensor.iloc[0]), match.group(1))] = identity
            recording_splits = {}
            for name, identities in definition.get("recording_splits", {}).items():
                for identity in identities:
                    if identity in recording_splits:
                        raise ValueError("ALiMU physical recording assigned multiple splits")
                    recording_splits[identity] = name
            assignments = {}
            for identity, frame in frames.items():
                if "source_kind" in frame and not frame.source_kind.eq("real").all():
                    raise ValueError("Unified ALiMU view requires measured real data")
                recording = recordings[identity]
                excluded = definition.get("exclude_recordings", {}).get(recording)
                split = (
                    recording_splits.get(recording)
                    if recording_splits
                    else split_by_sequence.get(identity)
                )
                reference_sha = columns_digest(frame, ["timestamp_s", *HIGH])
                frame.attrs["reference_recording_id"] = recording
                frame.attrs["reference_measurement_sha256"] = reference_sha
                frame.attrs["physical_group_id"] = f"alimu/VN100/{reference_sha}"
                if excluded:
                    kept_id = lookup[(str(frame.low_sensor.iloc[0]), excluded)]
                    kept = frames[kept_id]
                    fields = ["timestamp_s", *LOW]
                    prefix = kept.iloc[: len(frame)]
                    if len(frame) >= len(kept) or not np.array_equal(
                        frame[fields].to_numpy(), prefix[fields].to_numpy()
                    ):
                        raise ValueError(
                            f"Configured ALiMU duplicate is not a strict input/time prefix: {identity}"
                        )
                    frame.attrs["archive_duplicate"] = {
                        "canonical_sequence_id": kept_id,
                        "canonical_recording_id": excluded,
                        "reason": "shorter_timestamp_and_low_IMU_prefix_with_reference_variant",
                        "input_measurement_sha256": columns_digest(frame, fields),
                        "canonical_prefix_sha256": columns_digest(prefix, fields),
                        "reference_prefix_equal": bool(
                            np.array_equal(frame[HIGH].to_numpy(), prefix[HIGH].to_numpy())
                        ),
                        "reference_prefix_max_difference": np.max(
                            abs(frame[HIGH].to_numpy() - prefix[HIGH].to_numpy()), axis=0
                        ).tolist(),
                    }
                elif split not in SPLITS:
                    raise ValueError(f"Unassigned ALiMU sequence: {identity}")
                else:
                    if reference_sha in assignments and assignments[reference_sha] != split:
                        raise ValueError(
                            "ALiMU reference recording occurs in multiple split assignments"
                        )
                    assignments[reference_sha] = split
                yield (
                    dataset,
                    definition,
                    path,
                    str(identity),
                    split,
                    frame,
                )
        elif dataset == "penncosyvio":
            upstream = {}
            if definition.get("upstream_manifest"):
                upstream_path = _source_path(root, definition["upstream_manifest"])
                logger.info("READS Penn upstream provenance {}", upstream_path)
                hashes[_relative(upstream_path, root)] = file_sha256(upstream_path)
                upstream = {
                    r["sequence_id"]: r
                    for r in json.loads(upstream_path.read_text(encoding="utf-8"))["records"]
                    if r["dataset_id"] == dataset and r["pair_id"] == "tango_top_to_adis16488"
                }
            for split, identities in definition["splits"].items():
                for identity in identities:
                    path = _source_path(root, definition["root"]) / f"{identity}.parquet"
                    logger.info("READS strict Penn pair {}", path)
                    frame = pd.read_parquet(path)
                    if (
                        "training_eligible" not in frame
                        or not frame.training_eligible.eq(True).all()
                    ):
                        raise ValueError(f"Ineligible Penn source {path}")
                    if upstream:
                        previous = upstream[identity]
                        metrics_path = _source_path(root, previous["metrics_path"])
                        logger.info("READS Penn inherited filter delays {}", metrics_path)
                        hashes[_relative(metrics_path, root)] = file_sha256(metrics_path)
                        inherited = json.loads(metrics_path.read_text(encoding="utf-8"))
                        aligned = inherited["transformed_alignment"]
                        gyro_delay = float(aligned["low_resample"]["algorithmic_latency_s"])
                        accel_stage = inherited["source_preparation"]["penn_accel_to_gyro"][
                            "resample"
                        ]
                        accel_delay = gyro_delay + float(accel_stage["algorithmic_latency_s"])
                        high_delay = float(aligned["high_resample"]["algorithmic_latency_s"])
                        frame.attrs["inherited_latency_s"] = {
                            **{c: gyro_delay for c in LOW[:3]},
                            **{c: accel_delay for c in LOW[3:]},
                            **dict.fromkeys(HIGH, high_delay),
                        }
                        frame.attrs["upstream_metrics_path"] = _relative(metrics_path, root)
                        frame.attrs["upstream_metrics_sha256"] = hashes[
                            _relative(metrics_path, root)
                        ]
                        frame.attrs["latency_scope"] = (
                            "known resampling waits/group delays; lever-arm causal phase response is not a constant delay"
                        )
                    trim = float(definition.get("upstream_warmup_trim_s", 0))
                    if trim:
                        frame["_label_valid"] = (
                            frame.timestamp_s >= frame.timestamp_s.iloc[0] + trim
                        )
                        frame.attrs["upstream_warmup_trim_s"] = trim
                    yield dataset, definition, path, identity, split, frame
        elif dataset == "smartpnt_msf":
            mapping = {
                identity: split
                for split, identities in definition["splits"].items()
                for identity in identities
            }
            if definition.get("imu_root"):
                for path in sorted(
                    _source_path(root, definition["imu_root"]).glob("*_imu_raw.parquet")
                ):
                    identity = path.name.removesuffix("_imu_raw.parquet")
                    gt_path = (
                        _source_path(root, definition["groundtruth_root"])
                        / f"{identity}_groundtruth.parquet"
                    )
                    logger.info(
                        "READS native SmartPNT IMU {} and ECEF navigation truth {}", path, gt_path
                    )
                    hashes[_relative(gt_path, root)] = file_sha256(gt_path)
                    frame = native_smartpnt_frame(pd.read_parquet(path), pd.read_parquet(gt_path))
                    frame.attrs["navigation_source_path"] = _relative(gt_path, root)
                    frame.attrs["navigation_source_sha256"] = hashes[_relative(gt_path, root)]
                    yield dataset, definition, path, identity, mapping.get(identity), frame
                continue
            for path in sorted(
                _source_path(root, definition["root"]).glob("*_imu_gt_aligned.parquet")
            ):
                identity = path.name.removesuffix("_imu_gt_aligned.parquet")
                logger.info("READS aligned SmartPNT navigation source {}", path)
                yield (
                    dataset,
                    definition,
                    path,
                    identity,
                    mapping.get(identity),
                    pd.read_parquet(path),
                )
        else:
            raise ValueError(f"Unsupported source adapter {dataset}")


def _normalization(records: list[dict[str, Any]], staging: Path, output: Path, root: Path) -> dict:
    result = {}
    for task in ("paired_signal", "navigation"):
        selected = [r for r in records if r["task"] == task and r["split"] == "train"]
        if not selected:
            continue
        columns = LOW + (HIGH if task == "paired_signal" else POSITION + VELOCITY)
        total = 0
        sums, squares = np.zeros(len(columns)), np.zeros(len(columns))
        for record in selected:
            path = staging / _source_path(root, record["path"]).relative_to(output)
            values = pd.read_parquet(path, columns=columns).to_numpy(dtype=np.float64)
            total += len(values)
            sums += values.sum(axis=0)
            squares += np.square(values).sum(axis=0)
        mean = sums / total
        std = np.sqrt(np.maximum(squares / total - mean**2, 0))
        result[task] = {
            "scope": "training_records_only",
            "rows": total,
            "columns": columns,
            "mean": mean.tolist(),
            "std": np.where(std < 1e-8, 1.0, std).tolist(),
        }
    return result


def prepare_unified_dataset(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build a new immutable dataset version; refuse accidental overwrite."""
    root = (project_root or project_root_from_file(__file__)).resolve()
    output = _source_path(root, config["output_root"]).resolve()
    if not output.is_relative_to(root / "data/processed"):
        raise ValueError("output_root must be inside project data/processed")
    if output.exists():
        raise FileExistsError(f"Dataset version already exists; choose a new output_root: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"An incomplete build exists: {staging}; inspect before recovery")
    staging.mkdir(parents=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")  # noqa: UP017
    run = root / "runs" / f"{config['dataset_id']}_build_{stamp}"
    run.mkdir(parents=True)
    config_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    for destination in (staging / "config_snapshot.yaml", run / "config.yaml"):
        logger.info("WRITES config snapshot {}", destination)
        destination.write_bytes(config_text.encode("utf-8"))
    hashes: dict[str, str] = {}
    records, duplicates, skipped = [], [], []
    canonical = {}
    for dataset, definition, path, identity, split, frame in _sources(config, root, hashes):
        source_rel = _relative(path, root)
        if source_rel not in hashes:
            hashes[source_rel] = file_sha256(path)
        digest = measurement_digest(frame)
        if frame.attrs.get("archive_duplicate"):
            duplicates.append(
                {
                    "dataset_id": dataset,
                    "original_sequence_id": identity,
                    "source_path": source_rel,
                    "source_file_sha256": hashes[source_rel],
                    "measurement_sha256": digest,
                    "action": "retained_original_excluded_from_training_export",
                    **frame.attrs["archive_duplicate"],
                }
            )
            logger.info("SKIPS archived partial input duplicate {} {}", dataset, identity)
            continue
        duplicate_key = (dataset, digest)
        if duplicate_key in canonical:
            duplicate = {
                "dataset_id": dataset,
                "original_sequence_id": identity,
                "canonical_sequence_id": canonical[duplicate_key],
                "measurement_sha256": digest,
                "source_path": source_rel,
                "source_file_sha256": hashes[source_rel],
                "action": "retained_original_excluded_from_training_export",
            }
            duplicates.append(duplicate)
            logger.info("SKIPS duplicate {} {} -> {}", dataset, identity, canonical[duplicate_key])
            continue
        canonical[duplicate_key] = identity
        if split not in SPLITS:
            raise ValueError(f"Unassigned unique source sequence {dataset}/{identity}")
        task = "navigation" if dataset == "smartpnt_msf" else "paired_signal"
        original_rows = len(frame)
        if task == "navigation":
            frame = frame.rename(columns={old: LOW[i] for i, old in enumerate(SMART_INPUT)})
            policy = config.get("navigation_ambiguity", "Fixed")
            if policy:
                if "gt_ambiguity_status" not in frame:
                    raise ValueError("Navigation ambiguity policy requires gt_ambiguity_status")
                frame["_label_valid"] = frame.get(
                    "_label_valid", True
                ) & frame.gt_ambiguity_status.eq(policy)
            labels = POSITION + VELOCITY + ATTITUDE
            signals = LOW
        else:
            signals, labels = LOW + HIGH, []
        segments, metrics = regularize_segments(
            frame,
            signals,
            rate_hz=float(config["rate_hz"]),
            native_rate_hz=float(definition["native_rate_hz"]),
            min_rows=int(config.get("window_size", 80)),
            fir_taps=int(config.get("fir_taps", 63)),
            max_gap_periods=float(config.get("max_gap_periods", 1.5)),
            label_columns=labels,
            angle_columns=ATTITUDE if labels else [],
        )
        stage_delay = metrics["availability_latency_s"]
        inherited_delays = frame.attrs.get("inherited_latency_s", {})
        total_delays = {
            column: stage_delay + inherited_delays.get(column, 0.0) for column in signals
        }
        metrics["incremental_availability_latency_s"] = stage_delay
        metrics["inherited_latency_s"] = inherited_delays
        metrics["total_known_latency_by_channel_s"] = total_delays
        metrics["availability_latency_s"] = max(total_delays.values())
        metrics["latency_scope"] = frame.attrs.get(
            "latency_scope", "known resampling waits and linear FIR group delays"
        )
        if task == "navigation":
            native_times = frame.timestamp_s.to_numpy(dtype=float)
            position = frame[POSITION].to_numpy(dtype=float)
            velocity = frame[VELOCITY].to_numpy(dtype=float)
            closure = np.linalg.norm(
                np.diff(position, axis=0)
                - (velocity[1:] + velocity[:-1]) * 0.5 * np.diff(native_times)[:, None],
                axis=1,
            )
            valid = frame._label_valid.to_numpy(dtype=bool)
            selected_closure = closure[valid[1:] & valid[:-1]]
            metrics["navigation_label_quality"] = {
                "position_velocity_step_residual_p99_m": float(np.quantile(selected_closure, 0.99))
                if len(selected_closure)
                else None,
                "position_velocity_step_residual_median_m": float(np.median(selected_closure))
                if len(selected_closure)
                else None,
                "convention": frame.attrs.get("navigation_frame", {}).get(
                    "convention", "legacy_aligned_input"
                ),
            }
        if not segments:
            skipped.append(
                {
                    "dataset_id": dataset,
                    "sequence_id": identity,
                    "reason": "no_valid_windows",
                    "metrics": metrics,
                }
            )
            logger.warning("SKIPS sequence without enough valid samples {}/{}", dataset, identity)
        for number, part in enumerate(segments):
            part["available_timestamp_s"] = part.timestamp_s + metrics["availability_latency_s"]
            safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", identity)
            segment_id = f"{dataset}__{safe}__seg{number:04d}"
            part["sequence_id"] = segment_id
            part["original_sequence_id"] = identity
            part["dataset_id"] = dataset
            part["task_view"] = task
            part["has_signal_label"] = task == "paired_signal"
            part["has_navigation_label"] = task == "navigation"
            part["training_eligible"] = True
            part["sample_rate_hz"] = float(config["rate_hz"])
            low_sensor = (
                str(part.low_sensor.iloc[0])
                if "low_sensor" in part
                else definition.get("low_sensor", "SmartPNT_IMU_unspecified_grade")
            )
            high_sensor = (
                str(part.high_sensor.iloc[0])
                if "high_sensor" in part
                else definition.get("high_sensor", "navigation_groundtruth")
            )
            pair = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{low_sensor}_to_{high_sensor}")
            part["low_sensor"], part["high_sensor"], part["pair_id"] = low_sensor, high_sensor, pair
            if task == "navigation":
                for index, old in enumerate(SMART_INPUT):
                    new = LOW[index]
                    part[old] = part[new]  # Compatible with the existing real-GT loader.
                for column in ATTITUDE:
                    part[column.replace("_deg", "_rad")] = np.deg2rad(part[column])
            else:
                for reference in ("VN100", "ADIS16488", "STIM300"):
                    part[f"reference_{reference.lower()}"] = float(high_sensor == reference)
            suffix = "_imu_gt_aligned" if task == "navigation" else ""
            relative = Path("sequences") / task / dataset / f"{segment_id}{suffix}.parquet"
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            logger.info("WRITES {} rows to {}", len(part), destination)
            part.to_parquet(destination, index=False, compression="zstd")
            sha = file_sha256(destination)
            record = {
                "dataset_id": dataset,
                "pair_id": pair,
                "sequence_id": segment_id,
                "original_sequence_id": identity,
                "task": task,
                "task_view": task,
                "split": split,
                "path": _relative(output / relative, root),
                "strict_path": _relative(output / relative, root),
                "sha256": sha,
                "fingerprint": sha,
                "training_eligible": True,
                "rows": len(part),
                "start_s": float(part.timestamp_s.iloc[0]),
                "end_s": float(part.timestamp_s.iloc[-1]),
                "rate_hz": float(config["rate_hz"]),
                "low_sensor": low_sensor,
                "high_sensor": high_sensor,
                "reference_frame": definition.get("reference_frame", "upstream_corresponding_axes"),
                "source_path": source_rel,
                "source_sha256": hashes[source_rel],
                "source_rows": original_rows,
                "source_measurement_sha256": digest,
                "source_metadata": frame.attrs,
                "physical_group_id": frame.attrs.get("physical_group_id", f"{dataset}/{identity}"),
                "reference_recording_id": frame.attrs.get("reference_recording_id", identity),
                "reference_measurement_sha256": frame.attrs.get("reference_measurement_sha256"),
                "preprocessing": metrics,
            }
            records.append(record)
    for task in ("paired_signal", "navigation"):
        selected = [r for r in records if r["task"] == task]
        groups = {
            name: {r["physical_group_id"] for r in selected if r["split"] == name}
            for name in SPLITS
        }
        if any(
            groups[a] & groups[b]
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError("Source sequence leakage across splits")
        split_manifest = {
            "task_view": task,
            "split_kind": "leave_one_sequence_out",
            "group_key": "physical_recording",
            "splits": {
                name: [
                    {
                        **r,
                        "group_value": r["physical_group_id"],
                        "row_filter": "all",
                    }
                    for r in selected
                    if r["split"] == name
                ]
                for name in SPLITS
            },
        }
        filename = "paired_default.json" if task == "paired_signal" else "navigation_default.json"
        _write_json(staging / "splits" / filename, split_manifest)
    label_schema = {
        "schema_version": 1,
        "signal_units": {"gyro": "rad/s", "accel": "m/s^2"},
        "rate_hz": config["rate_hz"],
        "tasks": {
            "paired_signal": {
                "input_columns": LOW,
                "target_columns": HIGH,
                "label_kind": "measured_reference_IMU",
                "condition_columns": [
                    "reference_vn100",
                    "reference_adis16488",
                    "reference_stim300",
                ],
            },
            "navigation": {
                "input_columns": LOW,
                "target_columns": POSITION + VELOCITY + ATTITUDE,
                "label_kind": "navigation_groundtruth_not_high_grade_IMU",
                "position_unit": "m",
                "velocity_unit": "m/s",
                "attitude_unit": "degree; additional _rad columns in radians",
                "window_label": "position_delta: end-start in local ENU metres",
            },
        },
    }
    implementation = {}
    for p in [
        Path(__file__),
        Path(__file__).parents[1] / "preprocessing/unified_signals.py",
        Path(__file__).parents[1] / "preprocessing/smartpnt_native.py",
    ]:
        if p.is_relative_to(root):
            implementation[_relative(p, root)] = file_sha256(p)
    source_state = source_state_fingerprint(root)
    _write_json(run / "source_state.json", source_state)
    manifest = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "dataset_root": _relative(output, root),
        "rate_hz": config["rate_hz"],
        "build_run": _relative(run, root),
        "git_commit": git_commit(root),
        "implementation_sha256": implementation,
        "source_state_sha256": source_state["sha256"],
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "config_source": _relative(config_path, root) if config_path else None,
        "source_files": hashes,
        "records": records,
        "duplicates": duplicates,
        "skipped": skipped,
        "exclusions": config.get("exclusions", []),
        "counts": {
            task: {
                split: sum(r["rows"] for r in records if r["task"] == task and r["split"] == split)
                for split in SPLITS
            }
            for task in ("paired_signal", "navigation")
        },
    }
    _write_json(
        staging / "strict_manifest.json",
        {
            "records": [r for r in records if r["task"] == "paired_signal"],
            "build_run": manifest["build_run"],
        },
    )
    _write_json(
        staging / "archive/duplicates.json",
        {"duplicates": duplicates, "original_files_retained": True},
    )
    _write_json(staging / "label_schema.json", label_schema)
    _write_json(staging / "normalization.json", _normalization(records, staging, output, root))
    _write_json(staging / "manifest.json", manifest)
    _write_json(run / "data_manifest.json", manifest)
    _write_json(
        run / "metrics.json",
        {
            "counts": manifest["counts"],
            "records": len(records),
            "duplicates_excluded": len(duplicates),
        },
    )
    report = [
        "# 统一 IMU 数据集构建报告",
        "",
        f"数据集：{config['dataset_id']}；共同采样率：{config['rate_hz']} Hz。",
        "",
        "| 任务 | 训练采样 | 验证采样 | 测试采样 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for task, counts in manifest["counts"].items():
        report.append(f"| {task} | {counts['train']} | {counts['validation']} | {counts['test']} |")
    report += [
        "",
        f"去重归档 {len(duplicates)} 条；原始文件保留。记录 {len(records)} 个有限值、无跨间断窗口的片段。",
        "",
        "paired_signal 标签为实测参考器件；navigation 标签为导航真值，两者不能混作同一种监督。",
        "",
        "不同平台保留各自的参考坐标系。共同单位/速率不代表全球统一的物理坐标轴。ALiMU 使用上游配对轴约定，未额外宣称已获独立外参验证。",
        "",
        "SmartPNT 默认仅保留 Fixed ambiguity 标签段。这是保守数据筛选，不能证明所有导航真值无误差。",
        "",
        "SmartPNT 正式配置从原生 200 Hz IMU 直接抗混叠降至 40 Hz；导航位置及速度由 ECEF 转换为各轨迹固定原点 ENU，保留上游局部字段以便审计。IMU 单位保留原解析器的增量/dt 解释，器件等级未被推定。",
        "",
        "降采样含匹配的 FIR 低通和端点裁剪；有效信号时间已补偿群延迟，available_timestamp_s 记录新阶段及继承的已知重采样等待/群延迟，不能等同于设备完整物理相位响应。",
        "",
        "原生数据、源文件哈希、处理实现哈希和配置快照均保留；重新构建应使用新版本路径。",
    ]
    (run / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    logger.info("WRITES immutable dataset {}", output)
    staging.replace(output)
    verification = verify_unified_dataset(output / "manifest.json", root)
    _write_json(run / "verification.json", verification)
    return manifest


def source_state_fingerprint(root: Path) -> dict[str, Any]:
    """Fingerprint actual tracked and untracked implementation/config contents."""
    files = {}
    for folder in ("src", "scripts", "configs"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml", ".json"}:
                if "__pycache__" not in path.parts:
                    files[_relative(path, root)] = file_sha256(path)
    for name in ("pyproject.toml", "uv.lock", "AGENTS.md"):
        if (root / name).is_file():
            files[name] = file_sha256(root / name)
    try:
        result = subprocess.run(
            ["git", "-c", "core.quotepath=false", "status", "--porcelain"],
            cwd=root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        status = result.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        status = ["git_state_unavailable"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "sha256": digest,
        "files": files,
        "git_commit": git_commit(root),
        "git_status": status,
        "includes_untracked_source": True,
    }


def verify_unified_dataset(manifest_path: Path, root: Path | None = None, *, validate_payloads: bool = True) -> dict[str, Any]:
    """Verify full exported records, temporal support and physical label availability."""
    root = (root or project_root_from_file(__file__)).resolve()
    status_path = manifest_path.parent / "dataset_status.json"
    if (
        status_path.is_file()
        and json.loads(status_path.read_text(encoding="utf-8")).get("status")
        == "superseded_diagnostic"
    ):
        raise ValueError("Dataset is superseded diagnostic; use its declared replacement")
    logger.info("READS dataset verification manifest {}", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_sha256(manifest_path.parent / "config_snapshot.yaml") != manifest["config_sha256"]:
        raise ValueError("Dataset config snapshot checksum mismatch")
    count = 0
    source_assignments = {}
    for record in manifest["records"]:
        path = _source_path(root, record["path"])
        if not path.resolve().is_relative_to(root):
            raise ValueError("Record path escapes project")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
        task = record["task"]
        if validate_payloads:
            logger.info("READS full record validation {}", path)
            frame = pd.read_parquet(path)
            required = LOW + (HIGH if task == "paired_signal" else POSITION + VELOCITY + ATTITUDE)
            if len(frame) != record["rows"] or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
                raise ValueError(f"Invalid row count or required numeric label in {path}")
            if not frame.training_eligible.eq(True).all():
                raise ValueError(f"Ineligible record {path}")
            if not np.allclose(np.diff(frame.timestamp_s), 1 / float(manifest["rate_hz"]), atol=1e-7, rtol=0):
                raise ValueError(f"Non-uniform exported sampling in {path}")
            if task == "navigation" and (frame.has_signal_label.any() or any(c in frame for c in HIGH)):
                raise ValueError("Navigation truth cannot masquerade as measured IMU labels")
            if task == "paired_signal" and not frame.has_signal_label.all():
                raise ValueError("Measured paired records must have signal labels")
        key = (record["dataset_id"], record["original_sequence_id"])
        if key in source_assignments and source_assignments[key] != record["split"]:
            raise ValueError("Physical source sequence crosses split assignments")
        source_assignments[key] = record["split"]
        physical = (
            record["dataset_id"],
            record.get("physical_group_id", record["original_sequence_id"]),
        )
        if physical in source_assignments and source_assignments[physical] != record["split"]:
            raise ValueError("Physical reference recording crosses split assignments")
        source_assignments[physical] = record["split"]
        count += int(record["rows"])
    return {
        "passed": True,
        "checked_records": len(manifest["records"]),
        "checked_rows": count,
        "sha256": file_sha256(manifest_path),
        "payloads_validated": validate_payloads,
        "checks": ["config_checksum", "file_hashes", "source_split_disjointness"] + (["finite_labels", "uniform_sampling", "task_label_separation"] if validate_payloads else []),
    }
