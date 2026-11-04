"""Leakage-safe split manifests for strict multi-IMU Parquet records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from virtual_imu.utils.paths import utc_now_iso

PENN_SEQUENCES = ("af", "as", "bf", "bs")
SPLIT_NAMES = ("train", "validation", "test")


def penn_loso_manifests(
    strict_manifest: dict[str, Any],
    output_root: Path,
) -> list[Path]:
    """Write four Penn leave-one-sequence-out folds over immutable strict files."""

    records = _eligible_records(strict_manifest, "penncosyvio")
    by_sequence: dict[str, list[dict[str, Any]]] = {sequence: [] for sequence in PENN_SEQUENCES}
    for record in records:
        sequence = str(record.get("sequence_id"))
        if sequence in by_sequence:
            by_sequence[sequence].append(record)
    missing = [sequence for sequence, values in by_sequence.items() if not values]
    if missing:
        raise ValueError(f"Cannot generate Penn LOSO; missing eligible sequences: {missing}")

    output_dir = output_root / "penn_loso"
    paths: list[Path] = []
    for test_sequence in PENN_SEQUENCES:
        if test_sequence == "bs":
            validation_sequence = "bf"
            fold_id = "default_bs"
        else:
            index = PENN_SEQUENCES.index(test_sequence)
            validation_sequence = PENN_SEQUENCES[(index - 1) % len(PENN_SEQUENCES)]
            fold_id = f"test_{test_sequence}"
        train_sequences = [
            sequence
            for sequence in PENN_SEQUENCES
            if sequence not in {test_sequence, validation_sequence}
        ]
        manifest = {
            "created_utc": utc_now_iso(),
            "split_kind": "leave_one_sequence_out",
            "fold_id": fold_id,
            "dataset_id": "penncosyvio",
            "group_key": "sequence_id",
            "eligibility_required": True,
            "splits": {
                "train": _penn_entries(by_sequence, train_sequences),
                "validation": _penn_entries(by_sequence, [validation_sequence]),
                "test": _penn_entries(by_sequence, [test_sequence]),
            },
        }
        validate_no_split_leakage(manifest)
        path = output_dir / f"{fold_id}.json"
        logger.info("WRITES Penn LOSO split manifest {}", path)
        _write_json_atomic(path, manifest)
        paths.append(path)
    return paths


def fusion_temporal_manifest(
    strict_manifest: dict[str, Any],
    output_root: Path,
    guard_s: float,
) -> Path:
    """Write shared guarded temporal boundaries for all Fusion sensor pairs."""

    guard = float(guard_s)
    if not np.isfinite(guard) or guard < 0.0:
        raise ValueError("guard_s must be finite and non-negative")
    records = _eligible_records(strict_manifest, "fusionportable")
    dynamic = [record for record in records if not _is_static_record(record)]
    static = [record for record in records if _is_static_record(record)]
    if not dynamic:
        raise ValueError("Cannot generate Fusion split without eligible dynamic records")
    pair_ids = {str(record["pair_id"]) for record in dynamic}
    if len(pair_ids) < 2:
        raise ValueError("Fusion comparison requires at least two eligible dynamic pair IDs")
    start_s = max(float(record["start_s"]) for record in dynamic)
    end_s = min(float(record["end_s"]) for record in dynamic)
    usable_duration = end_s - start_s - 2.0 * guard
    if usable_duration <= 0.0:
        raise ValueError("Fusion dynamic overlap is too short for two guard intervals")
    train_end = start_s + 0.60 * usable_duration
    validation_start = train_end + guard
    validation_end = validation_start + 0.20 * usable_duration
    test_start = validation_end + guard
    boundaries = {
        "train": (start_s, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, end_s),
    }
    splits = {
        split: [_temporal_entry(record, *boundaries[split]) for record in dynamic]
        for split in SPLIT_NAMES
    }
    splits["train"].extend(_static_entry(record) for record in static)
    manifest = {
        "created_utc": utc_now_iso(),
        "split_kind": "within_sequence_temporal",
        "dataset_id": "fusionportable",
        "group_key": "timestamp_s",
        "guard_s": guard,
        "fractions_after_guards": [0.60, 0.20, 0.20],
        "shared_dynamic_overlap": {"start_s": start_s, "end_s": end_s},
        "eligibility_required": True,
        "static_training_only": True,
        "splits": splits,
    }
    validate_no_split_leakage(manifest)
    path = output_root / "fusion_guarded_temporal.json"
    logger.info("WRITES Fusion guarded temporal split manifest {}", path)
    _write_json_atomic(path, manifest)
    return path


def validate_no_split_leakage(manifest: dict[str, Any]) -> None:
    """Reject group overlap, temporal overlap, guard violations, and ineligible entries."""

    splits = manifest.get("splits")
    if not isinstance(splits, dict) or any(name not in splits for name in SPLIT_NAMES):
        raise ValueError("Split manifest requires train, validation, and test lists")
    for split_name in SPLIT_NAMES:
        entries = splits[split_name]
        if not isinstance(entries, list):
            raise ValueError(f"Split {split_name} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or not bool(entry.get("training_eligible", False)):
                raise ValueError(f"Split {split_name} contains an ineligible entry")

    split_kind = str(manifest.get("split_kind"))
    if split_kind == "leave_one_sequence_out":
        groups = {
            name: {str(entry["group_value"]) for entry in splits[name]}
            for name in SPLIT_NAMES
        }
        if (
            groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]
        ):
            raise ValueError("LOSO group overlap detected between splits")
        return
    if split_kind != "within_sequence_temporal":
        raise ValueError(f"Unsupported split_kind: {split_kind}")

    guard = float(manifest.get("guard_s", 0.0))
    intervals: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for split_name in SPLIT_NAMES:
        for entry in splits[split_name]:
            if entry.get("sequence_role") == "static_training_only":
                if split_name != "train":
                    raise ValueError("Static Fusion records may appear only in train")
                continue
            start = float(entry["time_start_s"])
            end = float(entry["time_end_s"])
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError("Temporal split entry must have a finite positive interval")
            key = (str(entry["pair_id"]), str(entry["sequence_id"]))
            intervals.setdefault(key, {})[split_name] = (start, end)
    for key, by_split in intervals.items():
        if set(by_split) != set(SPLIT_NAMES):
            raise ValueError(f"Temporal pair {key} is missing a split interval")
        train = by_split["train"]
        validation = by_split["validation"]
        test = by_split["test"]
        if validation[0] - train[1] < guard - 1e-9:
            raise ValueError(f"Temporal guard violation between train and validation for {key}")
        if test[0] - validation[1] < guard - 1e-9:
            raise ValueError(f"Temporal guard violation between validation and test for {key}")


def _eligible_records(
    strict_manifest: dict[str, Any],
    dataset_id: str,
) -> list[dict[str, Any]]:
    records = strict_manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Strict manifest records must be a list")
    return [
        record
        for record in records
        if isinstance(record, dict)
        and str(record.get("dataset_id")) == dataset_id
        and bool(record.get("training_eligible", False))
        and bool(record.get("strict_path"))
    ]


def _penn_entries(
    by_sequence: dict[str, list[dict[str, Any]]],
    sequences: list[str],
) -> list[dict[str, Any]]:
    entries = []
    for sequence in sequences:
        for record in by_sequence[sequence]:
            entries.append(
                {
                    "path": str(record["strict_path"]),
                    "dataset_id": "penncosyvio",
                    "pair_id": str(record["pair_id"]),
                    "sequence_id": sequence,
                    "group_key": "sequence_id",
                    "group_value": sequence,
                    "row_filter": "all",
                    "training_eligible": True,
                    "fingerprint": record.get("fingerprint"),
                }
            )
    return entries


def _temporal_entry(
    record: dict[str, Any],
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    return {
        "path": str(record["strict_path"]),
        "dataset_id": "fusionportable",
        "pair_id": str(record["pair_id"]),
        "sequence_id": str(record["sequence_id"]),
        "time_column": "timestamp_s",
        "time_start_s": start_s,
        "time_end_s": end_s,
        "start_inclusive": True,
        "end_inclusive": False,
        "training_eligible": True,
        "fingerprint": record.get("fingerprint"),
    }


def _static_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **_temporal_entry(record, float(record["start_s"]), float(record["end_s"])),
        "sequence_role": "static_training_only",
        "end_inclusive": True,
    }


def _is_static_record(record: dict[str, Any]) -> bool:
    return str(record.get("source_kind", "")).lower() == "static" or str(
        record.get("sequence_id", "")
    ).lower().startswith("static")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
