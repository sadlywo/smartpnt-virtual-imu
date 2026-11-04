"""Strict manifest-native loading for training, validation, and test frames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.io.multi_imu_splits import SPLIT_NAMES, validate_no_split_leakage
from virtual_imu.utils.paths import project_root_from_file

IDENTITY_FIELDS = ("dataset_id", "pair_id", "sequence_id")


@dataclass(frozen=True)
class TrainingSplitBundle:
    frames: dict[str, pd.DataFrame]
    split_manifest: dict[str, Any]
    strict_manifest: dict[str, Any]
    split_manifest_path: Path
    strict_manifest_path: Path
    input_records: tuple[dict[str, Any], ...]


def load_training_split_manifest(
    split_manifest_path: Path,
    strict_manifest_path: Path,
    *,
    split_names: list[str] | tuple[str, ...] | None = None,
) -> TrainingSplitBundle:
    """Validate strict split provenance and load its three data frames."""

    project_root = project_root_from_file(__file__)
    resolved_split_path = _resolve_project_path(split_manifest_path, project_root)
    resolved_strict_path = _resolve_project_path(strict_manifest_path, project_root)
    split_manifest = _read_manifest(resolved_split_path, "split")
    strict_manifest = _read_manifest(resolved_strict_path, "strict")
    validate_no_split_leakage(split_manifest)
    strict_records = _strict_record_index(strict_manifest)

    frames: dict[str, pd.DataFrame] = {}
    input_records: list[dict[str, Any]] = []
    selected = tuple(SPLIT_NAMES if split_names is None else split_names)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(SPLIT_NAMES):
        raise ValueError("split_names must select distinct declared train/validation/test roles")
    logger.info("SKIPS unrequested split payloads {}", sorted(set(SPLIT_NAMES) - set(selected)))
    for split_name in selected:
        record_frames = []
        for record in split_manifest["splits"][split_name]:
            identity = _record_identity(record, f"Split {split_name}")
            strict_record = strict_records.get(identity)
            if strict_record is None:
                raise ValueError(
                    f"Split {split_name} has no matching strict manifest record for {identity}"
                )
            _validate_strict_match(record, strict_record, split_name, project_root)
            input_path = _resolve_project_path(record["path"], project_root)
            if not input_path.is_file():
                raise FileNotFoundError(f"Split {split_name} Parquet path does not exist: {input_path}")

            logger.info("READS {} split Parquet {}", split_name, input_path)
            frame = pd.read_parquet(input_path)
            _require_training_eligible_rows(frame, input_path)
            frame = _preserve_record_identity(frame, record, input_path)
            record_frames.append(_apply_row_filter(frame, record, input_path))
            input_records.append(dict(record))
        frames[split_name] = pd.concat(record_frames, ignore_index=True) if record_frames else pd.DataFrame()

    return TrainingSplitBundle(
        frames=frames,
        split_manifest=split_manifest,
        strict_manifest=strict_manifest,
        split_manifest_path=resolved_split_path,
        strict_manifest_path=resolved_strict_path,
        input_records=tuple(input_records),
    )


def _read_manifest(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind.capitalize()} manifest does not exist: {path}")
    logger.info("READS {} manifest {}", kind, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{kind.capitalize()} manifest root must be an object")
    return payload


def _strict_record_index(
    strict_manifest: dict[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = strict_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Strict manifest records must be a list")
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Strict manifest records must be objects")
        identity = _record_identity(record, "Strict manifest")
        if identity in index:
            raise ValueError(f"Strict manifest contains duplicate record identity {identity}")
        index[identity] = record
    return index


def _record_identity(record: dict[str, Any], context: str) -> tuple[str, str, str]:
    values = tuple(str(record.get(field, "")).strip() for field in IDENTITY_FIELDS)
    missing = [field for index, field in enumerate(IDENTITY_FIELDS) if not values[index]]
    if missing:
        raise ValueError(f"{context} record is missing identity fields: {missing}")
    return values


def _validate_strict_match(
    record: dict[str, Any],
    strict_record: dict[str, Any],
    split_name: str,
    project_root: Path,
) -> None:
    fingerprint = record.get("fingerprint")
    strict_fingerprint = strict_record.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint or fingerprint != strict_fingerprint:
        raise ValueError(
            f"Split {split_name} fingerprint does not match its strict manifest record"
        )
    if not _is_true(strict_record.get("training_eligible")):
        raise ValueError(f"Split {split_name} matched an ineligible strict manifest record")
    if not record.get("path") or not strict_record.get("strict_path"):
        raise ValueError(f"Split {split_name} record and strict manifest record require paths")
    split_path = _resolve_project_path(record["path"], project_root)
    strict_path = _resolve_project_path(strict_record["strict_path"], project_root)
    if split_path != strict_path:
        raise ValueError(f"Split {split_name} path does not match its strict manifest record")


def _require_training_eligible_rows(frame: pd.DataFrame, path: Path) -> None:
    if "training_eligible" not in frame.columns:
        raise ValueError(f"Loaded Parquet is missing training_eligible: {path}")
    eligible = frame["training_eligible"].map(_is_true)
    if not bool(eligible.all()):
        raise ValueError(f"Loaded Parquet contains rows where training_eligible is not true: {path}")


def _preserve_record_identity(
    frame: pd.DataFrame,
    record: dict[str, Any],
    path: Path,
) -> pd.DataFrame:
    result = frame.copy()
    for field in IDENTITY_FIELDS:
        expected = str(record[field])
        if field in result.columns and not result[field].astype("string").eq(expected).fillna(False).all():
            raise ValueError(f"Loaded Parquet {field} does not match manifest record: {path}")
        result[field] = expected
    return result


def _apply_row_filter(
    frame: pd.DataFrame,
    record: dict[str, Any],
    path: Path,
) -> pd.DataFrame:
    row_filter = record.get("row_filter")
    if row_filter is not None and not (
        row_filter == "all"
        or (isinstance(row_filter, dict) and row_filter.get("type") == "all")
    ):
        raise ValueError(f"Unsupported row_filter for {path}: {row_filter!r}")

    has_start = "time_start_s" in record
    has_end = "time_end_s" in record
    if not has_start and not has_end:
        return frame.reset_index(drop=True)
    if not has_start or not has_end:
        raise ValueError(f"Temporal row filter for {path} requires time_start_s and time_end_s")

    time_column = str(record.get("time_column", "timestamp_s"))
    if time_column not in frame.columns:
        raise ValueError(f"Temporal row filter column {time_column!r} is missing from {path}")
    start = float(record["time_start_s"])
    end = float(record["time_end_s"])
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError(f"Temporal row filter for {path} requires finite increasing bounds")
    timestamps = pd.to_numeric(frame[time_column], errors="coerce")
    if bool(record.get("start_inclusive", True)):
        mask = timestamps.ge(start)
    else:
        mask = timestamps.gt(start)
    if bool(record.get("end_inclusive", False)):
        mask &= timestamps.le(end)
    else:
        mask &= timestamps.lt(end)
    return frame.loc[mask].reset_index(drop=True)


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _is_true(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)
