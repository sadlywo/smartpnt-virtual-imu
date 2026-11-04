"""Frozen unified-IMU loading and recording-disjoint protocol assembly."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

LOW = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
ROLES = ("train", "validation", "test")
PAIR_COLUMNS = ["dataset_id", "pair_id", "low_sensor", "high_sensor"]
SUPPORT_COLUMNS = [
    "timestamp_s",
    *PAIR_COLUMNS,
    "sequence_id",
    "original_sequence_id",
    *LOW,
    *HIGH,
]

LOGGER = logging.getLogger(__name__)
PROTOCOL_IDS = ("pooled", "alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf")
PENN_RECORDINGS = ("af", "as", "bf", "bs")


def _recording(record: dict[str, Any]) -> str:
    if record["dataset_id"] == "alimu":
        match = re.search(r"(?:dyn|stat)_\d+", str(record.get("reference_recording_id", "")))
        if not match:
            raise ValueError("ALiMU record lacks reference_recording_id")
        return match.group(0)
    return str(record["original_sequence_id"])


def _physical(record: dict[str, Any]) -> str:
    value = record.get("physical_group_id")
    if not value:
        raise ValueError("Every frozen record requires physical_group_id")
    return str(value)


def _protocol_specs(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    paired = [
        record
        for record in records
        if record.get("task") == "paired_signal" and record.get("training_eligible") is True
    ]
    alimu = [record for record in paired if record["dataset_id"] == "alimu"]
    penn = [record for record in paired if record["dataset_id"] == "penncosyvio"]
    if not alimu or not penn:
        raise ValueError("Both ALiMU and PennCOSYVIO paired records are required")

    def add(identifier: str, values: list[dict[str, Any]], assignment: dict[str, str] | None = None):
        splits = {role: [] for role in ROLES}
        for record in values:
            role = assignment[_recording(record)] if assignment else str(record["split"])
            if role not in ROLES:
                raise ValueError(f"Unknown split role: {role}")
            splits[role].append(record)
        if any(not entries for entries in splits.values()):
            raise ValueError(f"Protocol {identifier} has an empty split")
        physical = {role: {_physical(record) for record in entries} for role, entries in splits.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if physical[left] & physical[right]:
                raise ValueError(f"Protocol {identifier} leaks a physical recording")
        return {"splits": splits, "protocol_kind": "recording_disjoint"}

    specs = {
        "pooled": add("pooled", alimu + penn),
        "alimu_only": add("alimu_only", alimu),
        "penn_only": add("penn_only", penn),
    }
    for index, test in enumerate(PENN_RECORDINGS):
        validation = PENN_RECORDINGS[(index - 1) % len(PENN_RECORDINGS)]
        assignment = {
            name: "test" if name == test else "validation" if name == validation else "train"
            for name in PENN_RECORDINGS
        }
        specs[f"penn_loso_{test}"] = add(f"penn_loso_{test}", penn, assignment)
    return specs


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a mapping")
    for key in ("experiment_id", "data", "protocols", "search", "output", "seed"):
        if key not in config:
            raise ValueError(f"Experiment configuration is missing {key}")
    return config


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass
class ProtocolData:
    protocol_id: str
    frames: dict[str, pd.DataFrame]
    records: dict[str, list[dict[str, Any]]]
    audit: dict[str, Any]


class UnifiedDataset:
    """Load only the frozen paired records required by selected protocols."""

    def __init__(self, project_root: Path, manifest_path: str | Path, *, verify_hashes: bool = True):
        self.project_root = project_root.resolve()
        self.manifest_path = resolve(self.project_root, manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.rate_hz = float(self.manifest["rate_hz"])
        self.records = list(self.manifest["records"])
        self.verify_hashes = bool(verify_hashes)
        self._frames: dict[str, pd.DataFrame] = {}
        self._verified: set[str] = set()
        self.specs = _protocol_specs(self.records)

    def load_record(self, record: dict[str, Any]) -> pd.DataFrame:
        path = resolve(self.project_root, record["path"])
        key = str(path)
        if key in self._frames:
            return self._frames[key].copy(deep=False)
        if not path.is_file():
            raise FileNotFoundError(f"Frozen dataset record is missing: {path}")
        if self.verify_hashes and key not in self._verified:
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Frozen dataset record checksum differs: {path}")
            self._verified.add(key)
        frame = pd.read_parquet(path, columns=SUPPORT_COLUMNS)
        missing = sorted(set(SUPPORT_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Record {record['sequence_id']} is missing columns: {missing}")
        frame = frame.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        if len(frame) != int(record["rows"]):
            raise ValueError(f"Record {record['sequence_id']} row count differs from manifest")
        timestamps = frame["timestamp_s"].to_numpy(float)
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Record {record['sequence_id']} has duplicate/non-increasing time")
        if not np.allclose(np.diff(timestamps), 1.0 / self.rate_hz, atol=1e-6, rtol=1e-5):
            raise ValueError(f"Record {record['sequence_id']} is not continuous at {self.rate_hz} Hz")
        values = frame[[*LOW, *HIGH]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Record {record['sequence_id']} contains nonfinite IMU values")
        for column in ["sequence_id", "original_sequence_id", *PAIR_COLUMNS]:
            if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
                raise ValueError(f"Record {record['sequence_id']} has empty identity field {column}")
        self._frames[key] = frame
        return frame.copy(deep=False)

    def protocol(self, protocol_id: str, *, min_segment_rows: int = 320, stride: int = 40) -> ProtocolData:
        if protocol_id not in self.specs:
            raise ValueError(f"Unknown protocol: {protocol_id}")
        spec = self.specs[protocol_id]
        pieces: dict[str, list[pd.DataFrame]] = {role: [] for role in ROLES}
        audits: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
        for role in ROLES:
            for record in spec["splits"][role]:
                frame = self.load_record(record)
                n = len(frame)
                if n < min_segment_rows:
                    audits[role].append(
                        {
                            "sequence_id": record["sequence_id"],
                            "rows": n,
                            "skipped": True,
                            "reason": f"shorter_than_{min_segment_rows}",
                        }
                    )
                    LOGGER.info(
                        "SKIPS protocol {} {} short sequence {} (<{} rows)",
                        protocol_id,
                        role,
                        record["sequence_id"],
                        min_segment_rows,
                    )
                    continue
                keep = n if role == "train" else (n // stride) * stride
                selected = frame.iloc[:keep].copy()
                selected["split"] = role
                pieces[role].append(selected)
                audits[role].append(
                    {
                        "sequence_id": record["sequence_id"],
                        "original_sequence_id": record["original_sequence_id"],
                        "rows": n,
                        "eligible_rows": keep,
                        "skipped_rows": n - keep,
                        "skipped": False,
                    }
                )
        frames = {}
        for role in ROLES:
            if not pieces[role]:
                raise ValueError(f"Protocol {protocol_id} has no eligible {role} records")
            frames[role] = pd.concat(pieces[role], ignore_index=True)
        logical = {
            role: sorted({(str(r["dataset_id"]), str(r["pair_id"])) for r in spec["splits"][role]})
            for role in ROLES
        }
        audit = {
            "protocol_id": protocol_id,
            "protocol_kind": spec.get("protocol_kind"),
            "split_records": {role: audits[role] for role in ROLES},
            "logical_pairs": {role: logical[role] for role in ROLES},
            "rows": {role: len(frames[role]) for role in ROLES},
            "sequences": {role: int(frames[role]["sequence_id"].nunique()) for role in ROLES},
            "pairs": {role: int(frames[role].groupby(PAIR_COLUMNS, sort=False).ngroups) for role in ROLES},
            "rate_hz": self.rate_hz,
        }
        return ProtocolData(protocol_id, frames, spec["splits"], audit)


def pair_key(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset_id"].astype(str) + "::" + frame["pair_id"].astype(str)


def ordered_pairs(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    return [
        tuple(str(value) for value in row)
        for row in frame[PAIR_COLUMNS].drop_duplicates().sort_values(PAIR_COLUMNS).itertuples(index=False)
    ]


def select_pair(frame: pd.DataFrame, identity: tuple[str, str, str, str]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in zip(PAIR_COLUMNS, identity, strict=True):
        mask &= frame[column].astype(str).eq(value).to_numpy()
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No rows for pair {identity}")
    return selected.reset_index(drop=True)
