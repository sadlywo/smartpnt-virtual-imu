"""Manifest-verified, recording-disjoint paired IMU windows."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from virtual_imu.training.dataset import FEATURE_COLUMNS, LOW_IMU_COLUMNS, TARGET_COLUMNS

LOG = logging.getLogger(__name__)
ROLES = ("train", "validation", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_record_splits(records: list[dict]) -> None:
    groups = {role: set() for role in ROLES}
    for record in records:
        role = record["split"]
        if role not in groups or not record.get("physical_group_id"):
            raise ValueError("Missing physical recording identity or invalid split")
        groups[role].add(record["physical_group_id"])
    for i, left in enumerate(ROLES):
        for right in ROLES[i + 1 :]:
            if groups[left] & groups[right]:
                raise ValueError(f"Physical recording leak between {left} and {right}")
    if any(not group for group in groups.values()):
        raise ValueError("Every split requires at least one physical recording")


class PairedSource:
    """Inspect identity/schema/hash before exposing a requested split."""

    def __init__(self, root: Path, config: dict, pair_id: str):
        self.root = root
        self.config = config
        self.path = root / config["manifest"]
        LOG.info("READS manifest %s", self.path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [
            r
            for r in self.manifest["records"]
            if r["dataset_id"] == config["dataset_id"]
            and r["pair_id"] == pair_id
            and r.get("training_eligible") is True
            and r.get("task") == "paired_signal"
        ]
        if not self.records:
            raise ValueError(f"No measured six-axis paired_signal data for {pair_id}")
        validate_record_splits(self.records)
        LOG.info(
            "SKIPS %d records outside dataset/pair/paired_signal selection",
            len(self.manifest["records"]) - len(self.records),
        )

    def load(self, role: str) -> pd.DataFrame:
        pieces = []
        for record in self.records:
            if record["split"] != role:
                continue
            path = self.root / record["path"]
            LOG.info("READS %s split=%s rows=%s", path, role, record["rows"])
            if self.config.get("verify_hashes", True) and sha256(path) != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            f = pd.read_parquet(path)
            if len(f) != record["rows"]:
                raise ValueError(f"Row count differs from manifest: {path}")
            required = ["timestamp_s", "sequence_id", *LOW_IMU_COLUMNS, *TARGET_COLUMNS]
            if set(required) - set(f):
                raise ValueError(f"Missing paired IMU columns: {path}")
            for col in ("dataset_id", "pair_id", "sequence_id", "low_sensor", "high_sensor"):
                if col in f and not f[col].astype(str).eq(str(record[col])).all():
                    raise ValueError(f"Identity mismatch for {col}: {path}")
            f = f.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
            times = f.timestamp_s.to_numpy(float)
            if not np.isfinite(times).all() or not np.allclose(
                np.diff(times), 1 / self.manifest["rate_hz"], atol=1e-6, rtol=1e-5
            ):
                raise ValueError(f"Non-contiguous timestamps: {path}")
            if not np.isfinite(f[LOW_IMU_COLUMNS + TARGET_COLUMNS].to_numpy(float)).all():
                raise ValueError(f"Nonfinite paired signals: {path}")
            f["physical_group_id"] = record["physical_group_id"]
            pieces.append(f)
        return pd.concat(pieces, ignore_index=True)


class PairedWindows(Dataset):
    """All rows retained; windows never cross sequences; tail is covered once stitched."""

    def __init__(
        self, frame: pd.DataFrame, window_size: int, stride: int, stats: dict | None = None
    ):
        if not 0 < stride <= window_size:
            raise ValueError("Need 0 < stride <= window_size to cover every supported row")
        self.frame = frame.copy().reset_index(drop=True)
        if "temperature_c" not in self.frame:
            self.frame["temperature_c"] = 25.0
        self.window_size = window_size
        features = self.frame[FEATURE_COLUMNS].to_numpy(np.float32)
        self.low = self.frame[LOW_IMU_COLUMNS].to_numpy(np.float32)
        self.high = self.frame[TARGET_COLUMNS].to_numpy(np.float32)
        residual = self.high - self.low
        if stats is None:
            fstd = features.std(0)
            # Constant target channels use unit scale instead of amplification.
            rstd = residual.std(0)
            stats = {
                "feature_mean": features.mean(0).tolist(),
                "feature_std": np.where(fstd < 1e-8, 1.0, fstd).tolist(),
                "residual_mean": residual.mean(0).tolist(),
                "residual_std": np.where(rstd < 1e-8, 1.0, rstd).tolist(),
            }
        self.stats = stats
        self.x = (
            (features - np.array(stats["feature_mean"])) / np.array(stats["feature_std"])
        ).astype(np.float32)
        self.y = (
            (residual - np.array(stats["residual_mean"])) / np.array(stats["residual_std"])
        ).astype(np.float32)
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Nonfinite normalized training data")
        self.starts = []
        for sequence, group in self.frame.groupby("sequence_id", sort=False):
            if len(group) < window_size:
                LOG.info("SKIPS short sequence %s rows=%d", sequence, len(group))
                continue
            indexes = group.index.to_numpy()
            if not np.all(np.diff(indexes) == 1):
                raise ValueError("Sequence rows must be contiguous")
            start, end = int(indexes[0]), int(indexes[-1]) + 1
            starts = list(range(start, end - window_size + 1, stride))
            if starts[-1] != end - window_size:
                starts.append(end - window_size)
            self.starts.extend(starts)
        if not self.starts:
            raise ValueError("No eligible windows")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        stop = start + self.window_size
        return (
            torch.from_numpy(self.x[start:stop].T.copy()),
            torch.from_numpy(self.y[start:stop].T.copy()),
        )

    def stitch(self, corrections: np.ndarray, indexes=None) -> tuple[np.ndarray, np.ndarray]:
        selected = list(range(len(self))) if indexes is None else list(indexes)
        if corrections.shape != (len(selected), 6, self.window_size):
            raise ValueError("One correction is required per window")
        total = np.zeros_like(self.low, dtype=np.float64)
        count = np.zeros(len(self.frame), dtype=np.int64)
        for i, index in enumerate(selected):
            start = self.starts[index]
            correction = corrections[i]
            total[start : start + self.window_size] += correction.T
            count[start : start + self.window_size] += 1
        mask = count > 0
        total[mask] /= count[mask, None]
        return total, mask
