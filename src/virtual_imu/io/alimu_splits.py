"""Strict sequence-level splits for ALiMU generalization tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

STRICT_SPLIT_MODES = {"leave_sensor_out", "leave_motion_out"}


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"ALiMU strict split input is missing columns: {missing}")


def _sequence_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["sequence_id"].dropna().astype(str).unique().tolist())


def _train_val_sequence_ids(sequence_ids: list[str], *, val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    if not sequence_ids:
        raise ValueError("No non-holdout sequences are available for train/val splitting")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    ids = np.array(sorted(sequence_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    if len(ids) == 1 or val_fraction == 0:
        return set(ids.tolist()), set()
    val_count = int(round(len(ids) * val_fraction))
    val_count = min(max(1, val_count), len(ids) - 1)
    val_ids = set(ids[:val_count].tolist())
    train_ids = set(ids[val_count:].tolist())
    return train_ids, val_ids


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "sequences": int(frame["sequence_id"].nunique()) if "sequence_id" in frame else 0,
    }
    for col in ("low_sensor", "motion_label", "source_kind"):
        if col in frame:
            summary[col] = sorted(frame[col].dropna().astype(str).unique().tolist())
    return summary


def make_strict_split(
    df: pd.DataFrame,
    *,
    mode: str,
    holdout: str,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Create strict train/val/test DataFrames by holding out a sensor or motion class."""
    mode = mode.lower()
    if mode not in STRICT_SPLIT_MODES:
        raise ValueError(f"Unsupported strict split mode {mode!r}; expected one of {sorted(STRICT_SPLIT_MODES)}")
    split_column = "low_sensor" if mode == "leave_sensor_out" else "motion_label"
    _require_columns(df, ["sequence_id", split_column])

    values = df[split_column].astype(str)
    test_mask = values == str(holdout)
    if not test_mask.any():
        available = sorted(values.dropna().unique().tolist())
        raise ValueError(f"Holdout {holdout!r} was not found in {split_column}; available={available}")
    candidate = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()
    train_ids, val_ids = _train_val_sequence_ids(_sequence_ids(candidate), val_fraction=val_fraction, seed=seed)
    train = candidate[candidate["sequence_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    val = candidate[candidate["sequence_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_seq = set(train["sequence_id"].astype(str).unique().tolist())
    val_seq = set(val["sequence_id"].astype(str).unique().tolist())
    test_seq = set(test["sequence_id"].astype(str).unique().tolist())
    if train_seq & val_seq or train_seq & test_seq or val_seq & test_seq:
        raise RuntimeError("Strict split produced overlapping sequence_id values")
    return {"train": train, "val": val, "test": test}


def strict_split_manifest(
    *,
    source_path: str | Path,
    output_root: str | Path,
    mode: str,
    holdout: str,
    val_fraction: float,
    seed: int,
    split_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for a strict split run."""
    return {
        "dataset": "ALiMU Low-Grade & High-Grade IMU Dataset",
        "source_path": str(source_path),
        "output_root": str(output_root),
        "split_policy": {
            "mode": mode,
            "holdout": holdout,
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "unit": "sequence_id",
        },
        "splits": {name: _split_summary(frame) for name, frame in split_frames.items()},
    }


def write_strict_split(
    df: pd.DataFrame,
    *,
    source_path: str | Path,
    output_root: str | Path,
    mode: str,
    holdout: str,
    val_fraction: float = 0.2,
    seed: int = 7,
) -> dict[str, Any]:
    """Write strict ALiMU train/val/test Parquet files and a manifest."""
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    split_frames = make_strict_split(df, mode=mode, holdout=holdout, val_fraction=val_fraction, seed=seed)
    paths: dict[str, Path] = {}
    for split_name, frame in split_frames.items():
        path = out / f"{split_name}.parquet"
        frame.to_parquet(path, index=False)
        paths[f"{split_name}_path"] = path
        logger.info("Wrote strict {} split with {} rows to {}", split_name, len(frame), path)
    manifest = strict_split_manifest(
        source_path=source_path,
        output_root=out,
        mode=mode,
        holdout=holdout,
        val_fraction=val_fraction,
        seed=seed,
        split_frames=split_frames,
    )
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote strict split manifest to {}", manifest_path)
    return {**paths, "manifest_path": manifest_path, "manifest": manifest}
