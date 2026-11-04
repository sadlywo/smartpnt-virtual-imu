"""ALiMU low/high-grade IMU dataset adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.io.parquet_store import write_parquet

STANDARD_COLUMNS = [
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

REQUIRED_AXES = ("ax", "ay", "az", "gx", "gy", "gz")
GYRO_SCALES = {
    "raw": 1.0,
    "rad_per_s": 1.0,
    "radps": 1.0,
    "deg_per_s": np.pi / 180.0,
    "dps": np.pi / 180.0,
}


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sequence_id(path: Path, root: Path) -> str:
    relative = _relative_posix(path.with_suffix(""), root)
    return re.sub(r"[^A-Za-z0-9]+", "_", relative).strip("_")


def _source_kind(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "synthetic" in parts:
        return "synthetic"
    if "real" in parts:
        return "real"
    return "unknown"


def _motion_label(path: Path) -> str:
    stem = path.stem.lower()
    if "stat" in stem or "stationary" in stem:
        return "stationary"
    if "dyn" in stem or "dynamic" in stem or "uav" in stem:
        return "dynamic"
    return "unknown"


def _sensor_prefixes(columns: list[str]) -> list[str]:
    candidates = []
    for col in columns:
        if col.endswith("_ax"):
            prefix = col[: -len("_ax")]
            if all(f"{prefix}_{axis}" in columns for axis in REQUIRED_AXES):
                candidates.append(prefix)
    return candidates


def _detect_low_high_prefixes(columns: list[str]) -> tuple[str, str]:
    prefixes = _sensor_prefixes(columns)
    high_candidates = [
        prefix
        for prefix in prefixes
        if prefix.lower() in {"vn100", "imu_high"} or "high" in prefix.lower()
    ]
    low_candidates = [
        prefix
        for prefix in prefixes
        if prefix.lower() in {"imu_low"} or ("low" in prefix.lower() and prefix not in high_candidates)
    ]
    if not high_candidates:
        raise ValueError(f"Could not find ALiMU high-grade IMU columns in: {columns}")
    if not low_candidates:
        low_candidates = [prefix for prefix in prefixes if prefix not in high_candidates]
    if not low_candidates:
        raise ValueError(f"Could not find ALiMU low-grade IMU columns in: {columns}")
    return low_candidates[0], high_candidates[0]


def _gyro_scale(gyro_unit: str) -> float:
    key = gyro_unit.lower()
    if key not in GYRO_SCALES:
        raise ValueError(f"Unsupported gyro_unit={gyro_unit!r}; expected one of {sorted(GYRO_SCALES)}")
    return float(GYRO_SCALES[key])


def read_alimu_csv(
    path: str | Path,
    *,
    raw_root: str | Path,
    sample_rate_hz: float,
    gyro_unit: str = "deg_per_s",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one ALiMU CSV and map it to the project low/high IMU schema."""
    file_path = Path(path)
    root = Path(raw_root)
    df = pd.read_csv(file_path)
    low_prefix, high_prefix = _detect_low_high_prefixes(list(df.columns))
    scale = _gyro_scale(gyro_unit)
    timestamps = np.arange(len(df), dtype=np.float64) / float(sample_rate_hz)
    sequence_id = _sequence_id(file_path, root)
    source_kind = _source_kind(file_path)
    motion_label = _motion_label(file_path)
    relative = _relative_posix(file_path, root)

    out = pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "gx_low": pd.to_numeric(df[f"{low_prefix}_gx"], errors="coerce") * scale,
            "gy_low": pd.to_numeric(df[f"{low_prefix}_gy"], errors="coerce") * scale,
            "gz_low": pd.to_numeric(df[f"{low_prefix}_gz"], errors="coerce") * scale,
            "ax_low": pd.to_numeric(df[f"{low_prefix}_ax"], errors="coerce"),
            "ay_low": pd.to_numeric(df[f"{low_prefix}_ay"], errors="coerce"),
            "az_low": pd.to_numeric(df[f"{low_prefix}_az"], errors="coerce"),
            "gx_high": pd.to_numeric(df[f"{high_prefix}_gx"], errors="coerce") * scale,
            "gy_high": pd.to_numeric(df[f"{high_prefix}_gy"], errors="coerce") * scale,
            "gz_high": pd.to_numeric(df[f"{high_prefix}_gz"], errors="coerce") * scale,
            "ax_high": pd.to_numeric(df[f"{high_prefix}_ax"], errors="coerce"),
            "ay_high": pd.to_numeric(df[f"{high_prefix}_ay"], errors="coerce"),
            "az_high": pd.to_numeric(df[f"{high_prefix}_az"], errors="coerce"),
            "temperature_c": 25.0,
            "motion_label": motion_label,
            "sequence_id": sequence_id,
            "source_kind": source_kind,
            "low_sensor": low_prefix,
            "high_sensor": high_prefix,
            "source_file": relative,
        }
    )
    before = len(out)
    out = out.dropna(subset=[col for col in STANDARD_COLUMNS if col.endswith("_low") or col.endswith("_high")])
    out = out.loc[:, STANDARD_COLUMNS].reset_index(drop=True)
    metadata = {
        "source_file": relative,
        "rows": int(len(out)),
        "dropped_rows": int(before - len(out)),
        "source_kind": source_kind,
        "motion_label": motion_label,
        "sequence_id": sequence_id,
        "low_sensor": low_prefix,
        "high_sensor": high_prefix,
        "sample_rate_hz": float(sample_rate_hz),
        "gyro_unit": gyro_unit,
    }
    return out, metadata


def _discover_csvs(raw_root: Path, include: dict[str, bool]) -> list[Path]:
    files: list[Path] = []
    if include.get("real", True):
        files.extend(sorted((raw_root / "Real").rglob("*.csv")))
    if include.get("synthetic", False):
        files.extend(sorted((raw_root / "Synthetic").rglob("*.csv")))
    return files


def _split_sequence_ids(
    sequence_ids: list[str],
    *,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, set[str]]:
    if not sequence_ids:
        raise ValueError("No ALiMU sequences available for splitting")
    total_fraction = train_fraction + val_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError("Split fractions must sum to a positive value")
    train_fraction /= total_fraction
    val_fraction /= total_fraction
    ids = np.array(sorted(sequence_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    train_count = int(round(n * train_fraction))
    val_count = int(round(n * val_fraction))
    if train_fraction > 0 and n >= 3:
        train_count = max(1, train_count)
    if val_fraction > 0 and n - train_count >= 2:
        val_count = max(1, val_count)
    if train_count + val_count > n:
        val_count = max(0, n - train_count)
    train_ids = set(ids[:train_count].tolist())
    val_ids = set(ids[train_count : train_count + val_count].tolist())
    test_ids = set(ids[train_count + val_count :].tolist())
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "sequences": int(frame["sequence_id"].nunique()) if len(frame) else 0,
        "low_sensors": sorted(frame["low_sensor"].dropna().unique().tolist()) if len(frame) else [],
        "source_kinds": sorted(frame["source_kind"].dropna().unique().tolist()) if len(frame) else [],
    }


def _write_report(manifest: dict[str, Any], path: Path) -> None:
    lines = [
        "# ALiMU Preparation Report",
        "",
        f"- Raw root: `{manifest['raw_root']}`",
        f"- Processed root: `{manifest['processed_root']}`",
        f"- Included files: {manifest['file_count']}",
        f"- Skipped files: {len(manifest['skipped_files'])}",
        f"- Gyro unit conversion: `{manifest['gyro_unit']}` to project standard columns",
        "",
        "## Splits",
        "",
        "| split | rows | sequences | low sensors | source kinds |",
        "|---|---:|---:|---|---|",
    ]
    for split_name in ("train", "val", "test"):
        split = manifest["splits"][split_name]
        lines.append(
            f"| {split_name} | {split['rows']} | {split['sequences']} | "
            f"{', '.join(split['low_sensors'])} | {', '.join(split['source_kinds'])} |"
        )
    lines.extend(["", "## Notes", "", "- Split is performed by `sequence_id`, not by row."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_alimu_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare ALiMU CSV files into train/val/test Parquet splits."""
    raw_root = Path(config.get("raw_root", "data/raw/alimu/Low_Grade_High_Grade_IMU_dataset"))
    processed_root = Path(config.get("processed_root", "data/processed/alimu"))
    report_path = Path(config.get("report_path", "outputs/alimu/preparation_report.md"))
    include = dict(config.get("include", {"real": True, "synthetic": False}))
    split_cfg = dict(config.get("split", {}))
    gyro_unit = str(config.get("gyro_unit", "deg_per_s"))

    logger.info("Reading ALiMU CSVs from {}", raw_root)
    csv_files = _discover_csvs(raw_root, include)
    if not csv_files:
        raise FileNotFoundError(f"No ALiMU CSV files found under {raw_root} with include={include}")

    frames = []
    file_metadata = []
    skipped_files = []
    for file_path in csv_files:
        source_kind = _source_kind(file_path)
        sample_rate = float(
            config.get("synthetic_rate_hz", 100.0)
            if source_kind == "synthetic"
            else config.get("real_rate_hz", 40.0)
        )
        try:
            frame, metadata = read_alimu_csv(
                file_path,
                raw_root=raw_root,
                sample_rate_hz=sample_rate,
                gyro_unit=gyro_unit,
            )
            frames.append(frame)
            file_metadata.append(metadata)
            logger.info("Read {} rows from {}", len(frame), file_path)
        except Exception as exc:
            skipped_files.append({"source_file": _relative_posix(file_path, raw_root), "reason": str(exc)})
            logger.warning("Skipped {}: {}", file_path, exc)

    if not frames:
        raise ValueError(f"No ALiMU files could be parsed under {raw_root}")

    all_df = pd.concat(frames, ignore_index=True)
    split_ids = _split_sequence_ids(
        all_df["sequence_id"].drop_duplicates().tolist(),
        train_fraction=float(split_cfg.get("train", 0.7)),
        val_fraction=float(split_cfg.get("val", 0.15)),
        test_fraction=float(split_cfg.get("test", 0.15)),
        seed=int(split_cfg.get("seed", config.get("seed", 0))),
    )
    processed_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_path": write_parquet(all_df, processed_root / "all.parquet"),
    }
    split_frames = {}
    for split_name in ("train", "val", "test"):
        frame = all_df[all_df["sequence_id"].isin(split_ids[split_name])].reset_index(drop=True)
        split_frames[split_name] = frame
        paths[f"{split_name}_path"] = write_parquet(frame, processed_root / f"{split_name}.parquet")
        logger.info("Wrote {} rows to {}", len(frame), paths[f"{split_name}_path"])

    manifest = {
        "dataset": "ALiMU Low-Grade & High-Grade IMU Dataset",
        "source_url": "https://github.com/cmsahu97/Low_Grade_High_Grade_IMU_dataset",
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "file_count": len(file_metadata),
        "skipped_files": skipped_files,
        "files": file_metadata,
        "gyro_unit": gyro_unit,
        "real_rate_hz": float(config.get("real_rate_hz", 40.0)),
        "synthetic_rate_hz": float(config.get("synthetic_rate_hz", 100.0)),
        "splits": {name: _split_summary(frame) for name, frame in split_frames.items()},
        "columns": STANDARD_COLUMNS,
    }
    manifest_path = processed_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest, report_path)
    paths["manifest_path"] = manifest_path
    paths["report_path"] = report_path
    logger.info("Wrote ALiMU manifest to {}", manifest_path)
    return {**paths, "manifest": manifest}
