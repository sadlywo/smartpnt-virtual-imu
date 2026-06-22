"""Paired high/low IMU dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.sim.imu_error_models import apply_low_grade_errors
from virtual_imu.sim.imu_truth import make_high_grade_imu
from virtual_imu.utils.paths import git_commit, utc_now_iso


def generate_split(config: dict[str, Any], split_name: str, split_cfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Generate one paired simulation split."""
    high = make_high_grade_imu(
        duration_s=float(split_cfg["duration_s"]),
        rate_hz=float(config["rate_hz"]),
        seed=seed,
        sequence_id=str(split_cfg.get("sequence_id", split_name)),
        motion_cfg=config.get("motion", {}),
    )
    return apply_low_grade_errors(high, config.get("imu_error", {}), seed=seed + 1000)


def generate_paired_dataset(config: dict[str, Any], config_path: str | Path | None = None) -> dict[str, Path]:
    """Generate train/val/test paired IMU Parquet files from a simulation config."""
    seed = int(config.get("seed", 0))
    out_dir = Path(config.get("output_dir", "data/processed/sim"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for offset, (split_name, split_cfg) in enumerate(config.get("splits", {}).items()):
        logger.info("Generating simulation split {}", split_name)
        frame = generate_split(config, split_name, split_cfg, seed + offset)
        paths[split_name] = write_parquet(frame, out_dir / f"{split_name}.parquet")
        logger.info("Wrote {} rows to {}", len(frame), paths[split_name])

    manifest = {
        "created_utc": utc_now_iso(),
        "git_commit": git_commit(),
        "config_path": str(config_path) if config_path else None,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (out_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
