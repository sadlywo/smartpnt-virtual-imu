"""Conservative SmartPNT-MSF inspection and reading utilities."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_imu.io.generic_imu_reader import (
    BAG_SUFFIXES,
    CONFIG_SUFFIXES,
    TEXT_SUFFIXES,
    read_config_file,
    read_mapped_imu,
    read_table_preview,
)


def scan_dataset(root: str | Path, preview_limit: int = 5) -> dict[str, Any]:
    """Scan a dataset directory and return a human-readable inspection structure."""
    root_path = Path(root)
    if not root_path.exists():
        return {
            "exists": False,
            "message": f"Dataset root not found: {root_path}. Place SmartPNT-MSF files there first.",
        }
    files = sorted(p for p in root_path.rglob("*") if p.is_file())
    suffix_counts = Counter(p.suffix.lower() or "<none>" for p in files)
    sizes = [p.stat().st_size for p in files]
    previews: list[dict[str, Any]] = []
    for path in files:
        if len(previews) >= preview_limit:
            break
        suffix = path.suffix.lower()
        rel = path.relative_to(root_path).as_posix()
        if suffix in TEXT_SUFFIXES:
            frame, error = read_table_preview(path, nrows=5)
            previews.append(
                {
                    "path": rel,
                    "kind": "text-table",
                    "columns": list(frame.columns) if frame is not None else [],
                    "rows": frame.astype(str).to_dict(orient="records") if frame is not None else [],
                    "error": error,
                }
            )
        elif suffix in CONFIG_SUFFIXES:
            cfg = read_config_file(path)
            previews.append({"path": rel, "kind": "config", "keys": list(cfg.keys())[:20] if cfg else []})
        elif suffix in BAG_SUFFIXES:
            previews.append(
                {
                    "path": rel,
                    "kind": "rosbag-like",
                    "note": "ROS bag/db3 parsing is optional; install the 'bags' extra and configure topics.",
                }
            )
    return {
        "exists": True,
        "root": str(root_path),
        "file_count": len(files),
        "suffix_counts": dict(suffix_counts),
        "total_size_bytes": sum(sizes),
        "max_size_bytes": max(sizes) if sizes else 0,
        "previews": previews,
        "tree": [p.relative_to(root_path).as_posix() for p in files[:200]],
    }


class SmartPNTReader:
    """Config-driven low/high IMU reader for SmartPNT-MSF-like directories."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = Path(config.get("raw_root", "data/raw/smartpnt"))

    def inspect(self) -> dict[str, Any]:
        """Inspect the configured raw root."""
        return scan_dataset(self.root)

    def _resolve_source(self, key: str) -> Path:
        source = self.config.get(key)
        if not source:
            raise ValueError(
                f"{key} is not configured. Run inspect_dataset.py first, then set {key} in the YAML config."
            )
        path = Path(source)
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(f"Configured source does not exist: {path}")
        if path.suffix.lower() in BAG_SUFFIXES:
            raise NotImplementedError(
                f"{path} looks like a ROS bag/db3 file. Install optional bag support and add topic mapping first."
            )
        return path

    def read_pair(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read low- and high-grade IMU tables according to the YAML config."""
        low = read_mapped_imu(self._resolve_source("low_imu_source"), self.config, "low")
        high = read_mapped_imu(self._resolve_source("high_imu_source"), self.config, "high")
        return low, high
