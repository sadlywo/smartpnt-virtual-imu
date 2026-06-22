"""Generic text-table IMU reader used before dataset-specific adapters exist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from virtual_imu.utils.time import to_seconds

TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".log"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}
BAG_SUFFIXES = {".bag", ".db3", ".mcap"}


def read_table_preview(path: str | Path, nrows: int = 5) -> tuple[pd.DataFrame | None, str | None]:
    """Read the first rows of a CSV/TXT-like file, returning an error string on failure."""
    file_path = Path(path)
    try:
        if file_path.suffix.lower() == ".tsv":
            return pd.read_csv(file_path, sep="\t", nrows=nrows), None
        return pd.read_csv(file_path, sep=None, engine="python", nrows=nrows), None
    except Exception as first_error:
        try:
            return pd.read_csv(file_path, delim_whitespace=True, nrows=nrows), None
        except Exception as second_error:
            return None, f"{type(first_error).__name__}: {first_error}; {type(second_error).__name__}: {second_error}"


def read_config_file(path: str | Path) -> dict[str, Any] | None:
    """Read JSON/YAML calibration-like files when possible."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text)
        else:
            import json

            data = json.loads(text)
        return data if isinstance(data, dict) else {"value": data}
    except Exception:
        return None


def read_mapped_imu(path: str | Path, cfg: dict[str, Any], prefix: str) -> pd.DataFrame:
    """Read an IMU table and map configured columns to standardized names."""
    df, error = read_table_preview(path, nrows=100000000)
    if df is None:
        raise ValueError(f"Could not read table {path}: {error}")
    time_col = cfg["time_column"]
    gyro_cols = list(cfg["gyro_columns"])
    accel_cols = list(cfg["accel_columns"])
    required = [time_col, *gyro_cols, *accel_cols]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing configured columns: {missing}. Available: {list(df.columns)}")
    out = pd.DataFrame()
    out["timestamp_s"] = to_seconds(df[time_col].to_numpy(), cfg.get("time_unit", "auto"))
    out[f"gx_{prefix}"] = pd.to_numeric(df[gyro_cols[0]], errors="coerce")
    out[f"gy_{prefix}"] = pd.to_numeric(df[gyro_cols[1]], errors="coerce")
    out[f"gz_{prefix}"] = pd.to_numeric(df[gyro_cols[2]], errors="coerce")
    out[f"ax_{prefix}"] = pd.to_numeric(df[accel_cols[0]], errors="coerce")
    out[f"ay_{prefix}"] = pd.to_numeric(df[accel_cols[1]], errors="coerce")
    out[f"az_{prefix}"] = pd.to_numeric(df[accel_cols[2]], errors="coerce")
    temp_col = cfg.get("temperature_column")
    if temp_col and temp_col in df.columns:
        out["temperature_c"] = pd.to_numeric(df[temp_col], errors="coerce")
    return out.dropna(subset=["timestamp_s"]).sort_values("timestamp_s").reset_index(drop=True)
