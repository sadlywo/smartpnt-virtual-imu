"""Context features for ALiMU low-to-high sensor ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_imu.training.dataset import LOW_IMU_COLUMNS

SPEC_FEATURE_COLUMNS = [
    "spec_gyro_noise_density",
    "spec_accel_noise_density",
    "spec_gyro_bias_instability",
    "spec_accel_bias_instability",
    "spec_gyro_bandwidth_hz",
    "spec_accel_bandwidth_hz",
    "spec_gyro_range_dps",
    "spec_accel_range_g",
    "spec_temp_coeff",
]
STATIC_CONTEXT_COLUMNS = [f"static_low_mean_{column.removesuffix('_low')}" for column in LOW_IMU_COLUMNS]
STATIC_CONTEXT_COLUMNS.append("static_low_mean_temperature_c")
CONTEXT_FEATURE_COLUMNS = [*SPEC_FEATURE_COLUMNS, *STATIC_CONTEXT_COLUMNS]


def _sensor_name(value: Any) -> str:
    return str(value).strip()


def _stationary_mask(df: pd.DataFrame) -> pd.Series:
    if "motion_label" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["motion_label"].astype(str).str.lower().eq("stationary")


def _low_static_context(df: pd.DataFrame) -> pd.DataFrame:
    columns = [*LOW_IMU_COLUMNS, "temperature_c"]
    source = df[_stationary_mask(df)]
    chunks = []
    for sensor, group in df.groupby("low_sensor", sort=False):
        stationary = source[source["low_sensor"].astype(str) == str(sensor)]
        basis = stationary if not stationary.empty else group
        row = basis[columns].mean(numeric_only=True).to_dict()
        row["low_sensor"] = sensor
        chunks.append(row)
    if not chunks:
        return pd.DataFrame(columns=["low_sensor", *STATIC_CONTEXT_COLUMNS])
    stats = pd.DataFrame(chunks)
    rename = {column: f"static_low_mean_{column.removesuffix('_low')}" for column in LOW_IMU_COLUMNS}
    rename["temperature_c"] = "static_low_mean_temperature_c"
    return stats.rename(columns=rename)[["low_sensor", *STATIC_CONTEXT_COLUMNS]]


def _spec_frame(sensor_specs: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for sensor, specs in sensor_specs.items():
        row = {"low_sensor": _sensor_name(sensor)}
        for column in SPEC_FEATURE_COLUMNS:
            row[column] = float(specs.get(column.removeprefix("spec_"), 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def add_context_features(df: pd.DataFrame, sensor_specs: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Add datasheet-style sensor specs and low-only static context columns."""
    if "low_sensor" not in df.columns:
        raise ValueError("ALiMU context features require a low_sensor column")
    missing = [column for column in LOW_IMU_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"ALiMU context features require low IMU columns: {missing}")
    out = df.copy()
    if "temperature_c" not in out.columns:
        out["temperature_c"] = 25.0
    out["low_sensor"] = out["low_sensor"].map(_sensor_name)
    spec = _spec_frame(sensor_specs)
    if spec.empty:
        raise ValueError("sensor_specs must include at least one sensor")
    unknown = sorted(set(out["low_sensor"]) - set(spec["low_sensor"]))
    if unknown:
        raise ValueError(f"Missing sensor_specs entries for: {unknown}")
    static = _low_static_context(out)
    out = out.merge(spec, on="low_sensor", how="left", validate="many_to_one")
    out = out.merge(static, on="low_sensor", how="left", validate="many_to_one")
    return out


def prepare_context_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare context-enriched ALiMU split Parquets from a YAML-like config."""
    input_splits = config.get("input_splits", {})
    if not input_splits:
        raise ValueError("config.input_splits is required")
    out_dir = Path(config.get("output_dir", "data/processed/alimu_context"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_specs = config.get("sensor_specs", {})
    manifest: dict[str, Any] = {
        "input_splits": {},
        "output_splits": {},
        "sensor_specs": sensor_specs,
        "context_feature_columns": CONTEXT_FEATURE_COLUMNS,
    }
    for split, path_value in input_splits.items():
        path = Path(path_value)
        frame = pd.read_parquet(path)
        enriched = add_context_features(frame, sensor_specs)
        out_path = out_dir / f"{split}.parquet"
        enriched.to_parquet(out_path, index=False)
        manifest["input_splits"][split] = str(path)
        manifest["output_splits"][split] = str(out_path)
    manifest_path = out_dir / "context_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
