"""Comparison table utilities for ALiMU low-to-high IMU experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOW_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH_COLUMNS = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
VIRTUAL_COLUMNS = ["gx_virtual", "gy_virtual", "gz_virtual", "ax_virtual", "ay_virtual", "az_virtual"]


def _rmse_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "rmse_6d": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))),
        "gyro_rmse_3d": float(np.sqrt(np.mean(np.sum(err[:, :3] ** 2, axis=1)))),
        "accel_rmse_3d": float(np.sqrt(np.mean(np.sum(err[:, 3:] ** 2, axis=1)))),
        "mae_6d": float(np.mean(np.abs(err))),
    }


def _snr_improvement(reference_rmse: float, candidate_rmse: float) -> float:
    eps = 1e-12
    return float(20.0 * np.log10(max(reference_rmse, eps) / max(candidate_rmse, eps)))


def _baseline_rows(supervised_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in supervised_result.get("leaderboard", []):
        rows.append(
            {
                "method": row["baseline"],
                "family": "classical/system-id",
                "rmse_6d": float(row["rmse_6d"]),
                "gyro_rmse_3d": float(row["gyro_rmse_3d"]),
                "accel_rmse_3d": float(row["accel_rmse_3d"]),
                "mae_6d": float(row["mae_6d"]),
                "gyro_snr_improvement_db": np.nan,
                "accel_snr_improvement_db": np.nan,
            }
        )
    return rows


def _neural_row(prediction_df: pd.DataFrame, neural_name: str) -> dict[str, Any]:
    missing = [
        col
        for col in [*LOW_COLUMNS, *HIGH_COLUMNS, *VIRTUAL_COLUMNS]
        if col not in prediction_df.columns
    ]
    if missing:
        raise ValueError(f"Prediction table is missing required comparison columns: {missing}")
    low = prediction_df[LOW_COLUMNS].to_numpy(dtype=float)
    high = prediction_df[HIGH_COLUMNS].to_numpy(dtype=float)
    virtual = prediction_df[VIRTUAL_COLUMNS].to_numpy(dtype=float)
    low_metrics = _rmse_metrics(low, high)
    virtual_metrics = _rmse_metrics(virtual, high)
    return {
        "method": neural_name,
        "family": "neural",
        **virtual_metrics,
        "gyro_snr_improvement_db": _snr_improvement(
            low_metrics["gyro_rmse_3d"], virtual_metrics["gyro_rmse_3d"]
        ),
        "accel_snr_improvement_db": _snr_improvement(
            low_metrics["accel_rmse_3d"], virtual_metrics["accel_rmse_3d"]
        ),
    }


def build_comparison_table(
    supervised_result: dict[str, Any],
    prediction_df: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    neural_name: str = "TCN",
) -> pd.DataFrame:
    """Build a sorted comparison table from classical baselines and neural prediction files."""
    if isinstance(prediction_df, dict):
        neural_rows = [_neural_row(frame, name) for name, frame in prediction_df.items()]
    else:
        neural_rows = [_neural_row(prediction_df, neural_name)]
    rows = [*_baseline_rows(supervised_result), *neural_rows]
    table = pd.DataFrame(rows).sort_values("rmse_6d", kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def write_comparison_table(table: pd.DataFrame, out_path: str | Path) -> dict[str, Path]:
    """Write comparison table as CSV, JSON, and Markdown."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")
    json_path = out.with_suffix(".json")
    md_path = out.with_suffix(".md")
    table.to_csv(csv_path, index=False)
    json_path.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")

    lines = [
        "# ALiMU Low-to-High IMU Result Comparison",
        "",
        "| rank | method | family | rmse 6d | gyro rmse 3d | accel rmse 3d | mae 6d | gyro SNR gain dB | accel SNR gain dB |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in json.loads(table.to_json(orient="records")):
        gyro_gain = "" if pd.isna(row["gyro_snr_improvement_db"]) else f"{row['gyro_snr_improvement_db']:.3g}"
        accel_gain = "" if pd.isna(row["accel_snr_improvement_db"]) else f"{row['accel_snr_improvement_db']:.3g}"
        lines.append(
            f"| {int(row['rank'])} | {row['method']} | {row['family']} | "
            f"{row['rmse_6d']:.6g} | {row['gyro_rmse_3d']:.6g} | "
            f"{row['accel_rmse_3d']:.6g} | {row['mae_6d']:.6g} | "
            f"{gyro_gain} | {accel_gain} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}
