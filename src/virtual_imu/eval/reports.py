"""Evaluation report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from virtual_imu.eval.allan import allan_summary
from virtual_imu.eval.navigation import navigation_metrics
from virtual_imu.eval.preintegration import preintegration_metrics
from virtual_imu.eval.signal_metrics import signal_metrics, snr_improvement
from virtual_imu.eval.spectral_metrics import spectral_metrics


def motion_segment_metrics(df: pd.DataFrame, min_samples: int = 16) -> dict[str, Any]:
    """Evaluate signal/spectral metrics by motion_label when labels are available."""
    if "motion_label" not in df.columns:
        return {}
    segments: dict[str, Any] = {}
    for label, group in df.groupby("motion_label", sort=False):
        if len(group) < min_samples:
            segments[str(label)] = {"status": "insufficient samples", "samples": int(len(group))}
            continue
        low = signal_metrics(group, "low")
        virtual = signal_metrics(group, "virtual")
        segments[str(label)] = {
            "status": "ok",
            "samples": int(len(group)),
            "signal": {
                "low": low,
                "virtual": virtual,
                "improvement": snr_improvement(low, virtual),
            },
            "spectral": {
                "low": spectral_metrics(group, "low"),
                "virtual": spectral_metrics(group, "virtual"),
            },
            "preintegration": {
                "low": preintegration_metrics(group, "low"),
                "virtual": preintegration_metrics(group, "virtual"),
            },
        }
    return segments


def paired_device_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Report physical signal errors separately for each dataset/device mapping."""
    identities = ["dataset_id", "low_sensor", "high_sensor"]
    if not all(column in df for column in identities):
        return []
    results = []
    for values, group in df.groupby(identities, sort=True, dropna=False):
        axes = ("gx", "gy", "gz", "ax", "ay", "az")
        target = group[[f"{axis}_high" for axis in axes]].to_numpy(dtype=float)
        signal = {}
        for prefix in ("low", "virtual"):
            error = group[[f"{axis}_{prefix}" for axis in axes]].to_numpy(dtype=float) - target
            signal[prefix] = {
                "gyro_vector_rmse_radps": float(np.sqrt(np.mean(np.sum(error[:, :3]**2, axis=1)))),
                "accel_vector_rmse_mps2": float(np.sqrt(np.mean(np.sum(error[:, 3:]**2, axis=1)))),
                "axis_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            }
        results.append({**{key: values[i] for i, key in enumerate(identities)},
                        "samples": len(group), "signal": signal})
    return results


def _flatten(prefix: str, data: Any, rows: list[tuple[str, Any]]) -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, rows)
    elif isinstance(data, list):
        rows.append((prefix, json.dumps(data)))
    else:
        rows.append((prefix, data))


def evaluate_predictions(pred_path: str | Path, out_json: str | Path, min_allan_duration_s: float = 20.0) -> dict[str, Any]:
    """Evaluate a virtual IMU prediction file and write JSON/CSV/Markdown outputs."""
    df = pd.read_parquet(pred_path)
    high_cols = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
    if not all(col in df.columns for col in high_cols):
        raise ValueError("Evaluation requires high-grade reference columns in the prediction file")
    result: dict[str, Any] = {"prediction_path": str(pred_path)}
    low = signal_metrics(df, "low")
    virtual = signal_metrics(df, "virtual")
    result["signal"] = {"low": low, "virtual": virtual, "improvement": snr_improvement(low, virtual)}
    result["spectral"] = {"low": spectral_metrics(df, "low"), "virtual": spectral_metrics(df, "virtual")}
    result["allan"] = {
        "low": allan_summary(df, "low", min_duration_s=min_allan_duration_s),
        "virtual": allan_summary(df, "virtual", min_duration_s=min_allan_duration_s),
        "high": allan_summary(df, "high", min_duration_s=min_allan_duration_s),
    }
    result["preintegration"] = {
        "low": preintegration_metrics(df, "low"),
        "virtual": preintegration_metrics(df, "virtual"),
    }
    result["navigation"] = {"low": navigation_metrics(df, "low"), "virtual": navigation_metrics(df, "virtual")}
    result["motion_segments"] = motion_segment_metrics(df)
    result["paired_devices"] = paired_device_metrics(df)
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    write_csv_summary(result, out.with_suffix(".csv"))
    write_markdown_summary(result, out.with_suffix(".md"))
    return result


def write_csv_summary(metrics: dict[str, Any], path: str | Path) -> None:
    """Write flattened metrics CSV."""
    rows: list[tuple[str, Any]] = []
    _flatten("", metrics, rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_markdown_summary(metrics: dict[str, Any], path: str | Path) -> None:
    """Write a compact Markdown summary."""
    signal = metrics.get("signal", {})
    virtual = signal.get("virtual", {})
    low = signal.get("low", {})
    improvement = signal.get("improvement", {})
    lines = [
        "# Evaluation Summary",
        "",
        f"Prediction file: `{metrics.get('prediction_path')}`",
        "",
        "## Signal Metrics",
        "",
        "| Metric | Low | Virtual |",
        "|---|---:|---:|",
        f"| Gyro norm RMSE | {low.get('gyro_norm_rmse', float('nan')):.6g} | {virtual.get('gyro_norm_rmse', float('nan')):.6g} |",
        f"| Accel norm RMSE | {low.get('accel_norm_rmse', float('nan')):.6g} | {virtual.get('accel_norm_rmse', float('nan')):.6g} |",
        f"| Gyro SNR improvement |  | {improvement.get('gyro_snr_improvement_db', float('nan')):.3g} dB |",
        f"| Accel SNR improvement |  | {improvement.get('accel_snr_improvement_db', float('nan')):.3g} dB |",
        "",
        "## Allan Notes",
        "",
        f"- Low: {metrics.get('allan', {}).get('low', {}).get('status', 'unknown')}",
        f"- Virtual: {metrics.get('allan', {}).get('virtual', {}).get('status', 'unknown')}",
        f"- High: {metrics.get('allan', {}).get('high', {}).get('status', 'unknown')}",
    ]
    segments = metrics.get("motion_segments", {})
    devices = metrics.get("paired_devices", [])
    if devices:
        lines += ["", "## Dataset and Device Mappings", "",
                  "| Dataset | Mapping | Samples | Gyro RMSE low / virtual rad/s | Accel RMSE low / virtual m/s² |",
                  "|---|---|---:|---:|---:|"]
        for device in devices:
            low, virtual = device["signal"]["low"], device["signal"]["virtual"]
            lines.append(f"| {device['dataset_id']} | {device['low_sensor']} → {device['high_sensor']} | "
                         f"{device['samples']} | {low['gyro_vector_rmse_radps']:.6g} / {virtual['gyro_vector_rmse_radps']:.6g} | "
                         f"{low['accel_vector_rmse_mps2']:.6g} / {virtual['accel_vector_rmse_mps2']:.6g} |")
    if segments:
        lines.extend(
            [
                "",
                "## Motion Segments",
                "",
                "| Segment | Samples | Gyro RMSE Low | Gyro RMSE Virtual | Accel RMSE Low | Accel RMSE Virtual | Gyro Coherence Virtual |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, segment in segments.items():
            if segment.get("status") != "ok":
                lines.append(f"| {label} | {segment.get('samples', 0)} |  |  |  |  |  |")
                continue
            sig_low = segment["signal"]["low"]
            sig_virtual = segment["signal"]["virtual"]
            spec_virtual = segment["spectral"]["virtual"]
            lines.append(
                f"| {label} | {segment.get('samples', 0)} | "
                f"{sig_low.get('gyro_norm_rmse', float('nan')):.6g} | "
                f"{sig_virtual.get('gyro_norm_rmse', float('nan')):.6g} | "
                f"{sig_low.get('accel_norm_rmse', float('nan')):.6g} | "
                f"{sig_virtual.get('accel_norm_rmse', float('nan')):.6g} | "
                f"{spec_virtual.get('gyro_coherence', float('nan')):.3g} |"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
