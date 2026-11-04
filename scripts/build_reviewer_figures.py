"""Build complete, source-verified reviewer figures without inference or fitting.

Formal CLI accepts only complete nonsynthetic packages. Synthetic API validation
is permanently watermarked and cannot produce a formal report identity.
"""

# ruff: noqa: E402, RUF001 -- checkout/aux imports and scientific Unicode labels.
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from fontTools import subset
from fontTools.ttLib import TTFont
from matplotlib import colors, font_manager, ticker
from PIL import Image
from scipy.integrate import cumulative_trapezoid
from scipy.signal import coherence, welch

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reviewer_artifact_package import load_package, sha256_file
from reviewer_shared_figure_data import (
    BANK,
    analysis_profile,
    attach_shared,
    load_shared,
)
from reviewer_shared_figure_data import (
    COUNTS as SHARED_COUNTS,
)
from reviewer_shared_figure_data import (
    PROFILE as SHARED_PROFILE,
)
from reviewer_shared_figure_data import (
    SCOPE as SHARED_SCOPE,
)
from reviewer_temporal_support import classify_temporal_support
from virtual_imu.viz.style import COLORS, apply_paper_style

AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
LOW = [axis + "_low" for axis in AXES]
HIGH = [axis + "_high" for axis in AXES]
PAIRED_METRICS = [
    "gyro_vector_rmse_radps",
    "gyro_vector_mae_radps",
    "gyro_bias_norm_radps",
    "accel_vector_rmse_mps2",
    "accel_vector_mae_mps2",
    "accel_bias_norm_mps2",
]
NAV_METRICS = ["vector_rmse_m", "vector_mae_m", "bias_norm_m"]
METRIC_LABELS = {
    "gyro_vector_rmse_radps": "Gyroscope RMSE",
    "gyro_vector_mae_radps": "Gyroscope MAE",
    "gyro_bias_norm_radps": "Gyroscope bias norm",
    "accel_vector_rmse_mps2": "Accelerometer RMSE",
    "accel_vector_mae_mps2": "Accelerometer MAE",
    "accel_bias_norm_mps2": "Accelerometer bias norm",
    "vector_rmse_m": "Displacement RMSE",
    "vector_mae_m": "Displacement MAE",
    "bias_norm_m": "Displacement bias norm",
}
CLASSICAL = [
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "ridge_tuned",
    "fir_tuned",
    "arx_tuned",
]
_TRANSFER_CLASSICAL_TEMPORAL_METHODS = {
    "source_bias_mean_fixed": "bias_mean",
    "source_affine_fixed": "affine",
    "source_ridge_affine_fixed": "ridge_affine",
    "source_learned_fir_causal_fixed": "learned_fir_causal",
    "source_arx_linear_fixed": "arx_linear",
    "source_ridge_tuned": "ridge_tuned",
    "source_fir_tuned": "fir_tuned",
    "source_arx_tuned": "arx_tuned",
}
LABELS = {
    "tcn": "TCN",
    "tcn_original": "TCN",
    "plain_cnn": "CNN",
    "cnn_bilstm": "CNN–BiLSTM",
    "deepgyro_adapter": "DeepGyro",
    "tlio_resnet18_adapter": "TLIO ResNet18",
    "tartanimu_window_adapter": "TartanIMU",
    "learned_fir_causal": "FIR",
    "arx_linear": "ARX-style LOW lag/delta",
    "arx_tuned": "ARX-style LOW lag/delta [T]",
    "identity": "LOW / identity",
    "ridge_affine": "Ridge affine",
    "bias_mean": "Mean bias",
}
STATISTICS = "Neural: mean ± sample SD across 3 optimization seeds; classical: one deduplicated result; no recording CI."
TEMPORAL_SOURCE = Path(__file__).with_name("reviewer_temporal_support.py")
TEMPORAL_FIELDS = (
    "temporal_class",
    "temporal_label",
    "temporal_label_zh",
    "temporal_note",
    "temporal_order",
)
TEMPORAL_CONDITION = "Prepared 40 Hz inputs; eval/frozen statistics. Inference context is not an end-to-end online guarantee."


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def default_config():
    return {
        "seed": 20261024,
        "representative_model_seed": 20261010,
        "width_inches": 7.2,
        "png_dpi": 300,
        "tiff_dpi": 600,
        "pdf_renderer": "pdftoppm",
        "temporal_window_points": 800,
        "sample_rate_hz": 40.0,
        "spectral_nperseg": 256,
        "correlations_per_page": 12,
        "include_validation": True,
        "all_training_cells_required": 93,
        "shared_analysis_profile": SHARED_PROFILE,
        "temporal_context_display": "grouped_v1",
    }


def _slug(value):
    result = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return result[:100] + "_" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def _pair_display(value):
    """Keep the full routing identity in source data and use sensor names on paper."""
    identity = json.loads(value)
    if not identity:
        return "input pair (source CSV)"
    dataset = {"penncosyvio": "Penn", "alimu": "AL"}.get(
        identity["dataset_id"], identity["dataset_id"]
    )
    return f"{dataset}: {identity['low_sensor']} → {identity['high_sensor']}"


def label(row):
    family = str(row.get("model_family", row.get("method", "")))
    family = LABELS.get(family, family.replace("_", " "))
    variant = row.get("variant", "")
    suffix = {"default": " [D]", "tuned": " [T]"}.get(variant, "")
    scope = row.get("information_scope")
    if scope == SHARED_SCOPE and row.get("method", row.get("canonical_method")) != "identity":
        suffix += " [S]"
    elif scope == BANK:
        suffix += " [B]"
    return family + suffix


def _groups(frame, columns):
    if frame is None or frame.empty:
        return []
    actual = [column for column in columns if column in frame]
    return frame.groupby(actual, sort=True, dropna=False) if actual else [("all", frame)]


def select_low_excitation_segment(frame, window_points=800):
    """Choose median LOW-only dimensionless excitation, then earliest identity.

    Candidate windows do not overlap, never span segments, and use no HIGH,
    prediction, error, validation rank or downstream score.
    """
    candidates, values = [], []
    for name, group in frame.groupby("sequence_id", sort=True):
        group = group.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        if not np.isfinite(group[LOW].to_numpy(float)).all() or not np.allclose(
            np.diff(group.timestamp_s), 0.025, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("Temporal diagnostics require finite contiguous 40Hz segments")
        width = min(window_points, len(group))
        if width < 80:
            continue
        for start in range(0, len(group) - width + 1, width):
            candidates.append(
                {
                    "sequence_id": str(name),
                    "start_index": start,
                    "window_points": width,
                    "start_time_s": float(group.timestamp_s.iloc[start]),
                    "end_time_s": float(group.timestamp_s.iloc[start + width - 1]),
                }
            )
            values.append(group[LOW].iloc[start : start + width].to_numpy(float).std(axis=0))
    if not candidates:
        raise ValueError("No >=80-point contiguous segment for diagnostic selection")
    amplitudes = np.stack(values)
    scale = np.median(amplitudes, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    scores = np.mean((amplitudes / scale) ** 2, axis=1)
    selected = int(np.argmin(np.abs(scores - np.median(scores))))
    return {
        **candidates[selected],
        "low_excitation_score": float(scores[selected]),
        "candidate_count": len(candidates),
        "selection_rule": "closest_to_median_LOW_only_standardized_axis_excitation_then_earliest",
    }


def _temporal_pages(unit, config):
    frame, methods = unit["frame"], unit["methods"]
    selected = select_low_excitation_segment(frame, config["temporal_window_points"])
    group = (
        frame[frame.sequence_id.astype(str).eq(selected["sequence_id"])]
        .sort_values("timestamp_s")
        .reset_index(drop=True)
    )
    chosen = group.iloc[
        selected["start_index"] : selected["start_index"] + selected["window_points"]
    ]
    key = unit["protocol_id"] + "__" + unit["identity"]
    base = {
        "protocol_id": unit["protocol_id"],
        "identity": unit["identity"],
        "selection": selected,
        "seed": config["representative_model_seed"],
        "reference": "HIGH signal proxy; no independent-state truth",
    }
    pages, segment_rows = [], []
    if unit.get("information_scope") == SHARED_SCOPE:
        for name, segment in frame.groupby("sequence_id", sort=True):
            segment = segment.sort_values("timestamp_s")
            if not np.allclose(np.diff(segment.timestamp_s), 0.025, atol=1e-6, rtol=1e-5):
                raise ValueError("LOW integrated reference must not cross a segment gap")
            integrated = cumulative_trapezoid(
                segment[LOW].to_numpy(float) - segment[HIGH].to_numpy(float),
                segment.timestamp_s.to_numpy(float),
                axis=0,
                initial=0,
            )
            for sensor, axes, unit_name in (
                ("gyro", slice(0, 3), "rad"),
                ("accel", slice(3, 6), "m/s"),
            ):
                segment_rows.append(
                    {
                        "method": "identity",
                        "model_family": "identity",
                        "variant": "shared",
                        "information_scope": SHARED_SCOPE,
                        "sequence_id": str(name),
                        "sensor": sensor,
                        "metric": "integrated_signal_error_endpoint",
                        "unit": unit_name,
                        "mean": float(np.linalg.norm(integrated[-1, axes])),
                        "sample_sd": np.nan,
                        "samples": len(segment),
                        "start_s": float(segment.timestamp_s.iloc[0]),
                        "end_s": float(segment.timestamp_s.iloc[-1]),
                    }
                )
    families = {}
    for prefix, metadata in methods.items():
        if metadata.get("seed") not in (None, config["representative_model_seed"]):
            continue
        family = metadata.get("family") or metadata["method"]
        families.setdefault(family, []).append((prefix, metadata))
        for name, segment in frame.groupby("sequence_id", sort=True):
            segment = segment.sort_values("timestamp_s")
            if not np.allclose(np.diff(segment.timestamp_s), 0.025, atol=1e-6, rtol=1e-5):
                raise ValueError("Integrated error must not cross a segment gap")
            prediction = segment[[axis + "_" + prefix for axis in AXES]].to_numpy(float)
            error = prediction - segment[HIGH].to_numpy(float)
            integrated = cumulative_trapezoid(
                error, segment.timestamp_s.to_numpy(float), axis=0, initial=0
            )
            for sensor, axes, unit_name in (
                ("gyro", slice(0, 3), "rad"),
                ("accel", slice(3, 6), "m/s"),
            ):
                segment_rows.append(
                    {
                        "method": metadata["method"],
                        "model_family": family,
                        "variant": metadata.get("variant", "classical"),
                        "sequence_id": str(name),
                        "sensor": sensor,
                        "metric": "integrated_signal_error_endpoint",
                        "unit": unit_name,
                        "mean": float(np.linalg.norm(integrated[-1, axes])),
                        "sample_sd": np.nan,
                        "samples": len(segment),
                        "start_s": float(segment.timestamp_s.iloc[0]),
                        "end_s": float(segment.timestamp_s.iloc[-1]),
                    }
                )
    for family, entries in families.items():
        rows, spectral = [], []
        for prefix, metadata in entries:
            method = metadata["method"]
            for axis_index, axis in enumerate(AXES):
                time_values = chosen.timestamp_s.to_numpy(float) - chosen.timestamp_s.iloc[0]
                low, high = (
                    chosen[axis + "_low"].to_numpy(float),
                    chosen[axis + "_high"].to_numpy(float),
                )
                prediction = chosen[axis + "_" + prefix].to_numpy(float)
                if not np.isfinite(np.column_stack([low, high, prediction])).all():
                    raise ValueError("Nonfinite saved temporal predictions")
                integral = cumulative_trapezoid(prediction - high, time_values, initial=0)
                low_integral = cumulative_trapezoid(low - high, time_values, initial=0)
                rows.extend(
                    {
                        "method": method,
                        "axis": axis,
                        "sensor": "gyro" if axis_index < 3 else "accel",
                        "time_s": float(t),
                        "low": float(low_value),
                        "high": float(h),
                        "prediction": float(p),
                        "integrated_error": float(v),
                        "low_integrated_error": float(li),
                    }
                    for t, low_value, h, p, v, li in zip(
                        time_values, low, high, prediction, integral, low_integral, strict=True
                    )
                )
                nperseg = min(config["spectral_nperseg"], max(16, len(chosen) // 4))
                frequency, lp = welch(low - high, fs=40, nperseg=nperseg)
                _, pp = welch(prediction - high, fs=40, nperseg=nperseg)
                low_supported = min(np.std(low), np.std(high)) > 1e-12
                prediction_supported = min(np.std(prediction), np.std(high)) > 1e-12
                lc = np.full(len(frequency), np.nan)
                pc = np.full(len(frequency), np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    if low_supported:
                        _, lc = coherence(low, high, fs=40, nperseg=nperseg)
                    if prediction_supported:
                        _, pc = coherence(prediction, high, fs=40, nperseg=nperseg)
                spectral.extend(
                    {
                        "method": method,
                        "axis": axis,
                        "sensor": "gyro" if axis_index < 3 else "accel",
                        "frequency_hz": float(f),
                        "low_error_psd": float(low_psd),
                        "prediction_error_psd": float(p),
                        "low_coherence": float(lco),
                        "prediction_coherence": float(pco),
                        "low_coherence_status": "unsupported_unexcited_axis"
                        if not low_supported
                        else "supported"
                        if np.isfinite(lco)
                        else "unsupported_zero_spectral_power",
                        "prediction_coherence_status": "unsupported_unexcited_axis"
                        if not prediction_supported
                        else "supported"
                        if np.isfinite(pco)
                        else "unsupported_zero_spectral_power",
                        "nperseg": nperseg,
                    }
                    for f, low_psd, p, lco, pco in zip(frequency, lp, pp, lc, pc, strict=True)
                )
        wave, spectrum = pd.DataFrame(rows), pd.DataFrame(spectral)
        title = f"{unit['protocol_id']} · {unit['identity']} · {LABELS.get(family, family)}"
        coverage = {**base, "methods": [meta["method"] for _, meta in entries], "all_axes": AXES}
        pages.append(
            {
                "id": "signals_" + _slug(key + family),
                "kind": "signals",
                "title": title,
                "data": wave,
                "coverage": coverage,
                "source_tables": unit.get("source_tables", []),
                "caption": "Six physical axes in one contiguous segment chosen only by LOW excitation. Fixed seed20261010; all default/tuned methods have appendix pages. HIGH is a device reference, not independent truth.",
            }
        )
        pages.append(
            {
                "id": "spectral_" + _slug(key + family),
                "kind": "spectral",
                "title": title,
                "data": spectrum,
                "extra_data": wave,
                "coverage": coverage,
                "source_tables": unit.get("source_tables", []),
                "caption": "Welch residual PSD uses constant detrending; PSD/coherence are averaged over the three axes within each sensor group. Coherence is unsupported for unexcited axes or undefined spectral bins; source CSV keeps NaN with reasons and a three-axis mean is unsupported when any axis is undefined. Values below1e-24 are floored only for log display; unchanged raw PSD values are saved and this is not a hardware noise floor. Bias remains visible in integrated-error panels. Trapezoidal integrals are gyro/force signal error against HIGH (rad and m/s), not SO(3), inertial position, Allan deviation or trajectory ATE. No integration crosses a segment.",
            }
        )
    if segment_rows:
        pages.append(
            {
                "id": "segment_integrals_" + _slug(key),
                "kind": "segment_integrals",
                "title": key,
                "data": pd.DataFrame(segment_rows),
                "coverage": {**base, "all_segments": True},
                "source_tables": unit.get("source_tables", []),
                "caption": "All contiguous segments and fixed-seed methods are retained. Endpoint norm of integrated gyro/force error relative to HIGH, with separate units; no trajectory assembly or independent-state accuracy claim.",
            }
        )
    return pages


def _legacy_plan_pages(tables, config):
    pages = []
    metric_keys = ["task", "protocol_id", "split", "dataset_id", "low_sensor", "high_sensor"]
    for name, kind in (("metrics", "metrics"), ("deltas", "deltas"), ("transfer", "transfer")):
        frame = tables.get(name, pd.DataFrame()).copy()
        if frame.empty:
            continue
        if "aggregation_level" in frame:
            frame = frame[
                (frame.task.eq("paired") & frame.aggregation_level.eq("pair"))
                | (frame.task.eq("navigation") & frame.aggregation_level.eq("dataset"))
            ]
        if not config["include_validation"] and "split" in frame:
            frame = frame[frame.split.eq("test")]
        for key, group in _groups(frame, metric_keys):
            if kind == "deltas":
                group = group.rename(columns={"delta_mean": "mean", "delta_sample_sd": "sample_sd"})
            title = " · ".join(map(str, key if isinstance(key, tuple) else [key]))
            pages.append(
                {
                    "id": kind + "_" + _slug(title),
                    "kind": kind,
                    "title": title,
                    "data": group.copy(),
                    "coverage": {
                        "group": kind,
                        "rows": len(group),
                        "methods": sorted(group.model_family.unique()),
                        "variants": sorted(group.variant.unique()) if "variant" in group else [],
                    },
                    "caption": (
                        "Matched-seed tuned minus default; negative values are improvements for nonnegative error metrics. No test-based selection. "
                        if kind == "deltas"
                        else "All model variants and deduplicated classical methods; physical gyro/accel units remain separate. "
                    )
                    + STATISTICS,
                }
            )
    for cell, group in _groups(tables.get("training"), ["cell"]):
        cell = cell[0] if isinstance(cell, tuple) else cell
        runs = group[["job_id", "variant", "seed"]].drop_duplicates()
        if len(runs) != 6 or set(runs.variant) != {"default", "tuned"} or runs.seed.nunique() != 3:
            raise ValueError("Every training cell requires both variants and all three final seeds")
        pages.append(
            {
                "id": "training_" + _slug(cell),
                "kind": "training",
                "title": str(cell),
                "data": group.copy(),
                "coverage": {
                    "group": "training",
                    "cell": cell,
                    "run_ids": sorted(runs.job_id),
                    "variants": ["default", "tuned"],
                    "seeds": sorted(runs.seed.unique().tolist()),
                    "final_run_count": 6,
                    "loss_scope": "weighted configured train/validation loss separated by variant; common criterion explicitly named",
                },
                "caption": "All six final training histories; configured losses can differ in units/weights and are separated by variant. Shared validation criterion is labelled; early stopping gives unequal epoch counts. No seed is selected by test score.",
            }
        )
    for key, group in _groups(
        tables.get("ranks"), ["task", "split", "metric", "unit", "aggregation_level"]
    ):
        title = " · ".join(map(str, key))
        pages.append(
            {
                "id": "ranks_" + _slug(title),
                "kind": "ranks",
                "title": title,
                "data": group.copy(),
                "coverage": {
                    "group": "ranks",
                    "rows": len(group),
                    "protocols": sorted(group.protocol_id.unique()),
                },
                "caption": "Ranks of optimization-seed mean errors. Independent AL/Penn training, pooled evaluation slices, leave-recording-out Penn and rotated navigation trajectories retain separate protocol labels; equal ranks use average ties.",
            }
        )
    for key, group in _groups(
        tables.get("rank_correlations"),
        ["task", "split", "metric", "unit", "variant", "aggregation_level"],
    ):
        for start in range(0, len(group), config["correlations_per_page"]):
            chosen = group.iloc[start : start + config["correlations_per_page"]].copy()
            title = (
                " · ".join(map(str, key)) + f" · page{start // config['correlations_per_page'] + 1}"
            )
            pages.append(
                {
                    "id": "correlations_" + _slug(title),
                    "kind": "correlations",
                    "title": title,
                    "data": chosen,
                    "coverage": {"group": "correlations", "rows": len(chosen)},
                    "caption": "Descriptive Spearman and Kendall rank correlations retain distinct training-scope comparisons and pooled slices. Model ranks are not independent recording replicates; no p-value or recording confidence interval is inferred.",
                }
            )
    downstream = tables.get("downstream", pd.DataFrame())
    for key, group in _groups(downstream, ["protocol", "recording"]):
        for gains in (False, True):
            data = group.copy()
            data["mean"] = np.where(
                data.task.eq("attitude"),
                data.endpoint_rmse_deg_mean,
                data.position_endpoint_rmse_m_mean,
            )
            data["sample_sd"] = np.where(
                data.task.eq("attitude"),
                data.get("endpoint_rmse_deg_sample_sd", np.nan),
                data.get("position_endpoint_rmse_m_sample_sd", np.nan),
            )
            if gains:
                identity = data[data.method.eq("identity")][["task", "samples", "mean"]].rename(
                    columns={"mean": "low_mean"}
                )
                data = data.merge(identity, on=["task", "samples"], validate="many_to_one")
                data["gain_percent"] = 100 * (
                    1
                    - np.divide(
                        data["mean"],
                        data.low_mean,
                        out=np.full(len(data), np.nan),
                        where=data.low_mean > 1e-12,
                    )
                )
                data["gain_status"] = np.where(
                    ~np.isfinite(data.low_mean),
                    "unsupported_identity_reference",
                    np.where(
                        data.low_mean <= 1e-12,
                        "unsupported_near_zero_identity_rmse",
                        np.where(
                            np.isfinite(data["mean"]), "supported", "unsupported_endpoint_result"
                        ),
                    ),
                )
            title = " · ".join(map(str, key)) + (
                " · RMSE change vs LOW" if gains else " · absolute RMSE"
            )
            pages.append(
                {
                    "id": "downstream_" + _slug(title),
                    "kind": "downstream",
                    "title": title,
                    "data": data,
                    "gains": gains,
                    "coverage": {
                        "group": "downstream",
                        "protocol": key[0],
                        "recording": key[1],
                        "methods": sorted(data.method.unique()),
                        "horizons_samples": sorted(data.samples.unique().tolist()),
                        "pooled_bs_separate": key[0] == "pooled",
                    },
                    "caption": "Independent optical GT; all four predeclared horizons and all methods retained, including degradations. Four outer recordings are separate from pooled/bs same-recording reference. Oracle initial state and optical GT limits apply. "
                    + STATISTICS,
                }
            )
    for name in ("latency_neural", "latency_classical"):
        for key, group in _groups(
            tables.get(name),
            ["protocol_id", "pair_identity", "task", "device", "batch_size", "window_size"],
        ):
            title = " · ".join(map(str, key))
            pages.append(
                {
                    "id": name + "_" + _slug(title),
                    "kind": "latency",
                    "title": title,
                    "data": group.copy(),
                    "classical": name.endswith("classical"),
                    "coverage": {
                        "group": name,
                        "rows": len(group),
                        "measurement_ids": sorted(group.measurement_id.unique().tolist())
                        if "measurement_id" in group
                        else [],
                        "measurement_context": {
                            "device": str(group.device.iloc[0]),
                            "batch_size": int(group.batch_size.iloc[0]),
                            "window_size": int(group.window_size.iloc[0]),
                        },
                    },
                    "caption": "P50/P95 describe 300 timed calls after30 warmups, not optimization-seed SD. Neural FP32 network-only and classical float64 algorithm-forward are separate pages; classical lag/delta/summary transforms are included and fitted coefficients/bias/normalizers are not torch trainable parameters. Hardware and thread policy are recorded in source data. Architecture timing uses fixed seed20261010 and an explicit mapping, not test accuracy. "
                    + f"This page: {group.device.iloc[0]}, B={int(group.batch_size.iloc[0])} windows/call, W={int(group.window_size.iloc[0])} samples. "
                    + (
                        "Frozen classical CPU timing uses B=1/128, separate from neural timing."
                        if name.endswith("classical")
                        else "Frozen neural receipts use CPU B=1 only and CUDA B=1/128."
                    ),
                }
            )
    for unit in tables.get("temporal", []):
        pages.extend(_temporal_pages(unit, config))
    readable_pages = []
    for page in pages:
        data = page["data"]
        if page["kind"] in {"metrics", "transfer"} and data.task.iloc[0] == "paired":
            families = list(dict.fromkeys(data.loc[data.method.eq("model"), "model_family"]))
            if len(families) > 4:
                for start in range(0, len(families), 4):
                    selected = data[
                        ~data.method.eq("model")
                        | data.model_family.isin(families[start : start + 4])
                    ].copy()
                    readable_pages.append(
                        {
                            **page,
                            "id": page["id"] + f"__methods_{start // 4 + 1}",
                            "title": page["title"] + f" · methods page{start // 4 + 1}",
                            "data": selected,
                            "coverage": {
                                **page["coverage"],
                                "method_page": start // 4 + 1,
                                "page_model_families": families[start : start + 4],
                                "methods": sorted(selected.model_family.unique()),
                                "rows": len(selected),
                                "all_model_families": families,
                            },
                            "caption": page["caption"]
                            + " Method pages follow fixed family order; identical classical references repeat for comparison, never as extra statistical replicates.",
                        }
                    )
                continue
        if page["kind"] == "downstream":
            methods = list(dict.fromkeys(data.method))
            families = list(
                dict.fromkeys(method.split("/", 1)[0] for method in methods if "/" in method)
            )
            if len(families) > 4:
                for start in range(0, len(families), 4):
                    retained = {
                        method
                        for method in methods
                        if "/" not in method
                        or method.split("/", 1)[0] in families[start : start + 4]
                    }
                    readable_pages.append(
                        {
                            **page,
                            "id": page["id"] + f"__methods_{start // 4 + 1}",
                            "title": page["title"] + f" · methods page{start // 4 + 1}",
                            "data": data[data.method.isin(retained)].copy(),
                            "coverage": {
                                **page["coverage"],
                                "method_page": start // 4 + 1,
                                "page_methods": sorted(retained),
                                "methods": sorted(retained),
                                "all_methods_in_view": methods,
                            },
                            "caption": page["caption"]
                            + " Method pages retain all default/tuned families; classical/HIGH references repeat only for visual comparison.",
                        }
                    )
                continue
        readable_pages.append(page)
    pages = readable_pages
    identifiers = [page["id"] for page in pages]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Figure identifiers collide")
    source_keys = {
        "metrics": "metrics",
        "deltas": "deltas",
        "transfer": "transfer",
        "training": "training",
        "ranks": "ranks",
        "correlations": "rank_correlations",
        "downstream": "downstream",
    }
    for page in pages:
        if "source_tables" not in page:
            key = (
                "latency_classical"
                if page.get("classical")
                else "latency_neural"
                if page["kind"] == "latency"
                else source_keys.get(page["kind"], "")
            )
            page["source_tables"] = tables.get("_sources", {}).get(key, [])
        if page["kind"] == "training" and "_source_member" in page["data"]:
            page["source_tables"] = (
                page["data"][["_source_member", "_source_sha256"]]
                .drop_duplicates()
                .rename(columns={"_source_member": "path", "_source_sha256": "sha256"})
                .to_dict("records")
            )
    return pages


def _plan_pages_scoped(tables, config):
    """Plan scope-separated evidence before plotting; legacy figures remain available."""
    if tables.get("_analysis_profile") is None:
        return _legacy_plan_pages(tables, config)
    if tables["_analysis_profile"] != SHARED_PROFILE:
        raise ValueError("Unknown figure analysis_profile")
    if tables.get("_shared_coverage") != {**SHARED_COUNTS, "neural_split_support_references": 720}:
        raise ValueError("Shared figure coverage must retain 54/108/360/720 logical slots")
    pages = _legacy_plan_pages(tables, config)
    for page in pages:
        if page["kind"] == "transfer":
            page["data"] = page["data"].copy()
            page["data"].loc[page["data"].method.ne("model"), "information_scope"] = (
                "source_domain_shared"
            )
            page["coverage"]["classical_information_scope"] = "source_domain_shared"
            page["caption"] = (
                "Transfer classical controls use one source-domain shared calibration fitted from source TRAIN; they do not route by target device identity. "
                + page["caption"]
            )
        paired = page["kind"] == "metrics" and page["data"].task.iloc[0] == "paired"
        old_temporal = page["kind"] in {"signals", "spectral", "segment_integrals"}
        if paired or page["kind"] == "downstream" or old_temporal or page.get("classical"):
            page["id"] = "bank_supplement_" + page["id"]
            page["title"] = "Known-device bank (supplementary) · " + page["title"]
            page["coverage"]["information_scope"] = BANK
            page["caption"] = (
                "Supplementary known-device bank: coefficients are fitted/selected per device pair and selected using known identity. "
                "This condition is separate from the unified six-channel neural mapping and protocol-shared controls. "
                + page["caption"]
            )
            if paired:
                page["data"] = page["data"].copy()
                classical = ~page["data"].method.eq("model")
                page["data"].loc[classical, "information_scope"] = BANK
            elif page["kind"] == "downstream":
                page["data"] = page["data"].copy()
                classical = ~page["data"].method.str.contains("/", regex=False) & page[
                    "data"
                ].method.ne("HIGH")
                page["data"].loc[classical, "information_scope"] = BANK
    main_tables = {
        "metrics": tables["metrics_shared"],
        "downstream": tables["downstream_shared"],
        "temporal": tables["temporal_shared"],
        "_sources": {
            "metrics": tables.get("_sources", {}).get("metrics_shared", []),
            "downstream": tables.get("_sources", {}).get("downstream_shared", []),
        },
    }
    for page in _legacy_plan_pages(main_tables, config):
        page["id"] = "shared_main_" + page["id"]
        page["title"] = "Protocol-shared main comparison · " + page["title"]
        page["coverage"].update(
            information_scope=SHARED_SCOPE,
            identity_reference="common_LOW_once",
            shared_controls_index_sha256=tables["_shared_controls_index_sha256"],
        )
        page["caption"] = (
            "Main common-task comparison: unified six-channel neural models and one shared classical state per protocol, without device-ID coefficient routing. "
            "Neural n=3 optimization seeds; shared n=1, SD not applicable. Common LOW/identity appears once per comparison unit. "
            "TRAIN physical splits agree; raw-SI float64 OLS uses all N rows, whereas neural normalized FP32 window training has different loss/exposure. "
            "Context, causality and capacity can differ across algorithms. "
            "HIGH is a device proxy for signal panels; independent optical GT is used only for downstream propagation. "
            "ARX-style controls use LOW lag/delta, without HIGH feedback. " + page["caption"]
        )
        pages.append(page)
    sources = tables.get("_sources", {})
    cost_specs = [
        (
            "shared_state_inventory",
            "shared_parameters",
            ["protocol", "information_scope"],
            "Stored coefficients per effective state",
            ["coefficient_count", "coefficient_bytes", "serialized_state_bytes"],
        ),
        (
            "shared_bank_parameter_totals",
            "shared_parameters",
            ["protocol"],
            "Deployment totals across unique effective pair states",
            ["bank_coefficient_count", "bank_coefficient_bytes", "all_artifact_serialized_bytes"],
        ),
        (
            "shared_bank_parameter_totals",
            "shared_parameters",
            ["protocol"],
            "Deduplicated numerical storage and whole artifacts",
            [
                "numeric_unique_coefficient_count",
                "numeric_unique_coefficient_bytes",
                "all_artifact_serialized_bytes",
            ],
        ),
        (
            "shared_latency_summary",
            "shared_latency",
            ["protocol", "pair_identity_json", "batch_size_per_pair", "window_size"],
            "Measured single-state CPU forward",
            [],
        ),
        (
            "shared_workload_latency",
            "shared_workload",
            ["protocol", "batch_size_per_pair", "window_size"],
            "Measured complete P × B CPU workload",
            [],
        ),
        (
            "shared_alias_audit",
            "shared_alias",
            ["protocol"],
            "Evidence of shared/bank numerical relationship",
            [],
        ),
    ]
    for name, kind, keys, purpose, fields in cost_specs:
        for key, group in _groups(tables.get(name), keys):
            data = group.copy()
            # Common identity is one reference, never two independent controls.
            if "information_scope" in data and data.information_scope.nunique() > 1:
                data = data[
                    ~(data.canonical_method.eq("identity") & data.information_scope.eq(BANK))
                ]
            group_values = key if isinstance(key, tuple) else (key,)
            actual_keys = [column for column in keys if column in group]
            title = (
                purpose
                + " · "
                + " · ".join(
                    _pair_display(value) if column == "pair_identity_json" else str(value)
                    for column, value in zip(actual_keys, group_values, strict=True)
                )
            )
            page_sources = list(sources.get(name, []))
            if kind in {"shared_latency", "shared_workload"}:
                page_sources += sources.get("shared_latency_raw", [])
            pages.append(
                {
                    "id": name + "_" + _slug(title),
                    "kind": kind,
                    "title": title,
                    "data": data,
                    "fields": fields,
                    "source_tables": page_sources,
                    "coverage": {
                        "group": name,
                        "information_scope": SHARED_SCOPE,
                        "included_scopes": sorted(data.information_scope.unique()),
                        "rows": len(data),
                        "source_group_key": dict(zip(actual_keys, group_values, strict=True)),
                        "identity_reference": "common_LOW_once",
                        "shared_controls_index_sha256": tables["_shared_controls_index_sha256"],
                        "pair_counts": sorted(data.pair_count.unique().tolist())
                        if "pair_count" in data
                        else [],
                        "windows_per_call": sorted(data.windows_per_call.unique().tolist())
                        if "windows_per_call" in data
                        else [],
                        "batch_size_per_pair": sorted(data.batch_size_per_pair.unique().tolist())
                        if "batch_size_per_pair" in data
                        else [],
                        "window_size": sorted(data.window_size.unique().tolist())
                        if "window_size" in data
                        else [],
                    },
                    "caption": "Stored float64 coefficients include intercepts and are not statistical degrees of freedom or neural torch parameters. Raw SI uses zero fitted normalization coefficients. Per-state storage, deployment pair sums, exact numerical-content deduplication and whole serialized artifact sizes have distinct meanings. Whole bank artifacts may include other methods and repeated per-final copies; their bytes are not deployment-only coefficient bytes. CPU algorithm-forward includes lag/delta transforms and preloaded same-stream history. P50/P95 are actually measured call distributions, never sums of pair quantiles; throughput is a separate complete workload loop. Workload calls contain P × B windows, not one input. Neural FP32 network-only CPU/GPU figures remain separate. Alias labels require saved provenance and numerical proof; Penn equality is not assumed.",
                }
            )
    for page in pages:
        page["figure_contract"] = {
            "backend": "Python/matplotlib",
            "archetype": "quantitative grid",
            "claim": page["title"],
            "evidence": page["coverage"],
            "statistics": "Neural: 3 optimizer seeds; deterministic shared/bank: n=1, blank SD; no independent-recording CI",
            "review_risks": [
                "Keep information scopes distinct",
                "Keep physical units distinct",
                "Retain negative/unsupported outcomes",
                "No global SOTA or independent trial claim",
            ],
            "exports": {
                "width_inches": 7.2,
                "editable_svg_text": True,
                "embedded_pdf_fonts": True,
                "png_dpi": 300,
                "tiff_dpi": 600,
            },
        }
    if len({page["id"] for page in pages}) != len(pages):
        raise ValueError("Shared/bank figure identifiers collide")
    return pages


def annotate_temporal_frame(frame, *, task=None, protocol=None, model_types=None):
    """Add display metadata without changing source identities, values or ranks."""
    data = frame.copy()
    if data.empty or not {"canonical_method", "model_family", "method"}.intersection(data):
        return data
    rows, resolved_types = [], []
    model_types = model_types or {}
    for _, row in data.iterrows():
        from_display = False
        method = row.get("canonical_method")
        if pd.isna(method):
            method = row.get("model_family")
        if pd.isna(method) or method == "model":
            method = row.get("method")
            from_display = True
        if not isinstance(method, str):
            raise ValueError("Temporal display requires an exact method/family identity")
        variant = row.get("variant")
        if from_display:
            method = method.removesuffix(" [shared]")
            if "/" in method:
                method, display_variant = method.split("/", 1)
                if display_variant not in {"default", "tuned"}:
                    raise ValueError("Unknown temporal display variant: " + display_variant)
                if pd.isna(variant):
                    variant = display_variant
        source_protocol = row.get("protocol_id", row.get("protocol"))
        if pd.isna(source_protocol):
            source_protocol = protocol or ""
        source_protocol = str(source_protocol)
        row_task = task or row.get("task")
        if row_task in {"attitude", "strapdown"}:
            row_task = "paired"
        elif row_task is None or pd.isna(row_task):
            row_task = "navigation" if source_protocol.startswith("navigation") else "paired"
        elif row_task not in {"paired", "navigation"}:
            raise ValueError("Unknown temporal task in saved figure data: " + str(row_task))
        if source_protocol and source_protocol.startswith("navigation") != (
            row_task == "navigation"
        ):
            raise ValueError("Temporal task and protocol identity disagree")
        if method == "HIGH":
            if row_task != "paired":
                raise ValueError("HIGH is a measured paired reference, not a navigation method")
            rows.append(
                {
                    "temporal_class": "measured_reference",
                    "temporal_label": "Reference",
                    "temporal_label_zh": "观测HIGH参考",
                    "temporal_order": 3,
                    "temporal_note": "Synchronous measured HIGH device reference; no learned-method causal classification or end-to-end online claim.",
                }
            )
            resolved_types.append(None)
            continue
        temporal_method = method
        if method in _TRANSFER_CLASSICAL_TEMPORAL_METHODS:
            if (
                row_task != "paired"
                or source_protocol not in {"alimu_to_penn", "penn_to_alimu"}
                or variant != "classical"
            ):
                raise ValueError("Source classical temporal alias has incompatible context")
            # Resolve only display classification; saved method IDs/values stay intact.
            temporal_method = _TRANSFER_CLASSICAL_TEMPORAL_METHODS[method]
        model_type = row.get("model_type")
        if pd.isna(model_type):
            model_type = row.get("temporal_model_type")
        if pd.isna(model_type):
            source_protocol = {"alimu_to_penn": "alimu_only", "penn_to_alimu": "penn_only"}.get(
                source_protocol, source_protocol
            )
            model_type = model_types.get((source_protocol, method, variant))
        if pd.isna(model_type):
            model_type = None
        rows.append(classify_temporal_support(row_task, temporal_method, model_type=model_type))
        resolved_types.append(model_type)
    annotations = pd.DataFrame(rows, index=data.index)
    for field in TEMPORAL_FIELDS:
        if field in data and not data[field].equals(annotations[field]):
            raise ValueError("Saved temporal display metadata conflicts with the classifier")
        data[field] = annotations[field]
    data["temporal_model_type"] = resolved_types
    return data


def plan_pages(tables, config):
    pages = _plan_pages_scoped(tables, config)
    grouped_kinds = {
        "metrics",
        "deltas",
        "transfer",
        "latency",
        "shared_parameters",
        "shared_latency",
        "shared_workload",
    }
    for page in pages:
        page["data"] = annotate_temporal_frame(
            page["data"],
            protocol=page["coverage"].get("protocol_id", page["coverage"].get("protocol")),
            model_types=tables.get("_model_types"),
        )
        data = page["data"]
        if "temporal_class" not in data:
            if page["kind"] in {"ranks", "correlations"}:
                page["caption"] += (
                    " Original overall ranks/correlations remain unchanged; no within-class reranking."
                )
            continue
        labels = (
            data[["temporal_class", "temporal_label", "temporal_order"]]
            .drop_duplicates()
            .sort_values("temporal_order", kind="stable")
        )
        page["coverage"]["temporal_classes"] = labels.to_dict("records")
        page["coverage"]["temporal_support_sha256"] = sha256_file(TEMPORAL_SOURCE)
        page["coverage"]["temporal_groups_visible"] = page["kind"] in grouped_kinds
        page["caption"] += (
            " Temporal groups describe per-method inference context on prepared 40 Hz inputs in eval mode with frozen normalization/BatchNorm statistics. "
            "Causal uses current/past prepared samples; Window may use later samples within its supplied window. "
            "Navigation is window-endpoint displacement, not a paired per-sample causality claim. "
            "These labels are orthogonal to information scope and do not establish preprocessing/streaming or end-to-end online causality. "
            "Every method and original numerical value is retained; original overall ranks are not replaced by within-class ranks."
        )
        page.setdefault("figure_contract", {}).update(
            temporal_comparison_conclusion="Inference-context access differs across methods; visibly separate conditions without changing metrics or overall ranks.",
            temporal_support_sha256=sha256_file(TEMPORAL_SOURCE),
        )
    return pages


def _temporal_layout(data):
    """Stable class bands for a one-row-per-method plotting frame."""
    if "temporal_order" not in data:
        return data, np.arange(len(data), dtype=float), [], float(len(data))
    data = data.sort_values("temporal_order", kind="stable").reset_index(drop=True)
    positions, groups, cursor = [], [], 0.0
    for (category, label), group in data.groupby(["temporal_class", "temporal_label"], sort=False):
        header = cursor
        values = np.arange(len(group), dtype=float) + cursor + 1
        positions.extend(values)
        groups.append(
            {
                "class": category,
                "label": label,
                "header": header,
                "first": float(values[0]),
                "last": float(values[-1]),
            }
        )
        cursor = float(values[-1]) + 1.5
    return data, np.asarray(positions), groups, cursor


def _temporal_bands(ax, groups):
    shades = {
        "sample_causal_eval": "#F1F6FA",
        "window_lookahead_eval": "#FAF4EE",
        "window_endpoint": "#F1F6FA",
        "measured_reference": "#F5F5F5",
    }
    for group in groups:
        ax.axhspan(
            group["first"] - 0.45,
            group["last"] + 0.45,
            color=shades[group["class"]],
            zorder=-2,
            linewidth=0,
        )
        ax.text(
            0.02,
            group["header"],
            group["label"],
            transform=ax.get_yaxis_transform(),
            fontsize=6.7,
            fontweight="bold",
            va="center",
        )


def _footer_axis_geometry(fig):
    renderer = fig.canvas.get_renderer()
    footers = [item for item in fig.texts if str(item.get_gid()).startswith("figure_footer_")]
    labels = [
        item
        for ax in fig.axes
        for item in [
            ax.xaxis.label,
            ax.xaxis.get_offset_text(),
            *ax.get_xticklabels(minor=False),
            *ax.get_xticklabels(minor=True),
        ]
        if item.get_visible() and item.get_text()
    ]
    return (
        [(item, item.get_window_extent(renderer)) for item in footers],
        [(item, item.get_window_extent(renderer)) for item in labels],
    )


def _reserve_footer(fig, margin_px=8):
    """Keep notes separate from axis units/ticks on short and tall panels."""
    for index, item in enumerate(item for item in fig.texts if item is not fig._suptitle):
        item.set_gid(f"figure_footer_{index}")
        _fit_figure_text(fig, item)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cursor = fig.bbox.height * 0.02
    for item in sorted(
        (item for item in fig.texts if str(item.get_gid()).startswith("figure_footer_")),
        key=lambda item: item.get_position()[1],
    ):
        box = item.get_window_extent(renderer)
        if box.y0 < cursor:
            x, y = item.get_position()
            item.set_position((x, y + (cursor - box.y0) / fig.bbox.height))
            box = item.get_window_extent(renderer)
        cursor = box.y1 + 5
    footers, labels = _footer_axis_geometry(fig)
    if footers and labels:
        top = max(box.y1 for _, box in footers)
        lower = min(box.y0 for _, box in labels)
        if lower < top + margin_px:
            bottom = fig.subplotpars.bottom + (top + margin_px - lower) / fig.bbox.height
            if bottom >= fig.subplotpars.top - 0.1:
                raise ValueError("Figure has insufficient room for data and conditional notes")
            fig.subplots_adjust(bottom=bottom)
            fig.canvas.draw()


def _footer_axis_overlaps(fig):
    footers, labels = _footer_axis_geometry(fig)
    return [
        {"footer": footer.get_text(), "axis_label": label.get_text()}
        for footer, footer_box in footers
        for label, label_box in labels
        if footer_box.overlaps(label_box)
    ]


def _base(title, rows=1, columns=1, height=4.0, left=0.22, width=7.2):
    fig, axes = plt.subplots(rows, columns, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=left, right=0.97, top=0.88, bottom=0.16, wspace=0.37, hspace=0.45)
    heading = fig.suptitle(title, x=0.02, ha="left", y=0.98, fontsize=8.5, fontweight="bold")
    _fit_figure_text(fig, heading)
    renderer = fig.canvas.get_renderer()
    lower = heading.get_window_extent(renderer).y0 / fig.bbox.height
    fig.subplots_adjust(top=min(0.88, lower - 0.035))
    return fig, axes


def _fit_figure_text(fig, item):
    renderer = fig.canvas.get_renderer()
    font = item.get_fontproperties()
    limit = fig.bbox.width * 0.94
    lines, current = [], ""
    for word in item.get_text().split():
        candidate = current + " " + word if current else word
        if renderer.get_text_width_height_descent(candidate, font, ismath=False)[0] > limit:
            if current:
                lines.append(current)
                current = ""
            # A source identity can be an unbroken token; keep it on the canvas too.
            for character in word:
                if (
                    current
                    and renderer.get_text_width_height_descent(
                        current + character, font, ismath=False
                    )[0]
                    > limit
                ):
                    lines.append(current)
                    current = ""
                current += character
        else:
            current = candidate
    if current:
        lines.append(current)
    item.set_text("\n".join(lines))


def draw_page(page, config):
    kind, data = page["kind"], page["data"].copy()
    if kind in {"shared_parameters", "shared_latency", "shared_workload", "shared_alias"}:
        data = data.reset_index(drop=True)
        if kind == "shared_alias":
            positions, temporal_groups, extent = np.arange(len(data)), [], len(data)
        else:
            data, positions, temporal_groups, extent = _temporal_layout(data)
        names = []
        repeated_methods = data.canonical_method.duplicated(keep=False)
        for _index, row in data.iterrows():
            method = LABELS.get(row.canonical_method, row.canonical_method.replace("_", " "))
            method = method.replace("ARX-style LOW lag/delta", "ARX LOW lag/delta")
            scope = "shared" if row.information_scope == SHARED_SCOPE else "bank"
            pair = ""
            if (
                repeated_methods.iloc[_index]
                and page["coverage"]["group"] == "shared_state_inventory"
            ):
                identity = json.loads(row.pair_identity_json)
                pair = " · " + "/".join(
                    str(identity.get(key, "")) for key in ("low_sensor", "high_sensor")
                )
            names.append(textwrap.fill(method + " [" + scope + "]" + pair, width=38))
        if kind == "shared_alias":
            fig, axes = _base(page["title"], 1, 1, height=4.6, left=0.33)
            ax = axes[0, 0]
            statuses = [
                "common_identity",
                "exact_alias",
                "numerically_equivalent",
                "not_alias",
                "not_applicable_multi_pair_bank",
            ]
            for index, row in data.iterrows():
                if row.alias_status not in statuses:
                    raise ValueError("Unknown scientific alias status")
                ax.text(0.02, index, row.alias_status.replace("_", " "), va="center", fontsize=7)
            ax.set_yticks(range(len(data)), names)
            ax.set_ylim(len(data) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            fig, axes = _base(page["title"], 1, 3, height=max(4.6, extent * 0.31 + 1.5), left=0.32)
            fields = (
                page["fields"]
                if kind == "shared_parameters"
                else ["p50_ms", "p95_ms", "loop_throughput_windows_per_second"]
            )
            titles = {
                "coefficient_count": "Stored coefficients [scalars]",
                "coefficient_bytes": "Coefficient storage [bytes]",
                "serialized_state_bytes": "Whole state artifact [bytes]",
                "bank_coefficient_count": "Deployment coefficients [scalars]",
                "bank_coefficient_bytes": "Deployment coefficients [bytes]",
                "all_artifact_serialized_bytes": "Whole state artifacts [bytes]",
                "numeric_unique_coefficient_count": "Unique numeric coefficients [scalars]",
                "numeric_unique_coefficient_bytes": "Unique coefficient storage [bytes]",
                "p50_ms": "Call P50 [ms]",
                "p95_ms": "Call P95 [ms]",
                "loop_throughput_windows_per_second": "Loop throughput [windows/s]",
            }
            for column, field in enumerate(fields):
                ax = axes[0, column]
                _temporal_bands(ax, temporal_groups)
                values = pd.to_numeric(data[field], errors="coerce").to_numpy(float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(
                        "State/timing figure requires actual finite nonnegative measurements"
                    )
                for index, value in enumerate(values):
                    ax.scatter(
                        value,
                        positions[index],
                        s=13,
                        color=COLORS["virtual"]
                        if data.iloc[index].information_scope == SHARED_SCOPE
                        else COLORS["high"],
                    )
                ax.set_yticks(positions, names if column == 0 else [])
                ax.set_ylim(extent - 0.5, -0.5)
                ax.set_xlabel(textwrap.fill(titles[field], 20))
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                ax.tick_params(axis="y", length=0, labelsize=6.2)
                ax.grid(axis="y", visible=False)
        footer = (
            "Alias relationship follows audited state/support proofs; it does not add independent repeats."
            if kind == "shared_alias"
            else "n=1 deterministic state. Stored scalar count is not statistical DoF. Whole artifacts may contain other methods."
            if kind == "shared_parameters"
            else "CPU float64 algorithm-forward; same-stream history and transforms included. Measured distributions, not seed SD."
        )
        fig.text(0.02, 0.025, textwrap.fill(footer, 118), fontsize=6.2)
        if temporal_groups:
            fig.text(
                0.02,
                0.105 if kind in {"shared_latency", "shared_workload"} else 0.065,
                TEMPORAL_CONDITION,
                fontsize=6,
            )
        if kind in {"shared_latency", "shared_workload"}:
            sizes = data[
                ["pair_count", "batch_size_per_pair", "windows_per_call", "window_size"]
            ].drop_duplicates()
            if len(sizes) != 1:
                raise ValueError("Timing page cannot combine different P × B workloads")
            sizes = sizes.iloc[0]
            if sizes.windows_per_call != sizes.pair_count * sizes.batch_size_per_pair:
                raise ValueError("Timing workload windows differ from actual P × B")
            fig.text(
                0.02,
                0.062,
                f"P={int(sizes.pair_count)} pairs; B={int(sizes.batch_size_per_pair)} windows/pair; W={int(sizes.window_size)} samples; {int(sizes.windows_per_call)} windows/call.",
                fontsize=6.3,
            )
        _reserve_footer(fig)
        return fig
    if kind in {"metrics", "deltas", "transfer"}:
        metrics = PAIRED_METRICS if data.task.iloc[0] == "paired" else NAV_METRICS
        metrics = [metric for metric in metrics if metric in set(data.metric)]
        data["display"] = data.apply(label, axis=1)
        method_rows = data[
            ["display", *[field for field in TEMPORAL_FIELDS if field in data]]
        ].drop_duplicates()
        if method_rows.display.duplicated().any():
            raise ValueError("One displayed method has inconsistent temporal metadata")
        method_rows, y, temporal_groups, extent = _temporal_layout(method_rows)
        names = method_rows.display.tolist()
        rows = 2 if data.task.iloc[0] == "paired" else 1
        fig, axes = _base(
            page["title"],
            rows,
            3,
            height=max(4.0, rows * (extent * (0.18 if rows == 2 else 0.135) + 0.65) + 0.8),
            left=0.24,
        )
        for index, metric in enumerate(metrics):
            ax = axes.flat[index]
            _temporal_bands(ax, temporal_groups)
            selected = data[data.metric.eq(metric)].set_index("display").reindex(names)
            if selected.index.duplicated().any():
                raise ValueError("Repeated plot method/metric identity")
            for row, (_, value) in enumerate(selected.iterrows()):
                sd = value.get("sample_sd", np.nan)
                ax.errorbar(
                    value["mean"],
                    y[row],
                    xerr=sd if pd.notna(sd) else None,
                    marker="o",
                    ms=3,
                    lw=0.8,
                    capsize=1.5,
                    color=COLORS["accent"]
                    if value.get("variant") == "tuned"
                    else COLORS["virtual"]
                    if value.get("variant") == "default"
                    else COLORS["high"],
                )
                if pd.notna(value.get("reference_mean", np.nan)):
                    ax.scatter(
                        value.reference_mean,
                        y[row],
                        marker="s",
                        s=10,
                        facecolors="none",
                        edgecolors="#999999",
                    )
            ax.set_yticks(y, names if index % 3 == 0 else [])
            ax.set_ylim(extent - 0.5, -0.5)
            ax.tick_params(axis="y", length=0)
            metric_label = METRIC_LABELS[metric]
            ax.set_title(metric_label, fontsize=7.5)
            unit = str(selected.unit.dropna().iloc[0]) if "unit" in selected else "physical units"
            ax.set_xlabel(("Tuned − default " if kind == "deltas" else "") + f"[{unit}]")
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.grid(axis="y", visible=False)
            if kind == "deltas":
                ax.axvline(0, color="#999999", lw=0.7)
        fig.text(
            0.02,
            0.025,
            "Matched seeds: mean ± sample SD. Negative decreases error; no recording CI."
            if kind == "deltas"
            else "Circles: source-only transfer; hollow squares: target-trained reference. Seed SD, not recording CI."
            if kind == "transfer"
            else "D/T: default/tuned; S/B: shared/bank. Neural seed SD; classical n=1. No recording CI.",
            fontsize=6.5,
        )
        if temporal_groups:
            fig.text(0.02, 0.065, TEMPORAL_CONDITION, fontsize=6)
        _reserve_footer(fig)
        return fig
    if kind == "training":
        fig, axes = _base(page["title"], 2, 3, height=6.0, left=0.09)
        seeds = sorted(data.seed.unique())
        palette = [COLORS["virtual"], COLORS["low"], COLORS["accent"]]
        for variant, column in (("default", 0), ("tuned", 1)):
            part = data[data.variant.eq(variant)]
            for seed, color in zip(seeds, palette, strict=True):
                selected = part[part.seed.eq(seed)].sort_values("epoch")
                for row, value in ((0, "train_loss"), (1, "val_loss")):
                    axes[row, column].plot(
                        selected.epoch, selected[value], color=color, label=str(seed)
                    )
                    axes[row, column].set_xlabel("Epoch")
                    axes[row, column].set_ylabel("Configured " + value.replace("_", " "))
                    axes[row, column].set_title(variant, fontsize=7.5)
                axes[0, 2].plot(
                    selected.epoch,
                    selected.criterion,
                    color=color,
                    linestyle="-" if variant == "default" else "--",
                    label=f"{seed} {variant[0].upper()}",
                )
        if data.criterion_name.nunique() != 1:
            raise ValueError("A common validation panel cannot mix criteria")
        axes[0, 2].set_title(str(data.criterion_name.iloc[0]), fontsize=7)
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylabel("Shared validation criterion")
        handles, names = axes[0, 2].get_legend_handles_labels()
        fig.legend(
            handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.025),
            fontsize=5.5, ncol=3,
        )
        axes[1, 2].axis("off")
        text = ["Configured losses differ in scale."]
        for variant in ("default", "tuned"):
            part = data[data.variant.eq(variant)]
            if part.loss_recipe.nunique() != 1:
                raise ValueError("One variant mixes different configured loss recipes across seeds")
            recipe = json.loads(part.loss_recipe.iloc[0])
            active = {
                key.removesuffix("_weight"): value
                for key, value in recipe.items()
                if key.endswith("_weight") and value not in (0, None)
            }
            text.extend(
                [
                    f"{variant.upper()} · T={sorted(int(value) for value in part.window_size.unique())}",
                    "loss: " + str(recipe.get("loss", "weighted composite")),
                    f"lr0={recipe.get('lr', 'unrecorded')}",
                    f"WD={recipe.get('weight_decay', 'unrecorded')}",
                ]
            )
            if active:
                text.extend(
                    textwrap.wrap(
                        "weights: " + ", ".join(f"{key}={value}" for key, value in active.items()),
                        width=37,
                    )
                )
        axes[1, 2].text(0, 0.98, "\n".join(text), va="top", fontsize=6.1)
        return fig
    if kind == "downstream":
        names = list(dict.fromkeys(data.method))
        horizons = sorted(data.samples.unique())
        display_title = page["title"].replace("RMSE change vs LOW", "RMSE reduction vs LOW")
        fig, axes = _base(display_title, 1, 2, height=max(5.3, len(names) * 0.23 + 1.8), left=0.24)
        for index, task in enumerate(("attitude", "strapdown")):
            ax = axes[0, index]
            selected = data[data.task.eq(task)].set_index(["method", "samples"])
            matrix = np.array(
                [
                    [
                        selected.loc[(name, size), "gain_percent" if page["gains"] else "mean"]
                        for size in horizons
                    ]
                    for name in names
                ]
            )
            if page["gains"]:
                finite_values = matrix[np.isfinite(matrix)]
                maximum = max(1, np.abs(finite_values).max()) if len(finite_values) else 1
                norm = colors.TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
                cmap = colors.LinearSegmentedColormap.from_list(
                    "gain", [COLORS["low"], "white", COLORS["accent"]]
                )
            else:
                finite_values = matrix[np.isfinite(matrix)]
                norm = colors.Normalize(
                    vmin=float(finite_values.min()) if len(finite_values) else 0,
                    vmax=float(finite_values.max()) + 1e-15 if len(finite_values) else 1,
                )
                cmap = "Blues"
            image = ax.imshow(matrix, aspect="auto", norm=norm, cmap=cmap)
            for row, name in enumerate(names):
                for column, size in enumerate(horizons):
                    record = selected.loc[(name, size)]
                    value = matrix[row, column]
                    caption = (
                        "n/a"
                        if not np.isfinite(value)
                        else f"{value:+.0f}%"
                        if page["gains"]
                        else f"{value:.3g}"
                    )
                    if not page["gains"] and record.optimizer_runs == 3:
                        caption += f"\n±{record.sample_sd:.2g}"
                    ax.text(
                        column,
                        row,
                        caption,
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        color="white"
                        if norm(value) > 0.73 or (page["gains"] and norm(value) < 0.25)
                        else COLORS["high"],
                    )
            ax.set_yticks(
                range(len(names)),
                [
                    LABELS.get(name, name)
                    + (
                        " [S]"
                        if selected.loc[(name, horizons[0])].get("information_scope")
                        == SHARED_SCOPE
                        and name != "identity"
                        else " [B]"
                        if selected.loc[(name, horizons[0])].get("information_scope") == BANK
                        else ""
                    )
                    for name in names
                ]
                if index == 0
                else [],
            )
            ax.set_xticks(range(len(horizons)), [f"{(size - 1) / 40:.3f}" for size in horizons])
            ax.set_xlabel("Endpoint span [s]")
            ax.set_title(
                ("Attitude RMSE reduction vs LOW [%]" if task == "attitude" else "Position RMSE reduction vs LOW [%]")
                if page["gains"]
                else ("Attitude [°]" if task == "attitude" else "Position [m]"),
                fontsize=8,
            )
            ax.grid(False)
            fig.colorbar(
                image, ax=ax, orientation="horizontal", fraction=0.035, pad=0.09
            ).ax.tick_params(labelsize=6)
        fig.text(
            0.02,
            0.025,
            ("Reduction = 100 × (LOW − method) / LOW; positive improves, negative degrades. "
             "Seed SD is not recording CI. Pooled bs is a separate view.")
            if page["gains"] else
            "All methods/horizons, including degradation. Seed SD is not recording CI. Pooled bs is a separate view.",
            fontsize=6.3,
        )
        return fig
    if kind in {"ranks", "segment_integrals"}:
        if kind == "ranks":
            data["display"] = data.model_family.map(lambda name: LABELS.get(name, name))
            data["column"] = data.protocol_id + "/" + data.dataset_id + "/" + data.variant
            if "low_sensor" in data:
                data["column"] += "/" + data.low_sensor + "→" + data.high_sensor
            pivot = data.pivot(index="display", columns="column", values="rank")
            fig, axes = _base(page["title"], height=max(4.5, len(pivot) * 0.18 + 2.3), left=0.24)
            ax = axes[0, 0]
            ax.imshow(pivot, aspect="auto", cmap="Blues_r")
            for row in range(len(pivot)):
                for col in range(len(pivot.columns)):
                    if pd.notna(pivot.iloc[row, col]):
                        ax.text(
                            col,
                            row,
                            f"{pivot.iloc[row, col]:g}",
                            color="white"
                            if float(pivot.iloc[row, col])
                            <= (float(np.nanmin(pivot)) + float(np.nanmax(pivot))) / 2
                            else COLORS["high"],
                            ha="center",
                            va="center",
                            fontsize=6,
                        )
            ax.set_yticks(range(len(pivot)), pivot.index)
            ax.set_xticks(
                range(len(pivot.columns)),
                ["C" + str(i + 1) for i in range(len(pivot.columns))],
                rotation=45,
                ha="right",
                fontsize=6.5,
            )
            fig.text(
                0.02,
                0.025,
                "Columns: lexicographic protocol/dataset/variant/device identities in source CSV; all domains retained.",
                fontsize=6.2,
            )
            ax.grid(False)
            fig.subplots_adjust(bottom=0.35)
        else:
            fig, axes = _base(
                page["title"], 1, 2, height=max(5, data.method.nunique() * 0.17 + 1.8), left=0.24
            )
            for ax, sensor in zip(axes.flat, ("gyro", "accel"), strict=True):
                selected = data[data.sensor.eq(sensor)]
                pivot = selected.pivot(index="method", columns="sequence_id", values="mean")
                image = ax.imshow(pivot, aspect="auto", cmap="Blues")
                if len(pivot.columns) <= 8:
                    minimum, maximum = float(np.nanmin(pivot)), float(np.nanmax(pivot))
                    for row in range(len(pivot)):
                        for column in range(len(pivot.columns)):
                            value = float(pivot.iloc[row, column])
                            ax.text(
                                column,
                                row,
                                f"{value:.2g}",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color="white"
                                if value > minimum + 0.7 * (maximum - minimum)
                                else COLORS["high"],
                            )
                fig.colorbar(
                    image, ax=ax, orientation="horizontal", fraction=0.035, pad=0.12
                ).ax.tick_params(labelsize=6)
                ax.set_yticks(
                    range(len(pivot)),
                    [LABELS.get(name, name) for name in pivot.index] if sensor == "gyro" else [],
                )
                ax.set_xticks(range(len(pivot.columns)), range(1, len(pivot.columns) + 1))
                ax.set_xlabel("Contiguous segment index; mapping in CSV")
                ax.set_title(
                    "Integrated gyro error [rad]"
                    if sensor == "gyro"
                    else "Integrated force error [m/s]",
                    fontsize=7,
                )
                ax.grid(False)
        return fig
    if kind == "correlations":
        names = []
        for _, row in data.iterrows():
            sides = []
            for side in ("left", "right"):
                text = f"{row[side + '_protocol']}/{row[side + '_dataset']}"
                device = row.get(side + "_low_sensor")
                if pd.notna(device) and str(device) != "__all_devices__":
                    text += f" [{device}→{row[side + '_high_sensor']}]"
                sides.append(text)
            names.append(textwrap.fill(str(row.comparison) + ": " + " ↔ ".join(sides), width=47))
        fig, axes = _base(page["title"], 1, 2, height=max(4.4, len(data) * 0.55 + 1.3), left=0.43)
        for index, metric in enumerate(("spearman_rho", "kendall_tau_b")):
            ax = axes[0, index]
            ax.scatter(data[metric], np.arange(len(data)), s=18, color=COLORS["virtual"])
            ax.set_yticks(range(len(data)), names if index == 0 else [], fontsize=6.2)
            ax.set_ylim(len(data) - 0.5, -0.5)
            ax.set_xlim(-1.05, 1.05)
            ax.set_title(metric.replace("_", " "), fontsize=7)
            ax.axvline(0, color="#999999", lw=0.6)
        return fig
    if kind == "latency":
        data, y, temporal_groups, extent = _temporal_layout(data)
        data["display"] = data.apply(label, axis=1)
        names = data.display.tolist()
        fig, axes = _base(page["title"], 1, 3, height=max(4.2, extent * 0.19 + 1.3), left=0.25)
        fields = [
            "fitted_scalar_count" if page["classical"] else "parameters",
            "wall_p50_ms",
            "throughput",
        ]
        for index, field in enumerate(fields):
            ax = axes[0, index]
            _temporal_bands(ax, temporal_groups)
            x = data[field].to_numpy(float)
            ax.scatter(
                x,
                y,
                s=14,
                color=COLORS["high"] if page["classical"] else COLORS["virtual"],
            )
            if field == "wall_p50_ms":
                ax.hlines(y, x, data.wall_p95_ms, color=COLORS["virtual"], lw=0.9)
                ax.scatter(
                    data.wall_p95_ms,
                    y,
                    marker="|",
                    color=COLORS["virtual"],
                    s=22,
                )
            if np.all(x > 0):
                ax.set_xscale("log")
                lower, upper = ax.get_xlim()
                ax.xaxis.set_major_locator(ticker.FixedLocator(np.geomspace(lower, upper, 2)))
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, position: f"{value:.3g}"))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                tick_labels = ax.get_xticklabels()
                tick_labels[0].set_ha("left")
                tick_labels[-1].set_ha("right")
            ax.set_yticks(y, names if index == 0 else [])
            ax.set_ylim(extent - 0.5, -0.5)
            ax.set_xlabel(
                {
                    "fitted_scalar_count": "Fitted coefficients + bias",
                    "parameters": "Torch parameters",
                    "wall_p50_ms": "P50 → P95 wall [ms]",
                    "throughput": "Loop throughput [windows/s]",
                }[field],
                fontsize=6.5,
            )
            ax.grid(axis="y", visible=False)
        fig.text(
            0.02,
            0.03,
            "Fixed representative; P50/P95 are call quantiles. CPU/GPU and FP32/float64 scopes are separate.",
            fontsize=6.4,
        )
        if temporal_groups:
            fig.text(0.02, 0.075, TEMPORAL_CONDITION, fontsize=6)
        context = data[["device", "batch_size", "window_size"]].drop_duplicates()
        if len(context) != 1:
            raise ValueError("A latency page must preserve one device/batch/window context")
        context = context.iloc[0]
        fig.text(
            0.02,
            0.112,
            f"{context.device}; B={int(context.batch_size)} windows/call; W={int(context.window_size)} samples.",
            fontsize=6.2,
        )
        _reserve_footer(fig)
        return fig
    if kind == "signals":
        fig, axes = _base(page["title"], 2, 3, height=5.0, left=0.09)
        for index, axis in enumerate(AXES):
            ax = axes.flat[index]
            selected = data[data.axis.eq(axis)]
            first = selected[selected.method.eq(selected.method.iloc[0])]
            ax.plot(first.time_s, first.low, color=COLORS["low"], label="LOW", lw=0.8)
            ax.plot(first.time_s, first.high, color=COLORS["high"], label="HIGH", lw=0.9)
            for method, group in selected.groupby("method", sort=False):
                ax.plot(
                    group.time_s,
                    group.prediction,
                    label=method,
                    color=COLORS["accent"] if str(method).endswith("tuned") else COLORS["virtual"],
                    lw=0.9,
                )
            ax.set_title(axis, fontsize=7.5)
            ax.set_xlabel("Time in selected segment [s]")
            ax.set_ylabel("Angular rate [rad/s]" if index < 3 else "Specific force [m/s²]")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=6
        )
        return fig
    if kind == "spectral":
        fig, axes = _base(page["title"], 3, 2, height=6.2, left=0.11)
        wave = page["extra_data"]
        for column, sensor in enumerate(("gyro", "accel")):
            selected = data[data.sensor.eq(sensor)]
            for method, group in selected.groupby("method", sort=False):
                mean = group.groupby("frequency_hz", as_index=False).mean(numeric_only=True)
                color = COLORS["accent"] if str(method).endswith("tuned") else COLORS["virtual"]
                positive = mean.frequency_hz > 0
                axes[0, column].loglog(
                    mean.frequency_hz[positive],
                    np.maximum(mean.prediction_error_psd[positive], 1e-24),
                    color=color,
                    label=method,
                )
                # A mean over only some axes would change the diagnostic's meaning.
                coherence_values = group.groupby("frequency_hz").prediction_coherence.agg(
                    lambda values: values.mean() if values.notna().all() else np.nan
                )
                if np.isfinite(coherence_values).any():
                    axes[1, column].plot(coherence_values.index, coherence_values, color=color)
                else:
                    axes[1, column].text(
                        0.02,
                        0.12,
                        "Coherence unsupported: unexcited axis",
                        transform=axes[1, column].transAxes,
                        fontsize=6,
                    )
                trajectory = wave[wave.sensor.eq(sensor) & wave.method.eq(method)].pivot(
                    index="time_s", columns="axis", values="integrated_error"
                )
                axes[2, column].plot(
                    trajectory.index, np.linalg.norm(trajectory, axis=1), color=color
                )
            first = (
                selected[selected.method.eq(selected.method.iloc[0])]
                .groupby("frequency_hz")
                .mean(numeric_only=True)
            )
            axes[0, column].loglog(
                first.index[first.index > 0],
                np.maximum(first.low_error_psd[first.index > 0], 1e-24),
                color=COLORS["low"],
                label="LOW error",
            )
            low_coherence = (
                selected[selected.method.eq(selected.method.iloc[0])]
                .groupby("frequency_hz")
                .low_coherence.agg(lambda values: values.mean() if values.notna().all() else np.nan)
            )
            if np.isfinite(low_coherence).any():
                axes[1, column].plot(low_coherence.index, low_coherence, color=COLORS["low"])
            if page["coverage"].get("information_scope") == SHARED_SCOPE:
                low_integral = wave[
                    wave.sensor.eq(sensor) & wave.method.eq(selected.method.iloc[0])
                ].pivot(index="time_s", columns="axis", values="low_integrated_error")
                axes[2, column].plot(
                    low_integral.index,
                    np.linalg.norm(low_integral, axis=1),
                    color=COLORS["low"],
                    lw=1,
                )
            axes[1, column].set_ylim(-0.02, 1.02)
            for row in (0, 1):
                axes[row, column].set_xlabel("Frequency [Hz]")
                axes[row, column].set_xlim(0.1 if row == 0 else 0, 20)
            axes[0, column].set_ylabel(
                "Mean residual PSD [" + ("(rad/s)²/Hz" if sensor == "gyro" else "(m/s²)²/Hz") + "]"
            )
            axes[1, column].set_ylabel("Mean axis coherence")
            axes[2, column].set_ylabel(
                "Integrated error norm [" + ("rad" if sensor == "gyro" else "m/s") + "]"
            )
            axes[2, column].set_xlabel("Time [s]")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=6
        )
        return fig
    raise ValueError(f"Unknown page kind {kind}")


def _embed_fonts(path):
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    characters = "".join(
        "".join(node.itertext()) for node in root.iter() if node.tag.endswith("text")
    )
    css = []
    for weight in ("normal", "bold"):
        font_path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight=weight)
        )
        options = subset.Options()
        options.name_IDs = ["*"]
        options.recalc_timestamp = False
        font = TTFont(font_path, recalcTimestamp=False)
        sub = subset.Subsetter(options=options)
        sub.populate(text=characters)
        sub.subset(font)
        font.flavor = "woff"
        buffer = io.BytesIO()
        font.save(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        css.append(
            "@font-face{font-family:'DejaVu Sans';font-weight:"
            + weight
            + ";src:url(data:font/woff;base64,"
            + encoded
            + ") format('woff');}"
        )
    svg = path.read_text(encoding="utf-8")
    position = svg.index(">", svg.index("<defs")) + 1
    path.write_text(
        svg[:position] + '<style type="text/css">' + "".join(css) + "</style>" + svg[position:],
        encoding="utf-8",
    )


def _verify_directory(directory):
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "artifact_manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Figure artifact checksum/inventory differs")
    for name, expected in manifest["files"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Figure artifact checksum differs: {name}")
    return manifest


def _write_page(page, out, config, synthetic, resume):
    destination = out / "figures" / page["id"]
    payload = page["data"].to_csv(index=False, lineterminator="\n")
    extra = page.get("extra_data")
    extra_payload = extra.to_csv(index=False, lineterminator="\n") if extra is not None else None
    contract = {
        "id": page["id"],
        "kind": page["kind"],
        "synthetic": synthetic,
        "coverage": _safe(page["coverage"]),
        "data_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "extra_data_sha256": hashlib.sha256(extra_payload.encode()).hexdigest()
        if extra_payload
        else None,
        "config": config,
        "script_sha256": sha256_file(Path(__file__)),
        "source_tables": page["source_tables"],
        "figure_contract": page.get("figure_contract"),
        "temporal_support_sha256": sha256_file(TEMPORAL_SOURCE),
    }
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        _verify_directory(destination)
        if json.loads((destination / "contract.json").read_text(encoding="utf-8")) != contract:
            raise ValueError("Figure resume contract/source data differ")
        print(f"SKIPS: verified existing figure {page['id']}", flush=True)
        return json.loads((destination / "entry.json").read_text(encoding="utf-8"))
    staging = out / "_partial" / ("page_" + uuid.uuid4().hex[:12])
    staging.mkdir(parents=True)
    write_json(staging / "contract.json", contract)
    (staging / "source.csv").write_text(payload, encoding="utf-8", newline="")
    if extra_payload:
        (staging / "source_extra.csv").write_text(extra_payload, encoding="utf-8", newline="")
    fig = draw_page(page, config)
    temporal_headers = [
        item.get_text()
        for ax in fig.axes
        for item in ax.texts
        if item.get_text() in {"Causal", "Window", "Window endpoint", "Reference"}
    ]
    if page["coverage"].get("temporal_groups_visible"):
        expected_headers = {item["temporal_label"] for item in page["coverage"]["temporal_classes"]}
        if expected_headers - set(temporal_headers):
            raise ValueError("A required temporal group header is not visibly drawn")
    if synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC · NONFINAL",
            ha="center",
            va="center",
            rotation=25,
            fontsize=24,
            color="#aa6666",
            alpha=0.17,
        )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    footer_overlaps = _footer_axis_overlaps(fig)
    if footer_overlaps:
        plt.close(fig)
        raise ValueError("Figure footer overlaps axis labels: " + str(footer_overlaps))
    omitted_tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if tick.get_loc() < lower or tick.get_loc() > upper:
                    omitted_tick_labels.update((id(tick.label1), id(tick.label2)))
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or id(text) in omitted_tick_labels:
            continue
        box = text.get_window_extent(renderer)
        if (
            box.x0 < -1
            or box.y0 < -1
            or box.x1 > fig.bbox.width + 1
            or box.y1 > fig.bbox.height + 1
        ):
            outside.append(text.get_text())
    if outside:
        plt.close(fig)
        raise ValueError(f"Figure text outside canvas: {outside[:6]}")
    files = {}
    for suffix in ("svg", "pdf", "png", "tiff"):
        target = staging / ("figure." + suffix)
        settings = (
            {"dpi": config["png_dpi"]}
            if suffix == "png"
            else {"dpi": config["tiff_dpi"], "pil_kwargs": {"compression": "tiff_lzw"}}
            if suffix == "tiff"
            else {}
        )
        fig.savefig(target, **settings)
        if suffix == "svg":
            _embed_fonts(target)
        files[suffix] = "figures/" + page["id"] + "/figure." + suffix
    plt.close(fig)
    svg, pdf = (
        (staging / "figure.svg").read_text(encoding="utf-8"),
        (staging / "figure.pdf").read_bytes(),
    )
    if "data:font/woff;base64" not in svg or b"/FontFile2" not in pdf or b"/ToUnicode" not in pdf:
        raise ValueError("Editable embedded-font export checks failed")
    page_box = re.search(rb"/MediaBox\s*\[([^]]+)\]", pdf)
    if page_box is None:
        raise ValueError("PDF page box is missing")
    dimensions = [float(value) for value in page_box.group(1).split()]
    if abs((dimensions[2] - dimensions[0]) / 72 - config["width_inches"]) > 1e-4:
        raise ValueError("PDF page width changed")
    executable = shutil.which(config["pdf_renderer"])
    if executable is None:
        raise RuntimeError("Python figure QA requires the configured Poppler pdftoppm executable")
    subprocess.run(
        [
            executable,
            "-singlefile",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            "150",
            "-png",
            "figure.pdf",
            "pdf_render",
        ],
        check=True,
        capture_output=True,
        cwd=staging,
    )
    with (
        Image.open(staging / "figure.png") as png,
        Image.open(staging / "figure.tiff") as tiff,
        Image.open(staging / "pdf_render.png") as preview,
    ):
        png_dpi, tiff_dpi = png.info.get("dpi"), tiff.info.get("dpi")
        if (
            png_dpi is None
            or tiff_dpi is None
            or any(abs(float(value) - 300) > 0.1 for value in png_dpi)
            or any(abs(float(value) - 600) > 0.1 for value in tiff_dpi)
        ):
            raise ValueError("Raster DPI metadata differs")
        reference = png.convert("RGB").resize(preview.size, Image.Resampling.LANCZOS)
        difference = float(
            np.abs(
                np.asarray(reference).astype(float)
                - np.asarray(preview.convert("RGB")).astype(float)
            ).mean()
        )
        if difference > 20:
            raise ValueError("PDF render materially differs from publication PNG")
        raster = {
            "png_pixels": list(png.size),
            "tiff_pixels": list(tiff.size),
            "png_dpi": list(map(float, png_dpi)),
            "tiff_dpi": list(map(float, tiff_dpi)),
        }
    qa = {
        "status": "passed",
        "visual_review_status": "pending",
        "synthetic_fixture": synthetic,
        "text_outside_canvas": outside,
        "svg_editable_text": "<text" in svg,
        "svg_embedded_WOFF": True,
        "pdf_embedded_TrueType_ToUnicode": True,
        "pdf_media_box_points": dimensions,
        "pdf_actual_render_checked": True,
        "pdf_render_mean_absolute_pixel_difference_0_255": difference,
        "raster": raster,
        "scope": "Automatic export/render/geometry checks; actual visual review is tracked separately.",
        "temporal_support_sha256": sha256_file(TEMPORAL_SOURCE),
        "temporal_grouping": {
            "required": page["coverage"].get("temporal_groups_visible", False),
            "visible_headers": temporal_headers,
            "classes": page["coverage"].get("temporal_classes", []),
        },
        "footer_axis_overlaps": footer_overlaps,
    }
    if page["coverage"].get("information_scope") == SHARED_SCOPE:
        qa.update(
            analysis_profile=SHARED_PROFILE,
            shared_controls_index_sha256=page["coverage"]["shared_controls_index_sha256"],
        )
    if page["kind"] == "spectral":
        qa["spectral_display"] = {
            "welch_detrend": "constant",
            "log_display_floor": 1e-24,
            "raw_values_unchanged": True,
            "raw_values_below_display_floor": int(
                (page["data"][["low_error_psd", "prediction_error_psd"]] < 1e-24).to_numpy().sum()
            ),
            "interpretation": "Display floor is not a hardware noise floor; near-zero detrended PSD does not remove constant bias.",
        }
    write_json(staging / "qa.json", qa)
    entry = {
        "id": page["id"],
        "kind": page["kind"],
        "title": page["title"],
        "purpose": page["kind"],
        "caption": ("SYNTHETIC VALIDATION ONLY. " if synthetic else "") + page["caption"],
        "files": files,
        "sha256": {key: sha256_file(staging / ("figure." + key)) for key in files},
        "source_tables": page["source_tables"],
        "source_data": [
            {
                "path": "figures/" + page["id"] + "/source.csv",
                "sha256": sha256_file(staging / "source.csv"),
            }
        ],
        "coverage": page["coverage"],
        "figure_contract": page.get("figure_contract"),
        "temporal_support_sha256": sha256_file(TEMPORAL_SOURCE),
        "qa": {
            "status": "passed",
            "path": "figures/" + page["id"] + "/qa.json",
            "sha256": sha256_file(staging / "qa.json"),
            "visual_review_status": "pending",
        },
    }
    if extra_payload:
        entry["source_data"].append(
            {
                "path": "figures/" + page["id"] + "/source_extra.csv",
                "sha256": sha256_file(staging / "source_extra.csv"),
            }
        )
    write_json(staging / "entry.json", entry)
    write_json(
        staging / "artifact_manifest.json",
        {"files": {path.name: sha256_file(path) for path in staging.iterdir() if path.is_file()}},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    print(f"WRITES: {page['id']} ({page['kind']})", flush=True)
    return _safe(entry)


def _build(
    tables, out, config, *, synthetic, source_manifest_sha256, source_files, resume, page_ids=None
):
    apply_paper_style()
    matplotlib.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "savefig.bbox": None,
            "text.usetex": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    if config["width_inches"] != 7.2 or config["png_dpi"] != 300 or config["tiff_dpi"] != 600:
        raise ValueError("Publication exports require7.2in, PNG300 and TIFF600")
    pages = plan_pages(tables, config)
    if page_ids is not None:
        if not synthetic:
            raise ValueError("Formal figures cannot omit planned pages")
        if set(page_ids) - {page["id"] for page in pages}:
            raise ValueError("Unknown synthetic debug page identity")
        pages = [page for page in pages if page["id"] in page_ids]
    contract = {
        "synthetic": synthetic,
        "source_manifest_sha256": source_manifest_sha256,
        "config": config,
        "script_sha256": sha256_file(Path(__file__)),
        "package_helper_sha256": sha256_file(
            Path(__file__).with_name("reviewer_artifact_package.py")
        ),
        "page_ids": [page["id"] for page in pages],
        "source_files": source_files,
        "analysis_profile": tables.get("_analysis_profile"),
        "shared_controls_index_sha256": tables.get("_shared_controls_index_sha256"),
        "shared_coverage": tables.get("_shared_coverage"),
        "shared_adapter_sha256": sha256_file(
            Path(__file__).with_name("reviewer_shared_figure_data.py")
        ),
        "shared_public_validator_sha256": sha256_file(
            Path(__file__).with_name("build_reviewer_report.py")
        )
        if tables.get("_analysis_profile") == SHARED_PROFILE
        else None,
        "temporal_support_sha256": sha256_file(TEMPORAL_SOURCE),
        "figure_contracts": {page["id"]: page.get("figure_contract") for page in pages},
        "plot_runtime": {
            "python": sys.version,
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": version("scipy"),
            "fonttools": version("fonttools"),
            "pillow": version("Pillow"),
            "style_source_sha256": sha256_file(Path(apply_paper_style.__code__.co_filename)),
            "font_sha256": sha256_file(Path(font_manager.findfont("DejaVu Sans"))),
        },
    }
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        if not resume:
            raise FileExistsError("Existing figure artifacts require --resume")
        if json.loads((out / "contract.json").read_text(encoding="utf-8")) != contract:
            raise ValueError("Figure build resume source/config contract differs")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "contract.json", contract)
        (out / "config_snapshot.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
        (out / "source_snapshot_reviewer_temporal_support.py").write_bytes(
            TEMPORAL_SOURCE.read_bytes()
        )
        if tables.get("_analysis_profile") == SHARED_PROFILE:
            for filename in ("reviewer_shared_figure_data.py", "build_reviewer_report.py"):
                (out / ("source_snapshot_" + filename)).write_bytes(
                    Path(__file__).with_name(filename).read_bytes()
                )
    entries = [_write_page(page, out, config, synthetic, resume) for page in pages]
    index = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": synthetic,
        "synthetic_fixture": synthetic,
        "manifest_sha256": source_manifest_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "figures": entries,
        "source_files": source_files,
        "analysis_profile": tables.get("_analysis_profile"),
        "shared_controls_index_sha256": tables.get("_shared_controls_index_sha256"),
        "shared_coverage": tables.get("_shared_coverage"),
        "temporal_support_sha256": sha256_file(TEMPORAL_SOURCE),
        "temporal_support_source": {
            "path": "source_snapshot_reviewer_temporal_support.py",
            "sha256": sha256_file(TEMPORAL_SOURCE),
        },
        "coverage": {
            "planned_pages": len(pages),
            "training_cells": sum(page["kind"] == "training" for page in pages),
            "training_final_runs": sum(
                page["coverage"].get("final_run_count", 0) for page in pages
            ),
        },
        "qa_summary": {
            "status": "passed",
            "synthetic_fixture": synthetic,
            "pdf_render_checked": True,
            "visual_review_status": "passed"
            if all(entry["qa"]["visual_review_status"] == "passed" for entry in entries)
            else "pending",
        },
        "statistics": STATISTICS,
        "exclusions": [
            "Allan deviation",
            "assembled trajectory ATE",
            "test-selected model/seed/horizon",
            "recording CI from optimization seeds",
        ],
    }
    path = out / "figure_index.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _safe(index):
            raise ValueError("Existing figure index differs")
    else:
        write_json(path, index)
        (out / "figure_index.md").write_text(
            "# "
            + ("SYNTHETIC NONFINAL " if synthetic else "")
            + "Reviewer figures\n\n"
            + "\n\n".join(
                f"- **{entry['id']}** — {entry['caption']}\n  - PNG: {entry['files']['png']}\n  - PDF: {entry['files']['pdf']}"
                for entry in entries
            )
            + "\n",
            encoding="utf-8",
        )
    return _safe(index)


def build_synthetic(tables, out, *, page_ids=None, resume=False, config=None):
    """Private validation path; outputs can never be relabelled as real results."""
    settings = {**default_config(), **(config or {})}
    fingerprints = {}
    for name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            fingerprints[name] = hashlib.sha256(
                value.to_csv(index=False, lineterminator="\n").encode()
            ).hexdigest()
    return _build(
        tables,
        out,
        settings,
        synthetic=True,
        source_manifest_sha256="synthetic:" + digest(fingerprints),
        source_files={},
        resume=resume,
        page_ids=page_ids,
    )


class SourceReader:
    def __init__(self, package):
        self.package, self.sources = package, {}

    def path(self, value, expected=None):
        observed = self.package.verify_file(value, expected)
        member = self.package.relative_path(value)
        self.sources[member] = observed
        return self.package.resolve_path(value)

    def json(self, value, expected=None):
        return json.loads(self.path(value, expected).read_text(encoding="utf-8-sig"))

    def csv(self, value):
        return pd.read_csv(self.path(value))

    def reference(self, value):
        path = self.path(value)
        member = self.package.relative_path(path)
        return {"path": member, "sha256": self.sources[member]}


def verify_business_gate(reader, roots):
    """Verify packaged sources and complete core/auxiliary identity matrices.

    Canonical bulk sequences excluded from the delivery remain fingerprint
    references. All plots use saved prediction LOW/HIGH/output columns.
    """
    package = reader.package
    state = reader.json(roots["core"] / "pipeline.json")
    cells = state.get("cells", {})
    finals = [
        job for job in state.get("jobs", {}).values() if job.get("phase", "").startswith("final_")
    ]
    seeds = state.get("final_seeds", [])
    identities = [(job.get("cell"), job.get("phase"), job.get("seed")) for job in finals]
    expected = {
        (cell, "final_" + variant, seed)
        for cell in cells
        for variant in ("default", "tuned")
        for seed in seeds
    }
    if (
        state.get("status") != "complete"
        or state.get("smoke")
        or len(cells) != 93
        or len(finals) != 558
        or set(seeds) != {20261010, 20261011, 20261012}
        or len(set(identities)) != 558
        or set(identities) != expected
        or any(
            job.get("status") != "complete" or job.get("evaluation_status") != "complete"
            for job in finals
        )
        or any("selected" not in cell for cell in cells.values())
    ):
        raise ValueError(
            "Formal figures require all93 frozen selections and all558 final evaluations"
        )
    frozen_sources = state["contract"].get("source_sha256", {})
    if not frozen_sources:
        raise ValueError("Frozen source checksum roster is missing")
    for identity, expected_sha in frozen_sources.items():
        reader.path(identity, expected_sha)
    for cell_id, cell in cells.items():
        path = roots["core"] / "selections" / (cell_id.replace("/", "__") + ".json")
        if reader.json(path, cell["selection_sha256"]) != cell["selected"]:
            raise ValueError("Saved selection identity differs")
    final_ids = {job["id"] for job in finals}
    for job in state["jobs"].values():
        if job.get("config") and job.get("config_sha256"):
            reader.path(job["config"], job["config_sha256"])
        if job.get("status") == "complete":
            for name, expected_sha in job.get("artifact_sha256", {}).items():
                reader.path(str(job["run_dir"]).rstrip("/") + "/" + name, expected_sha)
        if job["id"] in final_ids:
            if not job.get("evaluation_artifact_sha256"):
                raise ValueError("Final evaluation artifact inventory is missing")
            for name, expected_sha in job["evaluation_artifact_sha256"].items():
                reader.path(name, expected_sha)
            reader.path(
                roots["core"] / "evaluations" / (job["id"] + ".json"), job["evaluation_sha256"]
            )
    core_sha = package.verify_file(roots["core"] / "pipeline.json")
    verify_completion_audit(reader, state, finals, core_sha)
    auxiliary = {}
    for name, filename, stages in (
        (
            "aux1",
            "postprocessing_state.json",
            {"transfer", "downstream", "latency_gpu", "latency_cpu"},
        ),
        ("aux2", "final_analysis_state.json", {"accuracy_tables", "classical_latency"}),
    ):
        report = reader.json(roots[name] / filename)
        if (
            report.get("status") != "complete"
            or report.get("core_pipeline_sha256") != core_sha
            or set(report.get("stages", {})) != stages
        ):
            raise ValueError("Formal figures require matching complete auxiliary stages")
        for stage, entry in report["stages"].items():
            if entry.get("status") != "complete" or not entry.get("artifact_sha256"):
                raise ValueError("Auxiliary stage is incomplete or lacks its artifact inventory")
            directory = roots[name] / stage
            declared = {
                package.relative_path(directory / relative): expected_sha
                for relative, expected_sha in entry["artifact_sha256"].items()
            }
            prefix = package.relative_path(directory) + "/"
            actual = {member for member in package.files if member.startswith(prefix)}
            if actual != set(declared):
                raise ValueError("Auxiliary stage artifact inventory/coverage differs")
            for member, expected_sha in declared.items():
                reader.path(member, expected_sha)
        auxiliary[name] = report
    return state, finals, auxiliary


def verify_completion_audit(reader, state, finals, core_sha):
    audit = reader.json("metadata/completion_audit.json")
    profile = audit.get("analysis_profile")
    counts = {
        "selected_cells": 93,
        "final_evaluated": 558,
        "transfer_models": 120,
        "downstream_views": 5,
    }
    if profile is not None:
        if profile != SHARED_PROFILE:
            raise ValueError("Unknown completion audit analysis_profile")
        counts.update(**SHARED_COUNTS, shared_paired_support_refs=720)
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "passed"
        or audit.get("synthetic")
        or audit.get("source_pipeline_sha256") != core_sha
        or audit.get("counts") != counts
        or audit.get("source_roster_check", {}).get("status") != "passed"
    ):
        raise ValueError("Required complete real checkpoint/prediction/source audit differs")
    jobs = state["jobs"]
    checkpoints = audit.get("checkpoint_checks", [])
    expected_checkpoints = {job["id"] for job in jobs.values() if job.get("status") == "complete"}
    if (
        len(checkpoints) != len(expected_checkpoints)
        or {row["job_id"] for row in checkpoints} != expected_checkpoints
    ):
        raise ValueError("Checkpoint finite audit omits a completed fit")
    for row in checkpoints:
        job = jobs[row["job_id"]]
        if (
            row.get("finite") is not True
            or row.get("tensor_count", 0) <= 0
            or row.get("tensor_elements", 0) <= 0
            or row["phase"] != job["phase"]
            or row["sha256"] != job["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint finite/shape/checksum evidence differs")
        expected_path = reader.package.resolve_path(
            str(job["run_dir"]).rstrip("/\\") + "/best.ckpt"
        )
        if reader.package.resolve_path(row["source_path"]) != expected_path:
            raise ValueError("Checkpoint audit path differs from its job identity")
        reader.path(row["source_path"], row["sha256"])
    predictions = audit.get("prediction_checks", [])
    expected_predictions = {
        (job["id"], split) for job in finals for split in ("validation", "test")
    }
    if (
        len(predictions) != 1116
        or {(row["job_id"], row["split"]) for row in predictions} != expected_predictions
    ):
        raise ValueError("Prediction audit omits a final validation/test role")
    for row in predictions:
        if (
            row.get("finite") is not True
            or row.get("identity_nonnull") is not True
            or row.get("rows", 0) <= 0
            or row.get("columns", 0) <= 0
            or not row.get("checked_numeric_columns")
            or not row.get("checked_identity_columns")
        ):
            raise ValueError("Saved prediction finite/identity/shape evidence differs")
        job = jobs[row["job_id"]]
        expected_path = reader.package.resolve_path(
            str(job["run_dir"]).rstrip("/\\")
            + "/evaluation/"
            + row["split"]
            + "_predictions.parquet"
        )
        expected_hashes = [
            value
            for source, value in job["evaluation_artifact_sha256"].items()
            if reader.package.resolve_path(source) == expected_path
        ]
        if reader.package.resolve_path(row["source_path"]) != expected_path or expected_hashes != [
            row["sha256"]
        ]:
            raise ValueError(
                "Prediction audit path/checksum differs from its exact job/split identity"
            )
        reader.path(row["source_path"], row["sha256"])
    if len(audit.get("selection_checks", [])) != 93 or len(audit.get("marker_checks", [])) != (
        4 if profile == SHARED_PROFILE else 3
    ):
        raise ValueError("Completion audit omits selection or terminal-marker evidence")
    for marker in audit["marker_checks"]:
        if marker.get("status") != "passed":
            raise ValueError("A required terminal success marker did not pass")
        reader.path(marker["source_path"], marker["sha256"])
    return audit


def _transfer_table(records):
    rows, classical, observed = [], {}, set()
    for record in records:
        slot = (record["direction"], record["model_family"], record["variant"], record["seed"])
        if (
            slot in observed
            or record["seed"] not in {20261010, 20261011, 20261012}
            or record["variant"] not in {"default", "tuned"}
        ):
            raise ValueError("Transfer checkpoint has a duplicate or invalid seed/variant identity")
        observed.add(slot)
        pair_ids = [
            tuple(pair[name] for name in ("dataset_id", "low_sensor", "high_sensor"))
            for pair in record["metrics"]["pairs"]
        ]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Duplicate target pair in one transfer checkpoint")
        for pair in record["metrics"]["pairs"]:
            identity = {name: pair[name] for name in ("dataset_id", "low_sensor", "high_sensor")}
            for method, values in pair["metrics"].items():
                if method == "model":
                    reference_pair = next(
                        item
                        for item in record["target_domain_reference"]["test"]["pairs"]
                        if all(item[name] == value for name, value in identity.items())
                    )
                    for metric in PAIRED_METRICS:
                        rows.append(
                            {
                                "task": "paired",
                                "protocol_id": record["direction"],
                                "split": "test",
                                "aggregation_level": "pair",
                                **identity,
                                "method": "model",
                                "model_family": record["model_family"],
                                "variant": record["variant"],
                                "seed": record["seed"],
                                "metric": metric,
                                "value": values[metric],
                                "reference_value": reference_pair["metrics"]["model"][metric],
                            }
                        )
                else:
                    key = (record["direction"], *identity.values(), method)
                    signature = {
                        "target_fingerprint": record["target_fingerprint"],
                        "classical_state_sha256": record["classical_state_sha256"],
                        "metrics": {metric: values[metric] for metric in PAIRED_METRICS},
                    }
                    if key in classical and classical[key]["signature"] != signature:
                        raise ValueError(
                            "Transfer classical results have different support/state/results"
                        )
                    classical[key] = {
                        "signature": signature,
                        "record": record,
                        "identity": identity,
                        "method": method,
                        "values": values,
                    }
    frame = pd.DataFrame(rows)
    keys = [
        "task",
        "protocol_id",
        "split",
        "aggregation_level",
        "dataset_id",
        "low_sensor",
        "high_sensor",
        "method",
        "model_family",
        "variant",
        "metric",
    ]
    result = frame.groupby(keys, as_index=False).agg(
        mean=("value", "mean"),
        sample_sd=("value", "std"),
        n=("seed", "nunique"),
        reference_mean=("reference_value", "mean"),
    )
    if not result.n.eq(3).all():
        raise ValueError("Every transfer family/variant requires three seeds")
    extra = []
    for value in classical.values():
        for metric in PAIRED_METRICS:
            extra.append(
                {
                    "task": "paired",
                    "protocol_id": value["record"]["direction"],
                    "split": "test",
                    "aggregation_level": "pair",
                    **value["identity"],
                    "method": value["method"],
                    "model_family": value["method"],
                    "variant": "classical",
                    "metric": metric,
                    "mean": value["values"][metric],
                    "sample_sd": np.nan,
                    "n": 1,
                    "reference_mean": np.nan,
                }
            )
    result = pd.concat([result, pd.DataFrame(extra)], ignore_index=True)
    result["unit"] = result.metric.map(
        lambda metric: "rad/s" if metric.startswith("gyro") else "m/s²"
    )
    return result


def _latency_table(reader, root, classical=False):
    report = reader.json(root / "latency.json")
    if report.get("status") != "complete":
        raise ValueError("Actual latency receipts must be complete")
    rows = []
    if classical:
        if report.get("final_run_mappings") != 558:
            raise ValueError("Classical timing omits the complete final-run mapping")
        for record in report["records"]:
            rows.append(
                {
                    **{
                        key: value
                        for key, value in record.items()
                        if np.isscalar(value) or value is None
                    },
                    "protocol_id": record["contract"]["protocol_id"],
                    "model_family": record["method"],
                    "variant": "classical",
                    "pair_identity": json.dumps(record["pair_identity"], sort_keys=True),
                    "device": "cpu",
                    "wall_p50_ms": record["wall"]["p50_ms"],
                    "wall_p95_ms": record["wall"]["p95_ms"],
                    "throughput": record["loop_throughput_windows_per_second"],
                    "measurement_id": digest(record["contract"]),
                    "environment": json.dumps(report["environment"], sort_keys=True),
                }
            )
    else:
        audit = reader.json(root / "checkpoint_inventory.json")
        runtime = reader.json(root / "environment.json")
        if (
            not runtime.get("torch")
            or not runtime.get("device")
            or (str(runtime["device"]).startswith("cuda") and not runtime.get("gpu"))
            or (runtime["device"] == "cpu" and not runtime.get("cpu_processor"))
        ):
            raise ValueError("Neural timing lacks actual runtime/hardware evidence")
        if len(audit["entries"]) != 558 or report["entries"] != 558:
            raise ValueError("Neural timing omits the full558 architecture mapping")
        measurements = {}
        for record in report["records"]:
            if record["seed"] != 20261010 or record["wall"]["count"] != 300:
                raise ValueError("Neural timing is not the fixed-seed complete call protocol")
            device = str(record["device"])
            batches = {1} if device == "cpu" else {1, 128} if device.startswith("cuda") else set()
            if record["batch_size"] not in batches:
                raise ValueError("Frozen neural timing requires CPU B=1 or CUDA B=1/128")
            measurements.setdefault(record["architecture_id"], []).append(record)
        for entry in audit["entries"]:
            if entry["seed"] != 20261010:
                continue
            for record in measurements[entry["architecture_id"]]:
                rows.append(
                    {
                        "protocol_id": entry["protocol_id"],
                        "task": entry["task"],
                        "model_family": entry["model_family"],
                        "model_type": entry.get("architecture", {})
                        .get("model_config", {})
                        .get("type"),
                        "variant": entry["variant"],
                        "pair_identity": "all input channels",
                        "seed": 20261010,
                        "device": record["device"],
                        "batch_size": record["batch_size"],
                        "window_size": record["window_size"],
                        "parameters": record["parameters"],
                        "parameter_bytes": record["parameter_bytes"],
                        "wall_p50_ms": record["wall"]["p50_ms"],
                        "wall_p95_ms": record["wall"]["p95_ms"],
                        "throughput": record["loop_throughput_windows_per_second"],
                        "precision": record["precision"],
                        "measurement_id": record["architecture_id"]
                        + "/"
                        + record["device"]
                        + "/"
                        + str(record["batch_size"]),
                        "timing_representative_entry_id": entry["timing_representative_entry_id"],
                        "environment": json.dumps(runtime, sort_keys=True),
                    }
                )
    return pd.DataFrame(rows)


def read_saved_results(reader, roots, state, finals, config):
    tables, sources = {}, {}
    aggregate = roots["aux2"] / "accuracy_tables"
    for key, filename in (
        ("metrics", "metrics_seed_summary.csv"),
        ("deltas", "matched_seed_delta_summary.csv"),
        ("ranks", "model_rankings.csv"),
        ("rank_correlations", "rank_correlations.csv"),
    ):
        tables[key] = reader.csv(aggregate / filename)
        sources[key] = [reader.reference(aggregate / filename)]
    validate_metric_coverage(tables["metrics"], state)
    registry = reader.csv(aggregate / "run_registry.csv")
    if len(registry) != 558 or registry.job_id.nunique() != 558:
        raise ValueError("Accuracy aggregation lacks all558 final runs")
    training, model_types = [], {}
    for job in finals:
        cell = state["cells"][job["cell"]]
        directory = reader.package.resolve_path(job["run_dir"])
        frozen = yaml.safe_load(
            reader.path(job["config"], job["config_sha256"]).read_text(encoding="utf-8")
        )
        model_type = frozen.get("model", {}).get("type")
        model_key = (
            cell["protocol"]["id"],
            cell["candidate"]["id"],
            job["phase"].removeprefix("final_"),
        )
        if model_key in model_types and model_types[model_key] != model_type:
            raise ValueError("Final optimization seeds disagree on their source model_type")
        model_types[model_key] = model_type
        path = directory / "metrics.csv"
        history = reader.csv(path)
        criterion = frozen["training"]["checkpoint_metric"]
        required = ["epoch", "train_loss", "val_loss", criterion]
        if (
            set(required) - set(history)
            or not np.isfinite(history[required].to_numpy(float)).all()
            or history.epoch.duplicated().any()
        ):
            raise ValueError("Saved final learning curves are incomplete/nonfinite")
        history = history.copy()
        history["criterion"] = history[criterion]
        for key, value in {
            "cell": job["cell"],
            "job_id": job["id"],
            "protocol_id": cell["protocol"]["id"],
            "task": cell["protocol"]["task"],
            "model_family": cell["candidate"]["id"],
            "variant": job["phase"].removeprefix("final_"),
            "seed": job["seed"],
            "criterion_name": criterion,
            "loss_recipe": json.dumps(frozen["training"], sort_keys=True),
            "window_size": frozen["data"]["window_size"],
            "_source_member": reader.package.relative_path(path),
            "_source_sha256": reader.package.verify_file(path),
        }.items():
            history[key] = value
        training.append(history)
    tables["training"] = pd.concat(training, ignore_index=True)
    tables["_model_types"] = model_types
    if tables["training"].cell.nunique() != config["all_training_cells_required"]:
        raise ValueError("All93 training curve cells must be retained")
    transfer_path = roots["aux1"] / "transfer/transfer_metrics.json"
    transfer = reader.json(transfer_path)
    if (
        transfer.get("status") != "complete"
        or transfer.get("records") != 120
        or len(transfer["source_model_records"]) != 120
    ):
        raise ValueError("Cross-domain figures must retain all120 source checkpoints")
    directions = {"alimu_to_penn": "alimu_only", "penn_to_alimu": "penn_only"}
    expected_transfer = {
        (direction, cell.split("/", 1)[1], variant, seed)
        for direction, protocol in directions.items()
        for cell in state["cells"]
        if cell.startswith(protocol + "/")
        for variant in ("default", "tuned")
        for seed in (20261010, 20261011, 20261012)
    }
    observed_transfer = [
        (record["direction"], record["model_family"], record["variant"], record["seed"])
        for record in transfer["source_model_records"]
    ]
    if (
        len(expected_transfer) != 120
        or len(set(observed_transfer)) != 120
        or set(observed_transfer) != expected_transfer
    ):
        raise ValueError(
            "Cross-domain direction/family/variant/seed matrix differs from the frozen core"
        )
    tables["transfer"] = _transfer_table(transfer["source_model_records"])
    sources["transfer"] = [reader.reference(transfer_path)]
    downstream_root = roots["aux1"] / "downstream"
    summary_path = downstream_root / "overview/all_fold_summary.csv"
    tables["downstream"] = reader.csv(summary_path)
    for stem in ("endpoint_rmse_deg", "position_endpoint_rmse_m"):
        if stem + "_sd" in tables["downstream"]:
            tables["downstream"] = tables["downstream"].rename(
                columns={stem + "_sd": stem + "_sample_sd"}
            )
    protocols = {"penn_loso_af", "penn_loso_as", "penn_loso_bf", "penn_only", "pooled"}
    if set(tables["downstream"].protocol) != protocols:
        raise ValueError("Downstream views require four outer recordings and separate pooled/bs")
    validate_downstream_coverage(tables["downstream"], state)
    sources["downstream"] = [reader.reference(summary_path)]
    tables["temporal"] = []
    for protocol in sorted(protocols):
        unit = downstream_root / "units" / protocol
        metadata = reader.json(unit / "metrics.json")
        if (
            metadata["neural_optimizer_runs"] != 60
            or metadata["neural_method_variants"] != 20
            or metadata["unique_classical_methods"] != 9
        ):
            raise ValueError(
                "Downstream methods omit a family/variant/seed or classical comparison"
            )
        signal_path = reader.path(unit / "evaluated_signals.parquet")
        signal = pd.read_parquet(signal_path)
        methods = {
            prefix: value
            for prefix, value in metadata["methods"].items()
            if value.get("seed") in (None, 20261010)
        }
        tables["temporal"].append(
            {
                "protocol_id": protocol,
                "identity": "Penn_" + metadata["recording"],
                "frame": signal,
                "methods": methods,
                "source_tables": [
                    reader.reference(signal_path),
                    reader.reference(unit / "metrics.json"),
                ],
            }
        )
    # Independent AL training supplies qualitative pages for every AL device
    # pair; all other training/evaluation scopes remain in the metric pages.
    al_jobs = [
        job for job in finals if job["cell"].startswith("alimu_only/") and job["seed"] == 20261010
    ]
    common, methods, evidence, extra_columns = None, {}, [], {}
    for index, job in enumerate(sorted(al_jobs, key=lambda value: (value["cell"], value["phase"]))):
        path = reader.path(str(job["run_dir"]).rstrip("/") + "/evaluation/test_predictions.parquet")
        frame = (
            pd.read_parquet(path)
            .sort_values(["dataset_id", "low_sensor", "high_sensor", "sequence_id", "timestamp_s"])
            .reset_index(drop=True)
        )
        physical = [
            "dataset_id",
            "low_sensor",
            "high_sensor",
            "sequence_id",
            "timestamp_s",
            *LOW,
            *HIGH,
        ]
        if common is None:
            common = frame[physical].copy()
            for name in CLASSICAL:
                for axis in AXES:
                    extra_columns[axis + "_classic_" + name] = frame[name + "_" + axis].to_numpy(
                        copy=True
                    )
                methods["classic_" + name] = {
                    "method": name,
                    "seed": None,
                    "family": None,
                    "variant": "classical",
                }
        elif not common[physical].equals(frame[physical]):
            raise ValueError("AL fixed-seed methods do not share physical temporal support")
        prefix = f"model_{index:03d}"
        for axis in AXES:
            extra_columns[axis + "_" + prefix] = frame["model_" + axis].to_numpy(copy=True)
        family, variant = job["cell"].split("/", 1)[1], job["phase"].removeprefix("final_")
        methods[prefix] = {
            "method": family + "/" + variant,
            "family": family,
            "variant": variant,
            "seed": 20261010,
        }
        evidence.append(reader.reference(path))
    if common is None or len(al_jobs) != 20:
        raise ValueError(
            "AL temporal diagnostics require every family/variant at fixedseed20261010"
        )
    common = pd.concat([common, pd.DataFrame(extra_columns, index=common.index)], axis=1)
    for identity, group in common.groupby(["dataset_id", "low_sensor", "high_sensor"], sort=True):
        tables["temporal"].append(
            {
                "protocol_id": "alimu_only",
                "identity": "__".join(map(str, identity)),
                "frame": group.copy(),
                "methods": methods,
                "source_tables": evidence,
            }
        )
    tables["latency_neural"] = pd.concat(
        [_latency_table(reader, roots["aux1"] / name) for name in ("latency_gpu", "latency_cpu")],
        ignore_index=True,
    )
    sources["latency_neural"] = [
        reader.reference(roots["aux1"] / name / file)
        for name in ("latency_gpu", "latency_cpu")
        for file in ("latency.json", "checkpoint_inventory.json", "environment.json")
    ]
    tables["latency_classical"] = _latency_table(
        reader, roots["aux2"] / "classical_latency", classical=True
    )
    sources["latency_classical"] = [
        reader.reference(roots["aux2"] / "classical_latency/latency.json")
    ]
    tables["_sources"] = sources
    return tables


def validate_metric_coverage(frame, state):
    primary = frame[
        (frame.task.eq("paired") & frame.aggregation_level.eq("pair"))
        | (frame.task.eq("navigation") & frame.aggregation_level.eq("dataset"))
    ]
    expected_protocols = {cell.split("/", 1)[0] for cell in state["cells"]}
    if set(primary.protocol_id) != expected_protocols or set(primary.split) != {
        "validation",
        "test",
    }:
        raise ValueError("Physical summary omits a protocol or validation/test role")
    for key, group in primary.groupby(
        ["protocol_id", "split", "task", "dataset_id", "low_sensor", "high_sensor"],
        dropna=False,
        sort=True,
    ):
        protocol, _, task, *_ = key
        families = {
            cell.split("/", 1)[1] for cell in state["cells"] if cell.startswith(protocol + "/")
        }
        expected = {(family, variant) for family in families for variant in ("default", "tuned")}
        classical = set(
            CLASSICAL if task == "paired" else ["zero", "train_mean", "ridge", "ridge_tuned"]
        )
        for metric in PAIRED_METRICS if task == "paired" else NAV_METRICS:
            selected = group[group.metric.eq(metric)]
            neural = selected[selected.method.eq("model")]
            others = selected[~selected.method.eq("model")]
            identities = list(zip(neural.model_family, neural.variant, strict=True))
            if (
                set(identities) != expected
                or len(identities) != len(expected)
                or not neural.n.eq(3).all()
                or not np.isfinite(neural[["mean", "sample_sd"]].to_numpy(float)).all()
            ):
                raise ValueError(
                    "Physical metric omits a neural family/variant or three-seed statistic"
                )
            if (
                set(others.method) != classical
                or len(others) != len(classical)
                or not others.n.eq(1).all()
                or not np.isfinite(others["mean"].to_numpy(float)).all()
            ):
                raise ValueError("Physical metric omits a deduplicated classical method")


def validate_downstream_coverage(frame, state):
    for protocol, group in frame.groupby("protocol", sort=True):
        families = {
            cell.split("/", 1)[1] for cell in state["cells"] if cell.startswith(protocol + "/")
        }
        methods = (
            {family + "/" + variant for family in families for variant in ("default", "tuned")}
            | set(CLASSICAL)
            | {"HIGH"}
        )
        expected = {
            (task, method, samples)
            for task in ("attitude", "strapdown")
            for method in methods
            for samples in (80, 200, 400, 800)
        }
        actual = list(zip(group.task, group.method, group.samples, strict=True))
        if len(families) != 10 or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(
                "Downstream summary omits/repeats a method, task or predeclared horizon"
            )
        if not np.allclose(group.horizon_s, (group.samples - 1) / 40, atol=1e-9, rtol=1e-9):
            raise ValueError("Downstream endpoint spans differ from the frozen four horizons")
        for task, stem in (
            ("attitude", "endpoint_rmse_deg"),
            ("strapdown", "position_endpoint_rmse_m"),
        ):
            selected = group[group.task.eq(task)]
            neural = selected.method.str.contains("/", regex=False)
            if selected.optimizer_runs.to_numpy().tolist() != np.where(neural, 3, 1).tolist():
                raise ValueError("Downstream optimizer/classical replication counts differ")
            if (
                not np.isfinite(selected[stem + "_mean"]).all()
                or (selected[stem + "_mean"] < 0).any()
            ):
                raise ValueError("Downstream RMSE must be finite and nonnegative")
            if not np.isfinite(selected.loc[neural, stem + "_sample_sd"]).all():
                raise ValueError("Downstream neural sample SD is missing")


def record_visual_review(out, identifiers, *, reviewer, notes):
    """Record actual image/PDF review after viewing, never infer it from export."""
    out = Path(out)
    index_path = out / "figure_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if set(identifiers) - {entry["id"] for entry in index["figures"]}:
        raise ValueError("Visual review names an unknown figure")
    for entry in index["figures"]:
        if entry["id"] not in identifiers:
            continue
        directory = out / "figures" / entry["id"]
        _verify_directory(directory)
        qa_path = out / entry["qa"]["path"]
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa.update(
            visual_review_status="passed",
            visual_review={
                "reviewer": reviewer,
                "notes": notes,
                "reviewed_utc": datetime.now(UTC).isoformat(),
            },
        )
        qa_path.write_text(
            json.dumps(_safe(qa), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        entry["qa"].update(visual_review_status="passed", sha256=sha256_file(qa_path))
        (directory / "entry.json").write_text(
            json.dumps(_safe(entry), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (directory / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "files": {
                        path.name: sha256_file(path)
                        for path in directory.iterdir()
                        if path.is_file() and path.name != "artifact_manifest.json"
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    index["qa_summary"]["visual_review_status"] = (
        "passed"
        if all(entry["qa"]["visual_review_status"] == "passed" for entry in index["figures"])
        else "pending"
    )
    index_path.write_text(
        json.dumps(_safe(index), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=SCRIPT_ROOT / "configs/reviewer_figures.yaml"
    )
    parser.add_argument("--core-root", type=Path)
    parser.add_argument("--aux1-root", type=Path)
    parser.add_argument("--aux2-root", type=Path)
    parser.add_argument("--aux3-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        f"READS: complete immutable package {args.results_root}; full member SHA verification",
        flush=True,
    )
    package = load_package(args.results_root, verify=True, allow_synthetic=False)
    profile = analysis_profile(package)
    roots = {
        "core": package.root_path("core_pipeline"),
        "aux1": package.root_path("aux1"),
        "aux2": package.root_path("aux2"),
    }
    if profile == SHARED_PROFILE:
        roots["aux3"] = package.root_path("aux3")
    elif args.aux3_root is not None:
        raise ValueError("Legacy profile cannot accept an aux3 root override")
    for name, override in (
        ("core", args.core_root),
        ("aux1", args.aux1_root),
        ("aux2", args.aux2_root),
        ("aux3", args.aux3_root),
    ):
        if override is not None and package.resolve_path(override) != roots[name]:
            raise ValueError(
                "Explicit source root differs from the package's exact identity mapping"
            )
    reader = SourceReader(package)
    state, finals, _ = verify_business_gate(reader, roots)
    settings = {**default_config(), **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    if settings != default_config():
        raise ValueError(
            "Formal figure contract cannot silently change seed, coverage, horizon or export policy"
        )
    print(
        "SKIPS: inference, test-selected method/seed/segment/horizon, canonical bulk loading, Allan/trajectory ATE fabrication",
        flush=True,
    )
    shared = None
    if profile == SHARED_PROFILE:
        shared = load_shared(
            reader, roots, finals, package.verify_file(roots["core"] / "pipeline.json")
        )
    tables = read_saved_results(reader, roots, state, finals, settings)
    if shared is not None:
        tables = attach_shared(tables, reader, roots, shared)
    index = _build(
        tables,
        args.out,
        settings,
        synthetic=False,
        source_manifest_sha256=package.manifest_sha256,
        source_files=reader.sources,
        resume=args.resume,
    )
    print(
        f"WRITES: {len(index['figures'])} complete figure pages; actual visual review remains explicitly tracked",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
