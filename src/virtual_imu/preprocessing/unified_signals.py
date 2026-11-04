"""Gap-aware common-rate signals with explicit offline preparation latency."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import signal


def regularize_segments(
    frame: pd.DataFrame,
    signal_columns: list[str],
    *,
    rate_hz: float,
    native_rate_hz: float,
    min_rows: int,
    fir_taps: int = 63,
    max_gap_periods: float = 1.5,
    label_columns: list[str] | None = None,
    angle_columns: list[str] | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Filter input/paired labels identically; interpolate navigation truth in time.

    Each finite source run is independent. FIR warmup is removed, its known
    group delay is attached to availability timestamps, and signal time is
    compensated. This is offline preparation, not zero-latency streaming.
    """
    if not np.isfinite([rate_hz, native_rate_hz, max_gap_periods]).all():
        raise ValueError("Rates and maximum gap must be finite")
    if rate_hz <= 0 or native_rate_hz <= 0 or max_gap_periods <= 0 or min_rows <= 0:
        raise ValueError("Rates, maximum gap and minimum rows must be positive")
    if rate_hz > native_rate_hz * 1.005:
        raise ValueError("Refusing upsampling: it creates no additional measured bandwidth")
    labels = list(label_columns or [])
    angles = set(angle_columns or [])
    required = ["timestamp_s", *signal_columns, *labels]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing signal/label fields: {missing}")
    data = frame.copy()
    times = data.timestamp_s.to_numpy(dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Source timestamps must be finite and increasing in source order")
    duplicate_mask = data.timestamp_s.duplicated(keep="first")
    data = data.loc[~duplicate_mask].reset_index(drop=True)
    times = data.timestamp_s.to_numpy(dtype=np.float64)
    finite = np.isfinite(data[[*signal_columns, *labels]].to_numpy(dtype=np.float64)).all(axis=1)
    if "_label_valid" in data:
        finite &= data._label_valid.to_numpy(dtype=bool)
    indices = np.flatnonzero(finite)
    downsampling = native_rate_hz > rate_hz * 1.005
    if fir_taps < 3 or fir_taps % 2 == 0:
        raise ValueError("fir_taps must be odd and at least three")
    delay = (fir_taps - 1) / (2 * native_rate_hz) if downsampling else 0.0
    metrics: dict[str, Any] = {
        "input_rows": len(frame),
        "duplicate_timestamps_removed": int(duplicate_mask.sum()),
        "invalid_or_masked_rows": int((~finite).sum()),
        "native_rate_hz": native_rate_hz,
        "rate_hz": rate_hz,
        "anti_alias_applied": downsampling,
        "fir_taps": fir_taps if downsampling else 0,
        "fir_group_delay_s": delay,
        "interpolation_latency_s": 1 / native_rate_hz,
        "availability_latency_s": delay + 1 / native_rate_hz,
        "discarded_short_segments": 0,
    }
    if not len(indices):
        return [], {**metrics, "segments": 0, "output_rows": 0}
    boundaries = (
        np.flatnonzero(
            (np.diff(indices) > 1) | (np.diff(times[indices]) > max_gap_periods / native_rate_hz)
        )
        + 1
    )
    segments = []
    origin = float(times[0])
    kernel = signal.firwin(fir_taps, 0.8 * rate_hz / 2, fs=native_rate_hz) if downsampling else None
    for selected in np.split(indices, boundaries):
        part = data.iloc[selected].reset_index(drop=True)
        relative = part.timestamp_s.to_numpy(dtype=float) - origin
        values = part[signal_columns].to_numpy(dtype=float)
        warmup = fir_taps - 1 if downsampling else 0
        if len(part) <= warmup + 1:
            metrics["discarded_short_segments"] += 1
            continue
        if downsampling:
            values = signal.lfilter(kernel, [1.0], values, axis=0)
        support_time = relative[warmup:] - delay
        values = values[warmup:]
        start = np.ceil((support_time[0] - 1e-8) * rate_hz) / rate_hz
        end = support_time[-1]
        count = int(np.floor((end - start + 1e-8) * rate_hz)) + 1
        if count < min_rows:
            metrics["discarded_short_segments"] += 1
            continue
        grid = start + np.arange(count, dtype=float) / rate_hz
        nearest = np.searchsorted(relative, grid).clip(1, len(relative) - 1)
        left = nearest - 1
        nearest = np.where(
            abs(relative[left] - grid) < abs(relative[nearest] - grid), left, nearest
        )
        out = part.iloc[nearest].reset_index(drop=True).copy()
        out["timestamp_s"] = grid
        out["source_timestamp_s"] = origin + grid
        out["available_timestamp_s"] = grid + delay + 1 / native_rate_hz
        for index, column in enumerate(signal_columns):
            out[column] = np.interp(grid, support_time, values[:, index])
        for column in labels:
            truth = part[column].to_numpy(dtype=float)
            if column in angles:
                truth = np.rad2deg(np.unwrap(np.deg2rad(truth)))
            interpolated = np.interp(grid, relative, truth)
            if column in angles:
                interpolated = (
                    interpolated % 360 if "heading" in column else (interpolated + 180) % 360 - 180
                )
            out[column] = interpolated
        out.drop(columns=["_label_valid"], errors="ignore", inplace=True)
        segments.append(out)
    return segments, {
        **metrics,
        "segments": len(segments),
        "output_rows": sum(len(part) for part in segments),
    }
