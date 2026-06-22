"""Timestamp utilities."""

from __future__ import annotations

import numpy as np


def detect_time_unit(values: np.ndarray) -> str:
    """Detect likely timestamp unit from magnitude."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "s"
    median_abs = float(np.median(np.abs(finite)))
    if median_abs > 1e17:
        return "ns"
    if median_abs > 1e14:
        return "us"
    if median_abs > 1e11:
        return "ms"
    return "s"


def to_seconds(values: np.ndarray, unit: str = "auto") -> np.ndarray:
    """Convert timestamps to seconds with optional automatic unit detection."""
    unit = detect_time_unit(values) if unit == "auto" else unit
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if unit not in scale:
        raise ValueError(f"Unsupported timestamp unit: {unit}")
    seconds = values.astype(float) * scale[unit]
    if seconds.size:
        seconds = seconds - seconds[0]
    return seconds
