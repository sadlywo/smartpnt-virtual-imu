"""Classical filtering baselines."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, savgol_filter


def moving_average(x: np.ndarray, window: int = 5, causal: bool = True) -> np.ndarray:
    """Apply moving average per channel."""
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    out = np.zeros_like(x, dtype=float)
    for axis in range(x.shape[1]):
        if causal:
            out[:, axis] = lfilter(kernel, [1.0], x[:, axis])
        else:
            out[:, axis] = np.convolve(x[:, axis], kernel, mode="same")
    return out


def butterworth_lowpass(x: np.ndarray, rate_hz: float, cutoff_hz: float, order: int = 3) -> np.ndarray:
    """Apply a causal Butterworth low-pass filter."""
    b, a = butter(order, cutoff_hz / (0.5 * rate_hz), btype="low")
    return np.column_stack([lfilter(b, a, x[:, axis]) for axis in range(x.shape[1])])


def savitzky_golay(x: np.ndarray, window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Apply a non-causal Savitzky-Golay smoothing filter."""
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        window = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 4
    return savgol_filter(x, window_length=window, polyorder=polyorder, axis=0, mode="interp")
