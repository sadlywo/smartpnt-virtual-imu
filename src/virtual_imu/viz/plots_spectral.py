"""PSD comparison plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_psd_comparison(df: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    """Plot PSD of gyro and accel norms."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    fs = 1.0 / max(dt, 1e-9)
    nperseg = min(512, max(16, len(df) // 4))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax_obj, sensor, axes, unit in [
        (axs[0], "gyro", ["gx", "gy", "gz"], "(rad/s)^2/Hz"),
        (axs[1], "accel", ["ax", "ay", "az"], "(m/s^2)^2/Hz"),
    ]:
        for prefix, color in [("low", COLORS["low"]), ("virtual", COLORS["virtual"]), ("high", COLORS["high"])]:
            values = np.linalg.norm(df[[f"{axis}_{prefix}" for axis in axes]].to_numpy(float), axis=1)
            freq, psd = welch(values, fs=fs, nperseg=nperseg)
            ax_obj.semilogy(freq, psd + 1e-18, color=color, label=prefix)
        ax_obj.set_xlabel("Frequency [Hz]")
        ax_obj.set_ylabel(f"{sensor} PSD [{unit}]")
        ax_obj.set_xlim(0, fs / 2)
    axs[0].legend(frameon=False)
    stem = "psd_comparison"
    save_figure(fig, out_dir, stem)
    return stem, "PSD comparison of low-grade, virtual, and high-grade IMU norm signals."
