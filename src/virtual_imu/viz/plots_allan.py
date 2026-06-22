"""Allan deviation plots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.allan import allan_deviation
from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_allan(df: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    """Plot Allan deviation curves for gyro and accel norms."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    rate_hz = 1.0 / max(dt, 1e-9)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax_obj, _sensor, axes, ylabel in [
        (axs[0], "gyro", ["gx", "gy", "gz"], "Allan deviation [rad/s]"),
        (axs[1], "accel", ["ax", "ay", "az"], "Allan deviation [m/s^2]"),
    ]:
        for prefix, color in [("low", COLORS["low"]), ("virtual", COLORS["virtual"]), ("high", COLORS["high"])]:
            values = np.linalg.norm(df[[f"{axis}_{prefix}" for axis in axes]].to_numpy(float), axis=1)
            taus, adev = allan_deviation(values, rate_hz)
            if len(taus):
                ax_obj.loglog(taus, adev, color=color, label=prefix)
        ax_obj.set_xlabel("Averaging time [s]")
        ax_obj.set_ylabel(ylabel)
    axs[0].legend(frameon=False)
    stem = "allan_deviation"
    save_figure(fig, out_dir, stem)
    return stem, "Log-log Allan deviation curves for low-grade, virtual, and high-grade IMU norm signals."
