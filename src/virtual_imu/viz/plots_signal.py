"""Signal-domain plots."""

from __future__ import annotations

import pandas as pd

from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_time_series(df: pd.DataFrame, sensor: str, out_dir: str) -> tuple[str, str]:
    """Plot low/virtual/high time series for gyro or accel."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    axes = ["gx", "gy", "gz"] if sensor == "gyro" else ["ax", "ay", "az"]
    ylabel = "Angular rate [rad/s]" if sensor == "gyro" else "Specific force [m/s^2]"
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
    t = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    for ax_obj, axis in zip(axs, axes):  # noqa: B905
        ax_obj.plot(t, df[f"{axis}_low"], color=COLORS["low"], label="low")
        if f"{axis}_virtual" in df:
            ax_obj.plot(t, df[f"{axis}_virtual"], color=COLORS["virtual"], label="virtual")
        if f"{axis}_high" in df:
            ax_obj.plot(t, df[f"{axis}_high"], color=COLORS["high"], label="high", alpha=0.9)
        ax_obj.set_ylabel(f"{axis} {ylabel}")
    axs[-1].set_xlabel("Time [s]")
    axs[0].legend(frameon=False, ncol=3, loc="upper right")
    stem = f"imu_time_series_{sensor}"
    save_figure(fig, out_dir, stem)
    return stem, f"{sensor.capitalize()} time series comparing low-grade, virtual, and high-grade IMU signals."


def plot_error_comparison(df: pd.DataFrame, sensor: str, out_dir: str) -> tuple[str, str]:
    """Plot low/high and virtual/high residual norms."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    axes = ["gx", "gy", "gz"] if sensor == "gyro" else ["ax", "ay", "az"]
    ylabel = "Gyro error norm [rad/s]" if sensor == "gyro" else "Accel error norm [m/s^2]"
    low = df[[f"{axis}_low" for axis in axes]].to_numpy(float)
    virtual = df[[f"{axis}_virtual" for axis in axes]].to_numpy(float)
    high = df[[f"{axis}_high" for axis in axes]].to_numpy(float)
    t = df["timestamp_s"] - df["timestamp_s"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(t, np.linalg.norm(low - high, axis=1), color=COLORS["low"], label="low vs high")
    ax.plot(t, np.linalg.norm(virtual - high, axis=1), color=COLORS["virtual"], label="virtual vs high")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    stem = f"{sensor}_error_comparison"
    save_figure(fig, out_dir, stem)
    return stem, f"{sensor.capitalize()} residual norm shows whether the virtual IMU reduces error relative to the low-grade input."
