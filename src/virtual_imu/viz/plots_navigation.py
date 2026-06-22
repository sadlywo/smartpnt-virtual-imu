"""Preintegration and navigation plots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.navigation import integrate_dead_reckoning
from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_preintegration_error(df: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    """Plot cumulative IMU increment error proxies."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    t = df["timestamp_s"].to_numpy(float)
    dt = np.diff(t, prepend=t[0])[:, None]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for prefix, color in [("low", COLORS["low"]), ("virtual", COLORS["virtual"])]:
        gyro_err = df[[f"g{axis}_{prefix}" for axis in ["x", "y", "z"]]].to_numpy(float) - df[
            ["gx_high", "gy_high", "gz_high"]
        ].to_numpy(float)
        accel_err = df[[f"a{axis}_{prefix}" for axis in ["x", "y", "z"]]].to_numpy(float) - df[
            ["ax_high", "ay_high", "az_high"]
        ].to_numpy(float)
        inc = np.linalg.norm(np.cumsum(np.column_stack([gyro_err, accel_err]) * dt, axis=0), axis=1)
        ax.plot(t - t[0], inc, color=color, label=prefix)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative increment error")
    ax.legend(frameon=False)
    stem = "preintegration_error"
    save_figure(fig, out_dir, stem)
    return stem, "Cumulative preintegration error proxy for low-grade and virtual IMU signals relative to high-grade reference."


def plot_navigation_drift(df: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    """Plot simplified navigation drift."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    t = df["timestamp_s"].to_numpy(float)
    ref = integrate_dead_reckoning(df, "high")
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
    for prefix, color in [("low", COLORS["low"]), ("virtual", COLORS["virtual"])]:
        nav = integrate_dead_reckoning(df, prefix)
        axs[0].plot(t - t[0], np.linalg.norm(nav["attitude"] - ref["attitude"], axis=1), color=color, label=prefix)
        axs[1].plot(t - t[0], np.linalg.norm(nav["velocity"] - ref["velocity"], axis=1), color=color)
        axs[2].plot(t - t[0], np.linalg.norm(nav["position"] - ref["position"], axis=1), color=color)
    axs[0].set_ylabel("Attitude [rad]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[2].set_ylabel("Position [m]")
    axs[2].set_xlabel("Time [s]")
    axs[0].legend(frameon=False)
    stem = "navigation_drift"
    save_figure(fig, out_dir, stem)
    return stem, "Dead-reckoning drift of low-grade and virtual IMU against high-grade proxy integration."


def plot_trajectory(df: pd.DataFrame, out_dir: str) -> tuple[str, str]:
    """Plot 2D proxy trajectory comparison."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for prefix, color in [("low", COLORS["low"]), ("virtual", COLORS["virtual"]), ("high", COLORS["high"])]:
        nav = integrate_dead_reckoning(df, prefix)
        ax.plot(nav["position"][:, 0], nav["position"][:, 1], color=color, label=prefix)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(frameon=False)
    stem = "trajectory_comparison"
    save_figure(fig, out_dir, stem)
    return stem, "Proxy 2D trajectory from low-grade, virtual, and high-grade IMU integration."
