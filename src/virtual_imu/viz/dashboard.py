"""End-to-end figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_imu.viz.plots_allan import plot_allan
from virtual_imu.viz.plots_navigation import (
    plot_navigation_drift,
    plot_preintegration_error,
    plot_trajectory,
)
from virtual_imu.viz.plots_signal import plot_error_comparison, plot_time_series
from virtual_imu.viz.plots_spectral import plot_psd_comparison
from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_summary_bars(metrics: dict[str, Any], out_dir: str) -> tuple[str, str]:
    """Plot summary RMSE bars from evaluation metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    signal = metrics.get("signal", {})
    low = signal.get("low", {})
    virtual = signal.get("virtual", {})
    names = ["gyro RMSE", "accel RMSE"]
    low_values = [low.get("gyro_norm_rmse", np.nan), low.get("accel_norm_rmse", np.nan)]
    virtual_values = [virtual.get("gyro_norm_rmse", np.nan), virtual.get("accel_norm_rmse", np.nan)]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.bar(x - 0.18, low_values, width=0.36, color=COLORS["low"], label="low")
    ax.bar(x + 0.18, virtual_values, width=0.36, color=COLORS["virtual"], label="virtual")
    ax.set_xticks(x, names)
    ax.set_ylabel("Norm RMSE")
    ax.legend(frameon=False)
    stem = "summary_bar_metrics"
    save_figure(fig, out_dir, stem)
    return stem, "Summary RMSE bars compare low-grade and virtual IMU errors against the high-grade reference."


def plot_all_results(eval_path: str | Path, pred_path: str | Path, out_dir: str | Path) -> Path:
    """Generate all required paper-style figures and a figure index."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(pred_path)
    metrics = json.loads(Path(eval_path).read_text(encoding="utf-8")) if Path(eval_path).exists() else {}
    figures: list[tuple[str, str]] = []
    figures.append(plot_time_series(df, "gyro", str(root)))
    figures.append(plot_time_series(df, "accel", str(root)))
    figures.append(plot_error_comparison(df, "gyro", str(root)))
    figures.append(plot_error_comparison(df, "accel", str(root)))
    figures.append(plot_psd_comparison(df, str(root)))
    figures.append(plot_allan(df, str(root)))
    figures.append(plot_preintegration_error(df, str(root)))
    figures.append(plot_navigation_drift(df, str(root)))
    figures.append(plot_trajectory(df, str(root)))
    figures.append(plot_summary_bars(metrics, str(root)))
    index = root / "figure_index.md"
    lines = ["# Figure Index", "", "| Figure | Purpose | Caption Draft |", "|---|---|---|"]
    for stem, caption in figures:
        lines.append(f"| `{stem}.svg/pdf/png` | {stem.replace('_', ' ')} | {caption} |")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index
