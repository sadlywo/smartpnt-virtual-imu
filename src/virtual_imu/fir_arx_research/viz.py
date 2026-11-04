"""Paper-style figures for FIR/ARX experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from virtual_imu.viz.style import apply_paper_style

BLUE = "#0F4D92"
BLUE_LIGHT = "#DCE7F4"
TEAL = "#42949E"
RED = "#B64342"
RED_LIGHT = "#F4DEDB"
GOLD = "#9B7A24"
NEUTRAL = "#4D4D4D"
LIGHT_GREY = "#D8D8D8"

METHOD_LABELS = {
    "mean_bias": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "fir": "FIR",
    "arx": "ARX",
}

METHOD_COLORS = {
    "mean_bias": "#8A8A8A",
    "affine": "#B9A7E8",
    "ridge_affine": "#7C6CCF",
    "fir": BLUE,
    "arx": TEAL,
}


def _style() -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        files[suffix] = str(path)
    plt.close(fig)
    with Image.open(files["png"]) as image:
        pixels = image.size
    return {"stem": stem, "files": files, "png_pixels": list(pixels)}


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="#111111",
    )


def plot_protocol_improvements(pair_metrics: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    protocols = pair_metrics["protocol"].drop_duplicates().tolist()
    methods = [name for name in METHOD_LABELS if name in set(pair_metrics["model"])]
    protocol_summary = (
        pair_metrics[pair_metrics["model"].isin(methods)]
        .groupby(["protocol", "model"], as_index=False)
        .agg(
            gyro_pct=("test_gyro_improvement_vs_identity_pct", "mean"),
            accel_pct=("test_accel_improvement_vs_identity_pct", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
    x = np.arange(len(protocols))
    width = 0.11 if len(protocols) > 2 else 0.075
    for ax, metric, title in (
        (axes[0], "gyro_pct", "Gyroscope"),
        (axes[1], "accel_pct", "Accelerometer"),
    ):
        ax.axhline(0, color=NEUTRAL, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for index, method in enumerate(methods):
            values = []
            for protocol in protocols:
                row = protocol_summary[
                    (protocol_summary["protocol"] == protocol)
                    & (protocol_summary["model"] == method)
                ]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            offset = (index - (len(methods) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.4,
                label=METHOD_LABELS[method],
                zorder=2,
            )
        ax.set_xticks(x)
        if len(protocols) > 2:
            ax.set_xticklabels(protocols, rotation=25, ha="right")
        else:
            ax.set_xticklabels(protocols)
        ax.set_title(title, loc="left", pad=10, fontweight="normal")
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7, zorder=0)
        ax.set_xlabel("Recording-disjoint protocol")
    axes[0].set_ylabel("RMSE improvement vs LOW [%]")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(-0.05, -0.23),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    fig.text(
        0.01,
        0.01,
        "Macro averages across test sensor pairs. Values above zero indicate lower RMSE than the uncorrected LOW signal.",
        fontsize=5.8,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.92, bottom=0.34, wspace=0.10)
    return _save(fig, out_dir, "fig1_protocol_improvements")


def plot_pair_scatter(pair_metrics: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    methods = [name for name in METHOD_LABELS if name in set(pair_metrics["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    pair_ids = pair_metrics[["protocol", "dataset_id", "pair_id"]].drop_duplicates()
    pair_ids = pair_ids.sort_values(["protocol", "dataset_id", "pair_id"]).reset_index(drop=True)
    y_base = np.arange(len(pair_ids))
    for ax, metric, title in (
        (axes[0], "test_gyro_improvement_vs_identity_pct", "Gyroscope"),
        (axes[1], "test_accel_improvement_vs_identity_pct", "Accelerometer"),
    ):
        ax.axvline(0, color=NEUTRAL, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for index, method in enumerate(methods):
            ys, values = [], []
            for ypos, row in pair_ids.iterrows():
                match = pair_metrics[
                    (pair_metrics["protocol"] == row["protocol"])
                    & (pair_metrics["dataset_id"] == row["dataset_id"])
                    & (pair_metrics["pair_id"] == row["pair_id"])
                    & (pair_metrics["model"] == method)
                ]
                if len(match):
                    ys.append(ypos)
                    values.append(float(match[metric].iloc[0]))
            ax.scatter(
                values,
                np.asarray(ys) + (index - (len(methods) - 1) / 2) * 0.08,
                s=13,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.86,
                label=METHOD_LABELS[method],
                zorder=3,
            )
        ax.set_title(title, loc="left", pad=10, fontweight="normal")
        ax.grid(axis="x", color=LIGHT_GREY, lw=0.4, alpha=0.7)
        ax.set_xlabel("RMSE improvement vs LOW [%]")
        ax.set_yticks(y_base)
        ax.set_ylim(len(pair_ids) - 0.5, -0.5)
    labels = [f"{row.protocol} | {row.pair_id}" for row in pair_ids.itertuples(index=False)]
    axes[0].set_yticklabels(labels)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(-0.03, -0.20),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    if len(pair_ids) <= 3:
        fig.set_size_inches(7.2, 3.2)
        fig.subplots_adjust(left=0.31, right=0.99, top=0.86, bottom=0.31, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.31, right=0.99, top=0.90, bottom=0.28, wspace=0.10)
    return _save(fig, out_dir, "fig2_pair_level_improvements")


def plot_minimal_vs_full(pair_metrics: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    protocols = pair_metrics["protocol"].drop_duplicates().tolist()
    models = [name for name in ("raw_fir", "raw_arx", "full_fir", "full_arx", "minimal_fir_arx") if name in set(pair_metrics["model"])]
    labels = {"raw_fir": "Raw FIR", "raw_arx": "Raw ARX", "full_fir": "All-component FIR", "full_arx": "All-component ARX", "minimal_fir_arx": "Minimal routed ARX"}
    colors = {"raw_fir": "#8A8A8A", "raw_arx": "#B0B0B0", "full_fir": BLUE_LIGHT, "full_arx": TEAL, "minimal_fir_arx": "#9B7A24"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    x = np.arange(len(protocols))
    width = 0.13
    for ax, metric, title in (
        (axes[0], "test_gyro_improvement_vs_identity_pct", "Gyroscope"),
        (axes[1], "test_accel_improvement_vs_identity_pct", "Accelerometer"),
    ):
        ax.axhline(0, color=NEUTRAL, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for index, model in enumerate(models):
            values = []
            for protocol in protocols:
                rows = pair_metrics[(pair_metrics["protocol"] == protocol) & (pair_metrics["model"] == model)]
                values.append(float(rows[metric].mean()) if len(rows) else np.nan)
            offset = (index - (len(models) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=colors[model], edgecolor="white", linewidth=0.35, label=labels[model], zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=25, ha="right")
        ax.set_title(title, loc="left", pad=10, fontweight="normal")
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7, zorder=0)
        ax.set_xlabel("Recording-disjoint protocol")
    axes[0].set_ylabel("Test improvement vs LOW [%]")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.04, -0.22), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.7)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.32, wspace=0.10)
    return _save(fig, out_dir, "fig1_minimal_vs_full")


def plot_minimal_frontier(selections: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    rows = []
    for row in selections.itertuples(index=False):
        if getattr(row, "fallback", False) or not isinstance(getattr(row, "frontier", None), str):
            continue
        for item in __import__("json").loads(row.frontier):
            rows.append({
                "protocol": row.protocol,
                "scene": int(row.scene),
                "component_count": int(item["component_count"]),
                "validation_score": float(item["validation_score"]),
            })
    frontier = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    if frontier.empty:
        ax.text(0.5, 0.5, "No eligible frontier", ha="center", va="center", transform=ax.transAxes)
    else:
        grp = frontier.groupby("component_count")["validation_score"].agg(["mean", "min", "max"]).reset_index()
        ax.fill_between(grp["component_count"], grp["min"], grp["max"], color=BLUE_LIGHT, alpha=0.35, linewidth=0)
        ax.plot(grp["component_count"], grp["mean"], color=BLUE, marker="o", markersize=3.2)
        ax.set_xlabel("Causal component count")
        ax.set_ylabel("Identity-normalized validation score")
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    ax.set_title("Minimal-component accuracy frontier", loc="left", fontweight="normal", pad=10)
    _panel_label(ax, "a")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.88, bottom=0.20)
    return _save(fig, out_dir, "fig2_minimal_component_frontier")
