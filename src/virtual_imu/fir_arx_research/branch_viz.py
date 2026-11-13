"""Nature-style cross-branch summary figures for FIR/ARX preprocessing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from virtual_imu.fir_arx_research.viz import BLUE, BLUE_LIGHT, GOLD, LIGHT_GREY, NEUTRAL, RED, TEAL, _panel_label, _style


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


def plot_branch_effects(comparison: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    selected = comparison.copy()
    selected["branch_label"] = selected["branch"].map(
        {"vmd": "VMD joint", "tda": "TDA routed", "minimal": "Minimal"}
    )
    selected.loc[selected["condition"].eq("vmd_low"), "branch_label"] = "VMD low"
    selected.loc[selected["condition"].eq("vmd_high"), "branch_label"] = "VMD high"
    selected = selected[selected["model"].eq("fir") | selected["model"].eq("fir_arx")]
    agg = selected.groupby(["branch_label", "protocol"], as_index=False).agg(
        gyro=("gyro_delta_pct", "median"),
        accel=("accel_delta_pct", "median"),
    )
    branch_order = ["Minimal", "TDA routed", "VMD low", "VMD joint", "VMD high"]
    protocols = ["alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf", "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    x = np.arange(len(protocols))
    width = 0.12
    colors = {"Minimal": GOLD, "TDA routed": TEAL, "VMD low": BLUE_LIGHT, "VMD joint": BLUE, "VMD high": RED}
    for ax, metric, title in (
        (axes[0], "gyro", "Gyroscope"),
        (axes[1], "accel", "Accelerometer"),
    ):
        ax.axhline(0, color=NEUTRAL, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for index, branch in enumerate(branch_order):
            values = []
            for protocol in protocols:
                row = agg[(agg["branch_label"] == branch) & (agg["protocol"] == protocol)]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            offset = (index - (len(branch_order) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[branch],
                edgecolor="white",
                linewidth=0.3,
                label=branch,
                zorder=2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=25, ha="right")
        ax.set_title(title, loc="left", pad=10, fontweight="normal")
        ax.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7, zorder=0)
        ax.set_xlabel("Recording-disjoint protocol")
    axes[0].set_ylabel("Delta vs raw FIR/ARX [%]")
    axes[0].set_yscale("symlog", linthresh=5.0)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(-0.03, -0.22),
        ncol=5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    fig.text(
        0.01,
        0.01,
        "Bars show median delta across available FIR/ARX variants. Symlog y axis preserves large VMD degradations; negative is lower RMSE.",
        fontsize=5.7,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.39, wspace=0.10)
    return _save(fig, out_dir, "fig1_branch_effects")


def plot_branch_summary(summary: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    _style()
    selected = summary[summary["model"].isin(["fir", "fir_arx"])].copy()
    selected["label"] = selected["condition"].map(
        {"vmd_low": "VMD low", "vmd_joint": "VMD joint", "vmd_high": "VMD high", "tda_routed": "TDA routed", "scenario_minimal": "Minimal"}
    )
    labels = ["Minimal", "TDA routed", "VMD low", "VMD joint", "VMD high"]
    present = [label for label in labels if label in selected["label"].values]
    selected = selected.set_index("label").loc[present]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(selected))
    values = selected["joint_score_delta_median"].to_numpy()
    colors = [GOLD, TEAL, BLUE_LIGHT, BLUE, RED]
    color_map = {"Minimal": GOLD, "TDA routed": TEAL, "VMD low": BLUE_LIGHT, "VMD joint": BLUE, "VMD high": RED}
    ax.scatter(values, y, s=34, color=[color_map[label] for label in selected.index], edgecolor="white", linewidth=0.5, zorder=3)
    for ypos, value, wins, total in zip(
        y,
        values,
        selected["branch_win_protocols"].to_numpy(),
        selected["protocols"].to_numpy(),
        strict=True,
    ):
        ax.annotate(
            f"{value:+.2f}%  ({int(wins)}/{int(total)})",
            (value, ypos),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.1,
            color=NEUTRAL,
        )
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(selected.index)
    ax.set_xlabel("Median joint delta vs raw control [%]")
    ax.set_xscale("symlog", linthresh=5.0)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    ax.set_title("Branch-level signed effect", loc="left", fontweight="normal", pad=10)
    _panel_label(ax, "a")
    fig.text(
        0.01,
        0.01,
        "Parentheses: protocols where the branch jointly improves gyro and accel error. Negative is better.",
        fontsize=5.8,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.23)
    return _save(fig, out_dir, "fig2_branch_summary")
    if not comparison["model"].isin(["fir", "fir_arx"]).any():
        raise ValueError("Branch comparison has no FIR-family rows")
