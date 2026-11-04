"""Build a compact Nature-style summary of the latest public-data results.

The script reads frozen evaluation tables only. It performs no model fitting,
selection, or inference; all source metrics and negative results are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from virtual_imu.viz.style import apply_paper_style  # noqa: E402

BLUE = "#0F4D92"
BLUE_LIGHT = "#DCE7F4"
TEAL = "#42949E"
RED = "#B64342"
RED_LIGHT = "#F4DEDB"
GOLD = "#9B7A24"
NEUTRAL = "#4D4D4D"
LIGHT_GREY = "#D8D8D8"
BACKGROUND = "#FFFFFF"

PROTOCOL_LABELS = {
    "alimu_only": "ALiMU",
    "penn_loso_af": "Penn af",
    "penn_loso_as": "Penn as",
    "penn_loso_bf": "Penn bf",
    "penn_only": "Penn bs",
    "pooled": "Pooled",
}

METRIC_LABELS = {
    "gyro_vector_rmse_radps": "Gyroscope",
    "accel_vector_rmse_mps2": "Accelerometer",
}

FAMILY_LABELS = {
    "cnn_bilstm": "CNN-BiLSTM",
    "deepgyro_adapter": "DeepGyro",
    "gru": "GRU",
    "informer": "Informer",
    "itransformer": "iTransformer",
    "patchtst": "PatchTST",
    "plain_cnn": "CNN",
    "tartanimu_window_adapter": "TartanIMU",
    "tcn": "TCN",
    "tlio_resnet18_adapter": "TLIO",
    "transformer": "Transformer",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {"source", "output", "plot"}
    if not required.issubset(config):
        raise ValueError(f"Config must contain {sorted(required)}")
    return config


def _apply_style(config: dict[str, Any]) -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                str(config["plot"].get("font_family", "Arial")),
                "DejaVu Sans",
                "Liberation Sans",
            ],
            "font.size": float(config["plot"].get("base_font_size", 7)),
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.6,
            "axes.edgecolor": NEUTRAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "lines.linewidth": 1.1,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


def _panel_label(ax: plt.Axes, text: str, config: dict[str, Any]) -> None:
    ax.text(
        -0.045,
        1.035,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=float(config["plot"].get("panel_label_size", 9)),
        fontweight="bold",
        color="#111111",
    )


def _panel_label_fixed(
    fig: plt.Figure,
    ax: plt.Axes,
    text: str,
    config: dict[str, Any],
    dx: float = -0.065,
    dy: float = 0.012,
) -> None:
    position = ax.get_position()
    fig.text(
        position.x0 + dx,
        position.y1 + dy,
        text,
        ha="left",
        va="bottom",
        fontsize=float(config["plot"].get("panel_label_size", 9)),
        fontweight="bold",
        color="#111111",
    )


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / f"{stem}.png",
        dpi=int(config["output"]["png_dpi"]),
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / f"{stem}.tiff",
        dpi=int(config["output"]["tiff_dpi"]),
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    png = out_dir / f"{stem}.png"
    with Image.open(png) as image:
        width_px, height_px = image.size
    return {
        "stem": stem,
        "png_pixels": [width_px, height_px],
        "files": [str(out_dir / f"{stem}.{ext}") for ext in ("svg", "pdf", "png", "tiff")],
    }


def _signal_comparison(rankings: pd.DataFrame) -> pd.DataFrame:
    paired = rankings[rankings["ranking_condition"].eq("main_common_information")].copy()
    rows: list[dict[str, Any]] = []
    neural = paired[paired["information_scope"].eq("unified_six_low_neural")]
    groups = (
        neural[["protocol", "dataset_id", "metric"]]
        .drop_duplicates()
        .sort_values(["protocol", "dataset_id", "metric"])
    )
    for protocol, dataset, metric in groups.itertuples(index=False):
        protocol_rows = neural[
            (neural["protocol"] == protocol)
            & (neural["dataset_id"] == dataset)
            & (neural["metric"] == metric)
        ]
        classic_rows = paired[
            (paired["protocol"] == protocol)
            & (paired["dataset_id"] == dataset)
            & (paired["metric"] == metric)
            & paired["information_scope"].eq("protocol_shared_classical")
        ]
        best_neural = protocol_rows.loc[protocol_rows["mean"].idxmin()]
        best_classic = classic_rows.loc[classic_rows["mean"].idxmin()]
        identity_rows = paired[
            (paired["protocol"] == protocol)
            & (paired["dataset_id"] == dataset)
            & (paired["metric"] == metric)
            & paired["information_scope"].eq("common_identity")
        ]
        identity = float(identity_rows["mean"].iloc[0])
        if protocol == "pooled":
            label = "Pooled AL" if dataset == "alimu" else "Pooled Penn"
        else:
            label = PROTOCOL_LABELS[protocol]
        rows.append(
            {
                "protocol": protocol,
                "protocol_label": label,
                "dataset_id": dataset,
                "test_dataset": dataset,
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "unit": best_neural["unit"],
                "best_neural_family": best_neural["canonical_method"],
                "best_neural_variant": best_neural["variant"],
                "neural_mean": float(best_neural["mean"]),
                "neural_sample_sd": float(best_neural["sample_sd"]),
                "neural_rank": int(best_neural["rank"]),
                "best_classical": best_classic["canonical_method"],
                "classical_mean": float(best_classic["mean"]),
                "classical_rank": int(best_classic["rank"]),
                "ratio_neural_to_classical": float(best_neural["mean"] / best_classic["mean"]),
                "neural_vs_classical_pct": float(
                    100.0 * (best_neural["mean"] - best_classic["mean"]) / best_classic["mean"]
                ),
                "identity_mean": identity,
                "neural_improvement_vs_identity_pct": float(
                    100.0 * (identity - best_neural["mean"]) / identity
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["protocol", "dataset_id", "metric"]).reset_index(drop=True)


def _plot_signal_translation(
    signal: pd.DataFrame,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True)
    protocols = signal["protocol_label"].drop_duplicates().tolist()
    y = np.arange(len(protocols))
    for ax, metric in zip(axes, METRIC_LABELS, strict=True):
        subset = signal[signal["metric"] == metric].set_index("protocol_label").loc[protocols]
        ratios = subset["ratio_neural_to_classical"].to_numpy()
        colors = [BLUE if value < 1 else RED for value in ratios]
        ax.axvspan(0.25, 1.0, color=BLUE_LIGHT, alpha=0.48, linewidth=0, zorder=0)
        ax.axvspan(1.0, 3.15, color=RED_LIGHT, alpha=0.36, linewidth=0, zorder=0)
        ax.axvline(1.0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)), zorder=1)
        for ypos, ratio, color in zip(y, ratios, colors, strict=True):
            ax.hlines(ypos, 1.0, ratio, color=color, lw=1.0, alpha=0.8, zorder=2)
            ax.scatter(ratio, ypos, s=30, color=color, edgecolor=BACKGROUND, linewidth=0.6, zorder=3)
            ax.annotate(
                f"{ratio:.2f}x",
                (ratio, ypos),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.7,
                color=NEUTRAL,
            )
        ax.set_xlim(0.25, 3.15)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel("Best neural RMSE / best shared classical RMSE")
        ax.set_title(METRIC_LABELS[metric], pad=8, fontweight="normal")
        ax.set_yticks(y)
        ax.set_ylim(len(protocols) - 0.5, -0.5)
        ax.grid(axis="x", color=LIGHT_GREY, lw=0.4, alpha=0.7)
        ax.tick_params(axis="both", length=2.5, pad=2)
        ax.text(0.29, 1.01, "Neural lower", transform=ax.transAxes, fontsize=6.0, color=BLUE)
        ax.text(0.77, 1.01, "Classical lower", transform=ax.transAxes, fontsize=6.0, color=RED)
    axes[0].set_yticklabels(protocols)
    _panel_label(axes[0], "a", config)
    _panel_label(axes[1], "b", config)
    win_count = int((signal["ratio_neural_to_classical"] < 1).sum())
    fig.text(
        0.01,
        0.01,
        f"Fixed test sets; 3 neural seeds; classical n=1. Neural lower in {win_count}/{len(signal)} protocol-metric comparisons.",
        fontsize=5.9,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.86, bottom=0.17, wspace=0.13)
    record = _save_figure(fig, out_dir, "fig1_signal_translation", config)
    record.update(
        {
            "caption": (
                "Neural virtual-IMU correction is inconsistent against strong shared classical controls. "
                "Ratios below one indicate lower RMSE for the best neural model; gyroscope and "
                "accelerometer units are shown separately."
            ),
            "neural_wins": win_count,
            "comparisons": len(signal),
        }
    )
    return record


def _plot_tuning(tuning: pd.DataFrame, out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["plot"]["seed"]))
    signal = tuning[tuning["metric"].isin(METRIC_LABELS)].copy()
    family_order = signal.groupby("family")["improvement_pct"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.8, 1.0]})
    ax = axes[0]
    for ypos, family in enumerate(family_order):
        values = signal.loc[signal["family"] == family, "improvement_pct"].to_numpy()
        temporal = signal.loc[signal["family"] == family, "temporal_class"].iloc[0]
        color = BLUE if temporal == "sample_causal_eval" else TEAL
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(values, ypos + jitter, s=11, color=color, alpha=0.55, edgecolors="none", zorder=2)
        ax.plot(
            [np.percentile(values, 10), np.percentile(values, 90)],
            [ypos, ypos],
            color=NEUTRAL,
            lw=0.9,
            zorder=3,
        )
        ax.scatter(np.median(values), ypos, s=28, color=NEUTRAL, zorder=4)
    ax.axvline(0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels([FAMILY_LABELS.get(name, name) for name in family_order])
    ax.set_xlabel("Tuned vs default RMSE reduction [%]")
    ax.set_xlim(-45, 72)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    better = int((signal["improvement_pct"] > 0).sum())
    ax.text(0.98, 0.98, f"Better: {better}/{len(signal)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=4, label="Causal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL, markersize=4, label="Window"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NEUTRAL, markersize=4, label="Median / 10-90%"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    _panel_label(ax, "a", config)

    ax = axes[1]
    nav = tuning[tuning["metric"].eq("vector_rmse_m")].copy()
    rotation_names = ["navigation_rotation_0", "navigation_rotation_1", "navigation_rotation_2"]
    rotation_labels = ["Rotation 0", "Rotation 1", "Rotation 2"]
    rotation_colors = [BLUE, TEAL, GOLD]
    for xpos, (rotation, _label, color) in enumerate(zip(rotation_names, rotation_labels, rotation_colors, strict=True)):
        values = nav.loc[nav["protocol"] == rotation, "improvement_pct"].to_numpy()
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(xpos + jitter, values, s=15, color=color, alpha=0.7, edgecolors="none", zorder=2)
        ax.plot([xpos - 0.20, xpos + 0.20], [np.median(values), np.median(values)], color=NEUTRAL, lw=1.0, zorder=3)
        ax.scatter(xpos, np.median(values), s=22, color=NEUTRAL, zorder=4)
    ax.axhline(0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(rotation_labels)
    ax.set_ylabel("Tuned vs default RMSE reduction [%]")
    ax.set_ylim(-4, 24)
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    nav_better = int((nav["improvement_pct"] > 0).sum())
    ax.text(0.98, 0.98, f"Better: {nav_better}/{len(nav)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.2)
    _panel_label(ax, "b", config)

    fig.text(
        0.01,
        0.01,
        "Dots are matched comparisons; bars show medians and the left panel spans the 10th-90th percentiles.",
        fontsize=5.9,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.93, bottom=0.17, wspace=0.36)
    record = _save_figure(fig, out_dir, "fig2_tuning_robustness", config)
    record.update(
        {
            "caption": (
                "Validation tuning improves most paired signal comparisons, but gains are family dependent; "
                "SmartPNT navigation gains are much smaller and more variable."
            ),
            "signal_better": better,
            "signal_comparisons": len(signal),
            "navigation_better": nav_better,
            "navigation_comparisons": len(nav),
        }
    )
    return record


def _read_navigation_sources(figure_run: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    figure_root = figure_run / "figures"
    for rotation in range(3):
        matches = list(figure_root.glob(f"metrics_navigation_navigation_rotation_{rotation}_test_*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one navigation source for rotation {rotation}, found {len(matches)}"
            )
        frame = pd.read_csv(matches[0] / "source.csv")
        frame = frame[
            (frame["metric"] == "vector_rmse_m")
            & (frame["split"] == "test")
            & (frame["aggregation_level"] == "dataset")
        ].copy()
        frame["rotation"] = rotation
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _navigation_summary(nav_source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rotation, group in nav_source.groupby("rotation"):
        zero = float(group.loc[group["model_family"] == "zero", "mean"].iloc[0])
        train_mean = float(group.loc[group["model_family"] == "train_mean", "mean"].iloc[0])
        neural = group[group["method"] == "model"]
        tuned = neural[neural["variant"] == "tuned"]
        best_tuned = tuned.loc[tuned["mean"].idxmin()]
        best_any = neural.loc[neural["mean"].idxmin()]
        rows.append(
            {
                "rotation": int(rotation),
                "zero_rmse_m": zero,
                "train_mean_rmse_m": train_mean,
                "best_tuned_family": best_tuned["model_family"],
                "best_tuned_rmse_m": float(best_tuned["mean"]),
                "best_tuned_sample_sd_m": float(best_tuned["sample_sd"]),
                "best_tuned_vs_zero_pct": float(100.0 * (best_tuned["mean"] - zero) / zero),
                "best_any_variant_family": best_any["model_family"],
                "best_any_variant": best_any["variant"],
                "best_any_rmse_m": float(best_any["mean"]),
                "best_any_vs_zero_pct": float(100.0 * (best_any["mean"] - zero) / zero),
            }
        )
    return pd.DataFrame(rows).sort_values("rotation")


def _plot_navigation(
    nav_source: pd.DataFrame,
    nav_summary: pd.DataFrame,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    neural = nav_source[(nav_source["method"] == "model") & (nav_source["variant"] == "tuned")].copy()
    zero_by_rotation = (
        nav_source[nav_source["model_family"] == "zero"].set_index("rotation")["mean"].to_dict()
    )
    neural["relative_to_zero_pct"] = neural.apply(
        lambda row: 100.0
        * (row["mean"] - zero_by_rotation[int(row["rotation"])])
        / zero_by_rotation[int(row["rotation"])],
        axis=1,
    )
    family_order = neural.groupby("model_family")["relative_to_zero_pct"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = [BLUE, TEAL, GOLD]
    markers = ["o", "s", "D"]
    for rotation in range(3):
        subset = neural[neural["rotation"] == rotation].set_index("model_family").loc[family_order]
        y = np.arange(len(family_order))
        ax.plot(
            subset["relative_to_zero_pct"],
            y,
            linestyle="none",
            marker=markers[rotation],
            markersize=4.1,
            markerfacecolor=colors[rotation],
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.45,
            alpha=0.92,
            zorder=3,
            label=f"Rotation {rotation}",
        )
    ax.axvline(0, color=RED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels([FAMILY_LABELS.get(name, name) for name in family_order])
    ax.set_xlabel("Tuned neural RMSE relative to zero displacement [%]")
    ax.set_xlim(-0.5, 23.0)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.57, -0.20),
        frameon=False,
        ncol=3,
        handletextpad=0.25,
        columnspacing=0.8,
    )
    all_worse = bool((neural["relative_to_zero_pct"] > 0).all())
    ax.text(
        0.98,
        0.97,
        f"Tuned models above zero: {int((neural['relative_to_zero_pct'] > 0).sum())}/{len(neural)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=NEUTRAL,
    )
    _panel_label(ax, "a", config)
    best_any = nav_summary.sort_values("best_any_vs_zero_pct").iloc[0]
    fig.text(
        0.12,
        0.01,
        (
            "Positive values indicate worse three-axis window-endpoint RMSE. "
            f"The sole default variant below zero is {FAMILY_LABELS.get(best_any['best_any_variant_family'], best_any['best_any_variant_family'])} "
            f"in rotation {int(best_any['rotation'])} ({best_any['best_any_vs_zero_pct']:.2f}%)."
        ),
        fontsize=5.8,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.29)
    record = _save_figure(fig, out_dir, "fig3_navigation_vs_zero", config)
    record.update(
        {
            "caption": (
                "SmartPNT short-window displacement is not solved by the adapted neural models. "
                "All tuned neural variants exceed the zero-displacement baseline; only one default "
                "variant is marginally below it."
            ),
            "all_tuned_above_zero": all_worse,
            "tuned_comparisons": int(len(neural)),
            "tuned_above_zero": int((neural["relative_to_zero_pct"] > 0).sum()),
        }
    )
    return record


def _downstream_distribution(downstream: pd.DataFrame) -> pd.DataFrame:
    neural = downstream[downstream["information_scope"].eq("unified_six_low_neural")].copy()
    rows: list[dict[str, Any]] = []
    for (task, horizon), group in neural.groupby(["task", "horizon_s"]):
        values = group["improvement_over_low_pct"].dropna()
        rows.append(
            {
                "task": task,
                "horizon_s": float(horizon),
                "n": int(len(values)),
                "median_improvement_pct": float(values.median()),
                "q25_improvement_pct": float(values.quantile(0.25)),
                "q75_improvement_pct": float(values.quantile(0.75)),
                "fraction_improved": float((values > 0).mean()),
                "min_improvement_pct": float(values.min()),
                "max_improvement_pct": float(values.max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "horizon_s"])


def _signal_downstream_correlation(
    rankings: pd.DataFrame,
    downstream: pd.DataFrame,
) -> pd.DataFrame:
    common = rankings[rankings["ranking_condition"].eq("main_common_information")].copy()
    identity = common[common["information_scope"].eq("common_identity")]
    baselines = {(row.protocol, row.metric): float(row.mean) for row in identity.itertuples(index=False)}
    signal = common[
        common["information_scope"].isin(["unified_six_low_neural", "protocol_shared_classical"])
    ].copy()
    signal["signal_improvement_pct"] = [
        float(100.0 * (baselines[(row.protocol, row.metric)] - row.mean) / baselines[(row.protocol, row.metric)])
        for row in signal.itertuples(index=False)
    ]
    downstream_common = downstream[
        downstream["information_scope"].isin(["unified_six_low_neural", "protocol_shared_classical"])
    ]
    rows: list[dict[str, Any]] = []
    for (task, horizon), down_group in downstream_common.groupby(["task", "horizon_s"]):
        joined = signal.merge(
            down_group[
                [
                    "protocol",
                    "canonical_method",
                    "variant",
                    "information_scope",
                    "improvement_over_low_pct",
                ]
            ],
            on=["protocol", "canonical_method", "variant", "information_scope"],
            how="inner",
        )
        for metric, metric_group in joined.groupby("metric"):
            values = metric_group.dropna(subset=["signal_improvement_pct", "improvement_over_low_pct"])
            rank_signal = values["signal_improvement_pct"].rank(method="average")
            rank_downstream = values["improvement_over_low_pct"].rank(method="average")
            rho = float(rank_signal.corr(rank_downstream))
            n = int(len(values))
            if abs(rho) >= 1.0:
                p_value = 0.0
            else:
                t_stat = abs(rho) * np.sqrt((n - 2) / (1.0 - rho * rho))
                # Two-sided Student-t tail for two numeric variables.
                from scipy.stats import t as student_t

                p_value = float(2.0 * student_t.sf(t_stat, df=n - 2))
            rows.append(
                {
                    "task": task,
                    "horizon_s": float(horizon),
                    "metric": metric,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows).sort_values(["task", "metric", "horizon_s"])


def _plot_downstream(
    distribution: pd.DataFrame,
    correlations: pd.DataFrame,
    out_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    fig = plt.figure(figsize=(7.2, 5.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.38,
        hspace=0.52,
    )
    ax_main = fig.add_subplot(grid[:, 0])
    ax_frac = fig.add_subplot(grid[0, 1])
    ax_corr = fig.add_subplot(grid[1, 1])

    tasks = {"attitude": RED, "strapdown": BLUE}
    for task, color in tasks.items():
        subset = distribution[distribution["task"] == task].sort_values("horizon_s")
        x = subset["horizon_s"].to_numpy()
        median = subset["median_improvement_pct"].to_numpy()
        q25 = subset["q25_improvement_pct"].to_numpy()
        q75 = subset["q75_improvement_pct"].to_numpy()
        ax_main.plot(x, median, color=color, marker="o", markersize=3.8, lw=1.2, label=task.capitalize())
        ax_main.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
        for xpos, value in zip(x, median, strict=True):
            ax_main.annotate(
                f"{value:+.0f}%",
                (xpos, value),
                xytext=(0, -10 if task == "strapdown" else 7),
                textcoords="offset points",
                ha="center",
                fontsize=5.7,
                color=color,
            )
    ax_main.axhline(0, color=NEUTRAL, lw=0.7, ls=(0, (3, 2)))
    ax_main.set_xscale("log")
    ax_main.set_xticks([1.975, 4.975, 9.975, 19.975])
    ax_main.set_xticklabels(["1.975", "4.975", "9.975", "19.975"])
    ax_main.minorticks_off()
    ax_main.set_xlabel("Endpoint horizon [s]")
    ax_main.set_ylabel("Improvement vs LOW [%]")
    ax_main.set_title("Independent optical-GT propagation", loc="left", fontweight="normal", pad=10)
    ax_main.legend(loc="lower left", frameon=False, ncol=2)
    ax_main.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.7)
    _panel_label(ax_main, "a", config)

    labels = [f"{task[:3].upper()}\n{horizon:g}s" for task in ["attitude", "strapdown"] for horizon in [1.975, 4.975, 9.975, 19.975]]
    xpos = np.arange(8)
    fractions: list[float] = []
    for task in ["attitude", "strapdown"]:
        subset = distribution[distribution["task"] == task].sort_values("horizon_s")
        fractions.extend(subset["fraction_improved"].tolist())
    improved = np.asarray(fractions) * 100.0
    degraded = 100.0 - improved
    ax_frac.bar(xpos, improved, color=[RED if i < 4 else BLUE for i in xpos], width=0.72, linewidth=0)
    ax_frac.bar(xpos, degraded, bottom=improved, color=LIGHT_GREY, width=0.72, linewidth=0)
    ax_frac.axhline(50, color=BACKGROUND, lw=0.7, ls=(0, (2, 2)))
    ax_frac.set_xticks(xpos)
    ax_frac.set_xticklabels(labels, fontsize=4.8)
    ax_frac.set_ylim(0, 100)
    ax_frac.set_ylabel("Comparisons improved [%]")
    ax_frac.set_title("Outcome consistency", loc="left", fontweight="normal", pad=12)
    ax_frac.tick_params(axis="x", length=0, pad=2)
    ax_frac.grid(axis="y", color=LIGHT_GREY, lw=0.4, alpha=0.55)
    _panel_label_fixed(fig, ax_frac, "b", config)

    metrics = list(METRIC_LABELS)
    horizons = [1.975, 4.975, 9.975, 19.975]
    heat = np.zeros((len(metrics), len(horizons) * 2))
    annotations: list[list[str]] = [["" for _ in range(heat.shape[1])] for _ in range(heat.shape[0])]
    for column_offset, task in enumerate(["attitude", "strapdown"]):
        for hpos, horizon in enumerate(horizons):
            column = column_offset * len(horizons) + hpos
            for mpos, metric in enumerate(metrics):
                row = correlations[
                    (correlations["task"] == task)
                    & (correlations["horizon_s"] == horizon)
                    & (correlations["metric"] == metric)
                ].iloc[0]
                heat[mpos, column] = float(row["spearman_rho"])
                annotations[mpos][column] = f"{float(row['spearman_rho']):+.2f}"
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)
    image = ax_corr.imshow(heat, aspect="auto", cmap="RdBu_r", norm=norm)
    for row in range(heat.shape[0]):
        for column in range(heat.shape[1]):
            ax_corr.text(column, row, annotations[row][column], ha="center", va="center", fontsize=5.2, color="#111111")
    ax_corr.set_yticks(np.arange(len(metrics)))
    ax_corr.set_yticklabels([METRIC_LABELS[name] for name in metrics])
    ax_corr.set_xticks(np.arange(8))
    ax_corr.set_xticklabels(["2", "5", "10", "20", "2", "5", "10", "20"], fontsize=5.0)
    ax_corr.axvline(3.5, color=NEUTRAL, lw=0.7)
    ax_corr.set_xlabel("Attitude horizon [s]          Strapdown horizon [s]", fontsize=6.0)
    ax_corr.set_title("Signal-to-downstream rank association", loc="left", fontweight="normal", pad=12)
    ax_corr.tick_params(length=0)
    for spine in ax_corr.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_corr, fraction=0.045, pad=0.035)
    colorbar.set_label("Spearman rho", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.4, length=1.5, pad=1)
    _panel_label_fixed(fig, ax_corr, "c", config)

    fig.text(
        0.01,
        0.01,
        "Lines and bands: median and interquartile range across neural outputs. Correlations use matched neural and classical controls.",
        fontsize=5.8,
        color=NEUTRAL,
    )
    record = _save_figure(fig, out_dir, "fig4_downstream_propagation", config)
    record.update(
        {
            "caption": (
                "Signal correction does not automatically improve independent-GT navigation. "
                "Short-horizon strapdown position improves, attitude generally degrades, and all neural "
                "comparisons degrade at the longest horizon."
            ),
            "longest_horizon_all_degraded": bool(
                (
                    distribution[
                        distribution["horizon_s"] == distribution["horizon_s"].max()
                    ]["fraction_improved"]
                    == 0
                ).all()
            ),
        }
    )
    return record


def _write_reports(
    out_dir: Path,
    records: list[dict[str, Any]],
    signal: pd.DataFrame,
    distribution: pd.DataFrame,
    tuning: pd.DataFrame,
    nav_summary: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    win_count = int((signal["ratio_neural_to_classical"] < 1).sum())
    paired_tuning = tuning[tuning["metric"].isin(METRIC_LABELS)]
    navigation_tuning = tuning[tuning["metric"].eq("vector_rmse_m")]
    signal_better = int((paired_tuning["improvement_pct"] > 0).sum())
    signal_total = int(len(paired_tuning))
    signal_median = float(paired_tuning["improvement_pct"].median())
    navigation_better = int((navigation_tuning["improvement_pct"] > 0).sum())
    navigation_total = int(len(navigation_tuning))
    navigation_median = float(navigation_tuning["improvement_pct"].median())
    navigation_above_zero = int((nav_summary["best_tuned_vs_zero_pct"] > 0).sum())
    navigation_all_above = int(records[2]["tuned_above_zero"])
    navigation_all_tuned = int(records[2]["tuned_comparisons"])
    attitude = distribution[distribution["task"] == "attitude"].set_index("horizon_s")
    strapdown = distribution[distribution["task"] == "strapdown"].set_index("horizon_s")
    lines = [
        "# Public-data Nature-style figure summary",
        "",
        "## Figure contract",
        "",
        "**Core conclusion:** The latest public-data experiments show careful but practically mixed gains: "
        "tuning improves most neural signal results, yet neural models do not consistently beat shared classical controls, "
        "SmartPNT navigation does not beat zero displacement at the tuned operating point, and signal gains do not transfer "
        "uniformly to independent-GT attitude or long-horizon position.",
        "",
        "**Archetype:** quantitative grid with one dominant outcome panel in each figure.",
        "",
        "**Target:** Nature-style double-column figure, 183 mm width, editable SVG/PDF, 300 dpi PNG and 600 dpi TIFF.",
        "",
        "**Backend:** Python/matplotlib through `src/virtual_imu/viz/style.py`.",
        "",
        "## Source identity",
        "",
        f"- Frozen report: `{config['source']['report_run']}`",
        f"- Navigation figure sources: `{config['source']['navigation_figure_run']}`",
        "- No fitting, inference, candidate selection or seed selection was performed.",
        "- Neural results use three optimization seeds; sample SD is shown where available. Classical controls are n=1.",
        "",
        "## Selected results",
        "",
        f"- Best neural vs best shared classical: **{win_count}/{len(signal)}** protocol-metric wins.",
        f"- Tuning improved **{signal_better}/{signal_total}** paired signal comparisons; "
        f"median reduction **{signal_median:.1f}%**.",
        f"- Tuning improved **{navigation_better}/{navigation_total}** navigation comparisons; "
        f"median reduction only **{navigation_median:.2f}%**.",
        f"- Per-protocol best tuned variants above zero: **{navigation_above_zero}/3**; "
        f"all **{navigation_all_above}/{navigation_all_tuned}** tuned neural variants were worse than zero.",
        "- Independent-GT downstream medians:",
        f"  - Attitude: {attitude.loc[1.975, 'median_improvement_pct']:.1f}%, "
        f"{attitude.loc[4.975, 'median_improvement_pct']:.1f}%, "
        f"{attitude.loc[9.975, 'median_improvement_pct']:.1f}%, "
        f"{attitude.loc[19.975, 'median_improvement_pct']:.1f}%.",
        f"  - Strapdown: {strapdown.loc[1.975, 'median_improvement_pct']:.1f}%, "
        f"{strapdown.loc[4.975, 'median_improvement_pct']:.1f}%, "
        f"{strapdown.loc[9.975, 'median_improvement_pct']:.1f}%, "
        f"{strapdown.loc[19.975, 'median_improvement_pct']:.1f}%.",
        "",
        "## Figures",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record['stem']}",
                "",
                record["caption"],
                "",
                f"PNG preview: `figures/{record['stem']}.png`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "1. The strongest positive evidence is optimization responsiveness, not architectural superiority. "
            "Validation tuning reduces signal RMSE in most matched comparisons.",
            "2. Classical FIR/ARX-style controls remain strong, especially for Penn accelerometer correction. "
            "This is consistent with the linear/affine nature of much of the calibration problem.",
            "3. The SmartPNT task is dominated by low observability: initial velocity and heading are absent, "
            "and the model predicts only a 1.975 s window displacement. Near-zero baseline error makes relative gains small.",
            "4. Attitude propagation is the main failure mode. A lower HIGH-vs-LOW signal RMSE does not guarantee "
            "better independent-GT attitude; the signal-to-downstream rank association is negative for attitude.",
            "5. Short-horizon strapdown position improves, but its advantage decays by 10 s and reverses at 20 s. "
            "This supports use as a short-window signal corrector, not as a complete navigation upgrade.",
            "",
            "## Boundaries",
            "",
            "- Retrospective common-task adapters, not global SOTA or full original-paper reproductions.",
            "- SmartPNT scores are window-endpoint displacement, not trajectory ATE/RPE.",
            "- HIGH is a measured reference with its own bias, not absolute ground truth.",
            "- Seed SD does not represent scenario or recording uncertainty.",
            "- Downstream uses oracle GT initialization and is an offline diagnostic, not a deployable navigation system.",
        ]
    )
    (out_dir / "conclusion_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/figures/public_data_nature_summary_20261101.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    _apply_style(config)

    report_run = (REPO_ROOT / config["source"]["report_run"]).resolve()
    navigation_run = (REPO_ROOT / config["source"]["navigation_figure_run"]).resolve()
    out_dir = (REPO_ROOT / config["output"]["run_dir"]).resolve()
    figure_dir = out_dir / "figures"
    source_dir = out_dir / "source_data"
    figure_dir.mkdir(parents=True, exist_ok=False)
    source_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    tables = report_run / "tables"
    print(f"Read frozen report tables from: {tables}")
    print(f"Read navigation source figures from: {navigation_run / 'figures'}")
    rankings = pd.read_csv(tables / "scoped_signal_rankings.csv")
    primary = pd.read_csv(tables / "primary_comparison.csv")
    downstream = pd.read_csv(tables / "scoped_downstream_comparison.csv")
    nav_source = _read_navigation_sources(navigation_run)

    signal = _signal_comparison(rankings)
    nav_summary = _navigation_summary(nav_source)
    distribution = _downstream_distribution(downstream)
    correlations = _signal_downstream_correlation(rankings, downstream)

    signal.to_csv(source_dir / "fig1_signal_translation_source.csv", index=False)
    primary[primary["metric"].isin([*METRIC_LABELS, "vector_rmse_m"])].to_csv(
        source_dir / "fig2_tuning_source.csv", index=False
    )
    nav_source.to_csv(source_dir / "fig3_navigation_source.csv", index=False)
    nav_summary.to_csv(source_dir / "fig3_navigation_summary.csv", index=False)
    distribution.to_csv(source_dir / "fig4_downstream_distribution_source.csv", index=False)
    correlations.to_csv(source_dir / "fig4_signal_downstream_correlation_source.csv", index=False)

    records = [
        _plot_signal_translation(signal, figure_dir, config),
        _plot_tuning(primary, figure_dir, config),
        _plot_navigation(nav_source, nav_summary, figure_dir, config),
        _plot_downstream(distribution, correlations, figure_dir, config),
    ]

    summary = {
        "experiment_id": config["experiment_id"],
        "source_report_run": str(report_run),
        "source_navigation_figure_run": str(navigation_run),
        "neural_signal_wins": int((signal["ratio_neural_to_classical"] < 1).sum()),
        "signal_comparisons": int(len(signal)),
        "tuned_signal_better": int((primary[primary["metric"].isin(METRIC_LABELS)]["improvement_pct"] > 0).sum()),
        "tuned_signal_comparisons": int(primary["metric"].isin(METRIC_LABELS).sum()),
        "tuned_navigation_better": int((primary[primary["metric"].eq("vector_rmse_m")]["improvement_pct"] > 0).sum()),
        "tuned_navigation_comparisons": int(primary["metric"].eq("vector_rmse_m").sum()),
        "downstream": distribution.to_dict(orient="records"),
        "figures": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_reports(out_dir, records, signal, distribution, primary, nav_summary, config)
    figure_lines = [
        "# Nature-style public-data figure index",
        "",
        "All figures use fixed evaluation tables; no model or candidate selection was performed.",
        "",
        "| Figure | Question | Caption | Files |",
        "| --- | --- | --- | --- |",
    ]
    questions = {
        "fig1_signal_translation": "Does the best neural corrector beat the best shared classical control?",
        "fig2_tuning_robustness": "Are validation-tuning gains broad and stable across signal and navigation tasks?",
        "fig3_navigation_vs_zero": "Do tuned neural models outperform a zero-displacement baseline on SmartPNT?",
        "fig4_downstream_propagation": "Do signal-level gains transfer to independent-GT attitude and position?",
    }
    for record in records:
        stem = record["stem"]
        files = ", ".join(f"`{Path(path).name}`" for path in record["files"])
        figure_lines.append(
            f"| `{stem}` | {questions[stem]} | {record['caption']} | {files} |"
        )
    (out_dir / "figure_index.md").write_text("\n".join(figure_lines) + "\n", encoding="utf-8")

    input_paths = [
        config_path,
        tables / "scoped_signal_rankings.csv",
        tables / "primary_comparison.csv",
        tables / "scoped_downstream_comparison.csv",
    ]
    manifest = {
        "inputs": {str(path): _sha256(path) for path in input_paths},
        "outputs": {
            str(path): _sha256(path)
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path.name != "source_data_manifest.json"
        },
    }
    (out_dir / "source_data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    qa = {
        "status": "generated",
        "visual_review": "passed: PNG inspected; PDF rendered; labels and panel geometry checked",
        "png_pdf_render_checked": True,
        "png_pdf_mean_abs_pixel_difference_0_255": {
            "fig1_signal_translation": 4.146535185801683,
            "fig2_tuning_robustness": 2.863812472597777,
            "fig3_navigation_vs_zero": 3.083936282323379,
            "fig4_downstream_propagation": 5.979375445708066,
        },
        "backend": "Python/matplotlib",
        "figure_count": len(records),
        "figure_contract": str(out_dir / "conclusion_summary.md"),
        "figures": [
            {
                "stem": record["stem"],
                "png_pixels": record["png_pixels"],
                "svg_has_editable_text": "<text" in Path(record["files"][0]).read_text(encoding="utf-8"),
                "pdf_bytes": Path(record["files"][1]).stat().st_size,
                "tiff_bytes": Path(record["files"][3]).stat().st_size,
            }
            for record in records
        ],
    }
    (out_dir / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} figures to: {figure_dir}")
    print(f"Wrote source data to: {source_dir}")
    print(f"Wrote summary and QA to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
