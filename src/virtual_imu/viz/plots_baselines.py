"""Publication-style figures for three-tier baseline comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from virtual_imu.viz.style import COLORS, apply_paper_style

LAYER_COLORS = {
    "L1 classical": "#7f8c8d",
    "L1 filtering": "#d08c60",
    "L1 system ID": "#b8860b",
    "L2 public neural signal": "#4c78a8",
    "L3 public navigation adapted": "#2a9d8f",
    "L3 navigation-only reference": "#8e6bbd",
}

METRIC_SPECS = [
    ("gyro_norm_rmse", "Gyro norm RMSE\n[rad/s]", False),
    ("accel_norm_rmse", "Accel norm RMSE\n[m/s$^2$]", False),
    ("gyro_coherence", "Gyro coherence", True),
    ("preintegration_residual", "Preintegration\nresidual", False),
    ("position_drift_m_final", "Final position\ndrift [m]", False),
]


def _save_pub_figure(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Save a figure in repo and Nature-oriented publication formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in result.get("baselines", {}).items():
        meta = metrics.get("metadata", {})
        signal = metrics.get("signal", {}).get("virtual", {})
        spectral = metrics.get("spectral", {}).get("virtual", {})
        preint = metrics.get("preintegration", {}).get("virtual", {})
        nav = metrics.get("navigation", {}).get("virtual", {})
        rows.append(
            {
                "baseline": name,
                "layer": meta.get("layer", "unknown"),
                "family": meta.get("family", "unknown"),
                "causal": bool(meta.get("causal", False)),
                "trained": bool(meta.get("trained", False)),
                "uses_future": bool(meta.get("uses_future", False)),
                "gyro_norm_rmse": _finite(signal.get("gyro_norm_rmse")),
                "accel_norm_rmse": _finite(signal.get("accel_norm_rmse")),
                "gyro_coherence": _finite(spectral.get("gyro_coherence")),
                "preintegration_residual": _finite(preint.get("imu_preintegration_residual")),
                "position_drift_m_final": _finite(nav.get("position_drift_m_final")),
            }
        )
    return rows


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _sort_rows(rows: list[dict[str, Any]], metric: str = "position_drift_m_final") -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (np.inf if np.isnan(row[metric]) else row[metric], row["baseline"]))


def _metric_values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([row[metric] for row in rows], dtype=float)


def _layer_color(row: dict[str, Any]) -> str:
    return LAYER_COLORS.get(row["layer"], COLORS["accent"])


def _short_label(label: str) -> str:
    return label.replace("_", " ")


def plot_baseline_suite(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Generate Nature-style baseline comparison figures and an index."""
    apply_paper_style()
    mpl.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
        }
    )
    root = Path(out_dir)
    rows = _rows(result)
    if not rows:
        raise ValueError("No baseline rows available for plotting")

    figure_records = [
        _plot_summary_grid(rows, root),
        _plot_tradeoff(rows, root),
        _plot_rank_heatmap(rows, root),
    ]
    index = root / "baseline_figure_index.md"
    lines = ["# Baseline Figure Index", "", "| Figure | Purpose | Caption Draft |", "|---|---|---|"]
    for stem, caption in figure_records:
        lines.append(f"| `{stem}.svg/pdf/png/tiff` | {stem.replace('_', ' ')} | {caption} |")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def _plot_summary_grid(rows: list[dict[str, Any]], out_dir: Path) -> tuple[str, str]:
    ordered = _sort_rows(rows)
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 5.2), sharey=True)
    y = np.arange(len(ordered))
    labels = [_short_label(row["baseline"]) for row in ordered]
    colors = [_layer_color(row) for row in ordered]

    for idx, (ax, (metric, ylabel, higher_is_better)) in enumerate(zip(axes, METRIC_SPECS, strict=False)):
        values = _metric_values(ordered, metric)
        ax.barh(y, values, color=colors, height=0.68, edgecolor="#ffffff", linewidth=0.4)
        finite = np.isfinite(values)
        if finite.any():
            best_idx = int(np.nanargmax(values) if higher_is_better else np.nanargmin(values))
            ax.scatter(values[best_idx], best_idx, s=24, color="#1f2933", marker="*", zorder=3)
        ax.set_xlabel(ylabel)
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels(labels, fontsize=6.2)
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        if metric == "position_drift_m_final":
            ax.set_xscale("log")
            ax.set_xticks([3.0, 10.0, 30.0, 70.0])
            ax.set_xticklabels(["3", "10", "30", "70"])
            ax.minorticks_off()
        ax.tick_params(axis="x", labelsize=6.5)
        ax.margins(y=0.01)
    axes[0].invert_yaxis()

    seen = []
    for row in ordered:
        if row["layer"] not in seen:
            seen.append(row["layer"])
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=7,
            markerfacecolor=LAYER_COLORS[layer],
            markeredgecolor="none",
        )
        for layer in seen
    ]
    fig.legend(
        handles,
        seen,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.035),
        ncol=3,
        title="Baseline layer",
        title_fontsize=7,
        fontsize=6.5,
    )
    fig.text(
        0.2,
        0.17,
        "Star marks the best method per metric; final position drift uses a log-scaled x axis.",
        fontsize=7,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.2, right=0.995, top=0.98, bottom=0.31, wspace=0.42)
    stem = "baseline_summary_grid"
    _save_pub_figure(fig, out_dir, stem)
    caption = (
        "Three-tier baseline comparison across signal error, spectral coherence, preintegration residual, "
        "and proxy navigation drift; lower is better except coherence."
    )
    return stem, caption


def _plot_tradeoff(rows: list[dict[str, Any]], out_dir: Path) -> tuple[str, str]:
    ordered = _sort_rows(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for layer in dict.fromkeys(row["layer"] for row in ordered):
        layer_rows = [row for row in ordered if row["layer"] == layer]
        x = _metric_values(layer_rows, "gyro_coherence")
        y = _metric_values(layer_rows, "position_drift_m_final")
        all_rmse = _metric_values(ordered, "gyro_norm_rmse")
        fallback_rmse = float(np.nanmedian(all_rmse)) if np.isfinite(all_rmse).any() else 1.0
        rmse = np.nan_to_num(_metric_values(layer_rows, "gyro_norm_rmse"), nan=fallback_rmse)
        sizes = 36 + 70 * np.clip(1.0 / np.maximum(rmse, 1e-6), 0, 20) / 20
        ax.scatter(
            x,
            y,
            s=sizes,
            color=LAYER_COLORS.get(layer, COLORS["accent"]),
            alpha=0.92,
            edgecolor="#ffffff",
            linewidth=0.5,
            label=layer,
        )
    best_by_nav = {row["baseline"] for row in ordered[:3]}
    best_by_coherence = max(ordered, key=lambda row: -np.inf if np.isnan(row["gyro_coherence"]) else row["gyro_coherence"])
    neural_rows = [row for row in ordered if "neural" in row["layer"] or "navigation adapted" in row["layer"]]
    key_labels = {*best_by_nav, best_by_coherence["baseline"]}
    if neural_rows:
        best_neural = min(
            neural_rows,
            key=lambda row: np.inf if np.isnan(row["position_drift_m_final"]) else row["position_drift_m_final"],
        )
        key_labels.add(best_neural["baseline"])
    labels = [row for row in ordered if row["baseline"] in key_labels]
    offsets = {
        "learned_fir_causal": (-86, 5),
        "arx_linear": (6, 5),
        "haar_wavelet": (5, 8),
        "identity": (5, -2),
        "brossard_gyro_cnn": (-70, -8),
    }
    for row in labels:
        ax.annotate(
            row["baseline"],
            (row["gyro_coherence"], row["position_drift_m_final"]),
            xytext=offsets.get(row["baseline"], (5, 5)),
            textcoords="offset points",
            fontsize=6.5,
            color="#1f2933",
        )
    ax.set_xlabel("Gyro spectral coherence")
    ax.set_ylabel("Final position drift [m]")
    ax.set_yscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.2, title="Layer", title_fontsize=7)
    ax.text(
        0.02,
        0.04,
        "Better",
        transform=ax.transAxes,
        color="#2a9d8f",
        fontsize=7,
        fontweight="bold",
    )
    ax.annotate(
        "",
        xy=(0.28, 0.05),
        xytext=(0.07, 0.05),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#2a9d8f", "lw": 1.0},
    )
    ax.annotate(
        "",
        xy=(0.07, 0.05),
        xytext=(0.07, 0.26),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#2a9d8f", "lw": 1.0},
    )
    finite_x = _metric_values(ordered, "gyro_coherence")
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size:
        ax.set_xlim(max(0.0, finite_x.min() - 0.02), min(1.0, finite_x.max() + 0.02))
    finite_y = _metric_values(ordered, "position_drift_m_final")
    finite_y = finite_y[np.isfinite(finite_y) & (finite_y > 0)]
    if finite_y.size:
        ax.set_ylim(finite_y.min() * 0.82, finite_y.max() * 1.22)
    fig.subplots_adjust(left=0.08, right=0.73, bottom=0.2, top=0.96)
    stem = "baseline_gyro_nav_tradeoff"
    _save_pub_figure(fig, out_dir, stem)
    caption = (
        "Gyro coherence versus proxy navigation drift shows whether a baseline improves frequency consistency "
        "without sacrificing downstream navigation behavior."
    )
    return stem, caption


def _plot_rank_heatmap(rows: list[dict[str, Any]], out_dir: Path) -> tuple[str, str]:
    ordered = _sort_rows(rows)
    metric_labels = [spec[1].replace("\n", " ") for spec in METRIC_SPECS]
    matrix = []
    for metric, _ylabel, higher_is_better in METRIC_SPECS:
        values = _metric_values(ordered, metric)
        ranks = _rank(values, higher_is_better=higher_is_better)
        matrix.append(ranks)
    rank_matrix = np.asarray(matrix).T

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.29 * len(ordered) + 1.0)))
    image = ax.imshow(rank_matrix, cmap="viridis_r", aspect="auto", vmin=1, vmax=max(2, len(ordered)))
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([row["baseline"] for row in ordered], fontsize=6.5)
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=25, ha="right")
    ax.set_xlabel("Evaluation metric")
    ax.set_ylabel("Baseline")
    for i, row in enumerate(ordered):
        ax.text(
            len(metric_labels) + 0.08,
            i,
            row["layer"].replace(" public ", " "),
            va="center",
            fontsize=5.8,
            color=LAYER_COLORS.get(row["layer"], "#1f2933"),
        )
    for i in range(rank_matrix.shape[0]):
        for j in range(rank_matrix.shape[1]):
            ax.text(j, i, f"{rank_matrix[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="#ffffff")
    cbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("Rank (1 = best)")
    ax.set_xlim(-0.5, len(metric_labels) + 2.9)
    fig.tight_layout()
    stem = "baseline_rank_heatmap"
    _save_pub_figure(fig, out_dir, stem)
    caption = (
        "Metric-rank heatmap summarizes which baseline families dominate each criterion while preserving "
        "causal/offline and learned/non-learned distinctions in the row labels."
    )
    return stem, caption


def _rank(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, -np.inf if higher_is_better else np.inf)
    order = np.argsort(-clean if higher_is_better else clean)
    ranks = np.empty_like(clean, dtype=float)
    ranks[order] = np.arange(1, len(clean) + 1)
    return ranks
