"""Unified figure style for IMU research plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "high": "#1f2933",
    "low": "#c95f3f",
    "virtual": "#1f77b4",
    "accent": "#2a9d8f",
    "grid": "#d0d7de",
}

SIZES = {
    "single": (3.5, 2.4),
    "double": (7.2, 3.6),
    "tall": (7.2, 5.0),
}


def apply_paper_style() -> None:
    """Apply deterministic Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": COLORS["grid"],
            "lines.linewidth": 1.2,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_figure(size: str = "double") -> tuple[plt.Figure, plt.Axes]:
    """Create a styled figure and axis."""
    apply_paper_style()
    return plt.subplots(figsize=SIZES[size])


def save_figure(fig: plt.Figure, out_dir: str | Path, stem: str) -> list[Path]:
    """Save figure as SVG, PDF, and PNG."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
