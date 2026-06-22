---
name: imu-research-visualization
description: Generate consistent publication-quality IMU research figures for time series, PSD, Allan deviation, preintegration residuals, navigation drift, trajectory comparisons, metric bars, and paper captions. Use when Codex works on IMU experiment results, virtual IMU evaluation, SmartPNT-MSF figures, Allan variance plots, PSD plots, error curves, navigation trajectories, or manuscript-ready visualization in this repository.
---

# IMU Research Visualization

Use `src/virtual_imu/viz` as the source of truth for styling and plotting. Do not invent random visual styles for individual figures.

## Workflow

1. Read `references/figure_style.md` before adding or modifying IMU figures.
2. Generate figures through `python scripts/plot_results.py --eval <metrics.json> --pred <pred.parquet> --out outputs/figures` when possible.
3. Export every static figure as SVG, PDF, and PNG at 300 dpi.
4. Update `outputs/figures/figure_index.md` with filename, purpose, and a caption draft.
5. Keep axes labeled with units and legends outside dense data regions.

## Figure Requirements

- Use 3.5 inch width for single-column figures and 7.2 inch width for double-column figures.
- Use DejaVu Sans or another noncommercial default Matplotlib font.
- Use log-log axes for Allan deviation.
- Use appropriate frequency limits for PSD and label frequency in Hz.
- Show low, virtual, and high signals together when high-grade reference is available.
- Use equal aspect ratio for trajectories.
- Include a caption draft for every paper-facing figure.
