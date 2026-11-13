"""Frozen module replacement figures using only supplied run-local tables.

No model construction, checkpoint, prediction, dataset or prior-run reads.
Every static figure uses repository paper style and PNG300dpi/SVG/PDF exports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from virtual_imu.viz.style import COLORS, SIZES, apply_paper_style, save_figure

LOG = logging.getLogger(__name__)
UNITS = {"gyro_rmse": "Gyro RMSE [rad/s]", "accel_rmse": "Accel RMSE [m/s$^2$]"}
MARKERS = ("o", "s", "^")


def _save(fig, directory, stem, caption):
    paths = save_figure(fig, directory, stem)
    for path in paths:
        LOG.info("WRITES %s", path)
    return dict(stem=stem, caption=caption, paths=[str(path) for path in paths])


def _families(config):
    reference = config["reference_variant"]
    variants = config["variants"]
    front = [v for v in variants if v != reference and v.endswith("_lstm64")]
    back = [v for v in variants if v != reference and v.startswith("cnn_")]
    if set(front) & set(back) or set(front + back + [reference]) != set(variants):
        raise ValueError("Unclassified front/back replacement variant in config")
    return front, back


def _color(variant, config):
    if variant == config["reference_variant"]:
        return COLORS["high"]
    return COLORS["accent"] if variant.endswith("_lstm64") else COLORS["virtual"]


def _cells(frame, config):
    values = frame[frame.aggregation.eq("dataset")][["protocol", "dataset_id"]].drop_duplicates()
    return [
        (protocol, dataset)
        for protocol in config["protocols"]
        for dataset in sorted(values.loc[values.protocol.eq(protocol), "dataset_id"])
    ]


def _seed_points(ax, values, row, config, value_column, offset=0.0):
    values = values.sort_values("seed")
    # Fixed offsets, not random jitter; marker shape preserves seed identity.
    for index, seed in enumerate(config["seeds"]):
        observed = values.loc[values.seed.eq(seed), value_column].dropna()
        if observed.empty:
            continue
        displacement = (index - (len(config["seeds"]) - 1) / 2) * 0.07
        ax.scatter(
            observed,
            np.full(len(observed), row + offset + displacement),
            s=9,
            marker=MARKERS[index % len(MARKERS)],
            color=COLORS["high"],
            zorder=4,
        )


def _seed_legend(fig, config):
    handles = [
        Line2D(
            [],
            [],
            color=COLORS["high"],
            marker=MARKERS[i % len(MARKERS)],
            linestyle="none",
            markersize=3,
            label=f"seed {seed}",
        )
        for i, seed in enumerate(config["seeds"])
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False)


def _yaxis(ax, variants, column):
    ax.set_yticks(range(len(variants)), variants)
    ax.set_ylim(len(variants) - 0.5, -0.5)
    ax.tick_params(axis="y", labelleft=column == 0)
    ax.grid(axis="y", visible=False)


def _header(fig, config, text):
    if "synthetic" in config.get("experiment_id", "").lower():
        text = "SYNTHETIC / " + text
    fig.suptitle(text, fontsize=9)


def plot_physical(summary, metrics, directory, variants, stem, config):
    """TEST domain RMSE in physical units, mean/SD and exact seed markers."""
    cells = _cells(summary, config)
    fig, axes = plt.subplots(
        2, len(cells), figsize=SIZES["tall"], sharey=True, squeeze=False, layout="constrained"
    )
    for column, (protocol, dataset) in enumerate(cells):
        for row, (metric, label) in enumerate(UNITS.items()):
            ax = axes[row, column]
            frame = summary[
                summary.protocol.eq(protocol)
                & summary.dataset_id.eq(dataset)
                & summary.split.eq("test")
                & summary.aggregation.eq("dataset")
                & summary.metric.eq(metric)
            ]
            for index, variant in enumerate(variants):
                group = frame[frame.variant.eq(variant)]
                if len(group) != 1:
                    raise ValueError("Missing/ambiguous physical figure cell")
                value = group.iloc[0]
                ax.errorbar(
                    value["mean"],
                    index,
                    xerr=0.0 if np.isnan(value.sd) else value.sd,
                    color=_color(variant, config),
                    marker="|",
                    markersize=8,
                    capsize=2,
                    linewidth=1.3,
                    zorder=3,
                )
                seeds = metrics[
                    metrics.protocol.eq(protocol)
                    & metrics.dataset_id.eq(dataset)
                    & metrics.split.eq("test")
                    & metrics.aggregation.eq("dataset")
                    & metrics.variant.eq(variant)
                    & metrics.window.eq(320)
                ]
                _seed_points(ax, seeds, index, config, metric)
            raw = frame[frame.variant.eq("raw")]
            if not raw.empty:
                ax.axvline(raw["mean"].iloc[0], color=COLORS["low"], linestyle="--", linewidth=1)
            ax.set_xlim(left=0)
            ax.set_xlabel(label)
            _yaxis(ax, variants, column)
            if row == 0:
                ax.set_title(protocol + "\n" + dataset)
    _header(
        fig,
        config,
        stem.split("_")[0].title()
        + " replacements / TEST physical errors / "
        + config["reference_variant"],
    )
    _seed_legend(fig, config)
    return _save(
        fig,
        directory,
        stem,
        "LOW-to-HIGH physical TEST component RMSE for "
        + stem.split("_")[0]
        + " replacements versus "
        + config["reference_variant"]
        + ". Panels separate training scope, evaluation dataset and gyro/accel units. "
        "Identical fixed320 inputs at40Hz and central40 evaluation targets; pooled sample RMSE is preserved. "
        "Colored ticks/error bars show mean +/- sample SD (ddof=1), seed shapes show individual fits; "
        "the dashed orange line is deterministic Raw. Three optimization seeds are not recording confidence intervals. "
        "Reused controls retain prior-run training provenance; TEST does not select structures.",
    )


def plot_paired_percent(deltas, directory, variants, stem, config):
    """Signed within-seed RMSE percentages, sample SD and per-seed markers."""
    cells = _cells(deltas, config)
    fig, axes = plt.subplots(
        2, len(cells), figsize=SIZES["tall"], sharey=True, squeeze=False, layout="constrained"
    )
    for column, (protocol, dataset) in enumerate(cells):
        for row, (metric, _) in enumerate(UNITS.items()):
            ax = axes[row, column]
            frame = deltas[
                deltas.protocol.eq(protocol)
                & deltas.dataset_id.eq(dataset)
                & deltas.split.eq("test")
                & deltas.aggregation.eq("dataset")
                & deltas.metric.eq(metric)
                & deltas.window.eq(320)
            ]
            for index, variant in enumerate(variants):
                group = frame[frame.variant.eq(variant)]
                values = group.relative_delta_pct.dropna()
                if values.empty:
                    LOG.info(
                        "SKIPS undefined paired percentage %s/%s/%s/%s",
                        protocol,
                        dataset,
                        variant,
                        metric,
                    )
                    ax.text(
                        0.02, index, "undefined", transform=ax.get_yaxis_transform(), fontsize=7
                    )
                    continue
                sd = values.std(ddof=1) if len(values) > 1 else 0.0
                ax.errorbar(
                    values.mean(),
                    index,
                    xerr=sd,
                    color=_color(variant, config),
                    marker="|",
                    markersize=8,
                    capsize=2,
                    linewidth=1.3,
                    zorder=3,
                )
                _seed_points(ax, group, index, config, "relative_delta_pct")
            ax.axvline(0, color=COLORS["high"], linewidth=0.8)
            ax.set_xlabel(("Gyro" if row == 0 else "Accel") + " RMSE change [%]")
            _yaxis(ax, variants, column)
            if row == 0:
                ax.set_title(protocol + "\n" + dataset)
    _header(
        fig,
        config,
        stem.split("_")[0].title()
        + " replacements / paired TEST change / "
        + config["reference_variant"],
    )
    _seed_legend(fig, config)
    return _save(
        fig,
        directory,
        stem,
        "Physical TEST RMSE percentage change versus the same-seed "
        + config["reference_variant"]
        + " on exact protocol/split/dataset/pair/recording/window320 supports and common central40 targets. "
        "Compute 100*(treatment-reference)/reference per seed before mean +/- sample SD (ddof=1); "
        "seed shapes retain individual directions. Positive is worse; negative is better. "
        "Zero-reference percentages are undefined and skipped, never replaced by zero. "
        "These are optimization variations, not recording confidence intervals or TEST selection.",
    )


def _normalized(summary, metrics, directory, config):
    variants = list(config["variants"])
    fig, axes = plt.subplots(
        1,
        len(config["protocols"]),
        figsize=SIZES["tall"],
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    styles = (("validation", -0.16, COLORS["accent"]), ("test", 0.16, COLORS["virtual"]))
    for column, protocol in enumerate(config["protocols"]):
        ax = axes[0, column]
        for split, offset, color in styles:
            for index, variant in enumerate(variants):
                cell = summary[
                    summary.protocol.eq(protocol)
                    & summary.split.eq(split)
                    & summary.variant.eq(variant)
                    & summary.aggregation.eq("overall")
                    & summary.metric.eq("normalized_score")
                ]
                if len(cell) != 1:
                    raise ValueError("Missing/ambiguous normalized figure cell")
                value = cell.iloc[0]
                ax.errorbar(
                    value["mean"],
                    index + offset,
                    xerr=0.0 if np.isnan(value.sd) else value.sd,
                    color=color,
                    marker="|",
                    markersize=7,
                    capsize=2,
                    linewidth=1.2,
                )
                seeds = metrics[
                    metrics.protocol.eq(protocol)
                    & metrics.split.eq(split)
                    & metrics.variant.eq(variant)
                    & metrics.aggregation.eq("overall")
                ]
                _seed_points(ax, seeds, index, config, "normalized_score", offset)
        ax.axvline(1, color=COLORS["low"], linestyle="--", linewidth=0.8)
        ax.set_xlabel("Balanced normalized score [1]")
        ax.set_title(protocol)
        ax.set_xlim(left=0)
        _yaxis(ax, variants, column)
    _header(fig, config, "VALID and TEST / fixed320 / reference: " + config["reference_variant"])
    handles = [
        Line2D([], [], color=color, marker="|", label=split.upper()) for split, _, color in styles
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    return _save(
        fig,
        directory,
        "normalized_validation_test",
        "Balanced normalized VALID and TEST scores across all fixed320 replacements; each panel is one training scope. "
        "Each pair first averages gyro/Raw and accel/Raw RMSE ratios, followed by equal pairs and equal datasets. "
        "The pooled panel combines ALiMU and Penn with equal dataset weight, not separate TEST-domain selections. "
        "Ticks/error bars are seed mean +/- sample SD (ddof=1); small shapes are the same three seeds as the physical figures. "
        "Orange dashed Raw=1; lower is better. Only VALID ranks candidates; TEST is descriptive, and prior TEST was already seen.",
    )


def _compute(compute, directory, protocol, config):
    variants = list(config["variants"])
    fig, axes = plt.subplots(1, 2, figsize=SIZES["tall"], sharey=True, layout="constrained")
    for index, variant in enumerate(variants):
        frame = compute[
            compute.protocol.eq(protocol) & compute.variant.eq(variant) & compute.window.eq(320)
        ]
        parameters = frame[frame.metric.eq("parameters")].iloc[0]
        axes[0].errorbar(
            parameters["mean"],
            index,
            xerr=parameters.sd,
            marker="|",
            markersize=8,
            capsize=2,
            color=_color(variant, config),
        )
        for offset, metric, color in (
            (-0.13, "batch1_ms_mean", COLORS["accent"]),
            (0.13, "batch1_ms_median", COLORS["virtual"]),
        ):
            value = frame[frame.metric.eq(metric)].iloc[0]
            if value.cost_scope != "uniform_new_inference":
                raise ValueError("Cost figure requires uniformly new inference measurements")
            axes[1].errorbar(
                value["mean"],
                index + offset,
                xerr=value.sd,
                marker="|",
                markersize=8,
                capsize=2,
                color=color,
            )
    for column, ax in enumerate(axes):
        _yaxis(ax, variants, column)
        ax.set_xlim(left=0)
    axes[0].set_xlabel("Parameters [count]")
    axes[1].set_xlabel("Uniform batch1 latency [ms]")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    _header(fig, config, protocol + " / parameters and newly measured inference")
    handles = [
        Line2D([], [], color=color, marker="|", label=label)
        for color, label in (
            (COLORS["accent"], "per-fit mean"),
            (COLORS["virtual"], "per-fit median"),
        )
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    return _save(
        fig,
        directory,
        "compute_" + protocol,
        "Parameters and uniformly newly measured batch1 inference latency for "
        + protocol
        + ", including reference "
        + config["reference_variant"]
        + ". Both reused and new controls use current_run_uniform_measurement; "
        "latency mean/median is computed within each fit's repeated synchronized forwards, then summarized as seed mean +/- sample SD. "
        f"Warmup={config['evaluation']['latency_warmup']}, repeats={config['evaluation']['latency_repeats']}, input320/output central40. "
        "Timing is device/load dependent. Prior-run training time is retained separately as source_training in compute_summary.csv, "
        "not counted as new training. Parameter count alone does not equal effective context or a per-model HPO budget.",
    )


def render_module_replacement_figures(summary, deltas, compute, out_dir, config, metrics):
    """Export front/back physical/paired, VALID+TEST and per-scope cost figures."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    LOG.info("READS in-memory summaries and seed metrics; SKIPS dataset/model/checkpoint IO")
    front, back = _families(config)
    reference = config["reference_variant"]
    entries = []
    for family, variants in (("back", back), ("front", front)):
        if not variants:
            LOG.info("SKIPS empty %s family", family)
            continue
        entries.append(
            plot_physical(
                summary, metrics, directory, [reference, *variants], family + "_physical", config
            )
        )
        entries.append(
            plot_paired_percent(deltas, directory, variants, family + "_paired_percent", config)
        )
    entries.append(_normalized(summary, metrics, directory, config))
    for protocol in config["protocols"]:
        entries.append(_compute(compute, directory, protocol, config))
    text = [
        "# Module replacement figure index",
        "",
        "Source: this run's summary.csv, paired_deltas.csv, compute_summary.csv and seed metrics.",
        "",
        "Fixed320 input at40Hz, identical central40 support. Error bars: sample SD across optimization seeds, not recording CI. "
        "Positive paired change means worse. All PNGs are300dpi; SVG/PDF use repository paper style.",
        "",
    ]
    for entry in entries:
        stem = entry["stem"]
        links = " / ".join(f"[{ext.upper()}]({stem}.{ext})" for ext in ("png", "svg", "pdf"))
        text.extend([f"## {stem}", "", links, "", entry["caption"], ""])
    path = directory / "figure_index.md"
    LOG.info("WRITES %s", path)
    path.write_text("\n".join(text), encoding="utf-8")
    return entries
