"""Build traceable historical timing/downstream tables and publication figures.

Only completed historical receipts are read. No checkpoints, data sets,
training, model inference, method selection or new HPO results are accessed.
"""

# ruff: noqa: RUF001 -- Chinese captions and mathematical glyphs are intentional.

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from fontTools import subset
from fontTools.ttLib import TTFont
from matplotlib import colors, font_manager, ticker
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_imu.viz.style import COLORS, apply_paper_style

NEURAL = [
    "tcn_original",
    "plain_cnn",
    "gru",
    "transformer",
    "cnn_bilstm",
    "deepgyro_adapter",
    "tlio_resnet18_adapter",
]
TIMING_ORDER = [*NEURAL, "tartanimu_window_adapter"]
DOWNSTREAM_ORDER = [
    *NEURAL,
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "fir",
    "arx",
    "HIGH",
]
LABELS = {
    "tcn_original": "TCN",
    "plain_cnn": "CNN",
    "gru": "GRU",
    "transformer": "Transformer",
    "cnn_bilstm": "CNN–BiLSTM",
    "deepgyro_adapter": "DeepGyro backbone",
    "tlio_resnet18_adapter": "TLIO ResNet18",
    "tartanimu_window_adapter": "TartanIMU window",
    "identity": "LOW (identity)",
    "bias_mean": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "fir": "Causal FIR",
    "arx": "ARX",
    "HIGH": "HIGH reference",
}
HORIZONS = [80, 200, 400, 800]
SPANS = ["1.975", "4.975", "9.975", "19.975"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, Path):
            return str(item)
        return item

    path.write_text(
        json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_inputs(config, project):
    paths, verification = {}, []
    for name, spec in config["inputs"].items():
        path = project / spec["path"]
        observed = sha(path)
        if observed != spec["sha256"]:
            raise ValueError(f"Historical input checksum differs: {path}")
        print(f"READS: verified {name}: {path}", flush=True)
        paths[name] = path
        verification.append(
            {"name": name, "path": str(path), "sha256": observed, "bytes": path.stat().st_size}
        )
    return paths, verification


def timing_tables(paths, seed):
    reports = {
        device: json.loads(paths[f"{device}_timing"].read_text(encoding="utf-8"))
        for device in ("cpu", "gpu")
    }
    rows, lookup = [], {}
    expected = {
        (task, model)
        for task in ("paired", "navigation")
        for model in TIMING_ORDER
        if task == "navigation" or model != "tartanimu_window_adapter"
    }
    for device, report in reports.items():
        if report["status"] != "complete" or len(report["records"]) != (
            15 if device == "cpu" else 30
        ):
            raise ValueError("Historical timing receipt is incomplete")
        environment = report["environment"]
        if not environment["deterministic"] or environment["cpu_threads"] != (
            1 if device == "cpu" else 4
        ):
            raise ValueError("Historical timing runtime policy differs")
        for record in report["records"]:
            identity = (record["task"], record["candidate_id"])
            batch = record["batch_size"]
            if (
                identity not in expected
                or record["seed"] != seed
                or record["window_size"] != 80
                or record["precision"] != "float32"
            ):
                raise ValueError("Timing checkpoint/task identity differs from historical contract")
            if (
                record["checkpoint_expected_sha256"] != record["checkpoint_sha256"]
                or record["warmup"] != 30
                or record["iterations"] != 100
                or record["repeats"] != 3
            ):
                raise ValueError("Timing checksum or sampling budget differs")
            if record["input_channels"] != (9 if identity[0] == "paired" else 6) or record[
                "output_channels"
            ] != (6 if identity[0] == "paired" else 3):
                raise ValueError("Historical nine/six-channel tasks were mixed")
            key = (*identity, device, batch)
            if key in lookup:
                raise ValueError("Duplicate timing identity")
            lookup[key] = record
            row = {
                "task": identity[0],
                "model": identity[1],
                "label": LABELS[identity[1]],
                "device": device,
                "batch_size": batch,
                "seed": seed,
                "parameters": record["parameters"],
                "trainable_parameters": record["trainable_parameters"],
                "parameter_bytes": record["parameter_bytes"],
                "buffers": record["buffers"],
                "buffer_bytes": record["buffer_bytes"],
                "input_channels": record["input_channels"],
                "output_channels": record["output_channels"],
                "window_size": 80,
                "throughput_windows_per_second": record["loop_throughput_windows_per_second"],
                "cuda_event_throughput_windows_per_second": record[
                    "loop_cuda_event_windows_per_second"
                ],
                "gpu_peak_allocated_bytes": record["gpu_peak_allocated_bytes"],
                "gpu_forward_peak_increment_bytes": record["gpu_forward_peak_increment_bytes"],
                "gpu_model_resident_increment_bytes": record["gpu_model_resident_increment_bytes"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "checkpoint": record["checkpoint"],
                "cpu_threads": environment["cpu_threads"],
                "warmup": 30,
                "iterations": 100,
                "repeats": 3,
            }
            for statistic, raw in (("wall", "raw_wall_ms"), ("cuda_event", "raw_cuda_event_ms")):
                if record[statistic] is None:
                    continue
                values = np.asarray(record[raw], dtype=float)
                if values.shape != (3, 100) or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(
                        "Timing raw arrays must contain three repeats of100 finite positive calls"
                    )
                computed = {
                    "mean_ms": values.mean(),
                    "sample_sd_ms": values.std(ddof=1),
                    "p50_ms": np.percentile(values, 50),
                    "p95_ms": np.percentile(values, 95),
                    "min_ms": values.min(),
                    "max_ms": values.max(),
                }
                for name, value in computed.items():
                    if not np.isclose(value, record[statistic][name], rtol=1e-10, atol=1e-10):
                        raise ValueError("Independent timing replay differs")
                    row[f"{statistic}_{name}"] = value
            rows.append(row)
    wide = []
    for task, model in sorted(
        expected, key=lambda value: (value[0] != "paired", TIMING_ORDER.index(value[1]))
    ):
        records = [
            lookup[(task, model, device, batch)]
            for device, batch in (("cpu", 1), ("gpu", 1), ("gpu", 128))
        ]
        if (
            len({record["checkpoint_sha256"] for record in records}) != 1
            or len({record["parameters"] for record in records}) != 1
        ):
            raise ValueError("CPU/GPU/batch timing did not reuse the exact same checkpoint")
        cpu, gpu, bulk = records
        result = {
            "task": task,
            "model": model,
            "label": LABELS[model],
            "seed": seed,
            "parameters": cpu["parameters"],
            "trainable_parameters": cpu["trainable_parameters"],
            "parameter_bytes": cpu["parameter_bytes"],
            "buffers": cpu["buffers"],
            "buffer_bytes": cpu["buffer_bytes"],
            "input_channels": cpu["input_channels"],
            "output_channels": cpu["output_channels"],
            "window_size": 80,
            "checkpoint_sha256": cpu["checkpoint_sha256"],
        }
        for prefix, record in (("cpu_b1", cpu), ("gpu_b1", gpu), ("gpu_b128", bulk)):
            for kind in ("wall", "cuda_event"):
                if record[kind] is not None:
                    result.update(
                        {f"{prefix}_{kind}_{key}": value for key, value in record[kind].items()}
                    )
            result[prefix + "_throughput_windows_per_second"] = record[
                "loop_throughput_windows_per_second"
            ]
            result[prefix + "_gpu_peak_allocated_bytes"] = record["gpu_peak_allocated_bytes"]
            result[prefix + "_gpu_forward_peak_increment_bytes"] = record[
                "gpu_forward_peak_increment_bytes"
            ]
        result["gpu_over_cpu_wall_p50_ratio"] = gpu["wall"]["p50_ms"] / cpu["wall"]["p50_ms"]
        wide.append(result)
    return pd.DataFrame(rows), pd.DataFrame(wide), reports


def downstream_table(paths):
    summary = pd.read_csv(paths["downstream_summary"])
    trials = pd.read_csv(paths["downstream_trials"])
    windows = pd.read_csv(paths["downstream_windows"])
    expected = {
        (task, method, size)
        for task in ("attitude", "strapdown")
        for method in DOWNSTREAM_ORDER
        for size in HORIZONS
    }
    keys = ["task", "method", "samples"]
    if (
        len(summary) != 112
        or summary.duplicated(keys).any()
        or set(map(tuple, summary[keys].to_numpy())) != expected
    ):
        raise ValueError("Downstream source must retain all14 methods×4 horizons×2 tasks")
    cleaned = []
    for _, row in summary.iterrows():
        task, method, size = row[keys]
        metric = "endpoint_rmse_deg" if task == "attitude" else "position_endpoint_rmse_m"
        subset_rows = trials[
            (trials.task == task) & (trials.method == method) & (trials.samples == size)
        ]
        count = 3 if method in NEURAL else 1
        if (
            len(subset_rows) != count
            or row.optimizer_runs != count
            or not subset_rows[metric].notna().all()
        ):
            raise ValueError("Neural seed counts or deduplicated fixed evaluations differ")
        values = subset_rows[metric].to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1) if count > 1 else np.nan
        if not np.isclose(mean, row[metric + "_mean"], rtol=1e-10) or (
            count > 1 and not np.isclose(sd, row[metric + "_sd"], rtol=1e-10, atol=1e-12)
        ):
            raise ValueError("Independent seed aggregation differs from source")
        baseline = summary[
            (summary.task == task) & (summary.method == "identity") & (summary.samples == size)
        ].iloc[0][metric + "_mean"]
        gain = 100 * (1 - mean / baseline)
        if not np.isclose(gain, row[metric + "_improvement_over_low_pct"], atol=1e-8):
            raise ValueError("LOW-relative change differs")
        details = windows[
            (windows.task == task) & (windows.method == method) & (windows.samples == size)
        ]
        if len(details) != int(row.windows_per_run) * count:
            raise ValueError("Window-level source coverage differs")
        cleaned.append(
            {
                "task": task,
                "method": method,
                "label": LABELS[method],
                "samples": size,
                "horizon_s": (size - 1) / 40,
                "optimizer_runs": count,
                "windows_per_run": int(row.windows_per_run),
                "metric": metric,
                "unit": "deg" if task == "attitude" else "m",
                "mean": mean,
                "optimizer_seed_sample_sd": sd,
                "improvement_over_low_percent": gain,
                "independent_test_recordings": 1,
            }
        )
    return pd.DataFrame(cleaned), trials, windows


def _embedded_svg_font(path):
    text = path.read_text(encoding="utf-8")
    tree = ET.fromstring(text)
    characters = "".join(node.text or "" for node in tree.iter() if node.tag.endswith("text"))
    css = []
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    for weight, name in ((400, "normal"), (700, "bold")):
        source = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight=name)
        )
        options = subset.Options()
        options.name_IDs = ["*"]
        options.recalc_timestamp = False
        font = TTFont(source, recalcTimestamp=False)
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)
        subsetter.subset(font)
        font.flavor = "woff"
        stream = io.BytesIO()
        font.save(stream)
        encoded = base64.b64encode(stream.getvalue()).decode()
        css.append(
            "@font-face{font-family:'DejaVu Sans';font-weight:"
            + str(weight)
            + ";src:url(data:font/woff;base64,"
            + encoded
            + ") format('woff');}"
        )
    text = text.replace("</defs>", '<style type="text/css">' + "".join(css) + "</style></defs>", 1)
    path.write_text(text, encoding="utf-8")


def save_publication(fig, out, stem, caption, source, qa):
    fig.canvas.draw()
    bounds = fig.bbox
    clipped = []
    for item in fig.findobj(matplotlib.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(fig.canvas.get_renderer())
        if (
            box.width
            and box.height
            and (box.x0 < -1 or box.y0 < -1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1)
        ):
            clipped.append(item.get_text())
    if clipped:
        raise ValueError(f"Text outside figure canvas {stem}: {clipped}")
    files = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)):
        path = out / "figures" / f"{stem}.{extension}"
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        fig.savefig(path, dpi=dpi, facecolor="white", **options)
        if extension == "svg":
            _embedded_svg_font(path)
        files.append(path)
    svg = files[0].read_text(encoding="utf-8")
    pdf = files[1].read_bytes()
    text_count = sum(1 for node in ET.fromstring(svg).iter() if node.tag.endswith("text"))
    if (
        text_count < 10
        or "data:font/woff;base64" not in svg
        or b"/FontFile2" not in pdf
        or b"/ToUnicode" not in pdf
    ):
        raise ValueError("Editable/embedded font verification failed")
    raster = {}
    for path, expected_dpi in ((files[2], 300), (files[3], 600)):
        with Image.open(path) as image:
            measured = image.info.get("dpi")
            if measured is None or not all(abs(value - expected_dpi) < 0.1 for value in measured):
                raise ValueError("Raster DPI metadata differs")
            raster[path.suffix] = {
                "pixels": list(image.size),
                "dpi": [float(value) for value in measured],
            }
    qa[stem] = {
        "text_outside_canvas": clipped,
        "svg_text_nodes": text_count,
        "svg_embedded_WOFF_subsets": True,
        "pdf_FontFile2": True,
        "pdf_ToUnicode": True,
        "raster": raster,
        "width_inches": fig.get_figwidth(),
        "height_inches": fig.get_figheight(),
        "visual_review": "pending",
    }
    plt.close(fig)
    return {
        "id": stem,
        "caption_zh": caption,
        "source_data": source,
        "files": [str(path.relative_to(out)) for path in files],
    }


def figure_timing(table):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.25), gridspec_kw={"height_ratios": [7, 8]})
    fig.subplots_adjust(left=0.205, right=0.98, top=0.915, bottom=0.15, wspace=0.34, hspace=0.56)
    for row, task in enumerate(("paired", "navigation")):
        data = (
            table[table.task == task]
            .set_index("model")
            .loc[
                [
                    name
                    for name in TIMING_ORDER
                    if task == "navigation" or name != "tartanimu_window_adapter"
                ]
            ]
        )
        y = np.arange(len(data))
        for column, ax in enumerate(axes[row]):
            ax.set_ylim(len(data) - 0.5, -0.5)
            ax.set_yticks(y)
            ax.set_yticklabels([LABELS[name] for name in data.index] if column == 0 else [])
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
            ax.tick_params(axis="y", length=0, pad=5)
            ax.set_xscale("log")
            ax.text(
                -0.03,
                1.055,
                chr(97 + row * 3 + column),
                transform=ax.transAxes,
                ha="right",
                fontweight="bold",
                fontsize=9,
            )
            if column == 0:
                values = data.parameters.to_numpy() / 1e6
                ax.hlines(y, 0.02, values, colors="#bfc7cf", linewidth=1.2)
                ax.scatter(values, y, color=COLORS["high"], s=19, zorder=3)
                ax.set_xlim(0.02, 8)
                ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
                ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
                ax.set_title("Parameters", loc="left", pad=8)
                ax.set_xlabel("Parameters [million]")
            else:
                prefix = "cpu_b1" if column == 1 else "gpu_b1"
                median = data[prefix + "_wall_p50_ms"].to_numpy()
                p95 = data[prefix + "_wall_p95_ms"].to_numpy()
                color = COLORS["low"] if column == 1 else COLORS["virtual"]
                ax.hlines(y, median, p95, color=color, linewidth=1.5)
                ax.scatter(median, y, color=color, s=19, zorder=3)
                ax.plot(
                    p95, y, "|", color=color, markersize=7, linestyle="none", markeredgewidth=1.1
                )
                ax.set_xlim(0.2, 8)
                ax.set_xticks([0.3, 1, 3])
                ax.set_xticklabels(["0.3", "1", "3"])
                ax.set_title("CPU · 1 thread" if column == 1 else "RTX 4090 GPU", loc="left", pad=8)
                ax.set_xlabel("Batch 1 wall time [ms]")
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        box = axes[row, 0].get_position()
        fig.text(
            box.x0,
            box.y1 + 0.066,
            "Paired mapping: 9→6 channels, 80 samples"
            if task == "paired"
            else "Navigation network: 6→3 channels, 80 samples",
            fontsize=9,
            fontweight="bold",
        )
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", color="#333333", linestyle="none", markersize=4, label="P50"
            ),
            Line2D(
                [], [], marker="|", color="#333333", linestyle="none", markersize=7, label="P95"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.61, 0.04),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.205,
        0.013,
        "Historical seed 20261010 · 300 timed calls · FP32, strict determinism",
        fontsize=7,
    )
    return fig


def _heatmap_axes(gains=False):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.1))
    fig.subplots_adjust(left=0.185, right=0.98, top=0.915, bottom=0.255, wspace=0.72)
    for ax in axes:
        ax.grid(False)
        ax.set_xticks(range(4))
        ax.set_xticklabels(SPANS)
        ax.set_yticks(range(14))
        ax.set_yticklabels([LABELS[name] for name in DOWNSTREAM_ORDER])
        ax.tick_params(length=0, pad=4)
        ax.set_xlabel("Endpoint span [s]", labelpad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(6.5, color="#68727c", linewidth=0.7)
        ax.axhline(12.5, color="#68727c", linewidth=0.7)
    fig.text(
        0.185,
        0.976,
        "Independent optical GT · historical Penn bs test recording",
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        0.185,
        0.018,
        "Neural: 3 optimization seeds; fixed methods: 1 evaluation. Windows/run: 82 / 32 / 16 / 8.",
        fontsize=6.9,
    )
    return fig, axes


def figure_downstream(table, gains=False):
    fig, axes = _heatmap_axes(gains)
    for column, (task, unit, title) in enumerate(
        (("attitude", "°", "Attitude endpoint RMSE"), ("strapdown", "m", "Position endpoint RMSE"))
    ):
        ax = axes[column]
        subset_data = table[table.task == task].set_index(["method", "samples"])
        values = np.array(
            [
                [
                    subset_data.loc[
                        (name, size), "improvement_over_low_percent" if gains else "mean"
                    ]
                    for size in HORIZONS
                ]
                for name in DOWNSTREAM_ORDER
            ]
        )
        if gains:
            maximum = 50 * np.ceil(np.abs(values).max() / 50)
            norm = colors.TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum)
            cmap = colors.LinearSegmentedColormap.from_list(
                "project_gain", [COLORS["low"], "#ffffff", COLORS["accent"]]
            )
        else:
            norm = colors.LogNorm(
                vmin=0.8 if task == "attitude" else 0.4, vmax=25 if task == "attitude" else 220
            )
            cmap = colors.LinearSegmentedColormap.from_list(
                "project_absolute", ["#f4f7fa", "#8eb7d3", "#1f4b70"]
            )
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        for row, name in enumerate(DOWNSTREAM_ORDER):
            for index, size in enumerate(HORIZONS):
                record = subset_data.loc[(name, size)]
                if gains:
                    label = (
                        f"{values[row, index]:+.0f}%" if abs(values[row, index]) > 0.005 else "0%"
                    )
                else:
                    label = f"{record['mean']:.3g}"
                    if record.optimizer_runs == 3:
                        label += f"\n±{record.optimizer_seed_sample_sd:.2g}"
                dark = (
                    norm(values[row, index]) > 0.64
                    if not gains
                    else abs(norm(values[row, index]) - 0.5) > 0.36
                )
                ax.text(
                    index,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.9,
                    color="white" if dark else "#1f2933",
                    linespacing=1.05,
                )
        ax.add_patch(
            plt.Rectangle((-0.5, 6.5), 4, 1, fill=False, edgecolor="#34404a", linewidth=1.1)
        )
        ax.text(
            -0.03,
            1.045,
            chr(97 + column),
            transform=ax.transAxes,
            ha="right",
            fontweight="bold",
            fontsize=9,
        )
        ax.set_title(
            title + (" change vs LOW" if gains else f" [{unit}]"), loc="left", fontsize=8, pad=9
        )
        position = ax.get_position()
        cbax = fig.add_axes([position.x0, 0.125, position.width, 0.022])
        cb = fig.colorbar(image, cax=cbax, orientation="horizontal")
        if gains:
            cb.set_ticks([-maximum, -maximum / 2, 0, maximum / 2, maximum])
            cb.set_label("RMSE reduction vs LOW [%]", fontsize=7, labelpad=3)
        else:
            cb.set_ticks([1, 2, 5, 10, 20] if task == "attitude" else [0.5, 1, 5, 20, 100])
            cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            cb.set_label(f"Endpoint RMSE [{unit}] · lower is better", fontsize=7, labelpad=3)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_linewidth(0.5)
    fig.text(
        0.185,
        0.055,
        "Green: improvement; orange: degradation. Panel color ranges differ; all values are annotated."
        if gains
        else "Values: mean ± sample SD for neural models; fixed comparators have no seed interval. Log color scale.",
        fontsize=6.8,
    )
    return fig


def markdown_table(table, columns, labels):
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/reviewer_preliminary_figures.yaml")
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, verification = read_inputs(config, project)
    out = project / config["output_root"]
    print(f"WRITES: historical tables, figures, captions and QA under {out}", flush=True)
    print(
        "SKIPS: new HPO results, checkpoint/model/data access, training, significance tests, method/horizon selection",
        flush=True,
    )
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Use --overwrite only to regenerate this preliminary figure artifact directory"
        )
    for name in ("figures", "source_data", "source"):
        (out / name).mkdir(parents=True, exist_ok=True)
    long, wide, reports = timing_tables(paths, int(config["historical_checkpoint_seed"]))
    downstream, trials, windows = downstream_table(paths)
    long.to_csv(out / "source_data/timing_all45_records.csv", index=False)
    wide.to_csv(out / "parameters_latency_complete15.csv", index=False)
    downstream.to_csv(out / "source_data/downstream_all112_cells.csv", index=False)
    trials.to_csv(out / "source_data/downstream_all224_trials.csv", index=False)
    windows.to_csv(out / "source_data/downstream_all7728_windows.csv", index=False)
    apply_paper_style()
    matplotlib.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": None,
        }
    )
    qa = {}
    index = []
    index.append(
        save_publication(
            figure_timing(wide),
            out,
            "fig1_historical_parameters_latency",
            "图1｜已完成历史 checkpoint 的模型参数量与单窗前向耗时。上排为9→6配对信号映射，下排为6→3导航网络，均为80点窗口、固定种子20261010。点为300次计时调用的P50，短线到端帽为P95，均排除30次预热；不是优化种子误差区间。CPU为单线程Xeon8352V，GPU为RTX4090（4个CPU调度线程）；FP32、严格确定性、禁用TF32及Flash/高效/cuDNN注意力。仅测设备驻留输入的网络前向，不含数据传输、标准化、窗口收集或信号还原。CPU/GPU优劣不可推广到其他硬件、实现或策略。",
            "parameters_latency_complete15.csv",
            qa,
        )
    )
    index.append(
        save_publication(
            figure_downstream(downstream),
            out,
            "fig2a_historical_downstream_absolute",
            "图2a｜所有14种历史方法对独立光学GT的惯性传播诊断。展示全部1.975、4.975、9.975和19.975s预设窗口；左为SO(3)姿态终点RMSE（度），右为位置终点RMSE（米）。神经模型为3个优化种子的均值±样本SD；经典方法及HIGH各计算一次，不复制为3种子。窗口数分别为82/32/16/8，独立测试记录仅Penn bs一个。HIGH是测量参考而非姿态GT。每窗使用一次GT初态；位置初速度来自固定过去0.1s光学位置差，不是可部署完整导航系统。色标为对数；文字保留均值3位、SD2位有效数字。",
            "source_data/downstream_all112_cells.csv",
            qa,
        )
    )
    index.append(
        save_publication(
            figure_downstream(downstream, True),
            out,
            "fig2b_historical_downstream_gains_degradations",
            "图2b｜对同一LOW基线的RMSE相对变化，100×(1−方法RMSE/LOW RMSE)。绿色为改善，橙色为退化；完整保留14种方法×4个预设窗口。两面板采用各自对称范围并逐格标数，色深不可直接跨面板比较。短窗口位置改善与姿态及长窗口退化并存，不能只选正面窗口，也不能把匹配HIGH的信号误差降低等同于独立导航性能提升。没有显著性检验；优化种子SD不是独立记录置信区间。",
            "source_data/downstream_all112_cells.csv",
            qa,
        )
    )
    lines = [
        "# 已完成历史结果：参数、前向速度与独立GT诊断（阶段性）",
        "",
        "本目录只包含已完成历史测量，不含新一轮六通道HPO结果。配对历史模型使用9个输入通道（含3个参考标记），导航模型使用6个输入通道；不能直接当作新HPO的公平对照。",
        "",
        "计时设备：Intel Xeon Platinum8352V@2.10GHz（CPU单线程）；NVIDIA RTX4090（GPU路径4个CPU调度线程）。PyTorch2.4.0+cu121、FP32、严格确定性、数学注意力，禁用TF32及其他SDPA后端。每个任务/模型只使用种子20261010的实际checkpoint；30次预热，3×100次计时。P50/P95是调用分布，不是3个独立模型的均值/置信区间。",
        "",
        "## 完整参数和速度表",
        "",
        "下表均为实际网络前向。GPU B128吞吐使用独立整段循环计时；不能用带逐次同步的P50简单代替吞吐。网络耗时不含窗口收集、已有滤波可用延迟、传输、标准化或残差/位移还原。经典算法本次没有参数/前向计时收据，未虚构其速度。",
    ]
    for task, title in (
        ("paired", "配对信号映射：9→6，80点"),
        ("navigation", "导航网络：6→3，80点，训练目标为1.975s端点位移"),
    ):
        part = wide[wide.task == task].copy()
        part["params"] = part.parameters.map(lambda value: f"{value:,}")
        for prefix in ("cpu_b1", "gpu_b1"):
            part[prefix] = part.apply(
                lambda row, prefix=prefix: (
                    f"{row[prefix + '_wall_p50_ms']:.4f} / {row[prefix + '_wall_p95_ms']:.4f}"
                ),
                axis=1,
            )
        part["bulk"] = part.gpu_b128_throughput_windows_per_second.map(
            lambda value: f"{value:,.1f}"
        )
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                markdown_table(
                    part,
                    ["label", "params", "cpu_b1", "gpu_b1", "bulk"],
                    [
                        "模型",
                        "实际参数数",
                        "CPU B1 P50/P95(ms)",
                        "GPU B1 P50/P95(ms)",
                        "GPU B128吞吐(窗/s)",
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "严格策略下，小模型或特定卷积实现的CPU单窗耗时可能小于GPU；例如历史TCN配对模型为1.7168ms对5.9236ms。这是该实现和策略的测量结果，不能据此宣称CPU普遍优于GPU。GPU批量吞吐与单窗耗时须分别评价。",
            "",
            "## 独立GT下游完整结果",
            "",
            "全部14方法和4个预设窗口均保留。测试集只有Penn bs一个物理记录；神经模型的3优化种子不是3个独立场景。经典算法/HIGH去重为一次评价。下表无假设检验，不使用“统计显著”措辞。",
        ]
    )
    for task, title in (("attitude", "姿态终点RMSE（度）"), ("strapdown", "位置终点RMSE（米）")):
        table = []
        for method in DOWNSTREAM_ORDER:
            row = {"method": LABELS[method]}
            for size in HORIZONS:
                record = downstream[
                    (downstream.task == task)
                    & (downstream.method == method)
                    & (downstream.samples == size)
                ].iloc[0]
                row[str(size)] = f"{record['mean']:.5g}" + (
                    f" ± {record.optimizer_seed_sample_sd:.3g}"
                    if record.optimizer_runs == 3
                    else ""
                )
            table.append(row)
        lines.extend(
            [
                "",
                f"### {title}",
                "",
                markdown_table(
                    pd.DataFrame(table),
                    ["method", *[str(size) for size in HORIZONS]],
                    ["方法", *[span + "s" for span in SPANS]],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "历史TCN在1.975s的位置RMSE由LOW的1.2347m降为0.5048m（59.1%）；其姿态RMSE却由0.9676°增至2.1589°。19.975s的位置RMSE由LOW的88.733m增至182.364m。图表完整展示这些混合结论，不按测试分数挑选窗口或方法。",
            "",
            "独立审计显示HIGH相对光学姿态导数的平均角速度误差约1.04°/s，LOW约0.19°/s；该差异已存在于原始IMU文件。源域拟合的LOW→HIGH偏差映射会继承这一差异。因此，信号参考匹配和独立姿态/惯性位置质量是不同证据。光学GT约10cm的位置精度及过去0.1s初速度差分也限制本诊断的解释范围。",
            "",
            "## 图注与导出",
            "",
            *[line for item in index for line in [f"### {item['id']}", "", item["caption_zh"], ""]],
        ]
    )
    (out / "历史参数速度与下游诊断.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "figure_index.md").write_text(
        "# 历史结果图索引\n\n"
        + "\n\n".join(
            f"- **{item['id']}**: {item['caption_zh']}\n  - 源数据：{item['source_data']}\n  - 导出："
            + ", ".join(item["files"])
            for item in index
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(out / "figure_index.json", index)
    write_json(out / "qa.json", qa)
    write_json(out / "input_verification.json", verification)
    shutil.copy2(args.config, out / "config_snapshot.yaml")
    shutil.copy2(Path(__file__), out / "source/build_reviewer_preliminary_figures.py")
    write_json(
        out / "analysis_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "scope": "completed_historical_results_only_not_new_HPO",
            "inputs": verification,
            "source_script_sha256": sha(Path(__file__)),
            "config_sha256": sha(args.config),
            "processing_git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "processing_source_note": "Use the source snapshot SHA for uncommitted processing code; trained checkpoints retain their own receipt provenance.",
            "runtime": {
                "python": sys.version,
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "source_environments": {
                name: report["environment"] for name, report in reports.items()
            },
            "timing_records": len(long),
            "distinct_task_models": len(wide),
            "downstream_cells": len(downstream),
            "files": {
                path.relative_to(out).as_posix(): sha(path)
                for path in out.rglob("*")
                if path.is_file() and path.name != "analysis_manifest.json"
            },
        },
    )
    print(
        f"WRITTEN:15 model rows,45 timing rows,112 downstream cells,3 publication figures; visual QA pending under {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
