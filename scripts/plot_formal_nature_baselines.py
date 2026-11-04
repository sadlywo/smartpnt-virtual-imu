"""Publish fixed-run baseline tables and Nature-style quantitative figures.

Only reads frozen test evaluations. No fitting, inference, candidate selection,
alignment or training occurs. The supplied project YAML is the figure contract.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from virtual_imu.viz.style import apply_paper_style  # noqa: E402

LABELS = {
    "identity": "Identity",
    "bias_mean": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "learned_fir_causal": "Causal FIR",
    "arx_linear": "ARX (LOW history)",
    "model": "TCN",
    "zero": "Zero displacement",
    "train_mean": "Train mean",
    "ridge": "Ridge",
}
COLOURS = {
    "identity": "#404040",
    "bias_mean": "#8F8F8F",
    "affine": "#A5A0B8",
    "ridge_affine": "#88849F",
    "learned_fir_causal": "#65617F",
    "arx_linear": "#45405F",
    "model": "#1F77B4",
    "zero": "#404040",
    "train_mean": "#8F8F8F",
    "ridge": "#88849F",
}
CHANNELS = {
    "gyro": ("gyro_vector_rmse_radps", slice(0, 3), "rad/s"),
    "accel": ("accel_vector_rmse_mps2", slice(3, 6), "m/s^2"),
}


def log(kind: str, message: object) -> None:
    print(f"{kind}: {message}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    log("READS", path.relative_to(PROJECT).as_posix())
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def publication_style() -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": False,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
        }
    )


def panel_letter(ax, letter: str, x: float = -0.11, y: float = 1.075) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        color="black",
    )


def load_results(
    root: Path, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    state = read_json(root / "pipeline.json")
    expected = {(task, seed) for task in ("paired", "navigation") for seed in config["seeds"]}
    if (
        state["status"] != "complete"
        or {(entry["task"], entry["seed"]) for entry in state["runs"]} != expected
    ):
        raise ValueError("Expected the complete six-run frozen matrix")
    paired, navigation, decomposition = [], [], []
    sources = [root / "pipeline.json"]
    split = config["split"]
    for entry in state["runs"]:
        run = root / "experiments" / Path(entry["run_dir"]).name
        if entry["status"] != "evaluated":
            raise ValueError("Every task and seed must be evaluated")
        source_path = run / "source_bundle.json"
        source = read_json(source_path)
        if (
            source["source_fingerprint"] != config["source_fingerprint"]
            or source["dataset_manifest_sha256"] != config["dataset_manifest_sha256"]
        ):
            raise ValueError("Training source/data fingerprints differ from the figure contract")
        provenance_path = run / "provenance.json"
        provenance = read_json(provenance_path)
        if sha256(run / "data_manifest.json") != config["dataset_manifest_sha256"]:
            raise ValueError("Run data snapshot differs from the frozen dataset")
        for key, ref in provenance["unified_dataset"]["inputs"].items():
            filename = "data_manifest.json" if key == "dataset_manifest" else f"{key}.json"
            if sha256(run / filename) != ref["sha256"]:
                raise ValueError(f"Saved training manifest differs: {run.name}/{filename}")
        cfg_path = run / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if cfg["seed"] != entry["seed"]:
            raise ValueError("Seed/config mismatch")
        path = run / "evaluation/metrics.json"
        evaluated = read_json(path)["splits"][split]
        sources.extend([path, cfg_path, source_path, provenance_path, run / "data_manifest.json"])
        common = {"seed": entry["seed"], "split": split, "run": run.relative_to(PROJECT).as_posix()}
        if entry["task"] == "paired":
            for pair in evaluated["pairs"]:
                identity = {key: pair[key] for key in ("dataset_id", "low_sensor", "high_sensor")}
                if set(pair["metrics"]) != set(config["method_order"]):
                    raise ValueError("Paired methods differ from the fixed comparison")
                for method, metrics in pair["metrics"].items():
                    if not np.isfinite([metrics[k] for k, _, _ in CHANNELS.values()]).all():
                        raise ValueError("Non-finite paired metric")
                    row = {**common, **identity, "method": method, "samples": metrics["samples"]}
                    for metric, _, _ in CHANNELS.values():
                        row[metric] = metrics[metric]
                    row.update(
                        gyro_p95_radps=metrics["gyro_error_p95_radps"],
                        accel_p95_mps2=metrics["accel_error_p95_mps2"],
                    )
                    paired.append(row)
                    for channel, (metric, channels, unit) in CHANNELS.items():
                        mean = np.asarray(metrics["axis_mean_error"], float)[channels]
                        std = np.asarray(metrics["axis_error_std"], float)[channels]
                        offset_mse, centered_mse = float(mean @ mean), float(std @ std)
                        if not np.isclose(
                            offset_mse + centered_mse, metrics[metric] ** 2, rtol=1e-10, atol=1e-14
                        ):
                            raise ValueError(
                                "RMSE squared must equal squared mean residual plus centered MSE"
                            )
                        decomposition.append(
                            {
                                **common,
                                **identity,
                                "method": method,
                                "channel": channel,
                                "unit": unit,
                                "vector_rmse": metrics[metric],
                                "mean_residual_norm": np.sqrt(offset_mse),
                                "centered_vector_rmse": np.sqrt(centered_mse),
                                "mean_residual_mse_share_percent": 100
                                * offset_mse
                                / (offset_mse + centered_mse),
                            }
                        )
        else:
            if set(evaluated["metrics"]) != {"zero", "train_mean", "ridge", "model"}:
                raise ValueError("Navigation methods differ from the fixed comparison")
            for method, metrics in evaluated["metrics"].items():
                row = {**common, "method": method}
                for key in (
                    "vector_rmse_m",
                    "horizontal_rmse_m",
                    "vertical_rmse_m",
                    "component_rmse_m",
                    "error_norm_p95_m",
                ):
                    row[key] = metrics[key]
                if not np.isfinite(
                    list(row[k] for k in ("vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"))
                ).all():
                    raise ValueError("Non-finite navigation metric")
                navigation.append(row)
    paired_df, navigation_df, decomposition_df = map(
        pd.DataFrame, (paired, navigation, decomposition)
    )
    if len(paired_df) != 84 or len(navigation_df) != 12 or len(decomposition_df) != 168:
        raise ValueError("Incomplete baseline or seed coverage")
    for task, frame, groups in (
        ("paired", paired_df, ["low_sensor", "method"]),
        ("navigation", navigation_df, ["method"]),
    ):
        if not frame.groupby(groups)["seed"].nunique().eq(3).all():
            raise ValueError(f"Missing optimization seeds in {task}")
    return paired_df, navigation_df, decomposition_df, sources


def summarized(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    result = frame.groupby(groups, sort=False)[metrics].agg(["mean", "std", "count"])
    result.columns = ["_".join(parts) for parts in result.columns]
    return result.reset_index()


def relative_errors(paired: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (sensor, seed), group in paired.groupby(["low_sensor", "seed"], sort=False):
        baseline = group[group.method.eq("identity")].iloc[0]
        for _, row in group.iterrows():
            for channel, (metric, _, unit) in CHANNELS.items():
                records.append(
                    {
                        "low_sensor": sensor,
                        "seed": seed,
                        "method": row.method,
                        "channel": channel,
                        "unit": unit,
                        "improvement_percent": 100 * (1 - row[metric] / baseline[metric]),
                    }
                )
    return pd.DataFrame(records)


def show_three_seeds(ax, values: np.ndarray, y: float, color: str, scale: float = 1.0) -> None:
    values = np.asarray(values, float) * scale
    for offset, value in zip((-0.10, 0.0, 0.10), values, strict=True):
        ax.plot(
            value,
            y + offset,
            "o",
            markersize=2.4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.6,
            zorder=3,
        )
    ax.errorbar(
        values.mean(),
        y,
        xerr=values.std(ddof=1),
        fmt="s",
        markersize=3.6,
        color=color,
        elinewidth=0.9,
        capsize=2,
        markeredgewidth=0.6,
        zorder=4,
    )


def figure_absolute(paired: pd.DataFrame, config: dict):
    dim = config["figure_contract"]["dimensions"]
    fig, axes = plt.subplots(2, 4, figsize=(dim["width_inches"], dim["figure_1_height_inches"]))
    fig.subplots_adjust(left=0.154, right=0.985, bottom=0.115, top=0.91, hspace=0.60, wspace=0.37)
    sensors, methods = config["sensor_order"], config["method_order"]
    for row, (channel, (metric, _, _)) in enumerate(CHANNELS.items()):
        scale = 1000 if channel == "gyro" else 1
        for col, sensor in enumerate(sensors):
            ax = axes[row, col]
            data = paired[paired.low_sensor.eq(sensor)]
            for y, method in enumerate(methods):
                values = data[data.method.eq(method)].sort_values("seed")[metric].to_numpy()
                show_three_seeds(ax, values, y, COLOURS[method], scale)
            ax.set_yticks(
                range(len(methods)),
                [LABELS[m] for m in methods] if col == 0 else [""] * len(methods),
            )
            ax.tick_params(axis="y", length=0, pad=4)
            ax.set_ylim(len(methods) - 0.55, -0.55)
            ax.set_xlim(0, data[metric].max() * scale * 1.15)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            ax.set_xlabel("Gyro RMSE (mrad/s)" if row == 0 else "Accel RMSE (m/s²)", labelpad=5)
            if row == 0:
                label = "Tango Top" if sensor == "Tango Top IMU" else sensor
                target = "ADIS16488" if sensor == "Tango Top IMU" else "VN100"
                ax.set_title(f"{label} → {target}", pad=7)
            ax.spines["left"].set_visible(False)
            panel_letter(ax, chr(ord("a") + row * 4 + col), x=-0.16)
    fig.legend(
        [
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markeredgecolor="#404040",
                markerfacecolor="white",
                markersize=3,
            ),
            Line2D([], [], marker="s", color="#404040", markersize=3.5, linewidth=0.8),
        ],
        ["Individual optimization seeds (n = 3)", "Mean ± sample SD"],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.00),
        handletextpad=0.6,
        columnspacing=2.5,
    )
    return fig


def figure_relative(relative: pd.DataFrame, decomposition: pd.DataFrame, config: dict):
    dim = config["figure_contract"]["dimensions"]
    fig = plt.figure(figsize=(dim["width_inches"], dim["figure_2_height_inches"]))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.16,
        right=0.975,
        bottom=0.16,
        top=0.92,
        height_ratios=[1.65, 1],
        hspace=0.85,
        wspace=0.47,
    )
    sensors = config["sensor_order"]
    matrix_rows = []
    for col, channel in enumerate(CHANNELS):
        ax = fig.add_subplot(grid[0, col])
        part = relative[relative.method.eq("model") & relative.channel.eq(channel)]
        for y, sensor in enumerate(sensors):
            values = (
                part[part.low_sensor.eq(sensor)].sort_values("seed").improvement_percent.to_numpy()
            )
            colour = "#B76632" if values.mean() < 0 else "#1F77B4"
            show_three_seeds(ax, values, y, colour)
            ax.text(
                98,
                y,
                f"{values.mean():+.1f}%",
                va="center",
                ha="right",
                fontsize=6.5,
                color="black",
            )
        ax.axvline(0, color="#777777", linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlim(-60, 100)
        ax.set_xticks([-50, 0, 50, 100])
        ax.set_ylim(3.55, -0.55)
        ax.set_yticks(range(4), ["Tango Top" if s == "Tango Top IMU" else s for s in sensors])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("TCN improvement versus identity (%)")
        ax.set_title("Gyroscope" if channel == "gyro" else "Accelerometer", pad=9)
        panel_letter(ax, chr(ord("a") + col), x=-0.14)
        heat = fig.add_subplot(grid[1, col])
        values = np.empty((2, 4))
        for row, method in enumerate(("identity", "model")):
            for column, sensor in enumerate(sensors):
                subset = decomposition[
                    decomposition.method.eq(method)
                    & decomposition.channel.eq(channel)
                    & decomposition.low_sensor.eq(sensor)
                ]
                values[row, column] = subset.mean_residual_mse_share_percent.mean()
                matrix_rows.append(
                    {
                        "channel": channel,
                        "method": method,
                        "low_sensor": sensor,
                        "mean_residual_mse_share_percent_mean": values[row, column],
                        "mean_residual_mse_share_percent_sd": subset.mean_residual_mse_share_percent.std(
                            ddof=1
                        ),
                    }
                )
        colourmap = LinearSegmentedColormap.from_list(
            "mean_residual_share", ["#FFFFFF", "#C0D9E6", "#477C94"]
        )
        heat.imshow(
            values, cmap=colourmap, norm=Normalize(0, 100), aspect="auto", interpolation="nearest"
        )
        for (i, j), value in np.ndenumerate(values):
            heat.text(j, i, f"{value:.1f}%", ha="center", va="center", color="black", fontsize=6.5)
        heat.set_yticks([0, 1], ["Identity", "TCN"])
        heat.set_xticks(
            range(4), ["BMX160", "MPU9250", "LSM9DS1", "Tango Top"], rotation=22, ha="right"
        )
        heat.tick_params(length=0)
        for spine in heat.spines.values():
            spine.set_visible(False)
        heat.set_title("Mean-residual share of squared error (%)", pad=9)
        panel_letter(heat, chr(ord("c") + col), x=-0.14)
    return fig, pd.DataFrame(matrix_rows)


def figure_navigation(navigation: pd.DataFrame, config: dict):
    dim = config["figure_contract"]["dimensions"]
    fig, axes = plt.subplots(1, 3, figsize=(dim["width_inches"], dim["figure_3_height_inches"]))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.22, top=0.81, wspace=0.44)
    methods = ["zero", "train_mean", "ridge", "model"]
    metrics = ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]
    for col, (metric, title) in enumerate(
        zip(metrics, ["3D displacement", "Horizontal", "Vertical"], strict=True)
    ):
        ax = axes[col]
        for y, method in enumerate(methods):
            values = navigation[navigation.method.eq(method)].sort_values("seed")[metric].to_numpy()
            show_three_seeds(ax, values, y, COLOURS[method])
        ax.set_ylim(3.55, -0.55)
        ax.set_xlim(0, navigation[metric].max() * 1.18)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.set_yticks(range(4), [LABELS[m] for m in methods] if col == 0 else [""] * 4)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Window RMSE (m)")
        ax.set_title(title, pad=8)
        panel_letter(ax, chr(ord("a") + col), x=-0.16)
    fig.text(
        0.56,
        0.96,
        "SmartPNT Traj1 | 1.975 s endpoint displacement | 1,649 overlapping windows",
        fontsize=7,
        ha="center",
        va="top",
    )
    return fig


def export_figure(fig, folder: Path, stem: str) -> dict:
    # MaxNLocator can retain out-of-view ticks which the axis never draws;
    # remove these unused labels before checking the fixed figure canvas.
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            ticks = axis.get_majorticklocs()
            inside = (ticks >= min(limits)) & (ticks <= max(limits))
            if not inside.all():
                axis.set_ticks(ticks[inside])
    fig.canvas.draw()
    # Check the complete text geometry before exporting at exact physical size.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    clipped = []
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        extent = artist.get_window_extent(renderer=renderer)
        if (
            extent.x0 < bounds.x0 - 1
            or extent.y0 < bounds.y0 - 1
            or extent.x1 > bounds.x1 + 1
            or extent.y1 > bounds.y1 + 1
        ):
            clipped.append(artist.get_text())
    if clipped:
        raise ValueError(f"Text extends beyond fixed figure canvas: {stem}: {clipped}")
    outputs = []
    for suffix, dpi in (("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)):
        path = folder / f"{stem}.{suffix}"
        kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(path, dpi=dpi, bbox_inches=None, **kwargs)
        if path.stat().st_size == 0:
            raise ValueError("Empty figure export")
        outputs.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    svg = ET.parse(folder / f"{stem}.svg").getroot()
    text_nodes = list(svg.iter("{http://www.w3.org/2000/svg}text"))
    if len(text_nodes) < 10:
        raise ValueError("SVG does not preserve editable text nodes")
    result = {
        "stem": stem,
        "size_inches": fig.get_size_inches().tolist(),
        "editable_svg_text_nodes": len(text_nodes),
        "text_within_canvas": True,
        "files": outputs,
    }
    plt.close(fig)
    return result


def write_tables(
    output: Path,
    paired: pd.DataFrame,
    navigation: pd.DataFrame,
    decomposition: pd.DataFrame,
    relative: pd.DataFrame,
    config: dict,
) -> dict:
    tables = output / "source_data"
    tables.mkdir(exist_ok=True)
    paired_summary = summarized(
        paired,
        ["dataset_id", "low_sensor", "high_sensor", "method"],
        [metric for metric, _, _ in CHANNELS.values()],
    )
    nav_summary = summarized(
        navigation, ["method"], ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]
    )
    decomposition_summary = summarized(
        decomposition,
        ["low_sensor", "method", "channel", "unit"],
        [
            "vector_rmse",
            "mean_residual_norm",
            "centered_vector_rmse",
            "mean_residual_mse_share_percent",
        ],
    )
    relative_summary = summarized(
        relative, ["low_sensor", "method", "channel"], ["improvement_percent"]
    )
    items = {
        "paired_all_seeds": paired,
        "paired_summary": paired_summary,
        "navigation_all_seeds": navigation,
        "navigation_summary": nav_summary,
        "error_decomposition_all_seeds": decomposition,
        "error_decomposition_summary": decomposition_summary,
        "relative_improvement_all_seeds": relative,
        "relative_improvement_summary": relative_summary,
    }
    lines = [
        "# 冻结测试集的基线结果",
        "",
        "均值与样本标准差来自三个优化种子；不是独立采集重复。所有经典拟合只用训练集，同一配对下各方法使用相同测试时间点。",
        "",
    ]
    for channel, (metric, _, unit) in CHANNELS.items():
        lines += [
            f"## {'陀螺仪' if channel == 'gyro' else '加速度'}三维矢量 RMSE（{unit}）",
            "",
            "| 方法 | BMX160 → VN100 | MPU9250 → VN100 | LSM9DS1 → VN100 | Tango → ADIS16488 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for method in config["method_order"]:
            row = [LABELS[method]]
            for sensor in config["sensor_order"]:
                part = paired[paired.method.eq(method) & paired.low_sensor.eq(sensor)][metric]
                value = f"{part.mean():.6f}"
                if method == "model":
                    value += f" ± {part.std(ddof=1):.6f}"
                row.append(value)
            lines.append("| " + " | ".join(row) + " |")
        lines += [""]
    lines += [
        "## SmartPNT 窗口位移 RMSE（m）",
        "",
        "| 方法 | 三维 | 水平 | 垂直 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ("zero", "train_mean", "ridge", "model"):
        row = [LABELS[method]]
        for metric in ("vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"):
            values = navigation[navigation.method.eq(method)][metric]
            value = f"{values.mean():.4f}"
            if method == "model":
                value += f" ± {values.std(ddof=1):.4f}"
            row.append(value)
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Identity=不校正；Mean bias=训练均值偏移；Affine=当前六轴线性映射+截距；Ridge affine=加岭正则；Causal FIR=9 tap 因果 LOW 历史；ARX 为代码命名，实际使用12 tap LOW历史及LOW一阶差分，无过去 HIGH 输出反馈。TCN 为固定联合残差网络。",
        "",
        "固定经典基线在三个种子中相同，故无优化种子波动；不把重复写出的同一组基线当作三次独立实验。SmartPNT 只有1.975秒端点位移标签，零位移/均值预测不使用六轴输入，Ridge/TCN 使用 LOW 窗口。",
        "",
        "均值残差分解：RMSE² = ‖mean(error)‖² + Σ std(error_axis)²，std 使用总体定义（ddof=0）。这是测试区间经验统计，不是独立静态零偏或纯噪声标定。gyro/accel 保持各自单位，不能直接求和。",
        "",
    ]
    (output / "baseline_tables.md").write_text("\n".join(lines), encoding="utf-8")
    for name, frame in items.items():
        frame.to_csv(tables / f"{name}.csv", index=False, float_format="%.12g")
    return items


def write_captions(output: Path, figures: list[dict]) -> None:
    captions = [
        (
            "nature_figure_1_paired_baselines",
            "配对六轴信号的完整基线比较",
            "a–d为陀螺仪三维矢量RMSE（为易读显示为mrad/s；1mrad/s=0.001rad/s），e–h为加速度三维矢量RMSE（m/s²）。列依次为BMX160、MPU9250、LSM9DS1→VN100和Tango Top→ADIS16488。空心点为三个固定优化种子20261010/11/12，方点及误差线为均值±样本SD（ddof=1）；经典方法无种子波动。四列横轴范围分别设置，按数值比较；所有横轴从0开始。每种方法只在训练集拟合并在相同覆盖测试时间点评分。ALiMU测试来自dyn_08/dyn_09，Penn来自bs，三个器件视图不构成独立录制。FIR和代码所称ARX在各器件gyro/accel上均优于TCN，LSM9DS1 gyro的TCN还差于identity。无显著性检验。",
        ),
        (
            "nature_figure_2_improvement_and_error_structure",
            "相对改进与总误差结构",
            "a、b显示TCN相对identity的陀螺仪/加速度改进率100×(1−RMSE_TCN/RMSE_identity)；虚线为0，负值表示退化。方点/误差线为三个种子的均值±样本SD，空心点为各种子；LSM9DS1 gyro在三种子均退化。c、d显示同一测试区间中平均残差对总均方误差的占比：100×‖mean(error)‖²/RMSE²；颜色固定在0–100%，格内为三种子均值。剩余误差是去均值残差的波动部分，含动态响应、坐标/参考差异和随机误差，不是纯噪声。它没有分离温漂、零偏漂移、比例因子、安装误差或时钟误差，不是独立静态标定。",
        ),
        (
            "nature_figure_3_navigation_baselines",
            "SmartPNT导航探索的独立结果",
            "a–c为三维、水平和垂直窗口位移RMSE（m）。测试为Traj1的1649个重叠窗口，每个窗口80点、40Hz，端点间隔1.975秒。空心点为三个优化种子，方点/误差线为均值±样本SD；所有方法使用同一组端点。零位移不拟合，训练均值只用Traj3标签，Ridge/TCN输入为LOW六轴窗口；固定初始姿态/航向和速度未作为输入。TCN三种子均劣于zero和train-mean；不能累加重叠位移解释为完整轨迹，不是ATE/RPE或可靠全局惯导证据。三个优化种子不增加独立测试轨迹数量，无显著性检验。",
        ),
    ]
    lines = [
        "# Nature 风格科研图与图注",
        "",
        "主文件为可编辑文字的SVG与Type42字体PDF；PNG300dpi用于预览，TIFF600dpi为额外栅格副本。期刊主图仍应使用矢量文件。",
        "",
        "全部图沿用项目Python/matplotlib工作流与IMU样式基础，按Nature官方图稿指导调整为双栏约183mm、白底、5–7pt正文、8pt面板标签和统一方法色系。",
        "",
    ]
    for (stem, title, caption), record in zip(captions, figures, strict=True):
        record.update(title=title, caption=caption)
        lines += [
            f"## {title}",
            "",
            f"![{title}](figures/{stem}.png)",
            "",
            caption,
            "",
            f"[SVG](figures/{stem}.svg) · [PDF](figures/{stem}.pdf) · [TIFF](figures/{stem}.tiff)",
            "",
        ]
    lines += [
        "## 误差解释与范围",
        "",
        "主指标是相对参考IMU的三个轴矢量误差RMSE，包含方向误差；不是两向量模长的差。逐轴MAE描述典型绝对误差，mean(error)为平均残差偏移，std(error)为去均值波动，P95为误差范数95%分位。PSD/相干性/lag与积分是连续片段内的辅助诊断。当前结果无法将总残差独立分解为器件随机噪声、温漂、零偏漂移等固有误差来源。",
        "",
        "只有测试实测参考一致性，不能宣称VN100/ADIS硬件等级转移、未见器件泛化或高带宽/静态/温度补偿性能。三个种子描述优化波动，不是独立录制置信区间。",
        "",
        "[完整数值表](baseline_tables.md)；[逐种子源数据](source_data/paired_all_seeds.csv)；[误差分解源数据](source_data/error_decomposition_all_seeds.csv)。",
        "",
        "图稿规范参考：[Nature准备要求](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)；[Nature拼版与导出](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)。",
        "",
    ]
    (output / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root, output = PROJECT / config["source_run"], PROJECT / config["output_root"]
    log(
        "READS",
        f"{args.config}; six frozen evaluation JSONs, saved manifests/config/source bindings",
    )
    log(
        "WRITES",
        f"{config['output_root']}: source-data CSVs, baseline tables, figures, captions, config and hashed provenance",
    )
    log(
        "SKIPS",
        "training, inference, fitted-parameter edits, test-driven selection, original result bundle mutation, cross-unit pooling and confidence/significance claims",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "provenance").mkdir(exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    paired, navigation, decomposition, sources = load_results(root, config)
    relative = relative_errors(paired)
    write_tables(output, paired, navigation, decomposition, relative, config)
    publication_style()
    figures = [
        export_figure(
            figure_absolute(paired, config), output / "figures", "nature_figure_1_paired_baselines"
        )
    ]
    fig, panel_data = figure_relative(relative, decomposition, config)
    panel_data.to_csv(output / "source_data/figure_2_bias_fraction_matrix.csv", index=False)
    figures.append(
        export_figure(fig, output / "figures", "nature_figure_2_improvement_and_error_structure")
    )
    figures.append(
        export_figure(
            figure_navigation(navigation, config),
            output / "figures",
            "nature_figure_3_navigation_baselines",
        )
    )
    write_captions(output, figures)
    source_files = sorted(
        {*sources, config_path, Path(__file__).resolve(), PROJECT / "src/virtual_imu/viz/style.py"}
    )
    source_entries = [
        {
            "path": path.relative_to(PROJECT).as_posix(),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in source_files
    ]
    for path in (config_path, Path(__file__).resolve()):
        shutil.copy2(path, output / "provenance" / path.name)
    write_json(
        output / "figure_manifest.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_run": config["source_run"],
            "source_fingerprint": config["source_fingerprint"],
            "data_manifest_sha256": config["dataset_manifest_sha256"],
            "backend": "Python/matplotlib",
            "runtime": {
                "python": platform.python_version(),
                "matplotlib": matplotlib.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "seeds": config["seeds"],
            "contract": config["figure_contract"],
            "source_files": source_entries,
            "figures": figures,
            "visual_review": "pending; no completion claim until actual PNG/SVG/PDF inspection",
            "decomposition_identity_verified": True,
        },
    )
    log(
        "VERIFIED",
        "three figure sets with SVG/PDF/PNG/TIFF; finite frozen metrics, complete seed/method coverage, exact mean/centered-MSE decomposition, editable SVG text and all text within fixed canvas",
    )


if __name__ == "__main__":
    main()
