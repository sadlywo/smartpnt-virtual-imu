"""Traceable Nature-style figures for the completed retrospective IMU benchmark.

Read frozen evaluations only. No fitting, inference, checkpoint selection or edits
to original run identities occur. The plotting YAML fixes all representative choices.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid
from scipy.signal import coherence, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from virtual_imu.viz.style import apply_paper_style

MODELS = [
    "tcn_original",
    "plain_cnn",
    "gru",
    "transformer",
    "cnn_bilstm",
    "deepgyro_adapter",
    "tlio_resnet18_adapter",
    "tartanimu_window_adapter",
]
CLASSIC = ["identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear"]
LABEL = {
    "tcn_original": "TCN",
    "plain_cnn": "CNN",
    "gru": "GRU",
    "transformer": "Transformer",
    "cnn_bilstm": "CNN-BiLSTM",
    "deepgyro_adapter": "DeepGyro adapt.",
    "tlio_resnet18_adapter": "TLIO adapt.",
    "tartanimu_window_adapter": "TartanIMU adapt.",
    "identity": "Identity",
    "bias_mean": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "learned_fir_causal": "Causal FIR",
    "arx_linear": "ARX (LOW history)",
    "zero": "Zero",
    "train_mean": "Train mean",
    "ridge": "Ridge",
}
CAUSAL = {"tcn_original", "gru", "deepgyro_adapter"}
PAIR = ["dataset_id", "low_sensor", "high_sensor"]
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
METRICS = ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]
SENSORS = ["BMX160", "MPU9250", "LSM9DS1", "Tango Top IMU"]
COLORS = dict(
    zip(
        MODELS,
        ["#286994", "#6f4d8c", "#168b9e", "#a7709c", "#6c8c50", "#4b8a74", "#a17843", "#7264a6"],
        strict=True,
    )
)
LOW, HIGH = "#c46c42", "#222222"


def log(kind, text):
    print(f"{kind}: {text}", flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def finite(value, label):
    array = np.asarray(value, dtype=float)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"Empty/nonfinite plot data: {label}")
    return array


def ordered_pairs(frame):
    return sorted(
        frame[PAIR].drop_duplicates().itertuples(index=False, name=None),
        key=lambda x: SENSORS.index(x[1]),
    )


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class BenchmarkFigures:
    def __init__(self, config, config_path):
        self.config, self.config_path = config, config_path.resolve()
        self.root = Path(config["pipeline_root"]).resolve()
        self.old = Path(config["previous_pipeline_root"]).resolve()
        self.out = Path(config["report_dir"]).resolve()
        self.figdir, self.datadir = self.out / "figures", self.out / "figure_source_data"
        for p in [self.out, self.figdir, self.datadir]:
            p.mkdir(parents=True, exist_ok=True)
        self.entries, self.sources, self.figures, self.skips = [], {}, [], []
        self.seed = config["representative_seed"]
        self.selections, self.training_rows, self.parameters = [], [], []
        apply_paper_style()
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 7,
                "axes.labelsize": 7,
                "axes.titlesize": 7,
                "legend.fontsize": 6,
                "xtick.labelsize": 6.5,
                "ytick.labelsize": 6.5,
                "axes.grid": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "savefig.facecolor": "white",
                "lines.linewidth": 1.0,
            }
        )

    def bind(self, path):
        p = Path(path).resolve()
        if str(p) not in self.sources:
            self.sources[str(p)] = {
                "path": p.as_posix(),
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
        return p

    def read_json(self, path):
        return json.loads(self.bind(path).read_text(encoding="utf-8"))

    def table(self, name, rows):
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        p = self.datadir / f"{name}.csv"
        frame.to_csv(p, index=False, encoding="utf-8-sig")
        return frame

    def panel(self, rows, cols, height=5.6):
        return plt.subplots(rows, cols, figsize=(7.2, height), squeeze=False)

    def save(self, fig, stem, title, caption, selection=None):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_items = [*fig.texts, *[t for legend in fig.legends for t in legend.get_texts()]]
        for ax in fig.axes:
            text_items.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
            for axis in [ax.xaxis, ax.yaxis]:
                lo, hi = sorted(axis.get_view_interval())
                for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                    if lo <= tick.get_loc() <= hi:
                        text_items.extend([tick.label1, tick.label2])
        outside = []
        for item in text_items:
            if item.get_visible() and item.get_text():
                box = item.get_window_extent(renderer)
                if (
                    box.x0 < -1
                    or box.y0 < -1
                    or box.x1 > fig.bbox.width + 1
                    or box.y1 > fig.bbox.height + 1
                ):
                    outside.append(item.get_text())
        if outside:
            raise ValueError(f"Figure text outside fixed canvas {stem}: {outside}")
        paths = {}
        for ext, dpi in [("svg", 300), ("pdf", 300), ("png", 300), ("tiff", 600)]:
            p = self.figdir / f"{stem}.{ext}"
            options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
            fig.savefig(p, dpi=dpi, bbox_inches=None, **options)
            paths[ext] = {"path": p.as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
        plt.close(fig)
        self.figures.append(
            {
                "stem": stem,
                "title": title,
                "caption": caption,
                "files": paths,
                "selection": selection,
                "source_data_dir": self.datadir.as_posix(),
            }
        )
        log("WRITES", stem)

    def load(self):
        state = self.read_json(self.root / "pipeline.json")
        if state["status"] != "complete" or len(state["runs"]) != 45:
            raise ValueError("Requires complete 45-run matrix")
        self.bind(self.root / "matrix_config.yaml")
        self.bind(self.config_path)
        self.bind(Path(__file__))
        self.bind(Path("src/virtual_imu/viz/style.py"))
        seen = set()
        for raw in state["runs"]:
            ident = (raw["candidate_id"], raw["task"], raw["seed"])
            if ident in seen or raw["status"] != "evaluated":
                raise ValueError("Duplicate/incomplete result identity")
            seen.add(ident)
            run = (
                (self.old if raw.get("reused") else self.root)
                / "experiments"
                / Path(raw["run_dir"].replace("\\", "/")).name
            )
            cfg = yaml.safe_load(self.bind(run / "config.yaml").read_text(encoding="utf-8"))
            metrics = self.read_json(run / "evaluation/metrics.json")
            history = pd.read_csv(self.bind(run / "metrics.csv"))
            for name in [
                "provenance.json",
                "source_bundle.json",
                "data_manifest.json",
                "evaluation/provenance.json",
            ]:
                p = run / name
                if p.is_file():
                    self.bind(p)
            if cfg["seed"] != raw["seed"]:
                raise ValueError("Run seed/config mismatch")
            self.entries.append(
                {**raw, "run": run, "config": cfg, "metrics": metrics, "history": history}
            )
        for task, expected in [("paired", 21), ("navigation", 24)]:
            if sum(e["task"] == task for e in self.entries) != expected:
                raise ValueError("Unexpected task coverage")

    def metric_frames(self):
        paired, nav = [], []
        classical_seen = set()
        for e in self.entries:
            for split, result in e["metrics"]["splits"].items():
                base = {
                    "split": split,
                    "seed": e["seed"],
                    "candidate_id": e["candidate_id"],
                    "run_dir": e["run"].as_posix(),
                }
                if e["task"] == "paired":
                    for pair in result["pairs"]:
                        for method, m in pair["metrics"].items():
                            if method != "model":
                                key = (
                                    e["task"],
                                    split,
                                    e["seed"],
                                    *(pair[k] for k in PAIR),
                                    method,
                                )
                                if key in classical_seen:
                                    continue
                                classical_seen.add(key)
                            paired.append(
                                {
                                    **base,
                                    **{k: pair[k] for k in PAIR},
                                    "method": e["candidate_id"] if method == "model" else method,
                                    **{k: m[k] for k in METRICS},
                                    "gyro_mean_axis_mae_radps": float(np.mean(m["axis_mae"][:3])),
                                    "accel_mean_axis_mae_mps2": float(np.mean(m["axis_mae"][3:])),
                                    "gyro_mean_residual_norm_radps": float(
                                        np.linalg.norm(m["axis_mean_error"][:3])
                                    ),
                                    "accel_mean_residual_norm_mps2": float(
                                        np.linalg.norm(m["axis_mean_error"][3:])
                                    ),
                                }
                            )
                else:
                    for method, m in result["metrics"].items():
                        if method != "model":
                            key = (e["task"], split, e["seed"], method)
                            if key in classical_seen:
                                continue
                            classical_seen.add(key)
                        nav.append(
                            {
                                **base,
                                "method": e["candidate_id"] if method == "model" else method,
                                **{
                                    k: m[k]
                                    for k in [
                                        "vector_rmse_m",
                                        "horizontal_rmse_m",
                                        "vertical_rmse_m",
                                    ]
                                },
                            }
                        )
        return self.table("fig_paired_physical", paired), self.table("fig_navigation_physical", nav)

    def dotplot(self, ax, frame, methods, metric, scale=1.0, labels=True):
        for y, method in enumerate(methods):
            values = finite(frame[frame.method.eq(method)][metric], method) * scale
            if len(values) != 3:
                raise ValueError(f"Need three seed slots: {method}")
            color = COLORS.get(method, "#74777c")
            ax.scatter(
                values,
                y + np.linspace(-0.09, 0.09, 3),
                s=10,
                facecolors="white",
                edgecolors=color,
                linewidth=0.65,
                zorder=3,
            )
            ax.errorbar(
                values.mean(),
                y,
                xerr=values.std(ddof=1),
                color=color,
                fmt="s",
                markersize=3.5,
                capsize=2,
                linewidth=0.8,
                zorder=4,
            )
        ax.set_yticks(range(len(methods)), [LABEL[m] if labels else "" for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        if len(methods) == 13:
            ax.axhline(5.5, color="#bfc3c7", linewidth=0.65)

    def primary(self, paired, nav):
        pairs = ordered_pairs(paired)
        methods = CLASSIC + MODELS[:-1]
        for metric, name, xlabel, scale in [
            (METRICS[0], "gyro", "Gyro vector RMSE [mrad/s]", 1000),
            (METRICS[1], "accel", "Accel vector RMSE [m/s²]", 1),
        ]:
            fig, axs = self.panel(2, 2, 6.45)
            for i, (ax, keys) in enumerate(zip(axs.flat, pairs, strict=True)):
                frame = paired[paired.split.eq("test")]
                for k, v in zip(PAIR, keys, strict=True):
                    frame = frame[frame[k].eq(v)]
                self.dotplot(ax, frame, methods, metric, scale)
                ax.set(xlabel=xlabel, title=f"{chr(97 + i)}  {keys[1]} → {keys[2]}")
            fig.suptitle(
                "Three optimization seeds: hollow points; mean ± sample SD: squares",
                fontsize=7,
                y=0.985,
            )
            fig.subplots_adjust(
                left=0.16, right=0.985, top=0.93, bottom=0.075, wspace=0.55, hspace=0.4
            )
            self.save(
                fig,
                f"fig_01_paired_{name}",
                f"四种器件完整{name}基线比较",
                "冻结测试集全部13种方法，LOW和所有拟合方法在相同时间点计分。空心点为三个优化种子，方块/误差线为均值±样本SD；经典方法重复种子槽位不构成独立实验。gyro为清晰显示换算至mrad/s，CSV仍用rad/s；accel为m/s²。下方七种神经方法包含论文骨干适配，不是原论文完整方法复现。",
            )
        fig, axs = self.panel(1, 3, 4.6)
        methods = ["zero", "train_mean", "ridge", *MODELS]
        for i, (ax, metric, label) in enumerate(
            zip(
                axs.flat,
                ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"],
                ["3D", "Horizontal", "Vertical"],
                strict=True,
            )
        ):
            self.dotplot(ax, nav[nav.split.eq("test")], methods, metric, labels=i == 0)
            ax.set(xlabel=f"{label} window RMSE [m]", title=f"{chr(97 + i)}  {label}")
            ax.axhline(2.5, color="#bfc3c7", linewidth=0.65)
        fig.suptitle(
            "SmartPNT Traj1 | 1.975 s endpoint displacement | n = 1,649 windows", fontsize=7, y=0.98
        )
        fig.subplots_adjust(left=0.19, right=0.985, top=0.9, bottom=0.14, wspace=0.25)
        self.save(
            fig,
            "fig_02_navigation",
            "SmartPNT全部方法窗口位移比较",
            "SmartPNT固定测试轨迹的1,649个重叠窗口，11种方法的三维、水平、垂直RMSE。每点为1.975s首末ENU位移误差，未累计为轨迹；不是ATE/RPE。空心点与误差线含义同配对比较。",
        )

    def training(self):
        groups = []
        for task in ["paired", "navigation"]:
            for method in MODELS:
                entries = sorted(
                    [e for e in self.entries if e["task"] == task and e["candidate_id"] == method],
                    key=lambda e: e["seed"],
                )
                if entries:
                    groups.append((task, method, entries))
        for task, method, entries in groups:
            for e in entries:
                h, key = e["history"], e["config"]["training"]["checkpoint_metric"]
                best = h.loc[h[key].idxmin()]
                self.training_rows.append(
                    {
                        "candidate_id": method,
                        "task": task,
                        "seed": e["seed"],
                        "epochs": len(h),
                        "best_epoch": int(best.epoch),
                        "selection_metric": key,
                        "best_validation_value": float(best[key]),
                        "run_dir": e["run"].as_posix(),
                    }
                )
        self.table("fig_training_summary", self.training_rows)
        for page, start in enumerate(range(0, len(groups), 6), 1):
            chunk = groups[start : start + 6]
            rows = (len(chunk) + 2) // 3
            for kind in ["loss", "validation"]:
                fig, axs = self.panel(rows, 3, 2.6 * rows + 0.35)
                for ax in axs.flat:
                    ax.set_visible(False)
                for idx, (ax, (task, method, entries)) in enumerate(
                    zip(axs.flat, chunk, strict=False)
                ):
                    ax.set_visible(True)
                    for seed_index, e in enumerate(entries):
                        h = e["history"]
                        color = ["#286994", "#a17843", "#6c8c50"][seed_index]
                        if kind == "loss":
                            ax.plot(h.epoch, h.train_loss, color=color, linewidth=0.75)
                            ax.plot(
                                h.epoch, h.val_loss, color=color, linestyle="--", linewidth=0.75
                            )
                        else:
                            key = e["config"]["training"]["checkpoint_metric"]
                            ax.plot(
                                h.epoch, h[key], color=color, label=str(e["seed"]), linewidth=0.9
                            )
                            best = h.loc[h[key].idxmin()]
                            ax.scatter([best.epoch], [best[key]], color=color, s=9, zorder=4)
                    ax.set(
                        title=f"{chr(97 + idx)}  {LABEL[method]} | {task}",
                        xlabel="Epoch",
                        ylabel="Normalized objective [1]"
                        if kind == "loss"
                        else ("Grouped score [1]" if task == "paired" else "Component RMSE [m]"),
                    )
                    if kind == "loss":
                        ax.set_yscale("log")
                handles = [
                    Line2D([], [], color=c, label=str(s))
                    for c, s in zip(
                        ["#286994", "#a17843", "#6c8c50"], self.config["seeds"], strict=True
                    )
                ]
                if kind == "loss":
                    handles += [
                        Line2D([], [], color="#333333", label="Train"),
                        Line2D([], [], color="#333333", linestyle="--", label="Validation"),
                    ]
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    ncol=len(handles),
                    bbox_to_anchor=(0.5, 0.997),
                    fontsize=6.2,
                    frameon=False,
                )
                fig.subplots_adjust(
                    left=0.12,
                    right=0.985,
                    bottom=0.21 if rows == 1 else 0.11,
                    top=0.89 if rows == 1 else 0.93,
                    wspace=0.55,
                    hspace=0.65,
                )
                self.save(
                    fig,
                    f"fig_03_training_{kind}_{page}",
                    f"全部运行训练诊断：{kind}第{page}页",
                    "逐轮展示每个预先固定模型、任务和全部三个种子，覆盖实际训练轮数，不外推早停后的曲线。训练/验证loss是归一化目标，不能当作SI单位误差；验证选择图的点表示仅按验证集选出的最优epoch。种子不是独立录制。",
                )

    def secondary_errors(self, paired):
        pairs = ordered_pairs(paired)
        methods = CLASSIC + MODELS[:-1]
        for stem, metrics, description in [
            (
                "fig_09_mean_axis_mae",
                ["gyro_mean_axis_mae_radps", "accel_mean_axis_mae_mps2"],
                "Mean axis MAE",
            ),
            (
                "fig_10_mean_residual_norm",
                ["gyro_mean_residual_norm_radps", "accel_mean_residual_norm_mps2"],
                "Mean-residual norm",
            ),
        ]:
            fig, axs = self.panel(2, 4, 6.45)
            for row, metric in enumerate(metrics):
                for col, keys in enumerate(pairs):
                    frame = paired[paired.split.eq("test")]
                    for k, v in zip(PAIR, keys, strict=True):
                        frame = frame[frame[k].eq(v)]
                    ax = axs[row, col]
                    self.dotplot(
                        ax, frame, methods, metric, scale=1000 if row == 0 else 1, labels=col == 0
                    )
                    ax.set_xlabel(
                        f"{'Gyro' if row == 0 else 'Accel'} [{'mrad/s' if row == 0 else 'm/s²'}]",
                        fontsize=6.2,
                    )
                    ax.tick_params(axis="x", labelrotation=30)
                    if row == 0:
                        ax.set_title(str(keys[1]).replace(" IMU", ""))
            fig.suptitle(
                description + " | three optimization seeds; mean ± sample SD", fontsize=7, y=0.985
            )
            fig.subplots_adjust(
                left=0.17, right=0.985, bottom=0.11, top=0.92, hspace=0.45, wspace=0.4
            )
            caption = (
                "每个方法逐轴MAE的三轴平均，先在每个种子内平均各轴绝对误差，再展示种子均值±样本SD；它不是三维误差向量范数的MAE。"
                if "mae" in stem
                else "每个种子测试区间六轴残差均值分别取gyro与accel三维范数，再展示种子均值±样本SD。这是经验平均残差的幅度，不能视为独立静态零偏或全部系统误差来源。"
            )
            self.save(
                fig,
                stem,
                "各器件完整方法MAE比较" if "mae" in stem else "各器件完整方法平均残差比较",
                caption
                + "上排gyro换算为mrad/s，下排accel为m/s²；两种物理单位不相加。空心点为三个优化种子，经典方法重复输出不构成独立重复。",
            )

    def improvements(self, paired):
        frame = paired[paired.split.eq("test")]
        pairlist = ordered_pairs(frame)
        records = []
        fig, axs = self.panel(2, 3, 5.6)
        for row, metric in enumerate(METRICS):
            for col, baseline in enumerate(["identity", "learned_fir_causal", "arx_linear"]):
                matrix = np.zeros((7, 4))
                for j, keys in enumerate(pairlist):
                    part = frame
                    for k, v in zip(PAIR, keys, strict=True):
                        part = part[part[k].eq(v)]
                    base = part[part.method.eq(baseline)].set_index("seed")[metric]
                    for i, method in enumerate(MODELS[:-1]):
                        values = 100 * (
                            1 - part[part.method.eq(method)].set_index("seed")[metric] / base
                        )
                        matrix[i, j] = values.mean()
                        records.append(
                            {
                                "candidate_id": method,
                                **dict(zip(PAIR, keys, strict=True)),
                                "metric": metric,
                                "baseline": baseline,
                                "improvement_percent_mean": values.mean(),
                                "improvement_percent_sample_SD": values.std(ddof=1),
                            }
                        )
                ax = axs[row, col]
                # Neutral cells preserve numbers; direction encoded by small text, avoiding saturation of very adverse ratios.
                ax.imshow(
                    np.clip(matrix, -100, 100),
                    cmap="PuOr_r",
                    vmin=-100,
                    vmax=100,
                    aspect="auto",
                    alpha=0.25,
                )
                for i in range(7):
                    for j in range(4):
                        ax.text(j, i, f"{matrix[i, j]:+.0f}", ha="center", va="center", fontsize=6)
                ax.set_yticks(
                    range(7),
                    [LABEL[m] + (" *" if m in CAUSAL else "") for m in MODELS[:-1]]
                    if col == 0
                    else [],
                )
                ax.set_xticks(range(4), ["BMX", "MPU", "LSM", "Tango"], rotation=35)
                ax.set(title=f"{'Gyro' if row == 0 else 'Accel'} vs {LABEL[baseline]}")
                ax.tick_params(length=0)
        fig.suptitle(
            "Relative RMSE improvement (%) | positive = lower error | * causal per sample",
            fontsize=7,
            y=0.985,
        )
        fig.subplots_adjust(left=0.17, right=0.985, top=0.92, bottom=0.1, wspace=0.16, hspace=0.45)
        self.table("fig_relative_improvements", records)
        self.save(
            fig,
            "fig_04_improvements",
            "模型相对不校正及强经典基线的改善",
            "每格为三种子相对RMSE改善百分比的平均，正值为误差下降，负值为退化；准确数值及样本SD保存在CSV。颜色只在±100%内饱和，文字保留超过该范围的实际值；不跨物理单位相加。星号只标记逐点因果模型；其余方法使用完整窗口。论文骨干适配成绩不等于原方法SOTA复现。",
        )

    def predictions(self, task):
        result = {}
        for e in self.entries:
            if e["task"] != task or e["seed"] != self.seed:
                continue
            p = self.bind(e["run"] / "evaluation/test_predictions.parquet")
            if task == "paired":
                columns = [
                    "sequence_id",
                    "timestamp_s",
                    *PAIR,
                    *[f"{a}_{s}" for a in AXES for s in ["low", "high", "virtual"]],
                ]
            else:
                columns = None
            result[e["candidate_id"]] = pd.read_parquet(p, columns=columns)
        return result

    def integral_summary(self):
        records = []
        for e in self.entries:
            if e["task"] != "paired":
                continue
            for sequence in e["metrics"]["splits"]["test"]["sequences"]:
                d = sequence["diagnostics"]
                records.append(
                    {
                        "candidate_id": e["candidate_id"],
                        "seed": e["seed"],
                        **{k: sequence[k] for k in PAIR},
                        "sequence_id": sequence["sequence_id"],
                        "integral_windows": d["integral_windows"],
                        "gyro_integral_mean_norm_rad": d["delta_gyro_integral_error_rad"],
                        "accel_integral_mean_norm_mps": d["delta_accel_integral_error_mps"],
                    }
                )
        sequences = self.table("fig_integral_sequence_diagnostics", records)
        pooled = []
        for keys, group in sequences.groupby(["candidate_id", "seed", *PAIR], sort=True):
            weights = group.integral_windows.to_numpy(float)
            pooled.append(
                {
                    "method": keys[0],
                    "seed": keys[1],
                    **dict(zip(PAIR, keys[2:], strict=True)),
                    "gyro_integral_mean_norm_rad": np.average(
                        group.gyro_integral_mean_norm_rad, weights=weights
                    ),
                    "accel_integral_mean_norm_mps": np.average(
                        group.accel_integral_mean_norm_mps, weights=weights
                    ),
                    "integral_windows": int(weights.sum()),
                }
            )
        frame = self.table("fig_integral_pooled_by_window_count", pooled)
        pairs = ordered_pairs(frame)
        fig, axs = self.panel(2, 4, 4.8)
        for row, (metric, label) in enumerate(
            [
                ("gyro_integral_mean_norm_rad", "Mean |∫gyro error| [rad]"),
                ("accel_integral_mean_norm_mps", "Mean |∫accel error| [m/s]"),
            ]
        ):
            for col, keys in enumerate(pairs):
                part = frame
                for k, v in zip(PAIR, keys, strict=True):
                    part = part[part[k].eq(v)]
                ax = axs[row, col]
                self.dotplot(ax, part, MODELS[:-1], metric, labels=col == 0)
                ax.set_xlabel(label, fontsize=6.4)
                if row == 0:
                    ax.set_title(str(keys[1]).replace(" IMU", ""))
                ax.tick_params(axis="x", labelrotation=30)
        fig.suptitle(
            "Nonoverlapping 80-point signal integrals | reset at each continuous segment",
            fontsize=7,
            y=0.985,
        )
        fig.subplots_adjust(left=0.17, right=0.985, top=0.9, bottom=0.13, hspace=0.55, wspace=0.4)
        self.save(
            fig,
            "fig_08_integral_diagnostics",
            "按连续片段重置的80点信号积分误差",
            "使用冻结评估中的逐连续片段诊断：对每个不重叠80点窗口的预测−HIGH信号作梯形积分，计算三轴积分向量范数，再按窗口数加权平均。窗口首末间隔1.975s，各片段独立重置。点为优化种子，方块及误差线为均值±样本SD。这是平均范数，不是RMSE、SO(3)预积分或轨迹误差；仅展示七个配对神经方法。",
        )

    def parameter_table(self):
        path = self.root.parent / "server_deployment_20261024/benchmark_gpu_preflight.json"
        report = self.read_json(path)
        records = [
            {"candidate_id": "tcn_original", "task": "paired", "parameters": 147782},
            {"candidate_id": "tcn_original", "task": "navigation", "parameters": 146435},
        ]
        records.extend(
            {
                "candidate_id": x["candidate_id"],
                "task": x["task"],
                "parameters": x["first"]["parameters"],
            }
            for x in report["cases"]
        )
        metadata = {(e["candidate_id"], e["task"]): e for e in self.entries}
        for row in records:
            entry = metadata[(row["candidate_id"], row["task"])]
            row.update(
                temporal_context=entry["temporal_context"],
                fidelity=entry["fidelity"],
                label=LABEL[row["candidate_id"]],
            )
        self.table("fig_parameters_and_context", records)

    def paired_diagnostics(self):
        frames = self.predictions("paired")
        base = frames["tcn_original"]
        selections, integral_rows = [], []
        for keys, pair in base.groupby(PAIR, sort=True):
            eligible = []
            for seq, group in pair.groupby("sequence_id", sort=True):
                group = group.sort_values("timestamp_s")
                t = group.timestamp_s.to_numpy(float)
                if len(group) < self.config["minimum_continuous_samples"] or not np.allclose(
                    np.diff(t), 0.025, atol=1e-6, rtol=1e-5
                ):
                    continue
                first = group[
                    group.timestamp_s <= t[0] + self.config["representative_seconds"] + 1e-8
                ]
                high = first[[f"{a}_high" for a in AXES]].to_numpy(float)
                if np.max(high[:, :3].std(axis=0)) > 1e-4 or np.max(high[:, 3:].std(axis=0)) > 1e-3:
                    eligible.append((str(seq), first))
            if not eligible:
                raise ValueError(f"No eligible continuous representative for {keys}")
            seq, part = sorted(eligible, key=lambda x: x[0])[0]
            t = part.timestamp_s.to_numpy(float) - part.timestamp_s.iloc[0]
            selection = {
                **dict(zip(PAIR, keys, strict=True)),
                "sequence_id": seq,
                "seed": self.seed,
                "start_s": float(part.timestamp_s.iloc[0]),
                "end_s": float(part.timestamp_s.iloc[-1]),
                "samples": len(part),
                "eligible_segments": len(eligible),
                "rule": self.config["selection_rule"],
            }
            selections.append(selection)
            signals = {
                "high": part[[f"{a}_high" for a in AXES]].to_numpy(float),
                "low": part[[f"{a}_low" for a in AXES]].to_numpy(float),
            }
            for method, frame in frames.items():
                other = frame[frame.sequence_id.eq(seq)].sort_values("timestamp_s")
                for k, v in zip(PAIR, keys, strict=True):
                    other = other[other[k].eq(v)]
                other = other[
                    other.timestamp_s.between(
                        part.timestamp_s.iloc[0] - 1e-8, part.timestamp_s.iloc[-1] + 1e-8
                    )
                ]
                if len(other) != len(part) or not np.allclose(other.timestamp_s, part.timestamp_s):
                    raise ValueError(
                        "Model predictions do not cover the identical selected timestamps"
                    )
                signals[method] = other[[f"{a}_virtual" for a in AXES]].to_numpy(float)
                if not np.allclose(other[[f"{a}_high" for a in AXES]], signals["high"]):
                    raise ValueError("Reference disagreement across model predictions")
            tag = re.sub(r"[^A-Za-z0-9]+", "_", str(keys[1])).strip("_")
            for subset, models in [
                ("causal", [m for m in MODELS[:-1] if m in CAUSAL]),
                ("window", [m for m in MODELS[:-1] if m not in CAUSAL]),
            ]:
                self.signal_figure(signals, t, models, tag, subset, selection)
            for kind in ["psd", "coherence", "integral"]:
                fig, axs = self.panel(3, 2, 6.45)
                source_rows = []
                nperseg = min(256, len(t))
                methods = ["high", "low", *MODELS[:-1]] if kind == "psd" else ["low", *MODELS[:-1]]
                for j, axis in enumerate(AXES):
                    ax = axs[j % 3, j // 3]
                    ref = signals["high"][:, j]
                    excited = np.std(ref) > (1e-4 if j < 3 else 1e-3)
                    for method in methods:
                        y = finite(signals[method][:, j], f"{tag}/{axis}/{method}")
                        color = (
                            HIGH if method == "high" else LOW if method == "low" else COLORS[method]
                        )
                        label = (
                            "High"
                            if method == "high"
                            else "Low"
                            if method == "low"
                            else LABEL[method]
                        )
                        if kind == "psd":
                            f, power = welch(
                                y,
                                fs=40,
                                nperseg=nperseg,
                                noverlap=nperseg // 2,
                                window="hann",
                                detrend="constant",
                            )
                            mask = (f > 0) & (power > 0)
                            ax.loglog(f[mask], power[mask], color=color, linewidth=0.8, label=label)
                            source_rows.extend(
                                {
                                    "axis": axis,
                                    "method": method,
                                    "frequency_hz": float(x),
                                    "PSD": float(v),
                                }
                                for x, v in zip(f, power, strict=True)
                            )
                        elif kind == "coherence":
                            if not excited:
                                continue
                            f, c = coherence(
                                y,
                                ref,
                                fs=40,
                                nperseg=nperseg,
                                noverlap=nperseg // 2,
                                window="hann",
                                detrend="constant",
                            )
                            mask = np.isfinite(c)
                            ax.plot(f[mask], c[mask], color=color, linewidth=0.8, label=label)
                            source_rows.extend(
                                {
                                    "axis": axis,
                                    "method": method,
                                    "frequency_hz": float(x),
                                    "coherence": float(v) if np.isfinite(v) else None,
                                }
                                for x, v in zip(f, c, strict=True)
                            )
                        else:
                            integral = cumulative_trapezoid(y - ref, x=t, initial=0)
                            ax.plot(t, integral, color=color, linewidth=0.8, label=label)
                            source_rows.extend(
                                {
                                    "axis": axis,
                                    "method": method,
                                    "time_s": float(x),
                                    "additive_integral": float(v),
                                }
                                for x, v in zip(t, integral, strict=True)
                            )
                            integral_rows.append(
                                {
                                    **selection,
                                    "method": method,
                                    "axis": axis,
                                    "unit": "rad" if j < 3 else "m/s",
                                    "endpoint": float(integral[-1]),
                                    "max_abs": float(np.abs(integral).max()),
                                }
                            )
                    if kind == "psd":
                        ax.set(
                            xlim=(40 / nperseg, 20),
                            ylabel=f"{axis} PSD [{'rad²/s²/Hz' if j < 3 else 'm²/s⁴/Hz'}]",
                        )
                    elif kind == "coherence":
                        ax.set(xlim=(0, 20), ylim=(0, 1.03), ylabel=f"{axis} coherence [1]")
                        if not excited:
                            ax.text(
                                0.5,
                                0.5,
                                "Insufficient excitation",
                                transform=ax.transAxes,
                                ha="center",
                                fontsize=6,
                            )
                            self.skips.append(f"{seq}/{axis}: reference STD too low for coherence")
                    else:
                        ax.set_ylabel(f"{axis} ∫error [{'rad' if j < 3 else 'm/s'}]")
                        ax.axhline(0, color="#c9c9c9", linewidth=0.65)
                    if j % 3 == 2:
                        ax.set_xlabel(
                            "Frequency [Hz]"
                            if kind in ["psd", "coherence"]
                            else "Time from segment start [s]"
                        )
                handles = [
                    Line2D(
                        [],
                        [],
                        color=HIGH if m == "high" else LOW if m == "low" else COLORS[m],
                        label="High" if m == "high" else "Low" if m == "low" else LABEL[m],
                    )
                    for m in methods
                ]
                fig.legend(
                    handles=handles,
                    loc="upper center",
                    ncol=3,
                    bbox_to_anchor=(0.5, 0.982),
                    frameon=False,
                )
                fig.subplots_adjust(
                    left=0.12, right=0.985, top=0.87, bottom=0.08, hspace=0.4, wspace=0.35
                )
                self.table(f"fig_{kind}_{tag}", source_rows)
                detail = {
                    "psd": f"Welch功率谱，40Hz、Hann窗、nperseg={nperseg}、50%重叠、常数去趋势。",
                    "coherence": "与HIGH的幅值平方相干性，参数同Welch。仅参考轴STD超过gyro1e-4rad/s或accel1e-3m/s²时绘制；不是去噪强弱的独立证明。",
                    "integral": "对每轴信号−HIGH作梯形积分，片段起点重置为零。gyro积分rad、accel积分m/s；没有SO(3)、重力补偿或完整导航积分。",
                }[kind]
                self.save(
                    fig,
                    f"fig_06_{kind}_{tag}",
                    f"{keys[1]}连续片段{kind}",
                    f"固定种子{self.seed}，{keys[1]}→{keys[2]}，按预定客观规则选段，首{t[-1]:.3f}s，不跨片段拼接。{detail}",
                    selection,
                )
        self.selections.extend(selections)
        self.table("fig_paired_representative_segments", selections)
        self.table("fig_paired_representative_integrals", integral_rows)

    def signal_figure(self, signals, t, models, tag, subset, selection):
        fig, axs = self.panel(3, 2, 6.45)
        rows = []
        for j, axis in enumerate(AXES):
            ax = axs[j % 3, j // 3]
            for method in ["high", "low", *models]:
                y = finite(signals[method][:, j], method)
                color = HIGH if method == "high" else LOW if method == "low" else COLORS[method]
                ax.plot(
                    t,
                    y,
                    color=color,
                    linewidth=0.8,
                    label="High"
                    if method == "high"
                    else "Low"
                    if method == "low"
                    else LABEL[method],
                )
                rows.extend(
                    {"time_s": float(x), "axis": axis, "method": method, "value": float(v)}
                    for x, v in zip(t, y, strict=True)
                )
            ax.set_ylabel(f"{axis} [{'rad/s' if j < 3 else 'm/s²'}]")
            if j % 3 == 2:
                ax.set_xlabel("Time from segment start [s]")
        fig.legend(
            *axs[0, 0].get_legend_handles_labels(),
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.985),
            frameon=False,
        )
        fig.subplots_adjust(left=0.11, right=0.985, top=0.88, bottom=0.08, hspace=0.4, wspace=0.32)
        self.table(f"fig_signal_{tag}_{subset}", rows)
        self.save(
            fig,
            f"fig_05_signal_{tag}_{subset}",
            f"{tag}六轴时序：{subset}",
            f"固定种子{self.seed}和预先规则选定的相同连续片段，与LOW和HIGH逐轴比较。{subset}页为{'逐点因果' if subset == 'causal' else '使用完整窗口'}模型；两页必须共同阅读，未按测试误差挑选模型或片段。坐标与物理单位保留，不跨片段拼接。",
            selection,
        )

    def navigation_diagnostics(self):
        frames = self.predictions("navigation")
        base = frames["tcn_original"]
        seq = sorted(base.sequence_id.astype(str).unique())[0]
        part = base[base.sequence_id.eq(seq)].sort_values("start_time_s")
        part = part[
            part.start_time_s
            <= part.start_time_s.iloc[0] + self.config["representative_seconds"] + 1e-8
        ]
        t = part.start_time_s.to_numpy(float) - part.start_time_s.iloc[0]
        selection = {
            "seed": self.seed,
            "sequence_id": seq,
            "windows": len(part),
            "rule": "lexicographically first test segment; first <=20 seconds of window starts; no model-error selection",
        }
        for subset, models in [
            ("causal", [m for m in MODELS if m in CAUSAL]),
            ("window", [m for m in MODELS if m not in CAUSAL]),
        ]:
            fig, axs = self.panel(3, 2, 6.45)
            data = []
            for j, axis in enumerate(["x", "y", "z"]):
                key = f"delta_{axis}_local_m"
                truth = part[f"true_{key}"].to_numpy(float)
                axs[j, 0].plot(t, truth, color=HIGH, linewidth=1, label="GT")
                for method in [*models, "zero", "train_mean", "ridge"]:
                    if method in MODELS:
                        other = frames[method]
                        other = other[other.sequence_id.eq(seq)].sort_values("start_time_s")
                        other = other[
                            other.start_time_s.between(
                                part.start_time_s.iloc[0] - 1e-8, part.start_time_s.iloc[-1] + 1e-8
                            )
                        ]
                        assert len(other) == len(part) and np.allclose(
                            other.start_time_s, part.start_time_s
                        )
                        pred = other[f"pred_{key}"].to_numpy(float)
                        color, style = COLORS[method], "-"
                        axs[j, 0].plot(
                            t,
                            pred,
                            color=color,
                            label=LABEL[method],
                            marker=".",
                            markersize=2,
                            linewidth=0.8,
                        )
                    else:
                        pred = part[f"{method}_{key}"].to_numpy(float)
                        color, style = (
                            {"zero": "#444444", "train_mean": "#aaaaaa", "ridge": "#7d8d97"}[
                                method
                            ],
                            ":",
                        )
                    axs[j, 1].plot(
                        t,
                        pred - truth,
                        color=color,
                        linestyle=style,
                        label=LABEL[method],
                        linewidth=0.8,
                    )
                    data.extend(
                        {
                            "axis": axis,
                            "method": method,
                            "window_start_relative_s": float(x),
                            "truth_m": float(y),
                            "pred_m": float(z),
                            "error_m": float(z - y),
                        }
                        for x, y, z in zip(t, truth, pred, strict=True)
                    )
                for ax, label in zip(axs[j], ["window delta", "prediction − GT"], strict=True):
                    ax.set_ylabel(f"{'ENU'[j]} {label} [m]")
                    if j == 2:
                        ax.set_xlabel("Window start from segment start [s]")
                axs[j, 1].axhline(0, color="#cccccc", linewidth=0.6)
            handles = (
                [Line2D([], [], color=HIGH, label="GT")]
                + [Line2D([], [], color=COLORS[m], label=LABEL[m]) for m in models]
                + [
                    Line2D(
                        [],
                        [],
                        color={"zero": "#444444", "train_mean": "#aaaaaa", "ridge": "#7d8d97"}[m],
                        linestyle=":",
                        label=LABEL[m],
                    )
                    for m in ["zero", "train_mean", "ridge"]
                ]
            )
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=4,
                bbox_to_anchor=(0.5, 0.985),
                fontsize=6,
                frameon=False,
            )
            fig.subplots_adjust(
                left=0.12, right=0.985, top=0.87, bottom=0.09, hspace=0.42, wspace=0.45
            )
            self.table(f"fig_navigation_windows_{subset}", data)
            self.save(
                fig,
                f"fig_07_navigation_windows_{subset}",
                f"SmartPNT ENU窗口预测与误差：{subset}",
                f"固定种子{self.seed}、字典序首个测试连续段{seq}起始20s内的窗口。左为GT与各模型1.975s端点位移，右为模型及zero/train-mean/ridge误差。每个点是一段独立计分窗口，不累加、不构建轨迹，不是ATE/RPE；两页覆盖全部八个神经模型。",
                selection,
            )
        self.selections.append(selection)

    def finish(self):
        provenance = self.out / "plotting_provenance"
        provenance.mkdir(exist_ok=True)
        for path in [Path(__file__), self.config_path]:
            shutil.copy2(path, provenance / path.name)
        payload = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "pipeline_root": self.root.as_posix(),
            "backend": "Python/matplotlib",
            "figure_contract": self.config["figure_contract"],
            "figures": self.figures,
            "sources": list(self.sources.values()),
            "representative_selections": self.selections,
            "skips": self.skips,
            "visual_review": "pending root inspection",
            "statistics": "three optimization seeds, mean and sample SD; no independent-recording CI",
        }
        (self.out / "plotting_figure_manifest.json").write_text(
            json.dumps(safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# 扩展 IMU 模型科研图索引",
            "",
            "Python/matplotlib；PNG300dpi、SVG可编辑文本、PDF嵌入字体、TIFF600dpi。三优化种子的SD不是独立录制置信区间。",
            "",
        ]
        for f in self.figures:
            lines += [
                f"## {f['title']}",
                "",
                f"`{f['stem']}`",
                "",
                f["caption"],
                "",
                f"![{f['title']}](figures/{f['stem']}.png)",
                "",
            ]
        (self.out / "figure_index.md").write_text("\n".join(lines), encoding="utf-8")
        log(
            "VERIFIED",
            f"{len(self.entries)} source runs, {len(self.figures)} figures exported; pending root visual QA",
        )

    def run(self):
        self.load()
        paired, nav = self.metric_frames()
        self.primary(paired, nav)
        self.secondary_errors(paired)
        self.training()
        self.improvements(paired)
        self.integral_summary()
        self.parameter_table()
        if self.config.get("summary_only"):
            self.finish()
            return
        self.paired_diagnostics()
        self.navigation_diagnostics()
        self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Export summary/curve figures without reading large prediction files",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["summary_only"] = args.summary_only
    log(
        "READS",
        f"{args.config}, completed frozen 45-run benchmark and six linked original TCN runs",
    )
    log(
        "WRITES",
        f"{config['report_dir']}/figures, figure_source_data, figure_index and plotting provenance",
    )
    log(
        "SKIPS",
        "training, inference, test-based selection, original artifact mutations, unit pooling and trajectory accumulation",
    )
    BenchmarkFigures(config, args.config).run()


if __name__ == "__main__":
    main()
