"""Traceable physical figures from an immutable completed formal six-run archive.

No inference, training or test selection. Outputs belong to the explicit report.
"""

# ruff: noqa: RUF001
# Chinese output captions intentionally use Chinese punctuation and mathematical minus.
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml
from scipy.integrate import cumulative_trapezoid
from scipy.signal import coherence, welch

from virtual_imu.viz.style import COLORS, new_figure, save_figure

PAIR = ["dataset_id", "low_sensor", "high_sensor"]
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
METHODS = [
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "model",
]
LABELS = ["Identity", "Bias mean", "Affine", "Ridge", "FIR", "ARX", "Model"]
NAV = ["zero", "train_mean", "ridge", "model"]
NAV_LABELS = ["Zero", "Train mean", "Ridge", "Model"]
PALETTE = ["#c95f3f", "#b79e78", "#7d8590", "#2a9d8f", "#8867a9", "#6e8496", "#1f77b4"]


def log(action, message):
    print(f"{action}: {message}", flush=True)


def finite(values, name):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.isfinite(a).all():
        raise ValueError(f"Nonfinite or empty plotted source: {name}")
    return a


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Figures:
    def __init__(self, root, report):
        self.root, self.report = root.resolve(), report.resolve()
        self.out = self.report / "figures"
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed = 20261010
        self.sources, self.figures, self.skips, self.tables, self.entries = {}, [], [], [], []
        self.summary = {
            "representative_seed": self.seed,
            "selection_rule": "fixed seed; lexicographically first eligible continuous test segment; first <=20 s; never select by test error",
            "scope": "Seed SD describes optimization randomness, not independent recordings. Gyro and accel SI units remain separate. Additive integrals are not SO(3); navigation windows are never accumulated into trajectories.",
        }

    def bind(self, path):
        path = Path(path).resolve()
        key = path.as_posix()
        if key not in self.sources:
            log("READS", path)
            self.sources[key] = {"path": key, "bytes": path.stat().st_size, "sha256": digest(path)}
        return path

    def read_json(self, path):
        return json.loads(self.bind(path).read_text(encoding="utf-8"))

    def write_json(self, path, value):
        log("WRITES", path)
        path.write_text(
            json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def table(self, stem, rows):
        path = self.report / f"{stem}.csv"
        log("WRITES", path)
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.tables.append(
            {"path": path.as_posix(), "rows": len(frame), "columns": list(frame.columns)}
        )
        return frame

    def skip(self, reason):
        log("SKIPS", reason)
        self.skips.append(reason)

    def panel(self, rows=1, cols=1, height=3.6):
        fig, ax = new_figure("double")
        ax.remove()
        fig.set_size_inches(7.2, height)
        return fig, fig.subplots(rows, cols, squeeze=False)

    def save(self, fig, stem, title, caption, sources, selection=None):
        for ax in fig.axes:
            for line in ax.lines:
                if len(line.get_xdata()) and len(line.get_ydata()):
                    finite(line.get_xdata(), stem + " x")
                    finite(line.get_ydata(), stem + " y")
        paths = save_figure(fig, self.out, stem)
        for path in paths:
            log("WRITES", path)
        bindings = [self.root / "pipeline.json", self.root / "matrix_config.yaml", *sources]
        # Every figure binds frozen run configuration, manifest snapshots and evaluation provenance.
        for e in self.entries:
            if any(str(Path(p).resolve()).startswith(str(e["run"])) for p in sources):
                bindings.extend(
                    [
                        e["config_path"],
                        e["run"] / "provenance.json",
                        e["run"] / "evaluation" / "provenance.json",
                    ]
                )
                bindings.extend(
                    p
                    for p in [
                        e["run"] / n
                        for n in (
                            "data_manifest.json",
                            "split_manifest.json",
                            "strict_manifest.json",
                        )
                    ]
                    if p.is_file()
                )
        files = sorted({self.bind(p).as_posix() for p in bindings})
        self.figures.append(
            {
                "stem": stem,
                "title": title,
                "caption": caption,
                "files": {p.suffix[1:]: p.as_posix() for p in paths},
                "source_files": files,
                "selection": selection,
                "finite_plot_data": True,
            }
        )

    def load(self):
        state = self.read_json(self.root / "pipeline.json")
        if state.get("status") != "complete":
            raise ValueError("Requires completed formal pipeline")
        self.bind(self.root / "matrix_config.yaml")
        self.summary.update(
            pipeline_status=state["status"],
            source_git_commit=state.get("git_commit"),
            pipeline=self.root.as_posix(),
        )
        for entry in state["runs"]:
            run = self.root / "experiments" / Path(str(entry["run_dir"]).replace("\\", "/")).name
            cfgpath = run / "config.yaml"
            cfg = yaml.safe_load(self.bind(cfgpath).read_text(encoding="utf-8"))
            history = pd.read_csv(self.bind(run / "metrics.csv"))
            metrics = self.read_json(run / "evaluation" / "metrics.json")
            provenance = self.read_json(run / "provenance.json")
            self.bind(run / "evaluation" / "provenance.json")
            self.entries.append(
                {
                    **entry,
                    "run": run,
                    "config_path": cfgpath,
                    "config": cfg,
                    "history": history,
                    "metrics": metrics,
                    "provenance": provenance,
                }
            )
        if len(self.entries) != 6 or any(
            len([e for e in self.entries if e["task"] == t]) != 3 for t in ["paired", "navigation"]
        ):
            raise ValueError("Expected six completed runs: three seeds per task")
        if len({e["seed"] for e in self.entries}) != 3:
            raise ValueError("Expected three seeds")
        self.summary["runs"] = [
            {
                "task": e["task"],
                "seed": e["seed"],
                "run": e["run"].as_posix(),
                "epochs": len(e["history"]),
                "config": e["config_path"].as_posix(),
                "checkpoint_metric": e["config"]["training"]["checkpoint_metric"],
                "source_git_commit": e["provenance"].get("git_commit"),
            }
            for e in self.entries
        ]

    def training(self):
        summary = []
        for task in ["paired", "navigation"]:
            entries = sorted(
                [e for e in self.entries if e["task"] == task], key=lambda e: e["seed"]
            )
            sources = [p for e in entries for p in [e["run"] / "metrics.csv", e["config_path"]]]
            task_cn = "配对信号" if task == "paired" else "导航位移"
            fig, axes = self.panel(1, 3, 2.8)
            for ax, e in zip(axes.flat, entries, strict=True):
                h = e["history"]
                finite(h[["epoch", "train_loss", "val_loss"]], task + " losses")
                ax.plot(h.epoch, h.train_loss, color=COLORS["virtual"], label="Train")
                ax.plot(h.epoch, h.val_loss, color=COLORS["low"], label="Validation")
                ax.set(
                    xlabel="Epoch",
                    ylabel="Normalized objective [1]",
                    title=f"Seed {e['seed']}",
                    yscale="log",
                )
                ax.legend(loc="best")
                key = e["config"]["training"]["checkpoint_metric"]
                finite(h[key], task + " selection")
                best = h.loc[h[key].idxmin()]
                summary.append(
                    {
                        "task": task,
                        "seed": e["seed"],
                        "epochs": len(h),
                        "best_epoch": int(best.epoch),
                        "selection_metric": key,
                        "best_validation_value": float(best[key]),
                        "final_train_loss": float(h.train_loss.iloc[-1]),
                        "final_val_loss": float(h.val_loss.iloc[-1]),
                    }
                )
            fig.tight_layout()
            self.save(
                fig,
                f"{task}_training_loss",
                f"{task_cn}任务三种子训练与验证损失",
                f"{task_cn}任务全部三个优化种子的逐轮训练/验证目标。损失来自训练归一化空间，采用对数纵轴；曲线覆盖实际完成轮次，不延长早停后的曲线。损失不代表SI物理误差。",
                sources,
            )
            fig, axes = self.panel(1, 3, 2.8)
            for ax, e in zip(axes.flat, entries, strict=True):
                h = e["history"]
                if task == "paired":
                    cols = [
                        ("val_grouped_score", "Selection score", COLORS["high"]),
                        ("val_relative_gyro_rmse", "Gyro ratio", COLORS["virtual"]),
                        ("val_relative_accel_rmse", "Accel ratio", COLORS["low"]),
                    ]
                    ylabel = "Validation / identity RMSE [1]"
                else:
                    cols = [
                        ("val_rmse", "Component RMSE", COLORS["virtual"]),
                        ("val_vector_rmse", "Vector RMSE", COLORS["low"]),
                    ]
                    ylabel = "Validation displacement RMSE [m]"
                for col, label, color in cols:
                    finite(h[col], col)
                    ax.plot(h.epoch, h[col], color=color, label=label)
                key = e["config"]["training"]["checkpoint_metric"]
                best = h.loc[h[key].idxmin()]
                ax.axvline(best.epoch, color=COLORS["grid"], linestyle="--", linewidth=1)
                ax.set(
                    xlabel="Epoch",
                    ylabel=ylabel,
                    title=f"Seed {e['seed']} | epoch {int(best.epoch)}",
                )
                if task == "paired":
                    ax.axhline(1, color=COLORS["grid"], linestyle=":", linewidth=1)
                ax.legend(loc="best", fontsize=6)
            fig.tight_layout()
            caption = (
                "配对任务验证指标：按映射计算模型/identity陀螺与加速度向量RMSE比值，再按冻结协议分组平均，均为无量纲；低于1表示误差下降。虚线为仅按验证集选择的轮次。"
                if task == "paired"
                else "三个种子的验证集局部ENU端点位移分量RMSE与向量RMSE，单位m；虚线为验证集选出的轮次。val_rmse为分量RMSE，val_vector_rmse为向量RMSE；80点/40Hz窗口跨度1.975s，不累加为轨迹。"
            )
            self.save(
                fig,
                "paired_validation_score" if task == "paired" else "navigation_validation_rmse",
                f"{task_cn}任务验证选择指标",
                caption,
                sources,
            )
        self.summary["training"] = self.table("training_curve_summary", summary).to_dict("records")

    def metric_tables(self):
        prows, arows, nrows, narows = [], [], [], []
        for e in self.entries:
            for split, result in e["metrics"]["splits"].items():
                base = {
                    "seed": e["seed"],
                    "split": split,
                    "metrics_source": (e["run"] / "evaluation" / "metrics.json").as_posix(),
                }
                if e["task"] == "paired":
                    for p in result["pairs"]:
                        identity = {k: p[k] for k in PAIR}
                        for method, m in p["metrics"].items():
                            finite(
                                [m["gyro_vector_rmse_radps"], m["accel_vector_rmse_mps2"]],
                                "paired metrics",
                            )
                            prows.append(
                                {
                                    **base,
                                    **identity,
                                    "method": method,
                                    "samples": m["samples"],
                                    "gyro_vector_rmse_radps": m["gyro_vector_rmse_radps"],
                                    "accel_vector_rmse_mps2": m["accel_vector_rmse_mps2"],
                                }
                            )
                            for j, axis in enumerate(AXES):
                                arows.append(
                                    {
                                        **base,
                                        **identity,
                                        "method": method,
                                        "axis": axis,
                                        "unit": "rad/s" if j < 3 else "m/s^2",
                                        "rmse": m["axis_rmse"][j],
                                        "mae": m["axis_mae"][j],
                                        "bias": m["axis_mean_error"][j],
                                        "error_std": m["axis_error_std"][j],
                                    }
                                )
                else:
                    for method, m in result["metrics"].items():
                        nrows.append(
                            {
                                **base,
                                "method": method,
                                **{k: v for k, v in m.items() if not isinstance(v, list)},
                            }
                        )
                        for j, axis in enumerate(["east", "north", "up"]):
                            narows.append(
                                {
                                    **base,
                                    "method": method,
                                    "axis": axis,
                                    "unit": "m",
                                    "rmse": m["axis_rmse_m"][j],
                                    "mae": m["axis_mae_m"][j],
                                    "bias": m["axis_bias_m"][j],
                                }
                            )
        paired = self.table("paired_physical_all_seeds", prows)
        pa = self.table("paired_axis_rmse_mae_bias_all_seeds", arows)
        nav = self.table("navigation_physical_all_seeds", nrows)
        na = self.table("navigation_axis_rmse_mae_bias_all_seeds", narows)
        for name, frame, groups, metrics in [
            (
                "paired_physical_summary",
                paired,
                ["split", *PAIR, "method"],
                ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"],
            ),
            (
                "paired_axis_summary",
                pa,
                ["split", *PAIR, "method", "axis", "unit"],
                ["rmse", "mae", "bias"],
            ),
            (
                "navigation_physical_summary",
                nav,
                ["split", "method"],
                ["vector_rmse_m", "component_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"],
            ),
            (
                "navigation_axis_summary",
                na,
                ["split", "method", "axis", "unit"],
                ["rmse", "mae", "bias"],
            ),
        ]:
            # A grouping key named axis collides with GroupBy.axis in newer pandas;
            # named aggregation avoids probing that attribute on a selected group.
            out = frame.groupby(groups, sort=True, dropna=False).agg(
                {m: ["mean", "std", "count"] for m in metrics}
            )
            out.columns = [f"{col}_{stat}" for col, stat in out.columns]
            self.summary[name] = self.table(name, out.reset_index()).to_dict("records")
        changes = []
        for keys, group in paired[paired.split.eq("test")].groupby(PAIR, sort=True):
            for metric in ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]:
                by = group.pivot(index="seed", columns="method", values=metric)
                for baseline in METHODS[:-1]:
                    delta = 100 * (1 - by.model / by[baseline])
                    changes.append(
                        {
                            **dict(zip(PAIR, keys, strict=True)),
                            "metric": metric,
                            "baseline": baseline,
                            "improvement_percent_mean": delta.mean(),
                            "improvement_percent_sample_sd": delta.std(ddof=1),
                            "per_seed": delta.to_dict(),
                        }
                    )
        self.summary["paired_test_changes"] = changes
        return paired, nav

    def metric_bars(self, paired, nav):
        groups = list(paired[paired.split.eq("test")].groupby(PAIR, sort=True))
        if len(groups) != 4:
            raise ValueError("Expected four device mappings")
        for metric, sensor, unit, cn in [
            ("gyro_vector_rmse_radps", "gyro", "rad/s", "陀螺仪"),
            ("accel_vector_rmse_mps2", "accel", "m/s^2", "加速度计"),
        ]:
            fig, axes = self.panel(2, 2, 5)
            for ax, (keys, group) in zip(axes.flat, groups, strict=True):
                values = [
                    finite(group[group.method.eq(m)].sort_values("seed")[metric], metric)
                    for m in METHODS
                ]
                if any(len(a) != 3 for a in values):
                    raise ValueError("Expected 3 seeds per mapping/method")
                ax.bar(
                    range(7),
                    [a.mean() for a in values],
                    yerr=[a.std(ddof=1) for a in values],
                    color=PALETTE,
                    width=0.7,
                    capsize=2,
                    error_kw={"linewidth": 0.8},
                )
                for j, a in enumerate(values):
                    ax.scatter(
                        j + np.linspace(-0.12, 0.12, 3), a, color=COLORS["high"], s=6, zorder=4
                    )
                ax.set_xticks(range(7), LABELS, rotation=30, ha="right")
                ax.set_ylabel(f"Vector RMSE [{unit}]")
                ax.set_title(f"{keys[0]} | {keys[1]} -> {keys[2]}", fontsize=8)
                ax.set_ylim(bottom=0)
            fig.tight_layout()
            self.save(
                fig,
                f"paired_test_{sensor}_baselines",
                f"配对测试集{cn}向量误差及全部基线",
                f"四种映射测试集{cn}向量RMSE（{unit}）。柱为三个优化种子均值，误差棒为样本标准差（ddof=1），散点为单种子。Identity、均值偏置、仿射、岭回归、FIR和ARX均遵循冻结训练集拟合协议，在同一覆盖时间戳评分。种子差异不代表独立采集重复。",
                [
                    e["run"] / "evaluation" / "metrics.json"
                    for e in self.entries
                    if e["task"] == "paired"
                ],
            )
        fig, axes = self.panel(1, 3, 2.9)
        for ax, metric, label in zip(
            axes.flat,
            ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"],
            ["Vector RMSE [m]", "Horizontal RMSE [m]", "Vertical RMSE [m]"],
            strict=True,
        ):
            values = [
                finite(nav[nav.split.eq("test") & nav.method.eq(m)][metric], metric) for m in NAV
            ]
            ax.bar(
                range(4),
                [a.mean() for a in values],
                yerr=[a.std(ddof=1) for a in values],
                capsize=2,
                color=[PALETTE[i] for i in [0, 1, 3, 6]],
            )
            for j, a in enumerate(values):
                ax.scatter(
                    j + np.linspace(-0.12, 0.12, len(a)), a, s=8, color=COLORS["high"], zorder=4
                )
            ax.set_xticks(range(4), NAV_LABELS, rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        self.save(
            fig,
            "navigation_test_baselines",
            "导航测试集局部ENU窗口位移误差",
            "测试集重叠窗口局部ENU端点位移向量、水平与垂直RMSE，单位m。柱为三个种子均值，误差棒为样本标准差，点为单种子；比较零位移、训练目标均值、低IMU摘要岭回归与模型。窗口跨度1.975s，未累加为轨迹，不是全轨迹ATE/RPE。",
            [
                e["run"] / "evaluation" / "metrics.json"
                for e in self.entries
                if e["task"] == "navigation"
            ],
        )

    def representative_paired(self):
        e = next(e for e in self.entries if e["task"] == "paired" and e["seed"] == self.seed)
        path = e["run"] / "evaluation" / "test_predictions.parquet"
        frame = pd.read_parquet(self.bind(path))
        required = [
            "sequence_id",
            "timestamp_s",
            *PAIR,
            *[f"{a}_{s}" for a in AXES for s in ["low", "high", "virtual"]],
        ]
        if set(required) - set(frame.columns):
            raise ValueError("Missing paired prediction fields")
        finite(
            frame[["timestamp_s", *[f"{a}_{s}" for a in AXES for s in ["low", "high", "virtual"]]]],
            "paired predictions",
        )
        selections, integrals = [], []
        for keys, pair in frame.groupby(PAIR, sort=True):
            candidates = []
            for seq, group in pair.groupby("sequence_id", sort=True):
                group = group.sort_values("timestamp_s").reset_index(drop=True)
                t = group.timestamp_s.to_numpy(float)
                if len(t) < 160 or not np.allclose(np.diff(t), 0.025, atol=1e-6, rtol=1e-5):
                    self.skip(f"Representative {seq}: short or discontinuous")
                    continue
                first = group[group.timestamp_s <= t[0] + 20 + 1e-8].copy()
                high = first[[f"{a}_high" for a in AXES]].to_numpy(float)
                if (
                    np.max(high[:, :3].std(axis=0)) <= 1e-4
                    and np.max(high[:, 3:].std(axis=0)) <= 1e-3
                ):
                    self.skip(f"Representative {seq}: low excitation in first 20 s")
                    continue
                candidates.append((str(seq), first))
            if not candidates:
                raise ValueError(f"No continuous excited representative for {keys}")
            seq, part = sorted(candidates, key=lambda p: p[0])[0]
            t = part.timestamp_s.to_numpy(float) - part.timestamp_s.iloc[0]
            short = re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(map(str, keys))).strip("_")
            selection = {
                **dict(zip(PAIR, keys, strict=True)),
                "seed": self.seed,
                "sequence_id": seq,
                "start_time_s": float(part.timestamp_s.iloc[0]),
                "end_time_s": float(part.timestamp_s.iloc[-1]),
                "samples": len(part),
                "fs_hz": 40,
                "eligible_segments": len(candidates),
                "rule": self.summary["selection_rule"],
            }
            selections.append(selection)
            title = f"Seed {self.seed} | {keys[0]} | {keys[1]} -> {keys[2]}"
            caption = f"固定种子{self.seed}，{keys[0]}的{keys[1]}→{keys[2]}映射，连续测试段{seq}的起始{t[-1]:.3f}s。按字典序及连续性/最低激励要求确定段，不按测试误差挑选。"
            sources = [path, e["config_path"]]
            for kind in ["signal", "psd", "coherence", "integral"]:
                fig, axes = self.panel(3, 2, 6.45)
                diagnostic_rows = []
                nperseg = min(256, len(part))
                for j, a in enumerate(AXES):
                    ax = axes[j % 3, j // 3]
                    unit = "rad/s" if j < 3 else "m/s^2"
                    signals = [
                        ("high", "High reference", COLORS["high"]),
                        ("low", "Low / identity", COLORS["low"]),
                        ("virtual", "Virtual", COLORS["virtual"]),
                    ]
                    if kind in ["coherence", "integral"]:
                        signals = signals[1:]
                    for suffix, label, color in signals:
                        y = part[f"{a}_{suffix}"].to_numpy(float)
                        ref = part[f"{a}_high"].to_numpy(float)
                        if kind == "signal":
                            ax.plot(t, y, color=color, label=label, linewidth=1)
                        elif kind == "psd":
                            f, power = welch(
                                y,
                                fs=40.0,
                                nperseg=nperseg,
                                noverlap=nperseg // 2,
                                detrend="constant",
                                window="hann",
                            )
                            mask = (f > 0) & (power > 0)
                            finite(power[mask], "Welch PSD")
                            ax.loglog(f[mask], power[mask], color=color, label=label, linewidth=1)
                            diagnostic_rows.extend(
                                {"axis": a, "signal": suffix, "frequency_hz": x, "psd": v}
                                for x, v in zip(f, power, strict=True)
                            )
                        elif kind == "coherence":
                            f, c = coherence(
                                y,
                                ref,
                                fs=40.0,
                                nperseg=nperseg,
                                noverlap=nperseg // 2,
                                detrend="constant",
                                window="hann",
                            )
                            valid = np.isfinite(c)
                            if not valid.all():
                                self.skip(
                                    f"Undefined zero-power coherence bins: {seq} {a} {suffix} {(~valid).sum()}"
                                )
                            ax.plot(
                                f[valid],
                                c[valid],
                                color=color,
                                label=label + " vs high",
                                linewidth=1,
                            )
                            diagnostic_rows.extend(
                                {
                                    "axis": a,
                                    "signal": suffix,
                                    "frequency_hz": x,
                                    "coherence": v if np.isfinite(v) else None,
                                }
                                for x, v in zip(f, c, strict=True)
                            )
                        else:
                            yint = finite(
                                cumulative_trapezoid(y - ref, x=t, initial=0), "additive integral"
                            )
                            ax.plot(t, yint, color=color, label=label + " - high", linewidth=1)
                            integrals.append(
                                {
                                    **selection,
                                    "axis": a,
                                    "signal": suffix,
                                    "endpoint_additive_integral": yint[-1],
                                    "integral_unit": "rad" if j < 3 else "m/s",
                                    "max_abs_additive_integral": np.abs(yint).max(),
                                    "note": "segment-start reset; additive integral, not SO(3) or trajectory",
                                }
                            )
                    if kind == "signal":
                        ax.set_ylabel(f"{a} [{unit}]")
                    elif kind == "psd":
                        ax.set_ylabel(
                            f"{a} PSD [{'(rad/s)^2/Hz' if j < 3 else '(m/s^2)^2/Hz'}]", fontsize=7
                        )
                        ax.set_xlim(40 / nperseg, 20)
                    elif kind == "coherence":
                        ax.set(ylabel=f"{a} coherence [1]", xlim=(0, 20), ylim=(0, 1.03))
                    else:
                        ax.set_ylabel(
                            f"{a} additive integral [{'rad' if j < 3 else 'm/s'}]", fontsize=7
                        )
                        ax.axhline(0, color=COLORS["grid"], linestyle=":", linewidth=0.8)
                    if j % 3 == 2:
                        ax.set_xlabel(
                            "Frequency [Hz]"
                            if kind in ["psd", "coherence"]
                            else "Segment-relative time [s]"
                        )
                fig.legend(
                    *axes[0, 0].get_legend_handles_labels(),
                    loc="upper center",
                    ncol=3 if kind in ["signal", "psd"] else 2,
                    bbox_to_anchor=(0.5, 0.965),
                    frameon=False,
                )
                fig.suptitle(
                    title + (" | additive only; segment reset" if kind == "integral" else ""),
                    fontsize=8,
                    y=1,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                more = {
                    "signal": "六轴LOW、HIGH参考与VIRTUAL信号按SI单位比较。",
                    "psd": f"单一连续段内Welch功率谱：40Hz、Hann窗、nperseg={nperseg}、50%重叠、常数去趋势，上限20Hz；不拼接采集段。",
                    "coherence": f"LOW/HIGH和VIRTUAL/HIGH六轴相干性：40Hz、Hann窗、nperseg={nperseg}，仅在同一连续段估计。",
                    "integral": "对LOW−HIGH与VIRTUAL−HIGH误差作梯形积分，每段起点重置为0；陀螺积分rad，加速度积分m/s。仅为加性信号积分，无SO(3)预积分、姿态/重力补偿或轨迹导航积分。",
                }[kind]
                cn = {
                    "signal": "六轴信号",
                    "psd": "六轴功率谱",
                    "coherence": "六轴相干性",
                    "integral": "段内加性积分残差",
                }[kind]
                self.save(
                    fig,
                    f"paired_{kind}_{short}",
                    f"{keys[1]}至{keys[2]}{cn}（种子{self.seed}）",
                    caption + more,
                    sources,
                    selection,
                )
                if diagnostic_rows:
                    self.table(f"paired_{kind}_{short}", diagnostic_rows)
        self.table("paired_representative_integral_summary", integrals)
        self.summary["paired_representative_segments"] = selections
        self.summary["paired_representative_additive_integrals"] = integrals

    def representative_navigation(self):
        e = next(e for e in self.entries if e["task"] == "navigation" and e["seed"] == self.seed)
        path = e["run"] / "evaluation" / "test_predictions.parquet"
        frame = pd.read_parquet(self.bind(path))
        delta = ["delta_x_local_m", "delta_y_local_m", "delta_z_local_m"]
        true_cols, pred_cols = [f"true_{d}" for d in delta], [f"pred_{d}" for d in delta]
        finite(frame[[*true_cols, *pred_cols]], "navigation predictions")
        seq = sorted(frame.sequence_id.astype(str).unique())[0]
        group = (
            frame[frame.sequence_id.astype(str).eq(seq)]
            .sort_values("start_time_s")
            .reset_index(drop=True)
        )
        part = group[group.start_time_s <= group.start_time_s.iloc[0] + 20 + 1e-8]
        t = part.start_time_s.to_numpy(float) - part.start_time_s.iloc[0]
        selection = {
            "seed": self.seed,
            "sequence_id": seq,
            "start_time_s": float(part.start_time_s.iloc[0]),
            "end_time_s": float(part.end_time_s.iloc[-1]),
            "windows": len(part),
            "rule": self.summary["selection_rule"],
        }
        fig, axes = self.panel(3, 2, 6.1)
        for j, axis in enumerate(["East", "North", "Up"]):
            true, pred = part[true_cols[j]].to_numpy(float), part[pred_cols[j]].to_numpy(float)
            axes[j, 0].plot(
                t, true, color=COLORS["high"], label="GT window delta", marker=".", markersize=3
            )
            axes[j, 0].plot(
                t,
                pred,
                color=COLORS["virtual"],
                label="Model window delta",
                marker=".",
                markersize=3,
            )
            axes[j, 0].set_ylabel(f"{axis} delta [m]")
            for method, label, color in [
                ("zero", "Zero", PALETTE[0]),
                ("train_mean", "Train mean", PALETTE[1]),
                ("ridge", "Ridge", PALETTE[3]),
                ("pred", "Model", PALETTE[6]),
            ]:
                err = finite(
                    part[f"{method}_{delta[j]}"].to_numpy(float) - true, "navigation error"
                )
                axes[j, 1].plot(t, err, color=color, label=label, linewidth=1)
            axes[j, 1].set_ylabel(f"{axis} prediction - GT [m]")
            axes[j, 1].axhline(0, color=COLORS["grid"], linestyle=":", linewidth=0.8)
            if j == 2:
                for ax in axes[j]:
                    ax.set_xlabel("Window-start time from segment start [s]")
        axes[0, 0].legend(loc="best", fontsize=6)
        axes[0, 1].legend(loc="best", fontsize=6)
        fig.suptitle(
            f"Seed {self.seed} | first test segment | 1.975 s overlapping window deltas", fontsize=9
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.save(
            fig,
            "navigation_test_windows",
            f"导航ENU窗口位移与误差（种子{self.seed}）",
            f"固定种子{self.seed}首个字典序连续测试段{seq}，展示起始20s内窗口的ENU端点位移GT/模型值与各基线误差。每点为独立计分的1.975s端点位移，窗口步长1s；横轴在该段重置，绝不把重叠位移累加为轨迹。",
            [path, e["config_path"]],
            selection,
        )
        fig, axes = self.panel(1, 3, 2.9)
        spread = []
        for j, (ax, axis) in enumerate(zip(axes.flat, ["East", "North", "Up"], strict=True)):
            true, pred = frame[true_cols[j]].to_numpy(float), frame[pred_cols[j]].to_numpy(float)
            ax.scatter(true, pred, s=3, color=COLORS["virtual"], rasterized=True)
            low, high = min(true.min(), pred.min()), max(true.max(), pred.max())
            ax.plot([low, high], [low, high], color=COLORS["high"], linestyle="--", linewidth=0.8)
            ax.set(xlabel=f"GT {axis} window delta [m]", ylabel=f"Model {axis} window delta [m]")
            ax.set_aspect("equal", adjustable="box")
            spread.append(
                {
                    "axis": axis.lower(),
                    "windows": len(true),
                    "true_std_m": true.std(),
                    "pred_std_m": pred.std(),
                    "pred_to_true_std_ratio": pred.std() / max(true.std(), 1e-12),
                    "correlation": np.corrcoef(true, pred)[0, 1]
                    if min(true.std(), pred.std()) > 0
                    else None,
                }
            )
        fig.suptitle(f"Seed {self.seed} | all test windows; no trajectory accumulation", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        self.save(
            fig,
            "navigation_test_scatter",
            f"全部测试窗口ENU预测与GT散点（种子{self.seed}）",
            f"固定种子{self.seed}全部测试窗口ENU端点位移预测与GT；虚线为一致线，横纵尺度相同。各窗口保留独立含义，未构建轨迹；用于描述预测范围、偏置与变化幅度。",
            [path, e["config_path"]],
            {
                "seed": self.seed,
                "windows": len(frame),
                "rule": "all held-out windows of predeclared seed",
            },
        )
        self.summary["navigation_representative_segment"] = selection
        self.summary["navigation_test_prediction_spread"] = spread
        self.table("navigation_test_prediction_spread", spread)

    def finish(self):
        self.summary.update(
            figure_count=len(self.figures),
            image_file_count=3 * len(self.figures),
            tables=self.tables,
        )
        self.write_json(self.report / "figures_analysis.json", self.summary)
        index = [
            "# 完成实验的图表索引",
            "",
            f"共{len(self.figures)}幅图；每幅导出PNG（300dpi）、SVG与PDF。",
            "",
            "固定代表种子20261010，未按测试误差选择。汇总用三个优化种子；样本SD（ddof=1）只描述优化随机性，不代表独立采集不确定度。",
            "",
        ]
        for i, f in enumerate(self.figures, 1):
            index.extend(
                [
                    f"## 图{i}：{f['title']}",
                    "",
                    f"文件：`{f['stem']}.png/.svg/.pdf`",
                    "",
                    f"![{f['title']}]({f['stem']}.png)",
                    "",
                    f["caption"],
                    "",
                ]
            )
        index.extend(["## 数据表", ""])
        index.extend(f"- `{Path(t['path']).name}`：{t['rows']}行。" for t in self.tables)
        index.extend(
            [
                "",
                "## 追溯与限制",
                "",
                "源文件SHA256、冻结配置、manifest、源pipeline与git身份见`../figures_manifest.json`；分析汇总见`../figures_analysis.json`。",
                "",
                self.summary["scope"],
                "",
            ]
        )
        path = self.out / "figure_index.md"
        log("WRITES", path)
        path.write_text("\n".join(index), encoding="utf-8")
        self.bind(Path(__file__))
        self.bind(Path(__file__).resolve().parents[1] / "src" / "virtual_imu" / "viz" / "style.py")
        for f in self.figures:
            for p in f["files"].values():
                if not Path(p).is_file() or Path(p).stat().st_size <= 0:
                    raise ValueError(f"Missing or empty output {p}")
        self.write_json(
            self.report / "figures_manifest.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_pipeline": self.root.as_posix(),
                "source_git_commit": self.summary.get("source_git_commit"),
                "representative_seed": self.seed,
                "source_files": list(self.sources.values()),
                "figures": self.figures,
                "tables": self.tables,
                "skips": self.skips,
                "verification": {
                    "finite_plotted_source_arrays_and_line_coordinates": True,
                    "figure_count": len(self.figures),
                    "expected_image_files": len(self.figures) * 3,
                },
                "scope": self.summary["scope"],
            },
        )
        log(
            "SKIPS",
            "legacy normalized physical reports, outputs/latest, source edits, checkpoint inference, test-fit transformations, cross-segment spectra, full-trajectory accumulation",
        )
        log(
            "VERIFIED",
            f"{len(self.figures)} figures, {3 * len(self.figures)} nonempty image files, {len(self.tables)} CSV tables; finite plotted sources",
        )

    def run(self):
        self.load()
        self.training()
        paired, nav = self.metric_tables()
        self.metric_bars(paired, nav)
        self.representative_paired()
        self.representative_navigation()
        self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    Figures(args.pipeline_root, args.report_dir).run()


if __name__ == "__main__":
    main()
