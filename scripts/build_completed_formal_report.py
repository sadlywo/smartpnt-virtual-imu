"""Build the completed formal experiment report from frozen evaluation artifacts.

This script performs no training, inference, data repair, alignment or model selection.
The shared block representation produces portable Markdown, HTML and a CJK PDF.
"""

# ruff: noqa: RUF001  # Chinese punctuation is intentional in the Chinese report.
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
from statistics import mean, stdev

TITLE = "统一 IMU 正式训练完整报告"
SEEDS = [20261010, 20261011, 20261012]
METHODS = [
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "model",
]
LABELS = {
    "identity": "Identity",
    "bias_mean": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "learned_fir_causal": "Causal FIR",
    "arx_linear": "ARX",
    "model": "TCN",
    "zero": "零位移",
    "train_mean": "训练均值",
    "ridge": "Ridge",
}
GYRO = "gyro_vector_rmse_radps"
ACCEL = "accel_vector_rmse_mps2"
DEG = 180 / math.pi


def read_json(path: Path):
    print(f"READS {path}", flush=True)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    print(f"READS {path}", flush=True)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ms(values, digits=5):
    values = list(values)
    return f"{mean(values):.{digits}f} ± {(stdev(values) if len(values) > 1 else 0):.{digits}f}"


def fmt(value, digits=5):
    return "不可用" if value is None else f"{value:.{digits}f}"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Report:
    def __init__(self, out: Path):
        self.out = out
        self.blocks = []
        self.sources = set()

    def rel(self, p):
        return Path(os.path.relpath(Path(p).resolve(), self.out.resolve())).as_posix()

    def link(self, p, label=None):
        self.sources.add(str(Path(p).resolve()))
        return f"[{label or Path(p).name}]({self.rel(p)})"

    def h(self, text, level=1):
        self.blocks.append(("heading", level, text))

    def p(self, text, style="body"):
        self.blocks.append(("paragraph", style, text))

    def table(self, headers, rows, widths=None):
        self.blocks.append(("table", headers, [[str(x) for x in r] for r in rows], widths))

    def page(self):
        self.blocks.append(("page",))

    def code(self, text):
        self.blocks.append(("code", text))

    def figure(self, f, number):
        path = self.out / f["files"]["png"] if isinstance(f["files"], dict) else None
        if path is None or not path.is_file():
            candidates = list((self.out / "figures").glob(f["stem"] + ".png"))
            if not candidates:
                raise FileNotFoundError(f["stem"])
            path = candidates[0]
        self.blocks.append(("figure", self.rel(path), f"图 {number} | {f['title']}", f["caption"]))

    def write_csv(self, name, rows):
        p = self.out / "metric_tables" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"WRITES {p}", flush=True)
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return p


def load_runs(root):
    pipeline = read_json(root / "pipeline.json")
    if pipeline["status"] != "complete" or len(pipeline["runs"]) != 6:
        raise ValueError("Only the complete six-task/seed pipeline can be reported")
    runs = []
    for entry in pipeline["runs"]:
        if entry["status"] != "evaluated":
            raise ValueError("Every task/seed must be evaluated")
        folder = root / "experiments" / Path(entry["run_dir"]).name
        run = {
            **entry,
            "folder": folder,
            "evaluation": read_json(folder / "evaluation/metrics.json"),
            "history": read_csv(folder / "metrics.csv"),
        }
        history = run["history"]
        metric = "val_grouped_score" if entry["task"] == "paired" else "val_rmse"
        best = min(history, key=lambda x: float(x[metric]))
        run.update(
            best_epoch=int(float(best["epoch"])),
            best_metric=float(best[metric]),
            epochs=len(history),
            best_row=best,
        )
        runs.append(run)
    return pipeline, runs


def build_report(root: Path, out: Path, repo: Path):
    r = Report(out)
    pipeline, runs = load_runs(root)
    paired = [x for x in runs if x["task"] == "paired"]
    navigation = [x for x in runs if x["task"] == "navigation"]
    read_json(paired[0]["folder"] / "data_manifest.json")
    verification = read_json(root.parents[2] / "local_verification.json")
    provenance = read_json(paired[0]["folder"] / "provenance.json")
    bundle = read_json(paired[0]["folder"] / "source_bundle.json")
    launch_dir = root.parents[1] / pipeline["server_launch_run"]
    preflight = read_json(launch_dir / "preflight.json")
    scientific_audit = read_json(out / "scientific_audit.json")
    if not scientific_audit.get("passed"):
        raise ValueError("Scientific audit must pass before delivery")
    figures_file = out / "figures_manifest.json"
    figs = read_json(figures_file) if figures_file.is_file() else {"figures": []}
    figures = {x["stem"]: x for x in figs.get("figures", [])}
    fnum = 0

    def figure(stem):
        nonlocal fnum
        if stem in figures:
            fnum += 1
            r.figure(figures[stem], fnum)
        else:
            print(f"SKIPS unavailable figure {stem}", flush=True)

    pairkeys = [
        (p["dataset_id"], p["low_sensor"], p["high_sensor"])
        for p in paired[0]["evaluation"]["splits"]["test"]["pairs"]
    ]
    pairkeys.sort(key=lambda x: ["BMX160", "MPU9250", "LSM9DS1", "Tango Top IMU"].index(x[1]))

    def pair(run, key, split="test"):
        return next(
            p
            for p in run["evaluation"]["splits"][split]["pairs"]
            if tuple(p[k] for k in ("dataset_id", "low_sensor", "high_sensor")) == key
        )

    def vals(key, method, metric):
        return [pair(x, key)["metrics"][method][metric] for x in paired]

    def imp(key, method, metric):
        return [
            100
            * (
                1
                - pair(x, key)["metrics"]["model"][metric] / pair(x, key)["metrics"][method][metric]
            )
            for x in paired
        ]

    # Machine-readable supplements preserve every seed, method and physical axis.
    p_rows, axis_rows, seq_rows, nav_rows, nav_axis_rows, coverage_rows = [], [], [], [], [], []
    for run in paired:
        for split, result in run["evaluation"]["splits"].items():
            for p in result["pairs"]:
                for method, m in p["metrics"].items():
                    p_rows.append(
                        {
                            "seed": run["seed"],
                            "split": split,
                            "dataset_id": p["dataset_id"],
                            "low_sensor": p["low_sensor"],
                            "high_sensor": p["high_sensor"],
                            "method": method,
                            "samples": m["samples"],
                            GYRO: m[GYRO],
                            "gyro_vector_rmse_degps": m[GYRO] * DEG,
                            ACCEL: m[ACCEL],
                            "gyro_error_p95_radps": m["gyro_error_p95_radps"],
                            "accel_error_p95_mps2": m["accel_error_p95_mps2"],
                        }
                    )
                    for a, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
                        axis_rows.append(
                            {
                                "seed": run["seed"],
                                "split": split,
                                "dataset_id": p["dataset_id"],
                                "low_sensor": p["low_sensor"],
                                "method": method,
                                "axis": axis,
                                "unit": "rad/s" if a < 3 else "m/s2",
                                "rmse": m["axis_rmse"][a],
                                "mae": m["axis_mae"][a],
                                "mean_error": m["axis_mean_error"][a],
                                "error_std": m["axis_error_std"][a],
                            }
                        )
            for s in result["sequences"]:
                d = s["diagnostics"]
                seq_rows.append(
                    {
                        "seed": run["seed"],
                        "split": split,
                        "dataset_id": s["dataset_id"],
                        "low_sensor": s["low_sensor"],
                        "sequence_id": s["sequence_id"],
                        "samples": s["metrics"]["samples"],
                        GYRO: s["metrics"][GYRO],
                        ACCEL: s["metrics"][ACCEL],
                        "integral_windows": d["integral_windows"],
                        "gyro_integral_rad": d["delta_gyro_integral_error_rad"],
                        "accel_integral_mps": d["delta_accel_integral_error_mps"],
                        "gyro_lag_x_s": d["gyro_lag_s_by_axis"][0],
                        "gyro_lag_y_s": d["gyro_lag_s_by_axis"][1],
                        "gyro_lag_z_s": d["gyro_lag_s_by_axis"][2],
                    }
                )
            for c in result["coverage"]:
                coverage_rows.append({"seed": run["seed"], "split": split, **c})
    for run in navigation:
        for split, result in run["evaluation"]["splits"].items():
            for method, m in result["metrics"].items():
                nav_rows.append(
                    {
                        "seed": run["seed"],
                        "split": split,
                        "method": method,
                        **{k: v for k, v in m.items() if not isinstance(v, list)},
                    }
                )
                for a, axis in enumerate(("E", "N", "U")):
                    nav_axis_rows.append(
                        {
                            "seed": run["seed"],
                            "split": split,
                            "method": method,
                            "axis": axis,
                            "rmse_m": m["axis_rmse_m"][a],
                            "mae_m": m["axis_mae_m"][a],
                            "bias_m": m["axis_bias_m"][a],
                        }
                    )
    tables = {
        "配对完整指标": r.write_csv("paired_all_methods_all_seeds.csv", p_rows),
        "配对逐轴": r.write_csv("paired_axis_all_methods_all_seeds.csv", axis_rows),
        "配对连续片段": r.write_csv("paired_contiguous_segments_all_seeds.csv", seq_rows),
        "配对覆盖": r.write_csv("paired_coverage_all_seeds.csv", coverage_rows),
        "导航完整指标": r.write_csv("navigation_all_methods_all_seeds.csv", nav_rows),
        "导航逐轴": r.write_csv("navigation_axis_all_methods_all_seeds.csv", nav_axis_rows),
    }

    # Cover and executive interpretation.
    r.h(TITLE)
    r.p("ALiMU / PennCOSYVIO 配对六轴补偿 · SmartPNT-MSF 局部 ENU 位移探索", "subtitle")
    r.p("冻结方案，3 个优化随机种子，6 个已完成并评价的训练 run", "subtitle")
    r.p("报告日期：2026-10-24（Asia/Shanghai）", "small")
    r.table(
        ["审计项", "结果"],
        [
            ["Pipeline", pipeline["status"] + "；6 / 6 evaluated"],
            ["正式数据", "unified_imu_v2；944,840 行；145 个连续片段"],
            ["结果归档", "640 个文件逐一 SHA256 校验通过"],
            ["统计口径", "三个 seed 的均值 ± 样本标准差（ddof=1）"],
            ["结果来源", root.name],
        ],
    )
    r.p(
        "核心判定：TCN 相对原始 LOW 有选择性改善，但未超过已拟合的 FIR/ARX 经典基线；LSM9DS1 陀螺仪反而退化。SmartPNT TCN 的测试位移误差高于零位移与训练均值预测。当前结果不支持宣称网络已实现普遍的器件精度升级或可靠全局惯导。",
        "callout",
    )
    r.p(
        "阅读导航：第 1 节给出结论；第 2-4 节固定数据、模型与训练行为；第 5-7 节报告配对测试与连续诊断；第 8 节报告导航测试；第 9-11 节说明限制、复现与证据索引。图与原始指标使用相对路径链接，可随结果目录一同移动。",
        "small",
    )
    r.page()
    r.h("1  结果摘要与研究判定")
    summary = []
    for key in pairkeys:
        summary.append(
            [
                key[1] + " → " + key[2],
                ms(imp(key, "identity", GYRO), 2) + "%",
                ms(imp(key, "identity", ACCEL), 2) + "%",
                "FIR、ARX 均优于 TCN",
            ]
        )
    r.table(
        ["测试映射", "Gyro 改善 vs LOW", "Accel 改善 vs LOW", "经典基线判定"],
        summary,
        [0.28, 0.24, 0.24, 0.24],
    )
    lsm = next(k for k in pairkeys if k[1] == "LSM9DS1")
    r.p(
        f"四个器件映射中，LSM9DS1 陀螺仪从 Identity 的 {mean(vals(lsm, 'identity', GYRO)):.6f} rad/s 变为 TCN 的 {ms(vals(lsm, 'model', GYRO), 6)} rad/s，平均改善率为 {mean(imp(lsm, 'identity', GYRO)):.2f}%（负值代表退化）。其加速度改善不能抵消陀螺仪退化，也不能据联合分数隐藏这一现象。"
    )
    scores = [x["evaluation"]["splits"]["test"]["grouped_relative_score"] for x in paired]
    r.p(
        f"配对测试分组相对分数 S = {ms(scores, 6)}，Identity 的 S=1。该分数只用于概括相对误差，按数据集等权汇总，不等于新的物理精度单位。FIR 与 ARX 在四个映射的 gyro 和 accel 测试 RMSE 上都小于 TCN；因此本轮没有证明该 TCN 相对简单经典方法的额外价值。"
    )
    nav_model = [
        x["evaluation"]["splits"]["test"]["metrics"]["model"]["vector_rmse_m"] for x in navigation
    ]
    nav_zero = navigation[0]["evaluation"]["splits"]["test"]["metrics"]["zero"]["vector_rmse_m"]
    r.p(
        f"SmartPNT 1.975 s 端点位移：TCN 三维矢量 RMSE 为 {ms(nav_model, 4)} m，零位移为 {nav_zero:.4f} m。TCN 平均相对零位移改善率为 {100 * (1 - mean(nav_model) / nav_zero):.2f}%。三 seed 平均三维/水平 RMSE 劣于训练均值，优于当前 Ridge；不代表每个 seed、每个方向均优于 Ridge（20261010 的垂直误差高于 Ridge）。验证最佳出现在第 1-2 epoch，后续触发早停；这说明当前窗口到固定 ENU 的探索设定需要重新研究，不能当作已验证的惯导方案。"
    )
    r.p(
        "三次运行按预设 seed 全部报告，不选择测试最好的 seed。时域与频域代表图固定使用首个预设 seed=20261010，并按明确的连续片段规则选图。三个 seed 仅反映优化随机性；独立实验单位仍是物理录制/轨迹，不能把窗口或 seed 数当作独立采集样本量。"
    )
    r.p(
        "本报告没有根据测试结果修改架构、损失、对齐、基线参数或 checkpoint；也未重新训练。后续若探索新的假设，应预先冻结新的验证规则，并建立新的独立测试数据或明确测试已被观察的限制。",
        "small",
    )

    r.page()
    r.h("2  数据冻结、拆分与处理边界")
    r.p(
        "正式数据为 unified_imu_v2。所有输入/标签维持任务边界：ALiMU 与 Penn 提供实测六轴参考；SmartPNT 提供导航真值，未被转换为伪造高等级六轴标签。FusionPortable 与 NavINST 本轮未纳入，STIM300 条件通道虽然保留，但不代表已训练对应映射。"
    )
    r.table(
        ["数据 / 标签", "Train", "Validation", "Test"],
        [
            ["ALiMU / VN100", "dyn_01/02/04/05/06；stat_11", "dyn_07", "dyn_08/09"],
            ["Penn / ADIS16488", "af、as", "bf", "bs"],
            ["SmartPNT / Fixed 导航真值", "Traj3", "Traj4", "Traj1"],
        ],
        [0.25, 0.37, 0.18, 0.2],
    )
    r.table(
        ["任务/器件", "训练窗口", "验证窗口", "测试窗口"],
        [
            ["BMX160 → VN100", 4071, 373, 746],
            ["MPU9250 → VN100", 4071, 373, 746],
            ["LSM9DS1 → VN100", 4071, 373, 746],
            ["Tango Top IMU → ADIS16488", 245, 98, 163],
            ["配对合计", 12458, 1217, 2401],
            ["SmartPNT 导航", 2016, 3637, 1649],
        ],
    )
    r.p(
        "窗口均为 80 点、stride=40；不跨片段、缺口或 split。ALiMU 同一物理录制的三个低成本器件共享 VN100 参考，始终同属一个 split。正式划分测试未见过的录制，不测试未见过的器件型号。Penn 只有一个独立测试录制 bs，SmartPNT 只有一个测试轨迹 Traj1。"
    )
    r.p(
        "重复与归档：ALiMU dyn_03 是 dyn_04 的输入前缀，stat_10 是 stat_11 的输入前缀，三个低成本器件共六个较短变体已排除并归档；少数参考陀螺仪数值不同，不能称为配对文件完全一致。SmartPNT Traj2 与 Traj1 经哈希确认重复，Traj2 不参与划分。旧 v1 为已替代诊断数据，不参与正式训练。"
    )
    r.p(
        "所有信号采用共同 40 Hz 网格，gyro 为 rad/s，accel 为 m/s²。高采样率输入经 63 tap FIR 抗混叠后规整，切掉启动段；间隔大于 1.5 个原生周期、非有限值与真值质量切换均切分片段，不跨缺口插值。ALiMU 本身为 40 Hz 的记录不重复施加该滤波。"
    )
    r.p(
        "Penn 使用刚体参考系与杆臂处理，保留上游通道延迟并额外剔除 1.2 s 启动段。ALiMU 使用上游配对轴约定，外参未额外独立验证。SmartPNT 从原生 200 Hz 解析信号直接抗混叠；角/速度增量除包周期的单位解释通过周期与重力幅值检查，但不据此推定器件等级。"
    )
    r.p(
        "SmartPNT 以每条轨迹第一条 ECEF 真值及经纬度定义固定 ENU 原点。仅在两端均为 Fixed 且时间间隔合格的区间插值真值。姿态沿用上游 heading/pitch/roll 约定；未宣称已知 IMU 安装旋转。"
    )
    r.p(
        "便携数据说明："
        + r.link(out / "reproducibility/dataset_metadata/config_snapshot.yaml", "处理配置快照")
        + "；"
        + r.link(out / "reproducibility/dataset_metadata/label_schema.json", "标签与单位 schema")
        + "；"
        + r.link(out / "reproducibility/dataset_metadata/archive/duplicates.json", "重复源归档记录")
        + "。",
        "small",
    )
    r.p(
        "冻结清单："
        + r.link(paired[0]["folder"] / "data_manifest.json", "数据 manifest")
        + "；"
        + r.link(paired[0]["folder"] / "split_manifest.json", "配对 split")
        + "；"
        + r.link(navigation[0]["folder"] / "split_manifest.json", "导航 split")
        + "；"
        + r.link(paired[0]["folder"] / "data_verification.json", "145 段/944,840 行验证")
        + "。",
        "small",
    )

    r.page()
    r.h("3  模型、监督与固定训练方案")
    r.h("3.1  配对任务", 2)
    r.p(
        "输入为低成本六轴与参考类型 one-hot，共 9×80；当前已知 VN100 与 ADIS16488 参考。网络学习 r(t)=HIGH(t)-LOW(t)，输出六轴残差；先按训练集残差统计反归一化，再得到 VIRTUAL(t)=LOW(t)+r̂(t)。推理不读取参考测量。低成本器件型号用于分组，不是额外输入编码。"
    )
    r.p(
        "采用因果 TCN：64 隐藏通道，4 个残差扩张卷积块，每块两层 kernel=5 卷积，dilation=1/2/4/8，dropout=0.05。理论感受野为 1+2×(5-1)×(1+2+4+8)=121 点，长于 80 点输入；窗口外使用因果零填充，窗口起点历史不足。该结构的因果性不能证明完整预处理/分窗/合并链已经满足实时部署要求。"
    )
    r.p(
        "特征与残差统计仅使用 train 拟合，验证/测试复用；不能拿数据目录中 HIGH 的统计量替代残差目标统计。训练 sampler 令 ALiMU/Penn 各占约 50% 期望概率，每 epoch 抽 12,458 个窗口并允许重复。Penn 仅 245 个独有训练窗口，重采样提高曝光次数，不增加独立数据量。"
    )
    r.table(
        ["标准化残差空间 loss", "定义 / 固定权重"],
        [
            ["Signal", "(2×Huber(gyro)+Huber(accel))/3"],
            ["Difference", "一阶时间差分 Huber；0.02"],
            ["Spectrum", "窗口 log(1+FFT 幅值)差异；0.02"],
            ["Integral", "cumsum(residual)×0.025 s 的 Huber；0.005"],
            ["未启用", "Allan 权重=0；温度/静态上下文/独立 bias 分支关闭"],
        ],
        [0.28, 0.72],
    )
    r.p(
        "总 loss=Signal+0.02×Difference+0.02×Spectrum+0.005×Integral。Integral 是标准化窗口信号累积和约束，不是 SO(3) 预积分、导航滤波器或以米计的漂移。"
    )
    r.h("3.2  导航任务", 2)
    r.p(
        "单独训练，六轴 6×80 输入，不与配对任务联训。相同 TCN 取最后时刻的三个输出，监督 Δp=p_ENU(end)-p_ENU(start)，单位米；80 点首末间隔为 79/40=1.975 s。目标/输入归一化只用 Traj3 训练数据；loss 为标准化三维位移 Huber，速度与航向积分权重均为 0。"
    )
    r.table(
        ["优化设置", "配对", "导航"],
        [
            ["Optimizer", "AdamW", "AdamW"],
            ["学习率 / weight decay", "0.001 / 1e-5", "0.001 / 1e-5"],
            ["Batch / 最多 epoch", "128 / 60", "128 / 60"],
            ["梯度裁剪", "norm=1.0", "norm=1.0"],
            ["早停", "10 epoch 无严格改善", "10 epoch 无严格改善"],
            ["Scheduler", "ReduceLROnPlateau；patience=5；factor=0.5；min_lr=5e-5", "未配置"],
            ["Seed", "20261010/11/12", "20261010/11/12"],
        ],
        [0.28, 0.4, 0.32],
    )

    r.page()
    r.h("4  训练完成情况与模型选择")
    r.p(
        "配对 checkpoint、学习率调度与早停统一监控分组无量纲分数 val_grouped_score：每个映射先取 gyro/identity 与 accel/identity 的比值均值，再在数据集内平均器件，最后对 ALiMU/Penn 等权。重叠预测按物理时间点取算术平均；近零 Identity 分母下限为 1e-12，当前验证未出现触发组。"
    )
    r.p(
        "导航监控 val_rmse，定义为反归一化后所有三分量的总体 RMSE。三维矢量 RMSE=√3×component RMSE；val_vector_rmse 与 val_position_rmse_m 均是三维矢量口径，不能把它们与 val_rmse 直接混比。"
    )
    rows = []
    for run in runs:
        rows.append(
            [
                "配对" if run["task"] == "paired" else "导航",
                run["seed"],
                run["epochs"],
                run["best_epoch"],
                fmt(run["best_metric"], 6),
                "达到60上限" if run["epochs"] == 60 else "10轮无改善早停",
            ]
        )
    r.table(
        ["任务", "Seed", "完成 epoch", "最佳 epoch", "最佳验证值", "停止原因"],
        rows,
        [0.1, 0.17, 0.15, 0.15, 0.2, 0.23],
    )
    r.p(
        "配对三次均完成 60 epoch；最佳为第 60、58、58 epoch，达到预算上限不代表已收敛到最优。导航分别运行 12、11、12 epoch，最佳为第 2、1、2 epoch，其后连续 10 epoch 未改善而早停。最终测试均使用所选 best.ckpt，而非最后一轮或测试最优权重。"
    )
    figure("paired_training_loss")
    figure("paired_validation_score")
    r.page()
    r.h("4.1  导航训练行为与工程环境", 2)
    figure("navigation_training_loss")
    figure("navigation_validation_rmse")
    env = preflight["environment"]
    gpu = env["devices"][0]
    r.p(
        f"训练 pipeline 记录 Python {pipeline['python']}、Torch {pipeline['torch']}、CUDA runtime {pipeline['cuda_runtime']}，device={pipeline['device']}。服务器为 {env['platform']} / {env['machine']}，GPU 为 {gpu['name']}（设备报告总显存 {gpu['memory_bytes'] / 1024**3:.2f} GiB，非训练峰值占用），cuDNN={env['cudnn_version']}，NumPy={env['numpy']}，pandas={env['pandas']}，pyarrow={env['pyarrow']}，CPU threads=4。预检确认实际 CUDA 前向与 Conv1d 反向均可执行。"
    )
    r.p(
        "本报告没有新增吞吐、单窗口时延、峰值显存或嵌入式功耗测量。目录时间戳与日志用于运行追踪，不能替代受控性能基准。环境证据："
        + r.link(launch_dir / "preflight.json", "实际运行预检")
        + "；确定性积分兼容性回归："
        + r.link(root.parents[1] / "runs/gpu_integral_regression.log", "GPU 4 项回归通过")
        + "。"
    )
    r.p(
        "严格确定性模式固定 cuBLAS workspace、关闭 TF32、启用确定性卷积。Torch 2.4 CUDA cumsum 在该模式不受支持，因此窗口积分项使用等价下三角矩阵前缀求和；数值与梯度经 CPU/GPU 核对，数学 loss 与权重保持不变。这是运行兼容性修正，不是基于 test 调整训练目标。"
    )

    r.page()
    r.h("5  配对测试：逐器件与全部经典基线")
    r.p(
        "主指标为 sqrt(mean(||prediction-reference||²))，gyro 与 accel 分开计算；它不是两信号模长之差的 RMSE。下表列三个 seed 的均值 ± 样本 SD。经典基线确定性地使用同一训练数据，跨 seed 数值相同；SD=0 仅表示本轮重复不含基线优化随机性，不代表总体不确定性为零。"
    )
    r.p(
        "基线按器件/参考映射独立在 train 拟合：Mean bias 为 HIGH-LOW 平均差；Affine 为六轴线性映射加截距；Ridge affine 的 alpha=0.01；causal learned FIR 为 9 taps、alpha=0.01。代码中名为 ARX 的 arx_linear 实际使用 12 taps 低成本输入加当期一阶差分进行线性岭回归，不使用参考输出反馈；片段起点用本段首值填历史，每段重置。测试阶段没有按参考信号重估时差或对齐。所有方法在同一完整窗口覆盖时间点评价。"
    )
    for i, key in enumerate(pairkeys):
        if i:
            r.page()
        r.h(f"5.{i + 1}  {key[1]} → {key[2]}", 2)
        methods = [m for m in METHODS if m in pair(paired[0], key)["metrics"]]
        r.table(
            ["方法", "Gyro RMSE (rad/s)", "Gyro RMSE (deg/s)", "Accel RMSE (m/s²)"],
            [
                [
                    LABELS.get(m, m),
                    ms(vals(key, m, GYRO), 6),
                    ms([v * DEG for v in vals(key, m, GYRO)], 4),
                    ms(vals(key, m, ACCEL), 5),
                ]
                for m in methods
            ],
            [0.19, 0.28, 0.27, 0.26],
        )
        r.table(
            ["TCN 相对基线改善率", "Gyro (%)", "Accel (%)"],
            [
                [LABELS.get(m, m), ms(imp(key, m, GYRO), 2), ms(imp(key, m, ACCEL), 2)]
                for m in methods
                if m != "model"
            ],
            [0.38, 0.31, 0.31],
        )
        r.p(
            f"每次运行的测试覆盖点数为 {pair(paired[0], key)['samples']:,}。改善率=100×(1-RMSE_TCN/RMSE_baseline)，正值更好、负值退化。该映射 TCN 相对 Identity 的 gyro/accel 改善分别为 {mean(imp(key, 'identity', GYRO)):.2f}% / {mean(imp(key, 'identity', ACCEL)):.2f}%；比较经典基线时应以上表具体结果为准。"
        )
        slug = next(
            (
                s.removeprefix("paired_signal_")
                for s, f in figures.items()
                if s.startswith("paired_signal_") and key[1].lower().replace(" ", "_") in s.lower()
            ),
            None,
        )
        if slug:
            figure("paired_signal_" + slug)
    r.page()
    r.h("6  配对重复结果、逐轴与尾部误差")
    r.h("6.1  每个 seed 的 TCN 测试值", 2)
    rows = []
    for key in pairkeys:
        for run in paired:
            m = pair(run, key)["metrics"]["model"]
            rows.append(
                [key[1], run["seed"], fmt(m[GYRO], 6), fmt(m[GYRO] * DEG, 4), fmt(m[ACCEL], 5)]
            )
    r.table(
        ["器件", "Seed", "Gyro rad/s", "Gyro deg/s", "Accel m/s²"],
        rows,
        [0.23, 0.19, 0.2, 0.19, 0.19],
    )
    r.table(
        ["Seed", "Test 分组 S"],
        [
            [x["seed"], fmt(x["evaluation"]["splits"]["test"]["grouped_relative_score"], 6)]
            for x in paired
        ],
    )
    figure("paired_test_gyro_baselines")
    figure("paired_test_accel_baselines")
    r.page()
    r.h("6.2  六轴 RMSE / MAE / 平均误差", 2)
    r.p(
        "每格为三个 seed 的均值 ± 样本 SD；平均误差为 prediction-reference 的带符号均值。Gyro 单位为 rad/s，accel 为 m/s²。此处并列 Identity 与 TCN；包含全部经典基线、验证与测试的逐轴原值见 "
        + r.link(tables["配对逐轴"], "完整逐轴 CSV")
        + "。"
    )
    for i, key in enumerate(pairkeys):
        if i in (2,):
            r.page()
        r.h(key[1] + " → " + key[2], 3)
        rows = []
        for a, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
            for method in ("identity", "model"):
                rows.append(
                    [
                        axis,
                        LABELS[method],
                        *[
                            ms([pair(x, key)["metrics"][method][metric][a] for x in paired], 5)
                            for metric in ("axis_rmse", "axis_mae", "axis_mean_error")
                        ],
                    ]
                )
        r.table(["轴", "方法", "RMSE", "MAE", "平均误差"], rows, [0.07, 0.15, 0.26, 0.26, 0.26])
    r.page()
    r.h("6.3  误差范数 P95 与连续信号覆盖", 2)
    rows = []
    for key in pairkeys:
        for m in ("identity", "model"):
            rows.append(
                [
                    key[1],
                    LABELS[m],
                    ms(vals(key, m, "gyro_error_p95_radps"), 6),
                    ms(vals(key, m, "accel_error_p95_mps2"), 5),
                ]
            )
    r.table(["器件", "方法", "Gyro P95 (rad/s)", "Accel P95 (m/s²)"], rows, [0.23, 0.18, 0.3, 0.29])
    r.p(
        "P95 是每个时间点三维误差范数的 95% 分位数；它不是逐轴误差标准差或置信区间。峰值/局部尾部应回到逐片段输出检查；当前未定义事后筛选峰值时间段作为另一主指标。"
    )
    cov = paired[0]["evaluation"]["splits"]
    r.table(
        ["Split", "完整窗口", "覆盖时间点", "未覆盖片段尾点", "连续片段"],
        [
            [
                split,
                sum(c["windows"] for c in v["coverage"]),
                sum(c["covered_samples"] for c in v["coverage"]),
                sum(c["uncovered_samples"] for c in v["coverage"]),
                len(v["coverage"]),
            ]
            for split, v in cov.items()
        ],
    )
    r.p(
        "推理只使用完整 80 点窗口，stride=40，重叠区的物理预测做算术平均，每个 sequence_id/timestamp 仅计算一次。未覆盖尾点不补零、不加 padding 尾窗，也不计入主指标；基线同步只在覆盖时间点评价。窗口开始处因果历史不足，合并预测可能依赖不同窗口位置的可用历史，端点误差与实际流式缓存策略需单独审查。"
    )
    r.p(
        "覆盖 CSV："
        + r.link(tables["配对覆盖"])
        + "。逐片段 LOW/HIGH/VIRTUAL 保留在每个 run 的 evaluation/test_predictions.parquet；这些输出已经合并重叠，包含器件与数据集身份。"
    )

    r.page()
    r.h("7  真实连续片段诊断与图像证据")
    r.p(
        "所有频谱、coherence、互相关 lag 与积分都在真实连续片段内独立计算，不跨记录拼接。图固定使用 seed=20261010；时域展示首个符合条件的字典序连续测试片段、最多前 20 s。PSD/coherence/积分的精确选段、源文件与窗口规则由图 manifest 记录，代表图不按结果优劣选择。"
    )
    r.p(
        "下表逐段列 TCN 物理误差的三 seed 均值 ± 样本 SD。短名保留录制身份，完整 sequence_id 与所有诊断在 "
        + r.link(tables["配对连续片段"], "逐连续片段 CSV")
        + "；频谱明细保留于各 run 的 evaluation/metrics.json。"
    )
    seqs = paired[0]["evaluation"]["splits"]["test"]["sequences"]
    rows = []
    for s in seqs:
        ss = [
            next(
                v
                for v in x["evaluation"]["splits"]["test"]["sequences"]
                if v["sequence_id"] == s["sequence_id"]
            )
            for x in paired
        ]
        short = re.search(r"dyn_\d+", s["sequence_id"])
        short = short.group(0) if short else "bs"
        rows.append(
            [
                s["low_sensor"] + " / " + short,
                s["metrics"]["samples"],
                ms([v["metrics"][GYRO] for v in ss], 6),
                ms([v["metrics"][ACCEL] for v in ss], 5),
                ms([v["diagnostics"]["delta_gyro_integral_error_rad"] for v in ss], 5),
                ms([v["diagnostics"]["delta_accel_integral_error_mps"] for v in ss], 4),
            ]
        )
    r.table(
        ["器件/录制", "点数", "Gyro rad/s", "Accel m/s²", "角增量残余 rad", "速度增量残余 m/s"],
        rows,
        [0.22, 0.09, 0.18, 0.17, 0.17, 0.17],
    )
    r.p(
        "积分诊断使用每段内互不重叠的 80 点小窗，梯形积分端点跨度 1.975 s，窗口起点重置，再平均积分误差向量范数。它是加性六轴误差积分代理；未进行姿态传播、重力补偿或导航滤波，不能解释为 ATE/RPE、SO(3) 预积分或真实导航漂移。"
    )
    r.p(
        "陀螺 lag 在 ±0.5 s 范围内取互相关峰值，低激励轴标记不可用；该值仅诊断，没有回移信号改善测试分数。Coherence/PSD 对短片段与低激励不保证可辨识，图中任何高相关性也不等同于更低物理误差。"
    )
    # Main report contains compact diagnostics; complete SVG/PDF/PNG suite is linked.
    for prefix in ("paired_psd_", "paired_coherence_", "paired_integral_"):
        stems = [s for s in figures if s.startswith(prefix)]
        if stems:
            r.page()
            figure(stems[0])
    r.p(
        "全部器件的时域、PSD、coherence 与窗口积分图（SVG/PDF/PNG），含未嵌入正文的图："
        + r.link(out / "figures/figure_index.md", "完整图索引")
        + "；"
        + r.link(figures_file, "图与选择元数据")
        + "。",
        "small",
    )

    r.page()
    r.h("8  SmartPNT 导航探索：完整测试评价")
    r.p(
        "标签是局部固定 ENU 中 1.975 s 首末位移，而非绝对位置或完整轨迹。测试为 Traj1 的 1,649 个重叠窗口；所有方法使用同一批合格窗口与端点，Ridge/TCN 仅用 LOW 六轴输入，零位移与训练均值为常数预测。参考速度、姿态、初速度和温度均不作输入。"
    )
    nav_methods = ["zero", "train_mean", "ridge", "model"]

    def nv(method, metric):
        return [x["evaluation"]["splits"]["test"]["metrics"][method][metric] for x in navigation]

    r.table(
        ["方法", "三维 RMSE m", "水平 RMSE m", "垂直 RMSE m", "范数中位数 m", "范数 P95 m"],
        [
            [
                LABELS[m],
                *[
                    ms(nv(m, k), 4)
                    for k in (
                        "vector_rmse_m",
                        "horizontal_rmse_m",
                        "vertical_rmse_m",
                        "error_norm_median_m",
                        "error_norm_p95_m",
                    )
                ],
            ]
            for m in nav_methods
        ],
        [0.13, 0.19, 0.18, 0.16, 0.17, 0.17],
    )
    r.table(
        ["TCN 相对基线", "三维改善 (%)", "水平改善 (%)", "垂直改善 (%)"],
        [
            [
                LABELS[m],
                *[
                    ms(
                        [100 * (1 - a / b) for a, b in zip(nv("model", k), nv(m, k), strict=True)],
                        2,
                    )
                    for k in ("vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m")
                ],
            ]
            for m in nav_methods
            if m != "model"
        ],
    )
    r.p(
        "零位移无需拟合；训练均值只用 Traj3 标签均值；Ridge 仅使用每个低成本六轴窗口的 mean/std/end-minus-start 共 18 个统计量，特征标准化与系数在 train 拟合，alpha=0.01，截距不惩罚。该 Ridge 不是完整时序 FIR，也没有额外导航状态。"
    )
    r.p(
        "三维与分量口径：vector RMSE²=E_RMSE²+N_RMSE²+U_RMSE²；horizontal²=E²+N²；vertical=U_RMSE；component=vector/√3。TCN 的总体 component 测试 RMSE 为 "
        + ms(nv("model", "component_rmse_m"), 4)
        + " m。"
    )
    figure("navigation_test_baselines")
    r.page()
    r.h("8.1  每个 seed 与 ENU 逐轴误差", 2)
    r.table(
        ["Seed", "三维 RMSE m", "水平 RMSE m", "垂直 RMSE m", "Component m"],
        [
            [
                x["seed"],
                *[
                    fmt(x["evaluation"]["splits"]["test"]["metrics"]["model"][k], 4)
                    for k in (
                        "vector_rmse_m",
                        "horizontal_rmse_m",
                        "vertical_rmse_m",
                        "component_rmse_m",
                    )
                ],
            ]
            for x in navigation
        ],
    )
    rows = []
    for m in nav_methods:
        for a, axis in enumerate(("E", "N", "U")):
            rows.append(
                [
                    LABELS[m],
                    axis,
                    *[
                        ms(
                            [
                                x["evaluation"]["splits"]["test"]["metrics"][m][k][a]
                                for x in navigation
                            ],
                            4,
                        )
                        for k in ("axis_rmse_m", "axis_mae_m", "axis_bias_m")
                    ],
                ]
            )
    r.table(["方法", "轴", "RMSE m", "MAE m", "Bias m"], rows, [0.18, 0.07, 0.25, 0.25, 0.25])
    r.p(
        "各窗口预测、真值与基线输出保存在每个导航 run 的 evaluation/test_predictions.parquet。完整数字表："
        + r.link(tables["导航完整指标"])
        + "；"
        + r.link(tables["导航逐轴"])
        + "。"
    )
    figure("navigation_test_windows")
    r.page()
    r.h("8.2  位移散点与可识别性限制", 2)
    figure("navigation_test_scatter")
    r.p(
        "当前短窗口输入没有显式初始航向/姿态或初始速度。整体水平旋转后相似的机体系 IMU 可能对应不同固定 ENU 位移；不同初速度也可能给相同加速度但不同位移。因此该映射并非在任意运动下唯一可识别，网络可能主要学习 Traj3 的运动先验。测试劣于零位移与训练均值，限制了当前探索设定的有效性结论。"
    )
    r.p(
        "80 点窗口 stride=40，位移区间重叠且首末相差 79 个采样周期，不能把全部预测直接累加成轨迹。本轮没有 ATE、RPE、轨迹对齐、姿态/重力/初始状态一致的完整惯导评价。高等级配对任务与 SmartPNT 导航任务使用不同标签、参考类型与平台，不能把二者合并成已验证的跨平台下游收益。"
    )

    r.page()
    r.h("9  科学解释、已知限制与后续决策")
    limitations = [
        (
            "经典基线已经更强",
            "FIR/ARX 在全部四个映射的 gyro/accel 主指标上优于 TCN。部署选择应先考虑这些基线；当前不能仅凭相对 Identity 改善宣称深度模型更优。",
        ),
        (
            "存在稳定器件退化",
            "LSM9DS1 gyro 在三个 seed 下均应逐项检查；联合平均与加速度改善不能掩盖该退化。",
        ),
        (
            "独立采集数量有限",
            "ALiMU 仅 dyn_08/09 测试，Penn 仅 bs，SmartPNT 仅 Traj1。窗口重叠、三器件共享参考与 seed 重复均不增加独立物理采集数；本报告不据三 seed SD 给总体泛化置信区间或显著性结论。",
        ),
        (
            "带宽与硬件等级不可外推",
            "共同 40 Hz 数据不能证明原始高带宽噪声、硬件温漂或硬件等级。本版没有温度补偿、独立静态 test、静态上下文/bias 辅助分支，Allan 不是独立泛化证据。",
        ),
        (
            "实时链未实测",
            "TCN 因果结构、抗混叠等待、80 点窗口缓存与重叠合并需要共同计算端到端时延。未测峰值显存、吞吐、功耗与严格流式边界，不能由训练 GPU 推断端侧可用性。",
        ),
        (
            "测试已被观察",
            "可报告失败模式并提出假设；不能再按此测试选择损失/架构/对齐。新实验须事先写 YAML、定义验证规则并保留新的测试边界。",
        ),
    ]
    for title, text in limitations:
        r.h(title, 3)
        r.p(text)
    r.p(
        "可执行的后续研究方向：先验证 FIR/ARX 的连续动态响应与流式实现；若继续 TCN，预注册只信号 loss 与组合 loss、单数据集与联合训练的消融，并扩充独立采集。SmartPNT 应先定义可用初始状态、坐标规范化与姿态来源，再设计与这些输入一致的监督；不能在当前测试上调整初速/航向后报告为原方案改善。"
    )

    r.page()
    r.h("10  可复现性、来源与完整性")
    dataset_sha = provenance["unified_dataset"]["inputs"]["dataset_manifest"]["sha256"]
    r.table(
        ["项目", "固定值 / 证据"],
        [
            ["数据 manifest SHA256", dataset_sha],
            ["数据 build run", provenance["unified_dataset"]["build_run"]],
            [
                "数据 source_state_sha256",
                str(provenance["unified_dataset"].get("source_state_sha256")),
            ],
            ["服务器 source bundle", bundle["bundle_id"]],
            [
                "原来源包 Git base",
                bundle.get("git_commit", "unknown") + "；dirty=" + str(bundle.get("git_dirty")),
            ],
            ["训练来源包 source_fingerprint", bundle.get("source_fingerprint", "unknown")],
            ["归档 SHA256", verification["archive_sha256"]],
            [
                "结果文件校验",
                f"{verification['file_hashes_verified']} files；archive {verification['archive_bytes']:,} bytes",
            ],
            ["服务器 git_commit", pipeline["git_commit"]],
        ],
        [0.28, 0.72],
    )
    r.p(
        "服务器导出目录没有 .git，pipeline 的 git_commit=unknown，不能将它替换成当前本地 HEAD。原来源包记录 base commit="
        + bundle.get("git_commit", "unknown")
        + " 且 git_dirty="
        + str(bundle.get("git_dirty"))
        + "；所以仅 checkout 此提交不能重建当时工作区，必须使用来源包逐文件哈希。数据构建 source_state_sha256 是数据构建时源码状态的证据，不能自动当作此次训练代码的提交号；完整 source_bundle.json 与 bundle_manifest 保留源码条目的哈希。"
    )
    r.p(
        "SHA256 绑定源文件内容，不证明采集标签或标定在科学上绝对正确。下载校验确认本地副本与远程导出一致；它不重新训练或复算网络。报告表格直接从每个冻结 evaluation/metrics.json 重组，避免旧通用指标跨记录拼接或单位混用。"
    )
    r.p(
        f"独立科学审计 {scientific_audit['check_count']} 项通过：包括 checkpoint/config/source 哈希一致、权重有限、从已保存预测重算全部方法物理指标、复算验证分数并对应最佳 checkpoint；未发现审计错误。审计不修改原训练产物，也不把数值一致性当作外部泛化证据。"
        + r.link(out / "scientific_audit.json", "完整科学审计")
        + "；"
        + r.link(out / "reproducibility/model_registry.json", "六个模型及配置/数据/指标哈希登记")
        + "。"
    )
    r.h("10.1  原训练命令与重建报告", 2)
    r.p("原冻结候选的训练入口（重跑会创建新 run；不作为本次报告构建步骤执行）：")
    r.code("python scripts/run_formal_training.py --config configs/train_formal_matrix.yaml")
    r.p("在已下载目录上重建本文（只读来源，写报告与数字补充表，不训练）：")
    r.code(
        "python scripts/build_completed_formal_report.py --run-root runs/remote_results_formal_20261024/server/runs/formal_training_20261023_141916_733429 --output runs/remote_results_formal_20261024/report"
    )
    r.p(
        "运行需 reportlab 和可嵌入中文 TTF（默认 Windows SimHei；可用 --font 显式指定）。图先由报告图脚本按 src/virtual_imu/viz 风格生成，PDF 通过相同图片与共享文字表格构建。PDF/HTML 是阅读版本，Markdown 与 metric_tables 保留便于审计的文字与完整数字。"
    )
    r.p(
        "best.ckpt 保存最佳模型与归一化等元数据，但没有完整 optimizer/scheduler/RNG 状态，不能宣称无损断点续训。outputs/latest 不用于定位此结果；权威来源为下列固定 run 及 pipeline.json。"
    )

    r.page()
    r.h("11  证据与产物导航")
    r.p(
        "以下均为相对于报告文件的链接，需保留 runs/remote_results_formal_20261024 的目录结构。原始数据与模型文件未复制进文档正文。"
    )
    r.table(
        ["任务 / Seed", "run 与证据"],
        [
            [
                x["task"] + " / " + str(x["seed"]),
                r.link(x["folder"] / "config.yaml", "实际 YAML")
                + " · "
                + r.link(x["folder"] / "metrics.csv", "逐 epoch")
                + " · "
                + r.link(x["folder"] / "best.ckpt", "最佳权重")
                + " · "
                + r.link(x["folder"] / "evaluation/metrics.json", "完整评价")
                + " · "
                + r.link(x["folder"] / "evaluation/test_predictions.parquet", "测试预测")
                + " · "
                + r.link(
                    x["folder"] / "evaluation/baseline_train_state.json", "train-only 基线状态"
                )
                + " · "
                + r.link(x["folder"] / "provenance.json", "来源"),
            ]
            for x in runs
        ],
        [0.24, 0.76],
    )
    r.table(
        ["补充表", "内容"],
        [
            [r.link(path, label), "全部 seed、验证/测试、适用的方法及物理单位"]
            for label, path in tables.items()
        ],
        [0.46, 0.54],
    )
    r.p(
        "全局证据："
        + r.link(root / "pipeline.json", "完成状态")
        + "；"
        + r.link(root / "paired_summary.csv", "服务器配对汇总")
        + "；"
        + r.link(root / "navigation_summary.csv", "服务器导航汇总")
        + "；"
        + r.link(root.parents[2] / "local_verification.json", "本地完整性校验")
        + "；"
        + r.link(root.parents[1] / "bundle_manifest.json", "来源包 manifest")
        + "；"
        + r.link(paired[0]["folder"] / "source_bundle.json", "run 来源包副本")
        + "。"
    )
    r.p(
        "协议与解释依据（来源包内文档副本）："
        + r.link(out / "reproducibility/docs/TRAINING_PROTOCOL_V2.md", "正式训练协议")
        + "；"
        + r.link(out / "reproducibility/docs/UNIFIED_IMU_DATASET.md", "数据与标签说明")
        + "；"
        + r.link(out / "reproducibility/docs/SERVER_TRAINING.md", "服务器与确定性兼容性说明")
        + "。报告以实际 run/evaluation 为完成状态来源；协议中的历史“待补齐”措辞不覆盖当前实际产物。"
    )
    if figures_file.is_file():
        r.p(
            "图像审计："
            + r.link(figures_file, "图 manifest")
            + "；"
            + r.link(out / "figures_analysis.json", "图像数值分析")
            + "；"
            + r.link(out / "figures/figure_index.md", "全图索引")
            + "。正文展示代表图，完整套件保留每个器件的相同诊断，防止只展示有利的例子。"
        )
    return r


def inline_html(text):
    text = html.escape(str(text))
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


def normalize_glyphs(r):
    """Keep embedded SimHei text complete using explicit, portable notation."""

    def clean(x):
        if isinstance(x, str):
            return x.replace("²", "^2").replace("r̂", "r_hat").replace("̂", "").replace("−", "-")
        if isinstance(x, tuple):
            return tuple(clean(v) for v in x)
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x

    r.blocks = clean(r.blocks)


def write_markdown(r):
    lines = []
    for b in r.blocks:
        if b[0] == "heading":
            lines += ["#" * b[1] + " " + b[2], ""]
        elif b[0] == "paragraph":
            lines += [b[2], ""]
        elif b[0] == "table":
            lines += ["| " + " | ".join(b[1]) + " |", "| " + " | ".join(["---"] * len(b[1])) + " |"]
            lines += [
                "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |" for row in b[2]
            ]
            lines += [""]
        elif b[0] == "figure":
            lines += [f"![{b[2]}]({b[1]})", "", b[2] + "。" + b[3], ""]
        elif b[0] == "code":
            lines += ["```bash", b[1], "```", ""]
    p = r.out / "完整训练报告.md"
    print(f"WRITES {p}", flush=True)
    p.write_text("\n".join(lines), encoding="utf-8")


def write_html(r):
    elements = []
    for b in r.blocks:
        if b[0] == "heading":
            elements.append(f"<h{b[1]}>{html.escape(b[2])}</h{b[1]}>")
        elif b[0] == "paragraph":
            elements.append(f'<p class="{b[1]}">{inline_html(b[2])}</p>')
        elif b[0] == "table":
            elements.append(
                "<table><thead><tr>"
                + "".join("<th>" + inline_html(x) + "</th>" for x in b[1])
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join("<td>" + inline_html(x) + "</td>" for x in row) + "</tr>"
                    for row in b[2]
                )
                + "</tbody></table>"
            )
        elif b[0] == "figure":
            elements.append(
                f'<figure><img src="{b[1]}" alt="{html.escape(b[2])}"><figcaption><strong>{html.escape(b[2])}</strong>。{html.escape(b[3])}</figcaption></figure>'
            )
        elif b[0] == "code":
            elements.append("<pre>" + html.escape(b[1]) + "</pre>")
    text = (
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>"""
        + TITLE
        + """</title><style>body{font:16px/1.8 "Microsoft YaHei","Noto Sans CJK SC",sans-serif;color:#20313c;background:#f4f7f8;margin:0}main{max-width:1050px;margin:24px auto;background:white;padding:56px 64px;box-shadow:0 8px 30px #193c4612}h1{font-size:29px;border-bottom:2px solid #15767a;padding-bottom:12px;margin-top:48px}h2{font-size:23px;color:#12666b}h3{font-size:18px}p{margin:14px 0}a{color:#086b95}table{border-collapse:collapse;width:100%;margin:24px 0;font-size:13px}th{background:#183d4a;color:white}td,th{padding:9px 10px;border:1px solid #d7e1e5;text-align:left}tr:nth-child(even){background:#f3f7f8}figure{margin:36px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#45616e}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef4f5;padding:18px;font:13px/1.6 Consolas,monospace}.small{font-size:13px;color:#54707c}.subtitle{font-size:21px;color:#54707c}.callout{border-left:5px solid #d06b42;background:#fff4ec;padding:20px;font-weight:600}@media print{main{margin:0;padding:0;box-shadow:none}body{background:white}h1,h2,h3{break-after:avoid}table,figure{break-inside:avoid}}</style></head><body><main>"""
        + "\n".join(elements)
        + "</main></body></html>"
    )
    p = r.out / "完整训练报告.html"
    print(f"WRITES {p}", flush=True)
    p.write_text(text, encoding="utf-8")


def write_pdf(r, font):
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    rl_config.warnOnMissingFontGlyphs = 1
    pdfmetrics.registerFont(TTFont("CJK", str(font)))
    pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK")
    body = ParagraphStyle(
        "Body",
        fontName="CJK",
        fontSize=9.3,
        leading=15.3,
        spaceAfter=7,
        wordWrap="CJK",
        textColor=colors.HexColor("#20313c"),
        splitLongWords=True,
    )
    styles = {
        "body": body,
        "small": ParagraphStyle(
            "Small", parent=body, fontSize=7.6, leading=12, textColor=colors.HexColor("#526b77")
        ),
        "subtitle": ParagraphStyle("Sub", parent=body, fontSize=14, leading=23, spaceAfter=16),
        "callout": ParagraphStyle(
            "Callout",
            parent=body,
            fontSize=11,
            leading=19,
            borderPadding=12,
            backColor=colors.HexColor("#fff3e9"),
            spaceBefore=15,
            spaceAfter=18,
        ),
    }
    headings = {
        1: ParagraphStyle(
            "H1",
            parent=body,
            fontSize=19,
            leading=27,
            spaceBefore=8,
            spaceAfter=15,
            keepWithNext=True,
            textColor=colors.HexColor("#124d59"),
        ),
        2: ParagraphStyle(
            "H2",
            parent=body,
            fontSize=13,
            leading=21,
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
            textColor=colors.HexColor("#12666b"),
        ),
        3: ParagraphStyle(
            "H3",
            parent=body,
            fontSize=10.5,
            leading=17,
            spaceBefore=9,
            spaceAfter=6,
            keepWithNext=True,
        ),
    }
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.4, leading=11, spaceAfter=0)
    headcell = ParagraphStyle("Header", parent=cell, textColor=colors.white)

    def markup(s):
        t = html.escape(str(s))
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#096b8f">\1</link>', t)
        return t

    width = 170 * mm
    story = []
    for b in r.blocks:
        if b[0] == "heading":
            story.append(Paragraph(markup(b[2]), headings[min(3, b[1])]))
        elif b[0] == "paragraph":
            story.append(Paragraph(markup(b[2]), styles[b[1]]))
        elif b[0] == "page":
            story.append(PageBreak())
        elif b[0] == "code":
            story.append(
                Paragraph(
                    markup(b[1]),
                    ParagraphStyle(
                        "Code",
                        parent=styles["small"],
                        backColor=colors.HexColor("#edf3f5"),
                        borderPadding=10,
                        spaceBefore=8,
                        spaceAfter=14,
                    ),
                )
            )
        elif b[0] == "table":
            data = [[Paragraph(markup(x), headcell) for x in b[1]]] + [
                [Paragraph(markup(x), cell) for x in row] for row in b[2]
            ]
            ws = [width * x for x in b[3]] if b[3] else [width / len(b[1])] * len(b[1])
            table = Table(data, colWidths=ws, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183d4a")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f1f6f7")],
                        ),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5e1e6")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.extend([table, Spacer(1, 8)])
        elif b[0] == "figure":
            from PIL import Image as PILImage

            path = r.out / b[1]
            with PILImage.open(path) as im:
                w, h = im.size
            iw = width
            ih = iw * h / w
            if ih > 143 * mm:
                ih = 143 * mm
                iw = ih * w / h
            img = Image(str(path), width=iw, height=ih)
            caption = Paragraph(markup(b[2] + "。" + b[3]), styles["small"])
            story.extend([KeepTogether([Spacer(1, 8), img, Spacer(1, 5), caption]), Spacer(1, 12)])

    def page(canvas, doc):
        canvas.saveState()
        w, _h = A4
        canvas.setStrokeColor(colors.HexColor("#d1dde1"))
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 17 * mm, w - 20 * mm, 17 * mm)
        canvas.setFont("CJK", 7.2)
        canvas.setFillColor(colors.HexColor("#53717e"))
        canvas.drawString(20 * mm, 11.5 * mm, "统一 IMU 正式训练 | 冻结测试报告 | 2026-10-24")
        canvas.drawRightString(w - 20 * mm, 11.5 * mm, str(doc.page))
        canvas.restoreState()

    path = r.out / "完整训练报告.pdf"
    print(f"WRITES {path}", flush=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=23 * mm,
        title=TITLE,
        author="SmartPNT Virtual IMU",
        pageCompression=1,
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "runs/remote_results_formal_20261024/server/runs/formal_training_20261023_141916_733429"
        ),
    )
    ap.add_argument(
        "--output", type=Path, default=Path("runs/remote_results_formal_20261024/report")
    )
    ap.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"))
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(
        "SKIPS training, inference, data edits, test-driven tuning, latest snapshot inference, and source commits",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    r = build_report(args.run_root.resolve(), args.output.resolve(), repo)
    normalize_glyphs(r)
    write_markdown(r)
    write_html(r)
    if not args.no_pdf:
        write_pdf(r, args.font)
    files = [args.output / "完整训练报告.md", args.output / "完整训练报告.html"] + (
        [] if args.no_pdf else [args.output / "完整训练报告.pdf"]
    )
    audit = {
        "source_run": str(args.run_root),
        "seeds": SEEDS,
        "stats": "mean and sample SD ddof=1 across optimization seeds",
        "table_source": "each frozen evaluation/metrics.json",
        "outputs": [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in files],
    }
    path = args.output / "report_build_audit.json"
    print(f"WRITES {path}", flush=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
