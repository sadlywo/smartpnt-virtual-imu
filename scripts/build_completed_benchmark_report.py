"""Build a traceable Chinese report for the completed retrospective IMU matrix.

Reads only explicit run identities and audited tables; never trains, infers,
fits baselines, selects a candidate on test scores, or changes an old result.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev

TITLE = "统一 IMU 扩展模型完整对照报告"
SEEDS = [20261010, 20261011, 20261012]
GYRO = "gyro_vector_rmse_radps"
ACCEL = "accel_vector_rmse_mps2"
MODEL_ORDER = [
    "tcn_original",
    "plain_cnn",
    "gru",
    "transformer",
    "cnn_bilstm",
    "deepgyro_adapter",
    "tlio_resnet18_adapter",
    "tartanimu_window_adapter",
]
CLASSICAL = ["identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear"]
LABELS = {
    "tcn_original": "TCN（原固定候选）",
    "plain_cnn": "Temporal CNN",
    "gru": "GRU",
    "transformer": "Transformer",
    "cnn_bilstm": "CNN-BiLSTM",
    "deepgyro_adapter": "DeepGyro 骨干适配",
    "tlio_resnet18_adapter": "TLIO ResNet18 适配",
    "tartanimu_window_adapter": "TartanIMU 单窗口适配",
    "identity": "不校正 LOW",
    "bias_mean": "均值补偿",
    "affine": "仿射",
    "ridge_affine": "岭仿射",
    "learned_fir_causal": "因果 FIR",
    "arx_linear": "ARX（LOW 历史）",
    "zero": "零位移",
    "train_mean": "训练均值",
    "ridge": "Ridge",
}
SENSORS = ["BMX160", "MPU9250", "LSM9DS1", "Tango Top IMU"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    print(f"READS: {path}", flush=True)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    print(f"READS: {path}", flush=True)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def renderer(repo: Path):
    path = repo / "scripts/build_completed_formal_report.py"
    spec = importlib.util.spec_from_file_location("formal_report_rendering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    module.TITLE = TITLE
    return module


def ms(values, digits=6) -> str:
    values = list(values)
    if not values or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Empty or nonfinite report metric")
    if len(values) == 1 or max(values) == min(values):
        return f"{mean(values):.{digits}f}"
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def local_run(entry: dict, root: Path, old: Path) -> Path:
    folder = (
        old / "experiments" / Path(entry["run_dir"]).name
        if entry.get("reused")
        else root / "experiments" / Path(entry["run_dir"]).name
    )
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return folder


def source_runs(root: Path, old: Path):
    pipeline = read_json(root / "pipeline.json")
    entries = pipeline["runs"]
    assert pipeline["status"] == "complete" and len(entries) == 45
    assert len({(e["candidate_id"], e["task"], e["seed"]) for e in entries}) == 45
    assert sum(bool(e.get("reused")) for e in entries) == 6
    result = []
    for entry in entries:
        assert entry["status"] == "evaluated" and entry["seed"] in SEEDS
        folder = local_run(entry, root, old)
        history = read_csv(folder / "metrics.csv")
        selection = "val_grouped_score" if entry["task"] == "paired" else "val_rmse"
        best = min(history, key=lambda row: float(row[selection]))
        result.append(
            {
                **entry,
                "folder": folder,
                "history": history,
                "best_epoch": int(float(best["epoch"])),
                "epochs": len(history),
                "best_metric": float(best[selection]),
                "selection_metric": selection,
                "evaluation": read_json(folder / "evaluation/metrics.json"),
            }
        )
    return pipeline, result


def make_metric_maps(runs: list[dict]):
    paired, nav = {}, {}
    for run in runs:
        candidate, seed = run["candidate_id"], run["seed"]
        if run["task"] == "paired":
            for pair in run["evaluation"]["splits"]["test"]["pairs"]:
                sensor = pair["low_sensor"]
                for method, metrics in pair["metrics"].items():
                    key = (candidate if method == "model" else method, sensor, seed)
                    if key in paired:
                        if paired[key] != metrics:
                            raise ValueError(
                                f"Classical baseline differs between report sources: {key}"
                            )
                    paired[key] = metrics
        else:
            for method, metrics in run["evaluation"]["splits"]["test"]["metrics"].items():
                key = (candidate if method == "model" else method, seed)
                if key in nav and nav[key] != metrics:
                    raise ValueError(f"Navigation baseline differs between report sources: {key}")
                nav[key] = metrics
    return paired, nav


def figure_items(out: Path):
    paths = [
        out / "plotting_figure_manifest.json",
        out / "figure_manifest.json",
        out / "plotting_manifest.json",
        out / "figures/figure_manifest.json",
    ]
    for path in paths:
        if path.is_file():
            payload = read_json(path)
            figures = payload.get("figures", [])
            if figures:
                return path, figures
    raise FileNotFoundError("A completed new benchmark figure manifest is required")


def build(root: Path, old: Path, out: Path, repo: Path, *, require_figures: bool):
    render = renderer(repo)
    report = render.Report(out)
    pipeline, runs = source_runs(root, old)
    paired, nav = make_metric_maps(runs)
    analysis = read_json(out / "analysis.json")
    audit = read_json(out / "scientific_audit.json")
    passed = audit.get("passed", audit.get("status") in {"passed", "complete"})
    if not passed:
        raise ValueError("Independent scientific audit has not passed")
    report.sources.update(
        str(p.resolve())
        for p in [root / "pipeline.json", out / "analysis.json", out / "scientific_audit.json"]
    )
    new_server = root.parents[1]
    bundle = read_json(new_server / "bundle_manifest.json")
    workflow = (new_server / "runs/workflow_status.log").read_text(encoding="utf-8")
    assert "BENCHMARK_PASSED" in workflow and "WORKFLOW_EXIT=0" in workflow
    candidates = [
        candidate for candidate in MODEL_ORDER if any(r["candidate_id"] == candidate for r in runs)
    ]
    pair_models = [candidate for candidate in candidates if candidate != "tartanimu_window_adapter"]
    print("WRITES: Chinese report blocks and linked source index", flush=True)

    def metrics(method, sensor, metric):
        return [float(paired[(method, sensor, seed)][metric]) for seed in SEEDS]

    def nav_metrics(method, metric):
        return [float(nav[(method, seed)][metric]) for seed in SEEDS]

    # Independently check the principal conclusions against the frozen numbers.
    for candidate in pair_models:
        for sensor in SENSORS:
            for metric in [GYRO, ACCEL]:
                for baseline in ["learned_fir_causal", "arx_linear"]:
                    if mean(metrics(candidate, sensor, metric)) <= mean(
                        metrics(baseline, sensor, metric)
                    ):
                        raise ValueError(
                            "Principal paired conclusion disagrees with frozen metrics"
                        )
    for candidate in candidates:
        for baseline in ["zero", "train_mean"]:
            if mean(nav_metrics(candidate, "vector_rmse_m")) <= mean(
                nav_metrics(baseline, "vector_rmse_m")
            ):
                raise ValueError("Principal navigation conclusion disagrees with frozen metrics")

    def links(paths):
        return "；".join(report.link(p, p.name) for p in paths if p.is_file())

    report.h(TITLE)
    report.p("ALiMU / PennCOSYVIO 配对六轴映射 · SmartPNT-MSF 短窗口位移", "subtitle")
    report.p("39 次新增训练 + 6 次原 TCN 复用；45 / 45 均已评价", "subtitle")
    report.p("2026-10-24 · Asia/Shanghai · 回顾性固定协议比较", "small")
    report.table(
        ["完成核验", "结果"],
        [
            [
                "Pipeline / 工作流",
                "complete；45 个预期 task/model/seed 唯一；BENCHMARK_PASSED；退出码 0",
            ],
            ["配对方法", "7 个神经模型 + 6 个经典方法 = 13 个方法；四种器件映射"],
            ["导航方法", "8 个神经模型 + zero/train-mean/ridge = 11 个方法"],
            ["数据", "unified_imu_v2；944,840 行；145 个连续片段；划分及标签保持冻结"],
            ["统计", "三个优化种子，均值及样本 SD（ddof=1）；不是独立录制的置信区间"],
            [
                "完成时间",
                "2026-10-24 13:55:14（北京时间）；pipeline finished_at="
                + str(pipeline.get("finished_at")),
            ],
        ],
        [0.24, 0.76],
    )
    report.p(
        "本轮检验更多序列骨干在同一任务中的价值。论文骨干适配并非原论文完整复现；当前固定数据比较不代表全球 SOTA 排名。原测试结果已在新增候选前查看，因此明确标为回顾性扩展比较。",
        "callout",
    )
    report.p(
        "核心结果：七个神经方法的三种子均值，在四个器件的 gyro 与 accel 全部八项 RMSE 上都未超过 FIR/ARX；八个导航神经方法的三维位移 RMSE 均值都高于零位移与训练均值基线。扩展模型后仍未证明当前共同任务/训练协议带来超越强经典方法的收益。这个结论只针对本轮适配，不能用来否定各论文的原生完整方法。",
        "callout",
    )
    findings = analysis.get("findings", [])
    if isinstance(findings, dict):
        findings = [
            f"{key}：{value}"
            for key, value in findings.items()
            if isinstance(value, (str, int, float))
        ]
    for item in findings[:8]:
        if isinstance(item, str):
            report.p(item)
    report.p(
        "阅读重点：第 1 节给出全部已完成方法的物理指标；第 2-4 节说明数据、适配与训练控制；第 5-6 节给出训练身份及连续片段诊断；第 7-9 节列出误差解释、限制、复现和来源。所有候选都报告，不挑选测试最好的种子。",
        "small",
    )

    report.page()
    report.h("1  全部方法的测试对照")
    report.p(
        "表内每个神经模型使用预设的三个 seed=20261010/20261011/20261012。经典拟合只用 train，重复输出核对一致后归纳为一行；没有优化随机性的经典方法不人为添加误差条。gyro 和 accel 维持各自单位，不直接相加。"
    )
    for metric, title, digits in [
        (GYRO, "1.1  陀螺仪三维矢量 RMSE（rad/s）", 6),
        (ACCEL, "1.2  加速度三维矢量 RMSE（m/s^2）", 6),
    ]:
        report.h(title, 2)
        rows = [
            [LABELS[method], *(ms(metrics(method, sensor, metric), digits) for sensor in SENSORS)]
            for method in [*CLASSICAL, *pair_models]
        ]
        report.table(
            ["方法", "BMX160", "MPU9250", "LSM9DS1", "Tango Top"],
            rows,
            [0.26, 0.185, 0.185, 0.185, 0.185],
        )
        report.p(
            "前三个器件对应 VN100，Tango Top 对应 ADIS16488；矢量 RMSE=sqrt(mean(||prediction-reference||^2))。表的均值不等于将不同 seed 的预测拼成一条新信号。",
            "small",
        )
    report.h("1.3  相对原始 LOW 的变化", 2)
    for metric, title in [(GYRO, "Gyro 改善率（%）"), (ACCEL, "Accel 改善率（%）")]:
        rows = []
        for method in pair_models:
            cells = []
            for sensor in SENSORS:
                values = [
                    100
                    * (
                        1
                        - paired[(method, sensor, seed)][metric]
                        / paired[("identity", sensor, seed)][metric]
                    )
                    for seed in SEEDS
                ]
                cells.append(ms(values, 2))
            rows.append([LABELS[method], *cells])
        report.table([title, *SENSORS[:3], "Tango Top"], rows, [0.26, 0.185, 0.185, 0.185, 0.185])
    report.p(
        "正改善率代表低于原始 LOW，负值代表退化；它不是‘器件等级提升百分比’。相对 FIR/ARX 的完整变化在机器可读表中保留，不能只用不校正 LOW 作为容易超过的对手。"
    )
    report.p(
        "具体变化：CNN-BiLSTM 的三种子均值在八项配对 RMSE 上均优于不校正 LOW，其中 LSM9DS1 gyro 约为0.002447 rad/s，改善了原TCN该项退化。然而它在这些项目上仍未超过FIR/ARX。不同器件/信号的最低神经误差来自不同架构，不能把测试上的逐项最优拼成一个未经训练的‘组合模型’。"
    )
    report.h("1.4  SmartPNT 窗口端点位移（m）", 2)
    nav_order = ["zero", "train_mean", "ridge", *candidates]
    report.table(
        ["方法", "三维 RMSE", "水平 RMSE", "垂直 RMSE"],
        [
            [
                LABELS[m],
                *(
                    ms(nav_metrics(m, key), 4)
                    for key in ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]
                ),
            ]
            for m in nav_order
        ],
        [0.37, 0.21, 0.21, 0.21],
    )
    report.p(
        "这里只评价每个 1.975 秒窗口的 ENU 端点差。1649 个测试窗口相互重叠，不是 1649 条独立轨迹；不能将其重叠位移连续相加，也不能将这些指标改称 ATE/RPE。"
    )
    report.p(
        "导航三维结果中，TartanIMU单窗口适配的神经方法均值最低（约10.8315m），仍高于零位移10.2520m及训练均值10.6554m。垂直方向有局部收益，例如CNN约0.3082m、TLIO适配约0.3090m低于零位移约0.3266m；不能将‘三维均未超过基线’推广成每个方向都无改善。"
    )
    report.h("1.5  配对 MAE 与均值残差", 2)
    report.p(
        "MAE 下表定义为三个物理轴 MAE 的算术平均，gyro/accel分开；它与 mean(||error||) 的矢量范数 MAE 不同，后者在完整CSV中另列。均值残差范数定义为 ||mean(error_axis)||，不保留轴符号，逐轴有符号平均残差见逐轴表。"
    )
    for start, stop, unit, name in [(0, 3, "rad/s", "Gyro"), (3, 6, "m/s^2", "Accel")]:
        for kind, title in [("mae", "三个轴平均MAE"), ("bias", "均值残差范数")]:
            report.h(f"{name} {title}（{unit}）", 3)
            rows = []
            for method in [*CLASSICAL, *pair_models]:
                cells = []
                for sensor in SENSORS:
                    values = []
                    for seed in SEEDS:
                        metric = paired[(method, sensor, seed)]
                        value = (
                            mean(metric["axis_mae"][start:stop])
                            if kind == "mae"
                            else math.sqrt(
                                sum(v * v for v in metric["axis_mean_error"][start:stop])
                            )
                        )
                        values.append(value)
                    cells.append(ms(values, 6))
                rows.append([LABELS[method], *cells])
            report.table(
                ["方法", "BMX160", "MPU9250", "LSM9DS1", "Tango Top"],
                rows,
                [0.26, 0.185, 0.185, 0.185, 0.185],
            )
    report.h("1.6  导航逐轴 MAE 与有符号偏差（m）", 2)
    for key, title in [("axis_mae_m", "平均绝对误差"), ("axis_bias_m", "有符号平均误差")]:
        report.h(title, 3)
        report.table(
            ["方法", "East", "North", "Up"],
            [
                [
                    LABELS[method],
                    *(
                        ms([nav[(method, seed)][key][axis] for seed in SEEDS], 4)
                        for axis in range(3)
                    ),
                ]
                for method in nav_order
            ],
            [0.37, 0.21, 0.21, 0.21],
        )
    report.h("1.7  完整指标文件", 2)
    table_dir = out / "tables"
    for title, names in [
        (
            "均值/样本 SD 及全种子数值",
            [
                "paired_comparison.csv",
                "navigation_comparison.csv",
                "paired_per_seed.csv",
                "navigation_per_seed.csv",
            ],
        ),
        (
            "MAE、偏差、逐轴与连续片段",
            [
                "paired_axis_metrics_per_seed.csv",
                "paired_continuous_segments_per_seed.csv",
                "navigation_continuous_segments_per_seed.csv",
            ],
        ),
        (
            "改善率、训练登记和完整对照表",
            ["improvements.csv", "training_registry.csv", "complete_comparison.md"],
        ),
    ]:
        available = [table_dir / name for name in names if (table_dir / name).is_file()]
        if available:
            report.p(title + "：" + links(available), "small")
    report.p(
        "逐轴 MAE 和有符号平均残差分别报告，不能用矢量误差范数替代每个轴的偏差符号。连续片段表避免把采集缺口两侧拼接后计算时序诊断。"
    )

    report.page()
    report.h("2  数据、标签与拆分")
    report.p(
        "配对任务输入 LOW 六轴加参考类型的三个 one-hot 通道（共 9×80），预测 HIGH-LOW 的六轴标准化残差。反归一化后加回 LOW，得到 HIGH 的预测。推理不读取 HIGH 测量。VN100 与 ADIS16488 是实测参考，不是绝对无误差的真值。STIM300 类型通道保留在 schema 中，本轮没有对应样本映射。"
    )
    report.p(
        "SmartPNT 输入为六轴 6×80，输出固定 ENU 窗口首末位置差 3 维；其高等级六轴标签不可用，因此没有把导航真值伪造成 HIGH 六轴标签。没有真实温度输入，没有独立静态分支；AL stat_11 只在 train，不存在独立静态测试结论。AL 解析中的固定 25 值不是实测温度。"
    )
    report.table(
        ["数据集", "训练录制/轨迹", "验证", "测试"],
        [
            ["ALiMU → VN100", "dyn_01/02/04/05/06、stat_11", "dyn_07", "dyn_08/09"],
            ["Penn → ADIS16488", "af、as", "bf", "bs"],
            ["SmartPNT → ENU 位移", "Traj3", "Traj4", "Traj1"],
        ],
        [0.25, 0.38, 0.18, 0.19],
    )
    report.table(
        ["任务/器件", "Train 窗口", "Val 窗口", "Test 窗口"],
        [
            ["BMX160 / MPU9250 / LSM9DS1（各自）", 4071, 373, 746],
            ["Tango Top", 245, 98, 163],
            ["配对合计", 12458, 1217, 2401],
            ["SmartPNT", 2016, 3637, 1649],
        ],
        [0.46, 0.18, 0.18, 0.18],
    )
    report.p(
        "AL 三个 LOW 共享同次录制的 VN100，始终分配到同一个 split；同一录制的窗口不会跨 split。AL dyn_03 是 dyn_04 的 LOW 前缀，stat_10 是 stat_11 的 LOW 前缀，共六个较短器件变体归档；少数参考 gyro 不同，不能称整对文件完全重复。SmartPNT Traj2 与 Traj1 哈希确认重复并归档，不参加训练。原始数据保留。"
    )
    report.p(
        "共同采样率 40 Hz，gyro 为 rad/s，accel 为 m/s^2。高采样率 Penn/SmartPNT 输入经 63 tap FIR 抗混叠再规整；AL 原生 40 Hz 不重复滤波。大于 1.5 个原生周期的缺口及非有限/不合格真值区间切断片段，不跨缺口插值。80 点窗口、stride40，首末时间差为 79/40=1.975 秒。配对测试覆盖 96,320 个唯一时刻，227 个片段尾部点不满足完整窗口而未评价。"
    )
    report.p(
        "Penn 采用已记录的刚体参考系及杆臂处理；AL 使用上游轴约定，外参未另行独立标定。SmartPNT 只使用 Fixed 真值合格区间，ENU 原点按轨迹定义。FusionPortable 因严格时钟/因果滤波约束下缺少合格完整测试窗口而未纳入，NavINST 缺少必要标定也未纳入。不能把这两个数据集写成已训练来源。"
    )
    first_pair = next(r for r in runs if r["task"] == "paired")
    first_nav = next(r for r in runs if r["task"] == "navigation")
    report.p(
        "冻结数据证据："
        + links(
            [
                first_pair["folder"] / "data_manifest.json",
                first_pair["folder"] / "split_manifest.json",
                first_nav["folder"] / "split_manifest.json",
                first_pair["folder"] / "data_verification.json",
            ]
        ),
        "small",
    )

    report.page()
    report.h("3  模型范围与可复现程度")
    report.table(
        ["模型", "配对 / 导航参数", "上下文与适配范围"],
        [
            ["原 TCN", "147,782 / 146,435", "因果逐点；原六次运行，旧源码身份保留"],
            ["Temporal CNN", "91,206 / 89,667", "四层居中卷积；窗口方法，不是 Calib-Net 完整复现"],
            ["GRU", "39,750 / 38,979", "单向两层 hidden64；因果逐点"],
            ["Transformer", "563,334 / 562,947", "两层 d64/4heads/FFN2048；全窗口注意力，无 mask"],
            ["CNN-BiLSTM", "99,206 / 98,051", "CNN 加双向 LSTM；通用架构，不是 ALiMU 原方法复现"],
            ["DeepGyro 骨干", "77,766 / 77,043", "核对公开四层空洞卷积；共同输入/输出/损失适配"],
            [
                "TLIO ResNet18",
                "3,850,566 / 3,847,683",
                "完整四阶段骨干；改用稠密输出，无原滤波及协方差",
            ],
            [
                "TartanIMU 窗口",
                "不适用 / 4,537,859",
                "ResNet-LSTM；只有导航；单窗口、单头、无协方差/预训练",
            ],
        ],
        [0.23, 0.26, 0.51],
    )
    report.p(
        "参数量不同，因此这是固定任务/训练预算下的架构对照，不是等参数量或等计算量对照。合成 GPU 预检检查每个新增模型/任务的 batch128 前向、共同损失反向及 AdamW 更新，重复两次相同种子获得相同输出/模型指纹；它不测量科学性能，也不能作为真实部署延迟基准。"
    )
    report.p(
        "DeepGyro 采用 6→16→32→64→128→3 原骨干通道日程（本轮首尾通道按任务改变），kernel 7/7/7/7/1、dilation 1/4/16/64/1、BN/GELU/dropout。感受野 511 点大于本任务 80 点，大量左历史由边界复制填充。原方法的 gyro 3×3 校准、std 包装和姿态积分损失没有复现，不能把本轮成绩等同于原论文。"
    )
    report.p(
        "TLIO 原骨干经 stride32 下采样，80 点只剩约 3 个粗时间特征，本轮以最近邻映射回 80 点再输出六轴/三维，稠密头的时间分辨率有限。原方法均值/不确定度头、姿态辅助与融合滤波未复现。TartanIMU 以一个窗口构成一个 LSTM token，序列长度为 1，没有跨窗口记忆，也不验证原预训练及多平台学习收益。"
    )
    report.p(
        "配对逐点因果方法为 TCN、GRU、DeepGyro 适配。CNN、Transformer、CNN-BiLSTM、TLIO 稠密适配可利用同一窗口未来样本，应单独理解为离线/窗口处理方法；不能据其分数声称满足流式补偿。导航本来是整个已观测窗口到端点差的映射，时间接口与逐点信号补偿不同。"
    )
    report.p(
        "本次冻结的模型实现："
        + links(
            [
                new_server / "src/virtual_imu/models/tcn.py",
                new_server / "src/virtual_imu/models/gru.py",
                new_server / "src/virtual_imu/models/transformer.py",
                new_server / "src/virtual_imu/models/advanced.py",
                new_server / "src/virtual_imu/models/published_backbones.py",
                new_server / "src/virtual_imu/models/tartan_window.py",
            ]
        ),
        "small",
    )
    report.p(
        "CTIN 未公开可运行完整代码；Calib-Net 没有找到可核查官方网络代码；ALiMU 公开数据仓库未核实模型代码；AirIMU/AirIO 的原生融合/不确定度任务与本轮不同。本轮没有使用这些名字虚报原方法结果。旧 FSDA 简化模块未实现域适配目标且 reconstruction 无当前损失梯度，也不纳入论文方法比较。未修改 U-Net 的 CUDA 线性插值反向不符合严格确定性，因此未进入本次冻结名单。"
    )
    report.h("3.1  经典方法的真实定义", 2)
    report.table(
        ["方法", "训练与推理信息"],
        [
            ["Identity", "直接使用 LOW，无拟合"],
            ["Mean bias", "train 上 HIGH-LOW 均值偏移"],
            ["Affine", "当前 LOW 六轴到 HIGH 六轴的线性映射加截距"],
            ["Ridge affine", "同上，加岭正则"],
            ["Causal FIR", "9 tap 因果 LOW 历史，train 拟合"],
            ["ARX（代码名称）", "12 tap LOW 历史及 LOW 一阶差分；无过去 HIGH 输出反馈"],
            ["导航 zero / mean / ridge", "零位移、train 位移均值、train 窗口线性岭映射"],
        ],
        [0.27, 0.73],
    )
    report.p(
        "经典配对模型按器件对分别拟合；神经模型在 AL/Penn 联合训练，并以参考类型条件输入。这一容量/分组差异是比较条件的一部分，不能把它省略成完全相同学习问题。所有方法的评价时间点一致，经典拟合不读验证/测试标签。"
    )

    report.page()
    report.h("4  固定训练、验证选择与计算环境")
    report.table(
        ["控制项", "配对信号", "SmartPNT 导航"],
        [
            ["输入/目标", "9×80 → 6×80 标准化残差", "6×80 → 3 维标准化窗口位移"],
            ["种子/预算", "20261010/11/12；最多60轮；验证早停10", "同左"],
            ["优化器", "AdamW；lr0.001；wd1e-5；batch128；梯度裁剪1", "同左"],
            ["主损失", "(2×gyro Huber + accel Huber)/3", "标准化位移 Huber"],
            ["附加项", "差分0.02；频谱0.02；积分代理0.005；Allan0", "无速度/航向积分附加项"],
            ["LR 调度", "ReduceLROnPlateau：patience5，factor0.5，min5e-5", "沿用原任务无LR调度"],
            ["保存 checkpoint", "验证分数据集/器件组相对RMSE", "验证分量RMSE（m）"],
            ["采样/归一化", "train拟合；AL/Penn期望等权采样", "train拟合；轨迹窗口shuffle"],
        ],
        [0.22, 0.42, 0.36],
    )
    report.p(
        "共同训练损失用于隔离骨干选择；它不等于原论文的原生训练目标。积分项是在标准化残差信号上的加和代理，不是完整 SO(3) 预积分或物理单位的惯导损失。验证指标和测试指标分别保留，测试结果没有反向用于本轮 checkpoint、超参数或候选选择。"
    )
    report.p(
        "配对验证分数的顺序是：每个器件对在覆盖时间点计算gyro/accel物理矢量RMSE，分别除以相同时间点的LOW基线RMSE；两类比值平均，再对同一数据集内的器件对平均，最后对AL/Penn两个数据集等权平均。基线分母下限1e-12。该分数无量纲，只用于验证选择；它不把rad/s与m/s^2物理误差直接相加。重叠窗口在同一时刻的物理预测先作算术平均再评分。"
    )
    report.p(
        "环境：Ubuntu22.04 / Python3.12 / PyTorch2.4.0+cu121 / NumPy1.26.4 / RTX4090。关闭 TF32，CUBLAS_WORKSPACE_CONFIG=:4096:8，严格确定性算法，Transformer 使用数学 SDPA。重复相同种子仅控制随机过程；不同架构初始化消耗不同数量的随机数，导航 minibatch 顺序不保证跨架构逐批完全一致。"
    )
    report.p(
        "新模型源码与配置已打包冻结。原 TCN 只复用原固定候选的六次运行，先核验 data/training/model 等价及 checkpoint、指标、预测完整性；旧模型依然引用旧源码指纹，不能改写成由新源码训练。"
    )
    report.p(
        "配置及环境证据："
        + links(
            [
                root / "matrix_config.yaml",
                root / "configs/paired_source.yaml",
                root / "configs/navigation_source.yaml",
                new_server / "runs/server_deployment_20261024/benchmark_gpu_preflight.json",
                new_server / "bundle_manifest.json",
            ]
        ),
        "small",
    )
    report.p(
        "共同损失与训练实现："
        + links(
            [
                new_server / "src/virtual_imu/models/losses.py",
                new_server / "src/virtual_imu/training/trainer.py",
                new_server / "src/virtual_imu/training/gt_trainer.py",
                new_server / "src/virtual_imu/training/benchmark_pipeline.py",
            ]
        ),
        "small",
    )

    report.page()
    report.h("5  四十五次训练的最优 epoch 与来源")
    report.p(
        "下表逐一列出全部三种子。‘训练轮数<60’表示触发固定验证早停，所有 checkpoint 均来自相应验证指标的最优 epoch。未因测试分数决定保留某个种子。链接中的配置、模型和评价分别绑定实际 run 身份。"
    )
    report.p(
        "21次配对运行全部训练至60轮，未触发早停。导航大多在11-13轮结束、最优在第1-3轮；TLIO导航三种子分别训练34/19/30轮、最优24/9/20轮；TartanIMU导航均训练11轮、最优均为第1轮。这些早期最优与基线劣化共同提示当前导航监督条件和分布泛化问题，不能据此认为增加训练轮数就能解决。"
    )
    for candidate in candidates:
        for task, label in [("paired", "配对六轴"), ("navigation", "窗口位移")]:
            selected = sorted(
                [run for run in runs if run["candidate_id"] == candidate and run["task"] == task],
                key=lambda run: run["seed"],
            )
            if not selected:
                continue
            report.h(LABELS[candidate] + " / " + label, 3)
            report.table(
                ["种子", "轮数 / 最优", "最优验证值", "来源"],
                [
                    [
                        run["seed"],
                        f"{run['epochs']} / {run['best_epoch']}",
                        f"{run['best_metric']:.6f}",
                        "；".join(
                            [
                                report.link(run["folder"] / "config.yaml", "配置"),
                                report.link(run["folder"] / "best.ckpt", "模型"),
                                report.link(run["folder"] / "evaluation/metrics.json", "评价"),
                            ]
                        ),
                    ]
                    for run in selected
                ],
                [0.18, 0.19, 0.23, 0.40],
            )
    report.p(
        "完整训练登记、哈希与所有预测来源："
        + links(
            [
                table_dir / "training_registry.csv",
                out / "analysis.json",
                out / "scientific_audit.json",
            ]
        ),
        "small",
    )

    report.page()
    report.h("6  科研图与连续片段诊断")
    report.p(
        "以下图均从本轮实际完成结果生成。三种子曲线/误差条表示优化随机性；代表性时序固定 seed20261010，并按词典序首个合格连续测试片段选择，截取前至多20秒，不以模型测试误差挑选有利片段。各物理单位独立分轴显示。"
    )
    if require_figures:
        manifest_path, figures = figure_items(out)
        report.sources.add(str(manifest_path.resolve()))
        for number, figure in enumerate(figures, start=1):
            item = dict(figure)
            item.setdefault("title", item.get("stem", f"科研图 {number}"))
            item.setdefault(
                "caption", item.get("caption_draft", "源数据与图注见 figure_index.md。")
            )
            files = item.get("files", {})
            if isinstance(files, list):
                item["files"] = {Path(path).suffix.lstrip("."): path for path in files}
            if "files" not in item:
                item["files"] = {"png": "figures/" + item["stem"] + ".png"}
            if isinstance(item["files"], dict) and isinstance(item["files"].get("png"), dict):
                item["files"]["png"] = item["files"]["png"]["path"]
            report.figure(item, number)
    report.p(
        "信号积分误差在每个连续片段/评价窗口起点重置，gyro 的加和积分单位为 rad、accel 为 m/s；这些量没有姿态耦合和重力误差传播，不能当作真实姿态/位置漂移或 ATE。PSD 与相干性仅在同一有激励片段分析，未将缺口两侧拼接。PSD 的高频变化也可能来自不同参考噪声、滤波和动态响应，不自动证明硬件噪声被消除。"
    )

    report.page()
    report.h("7  误差来源、失败/退化与解释边界")
    report.p(
        "物理误差常见来源包括零偏及漂移、比例因子、轴不正交/交叉耦合、随机噪声与低频相关波动、非线性和振动响应，以及温度依赖。LOW-HIGH 误差还包含时间对齐、安装/坐标变换、杆臂、抗混叠/重采样差异及 HIGH 自身误差。本轮实测残差不能唯一分离这些来源；没有温度输入，不能给出温漂消除率。"
    )
    report.p(
        "严格可计算的统计恒等式为 vector_RMSE^2 = ||mean(error)||^2 + sum(population_std(error_axis)^2)。第一项是测试区间经验平均残差的平方，第二项是中心化残差均方；前者不是独立静态零偏标定，后者不是纯随机噪声估计。AL 未校正 accel 的第一项约占63%-68%，gyro约1%-2%，这只支持经验统计解释。"
    )
    report.p(
        "必须保留每种器件的退化项和不及强基线的结果。原 TCN 在四种器件 gyro/accel 共八项 RMSE 中七项优于 LOW，但 LSM9DS1 gyro 退化约37.5%，而 FIR/ARX 均优于原 TCN。新增方法是否改善这一局面，应由第1节全部方法表和完整补充指标判断，不通过把 gyro 与 accel 的物理数值混合来掩盖退化。"
    )
    report.p(
        "导航全局 ENU 位移存在初始朝向和速度的不充分可观测性；当前窗口输入没有独立已知的初始航向/速度。训练轨迹只有一条，验证和测试的运动分布可能不同。即使某个模型低于零位移，也不能由此证明可部署全局惯导，更不能把该位移任务改善当成 LOW→HIGH 六轴精度升级。"
    )
    report.p(
        "本轮45次都完成不表示每个方法效果都好。完整表保留所有候选，未删除不利种子；seed SD不等于独立录制的置信区间。AL三器件共享参考且只有两条动态测试录制，Penn仅一个测试录制，SmartPNT仅一条测试轨迹，泛化证据仍有限。"
    )

    report.h("8  后续实验建议（不改变本轮结果）")
    report.p(
        "优先建立新的独立采集/留出录制，核查器件间轴约定、时钟、安装及参考稳定性，再评估更多数据是否改变结论。针对配对任务，可在新协议中分开研究共同损失与单纯监督损失、等参数容量、联合与逐器件训练，以及更长历史/真实在线接口；不能在已查看测试集上反复选择最优组合后声称未见测试泛化。"
    )
    report.p(
        "导航下一步宜将任务改成可观测条件更清楚的局部坐标速度/位移，或显式提供初始朝向/速度，并按照官方方法需要的多窗口、原生损失和融合流程建立单独复现。TartanIMU的单token适配不宜据此否定其原论文方法，TLIO粗稠密头也不宜据此否定其原惯导系统。温度研究需真实温度记录及覆盖温区的独立测试。"
    )

    report.page()
    report.h("9  复现与证据索引")
    report.p(
        "本报告从不可变源结果读取数据，不进行训练、推理、基线重拟合或对齐修补。runs/<experiment_id>/ 为来源目录，outputs/latest 未用于推定身份。原六次复用记录使用原源码和模型来源。"
    )
    report.table(
        ["身份", "SHA256 / 指纹"],
        [
            ["新训练 bundle", "cd654e5003339b374d3fe2ebbe64976d6683e88fae3fc1f78da126c332984230"],
            ["新源码", bundle["source_fingerprint"]],
            ["共同数据 manifest", bundle["dataset_manifest_sha256"]],
            ["原 TCN 源码", "1a60647ce6ca7d84bf8729b032893a527af63bc731945f307d3b7aca8d8c4b2f"],
            [
                "原固定候选 bundle",
                "9ef6426fdd7d5fc924ee92be2b8597a31e594f9ea10b2861228f6bbfe1e4972a",
            ],
        ],
        [0.25, 0.75],
    )
    report.p(
        "来源文件："
        + links(
            [
                root / "pipeline.json",
                root / "matrix_config.yaml",
                root / "source_contract.json",
                new_server / "bundle_manifest.json",
                new_server / "runs/workflow_status.log",
                new_server / "runs/benchmark_export_20261024/verification.json",
                out / "scientific_audit.json",
            ]
        ),
        "small",
    )
    report.p(
        "完整对照表："
        + links(
            [
                root / "paired_summary.csv",
                root / "navigation_summary.csv",
                root / "paired_all_seeds.csv",
                root / "navigation_all_seeds.csv",
                table_dir / "complete_comparison.md",
            ]
        ),
        "small",
    )
    report.p(
        "论文来源：DeepGyro [RA-L 2020](https://doi.org/10.1109/LRA.2020.3003256)、TLIO [RA-L 2020](https://doi.org/10.1109/LRA.2020.3007421)、[TartanIMU 作者实现](https://github.com/superxslam/TartanIMU/tree/8a8025000f85d2c6b0692b88e15c2c91e761f7d7)。DeepGyro实现固定提交9948b664，MIT；TLIO为CathIAS/TLIO固定d7051c4f，BSD-3-Clause；TartanIMU固定8a802500，Apache-2.0。详见冻结扩展协议与逐方法文献来源记录。",
        "small",
    )
    report.code(
        "printf 'READS: fixed source bundle, curated corpus and YAMLs\\nWRITES: independent new run\\nSKIPS: old results and test-driven tuning\\n'\nbash scripts/server/bootstrap.sh --require-gpu\nbash scripts/server/run.sh benchmark --config configs/train_benchmark_matrix.yaml"
    )
    report.p(
        "以上命令用于从冻结包在独立位置复现实验，不是重新覆盖本次运行。部分优化器训练没有完整续跑状态，失败不能静默重训；本次所有预期项已完成，下载、核验、报告与后续监测阶段分别记录状态。",
        "small",
    )
    report.sources.update(str(run["folder"] / "evaluation/metrics.json") for run in runs)
    return render, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "runs/remote_results_benchmark_20261024/server/runs/benchmark_training_20261024_031951_978154"
        ),
    )
    parser.add_argument(
        "--old-run-root",
        type=Path,
        default=Path(
            "runs/remote_results_formal_20261024/server/runs/formal_training_20261023_141916_733429"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/remote_results_benchmark_20261024/report")
    )
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"))
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--allow-missing-figures", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(
        "READS: completed45run pipeline, exact old six reuse identities, independent scientific audit, all source metrics and new benchmark figure manifest",
        flush=True,
    )
    print(
        f"WRITES: {args.output}/完整模型对照报告.md/.html/.pdf and report_manifest.json", flush=True
    )
    print(
        "SKIPS: training, inference, fits, test selection, original result edits, credentials and latest snapshots",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    render, report = build(
        args.run_root.resolve(),
        args.old_run_root.resolve(),
        args.output.resolve(),
        repo,
        require_figures=not args.allow_missing_figures,
    )
    report.sources.update(
        str(path.resolve())
        for path in [
            Path(__file__),
            repo / "scripts/build_completed_formal_report.py",
            repo / "docs/BENCHMARK_EXPANSION_20261024.md",
        ]
    )
    render.normalize_glyphs(report)
    render.write_markdown(report)
    render.write_html(report)
    if not args.no_pdf:
        original_blocks = report.blocks
        report.blocks = []
        for block in original_blocks:
            if block[0] == "code":
                report.blocks.extend(("paragraph", "small", line) for line in block[1].splitlines())
            else:
                report.blocks.append(block)
        render.write_pdf(report, args.font)
        report.blocks = original_blocks
    outputs = []
    for suffix in ["md", "html", *([] if args.no_pdf else ["pdf"])]:
        source = args.output / f"完整训练报告.{suffix}"
        dest = args.output / f"完整模型对照报告.{suffix}"
        source.replace(dest)
        outputs.append({"path": dest.name, "bytes": dest.stat().st_size, "sha256": digest(dest)})
    manifest = {
        "title": TITLE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_run": str(args.run_root),
        "previous_run": str(args.old_run_root),
        "runs": 45,
        "new_runs": 39,
        "reused_runs": 6,
        "seeds": SEEDS,
        "statistics": "mean and sampleSD, ddof1, optimization seeds; not independent recordings",
        "outputs": outputs,
        "source_files": [
            {
                "path": Path(
                    os.path.relpath(Path(path).resolve(), args.output.resolve())
                ).as_posix(),
                "sha256": digest(Path(path)),
            }
            for path in sorted(report.sources)
            if Path(path).is_file()
        ],
        "visual_review": "pending; root inspects final rendered pages before delivery",
    }
    (args.output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"CREATED: {len(report.blocks)} content blocks; {len(outputs)} report outputs", flush=True
    )


if __name__ == "__main__":
    main()
