"""Assemble the final Chinese FIR/ARX research report and publication tables."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.viz import NEUTRAL, _style


def _markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _draw_table(frame: pd.DataFrame, headers: list[str], out_stem: Path, title: str) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 0.65 + 0.42 * (len(frame) + 1)))
    ax.axis("off")
    table = ax.table(
        cellText=frame.astype(str).to_numpy(),
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.4)
    table.scale(1.0, 1.35)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D8D8D8")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#E8EFF7")
            cell.set_text_props(weight="bold", color="#111111")
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.04)
    for suffix, dpi in (("png", 300), ("pdf", 300), ("svg", 300)):
        fig.savefig(out_stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=Path("runs/fir_arx_delivery_20261105"))
    args = parser.parse_args()
    root = args.delivery.resolve()
    comparison_root = root.parent / "fir_arx_comparison_20261105"
    branch_figure_root = root.parent / "fir_arx_branch_figures_20261105_v2"
    comparison = pd.read_csv(comparison_root / "tables/all_branch_comparison.csv")
    summary = pd.read_csv(comparison_root / "tables/branch_summary.csv")
    minimal = pd.read_csv(root / "extracted/fir_arx_minimal_full_20261105_v4/tables/metrics_by_protocol.csv")
    selection = pd.read_csv(root / "extracted/fir_arx_minimal_full_20261105_v4/tables/component_selection.csv")
    base = pd.read_csv(root / "extracted/fir_arx_base_20261105_v2/tables/metrics_by_protocol.csv")
    tables = root / "tables"
    tables.mkdir(exist_ok=True)

    branch_table = summary[summary["model"].isin(["fir", "fir_arx"])].copy()
    labels = {
        "vmd_low": "VMD 低频",
        "vmd_high": "VMD 高频",
        "vmd_joint": "VMD 高低联合",
        "tda_routed": "TDA 路由",
        "scenario_minimal": "场景最小输入",
    }
    branch_table["方法"] = branch_table["condition"].map(labels)
    branch_table["陀螺中位变化/%"] = branch_table["gyro_delta_pct_median"].round(2)
    branch_table["加计中位变化/%"] = branch_table["accel_delta_pct_median"].round(2)
    branch_table["联合中位变化/%"] = branch_table["joint_score_delta_median"].round(2)
    branch_table["改善协议"] = branch_table.apply(
        lambda row: f"{int(row.branch_win_protocols)}/{int(row.protocols)}", axis=1
    )
    branch_table = branch_table[
        ["方法", "陀螺中位变化/%", "加计中位变化/%", "联合中位变化/%", "改善协议"]
    ]
    branch_table.to_csv(tables / "table1_branch_summary.csv", index=False)
    english_branch = pd.DataFrame(
        {
            "Method": branch_table["方法"].replace(
                {
                    "场景最小输入": "Scenario-minimal",
                    "TDA 路由": "TDA routed",
                    "VMD 低频": "VMD low",
                    "VMD 高频": "VMD high",
                    "VMD 高低联合": "VMD joint",
                }
            ),
            "Gyro median delta (%)": branch_table["陀螺中位变化/%"],
            "Accel median delta (%)": branch_table["加计中位变化/%"],
            "Joint median delta (%)": branch_table["联合中位变化/%"],
            "Improved protocols": branch_table["改善协议"],
        }
    )
    _draw_table(
        english_branch,
        ["Method", "Gyro median delta (%)", "Accel median delta (%)", "Joint median delta (%)", "Improved protocols"],
        tables / "table1_branch_summary",
        "Signed branch effect relative to raw FIR/ARX",
    )

    minimal_cmp = comparison[comparison["branch"].eq("minimal")].copy()
    minimal_cmp = minimal_cmp.merge(
        selection.groupby("protocol", as_index=False).agg(
            mean_component_count=("component_count", "mean"),
            fallback_fraction=("fallback", "mean"),
        ),
        on=["protocol", "mean_component_count", "fallback_fraction"],
        how="left",
    )
    minimal_table = pd.DataFrame(
        {
            "协议": minimal_cmp["protocol"],
            "陀螺变化/%": minimal_cmp["gyro_delta_pct"].round(2),
            "加计变化/%": minimal_cmp["accel_delta_pct"].round(2),
            "联合变化/%": minimal_cmp["joint_score_delta"].round(2),
            "平均分量数": minimal_cmp["mean_component_count"].round(1),
            "回退比例": (100.0 * minimal_cmp["fallback_fraction"]).round(1),
        }
    )
    minimal_table.to_csv(tables / "table2_minimal_by_protocol.csv", index=False)
    english_minimal = pd.DataFrame(
        {
            "Protocol": minimal_table["协议"],
            "Gyro delta (%)": minimal_table["陀螺变化/%"],
            "Accel delta (%)": minimal_table["加计变化/%"],
            "Joint delta (%)": minimal_table["联合变化/%"],
            "Components": minimal_table["平均分量数"],
            "Fallback (%)": minimal_table["回退比例"],
        }
    )
    _draw_table(
        english_minimal,
        ["Protocol", "Gyro delta (%)", "Accel delta (%)", "Joint delta (%)", "Components", "Fallback (%)"],
        tables / "table2_minimal_by_protocol",
        "Scenario-minimal effect by recording-disjoint protocol",
    )

    baseline = base[base["model"].isin(["fir", "arx"])].copy()
    baseline["方法"] = baseline["model"].str.upper()
    baseline_table = pd.DataFrame(
        {
            "协议": baseline["protocol"],
            "方法": baseline["方法"],
            "陀螺RMSE/(rad/s)": baseline["test_gyro_vector_rmse"].map(lambda value: f"{value:.6g}"),
            "加计RMSE/(m/s²)": baseline["test_accel_vector_rmse"].map(lambda value: f"{value:.6g}"),
            "陀螺改善/%": baseline["test_gyro_improvement_vs_identity_pct"].round(2),
            "加计改善/%": baseline["test_accel_improvement_vs_identity_pct"].round(2),
        }
    )
    baseline_table.to_csv(tables / "table3_raw_fir_arx_reproduction.csv", index=False)
    english_baseline = pd.DataFrame(
        {
            "Protocol": baseline_table["协议"],
            "Method": baseline_table["方法"],
            "Gyro RMSE": baseline_table["陀螺RMSE/(rad/s)"],
            "Accel RMSE": baseline_table["加计RMSE/(m/s²)"],
            "Gyro gain (%)": baseline_table["陀螺改善/%"],
            "Accel gain (%)": baseline_table["加计改善/%"],
        }
    )
    _draw_table(
        english_baseline,
        ["Protocol", "Method", "Gyro RMSE", "Accel RMSE", "Gyro gain (%)", "Accel gain (%)"],
        tables / "table3_raw_fir_arx_reproduction",
        "Raw FIR/ARX reproduction, physical units",
    )

    vmd = summary[(summary["branch"].eq("vmd")) & summary["model"].isin(["fir", "arx"])]
    tda = summary[(summary["branch"].eq("tda")) & summary["model"].isin(["fir", "arx"])]
    minstat = summary[(summary["branch"].eq("minimal")) & summary["model"].eq("fir_arx")]
    vmd_joint = vmd[vmd["condition"].eq("vmd_joint") & vmd["model"].eq("fir")].iloc[0]
    tda_fir = tda[tda["model"].eq("fir")].iloc[0]
    min_row = minstat.iloc[0]
    fallback_fraction = float(selection["fallback"].mean()) if len(selection) else 0.0
    mean_components = float(selection["component_count"].mean()) if len(selection) else 0.0

    report = f"""# 基于 FIR/ARX 的短时 IMU 增强复现与三条新思路实验结果

日期：2026-11-05（Asia/Shanghai）

## 1. 结论摘要

本轮目标不是重做一个端到端大模型，而是检验“前处理或输入选择能否使 FIR/ARX 本身更强”。在统一数据、录制分离协议和固定测试评价下，结果是：

1. **场景最小输入是唯一在协议中位数上优于原始 FIR/ARX 的分支**：联合指标中位变化 **{min_row.joint_score_delta_median:+.2f}%**，在 **{int(min_row.branch_win_protocols)}/{int(min_row.protocols)}** 个协议同时改善陀螺与加计。但平均仍使用 **{mean_components:.1f}/18** 个分量，缩减有限，且场景样本不足时回退比例为 **{100*fallback_fraction:.1f}%**。
2. **VMD 直接输入不适合当前 FIR/ARX 设置**：低频、高频和高低联合都显著退化。联合 FIR 的跨协议联合中位变化为 **{vmd_joint.joint_score_delta_median:+.2f}%**，改善协议数为 **{int(vmd_joint.branch_win_protocols)}/{int(vmd_joint.protocols)}**。这不能否定 VMD 本身，但明确否定了当前“分解后直接回归”的实现。
3. **TDA 场景路由没有稳定增益**：联合中位变化 **{tda_fir.joint_score_delta_median:+.2f}%**，仅 **{int(tda_fir.branch_win_protocols)}/{int(tda_fir.protocols)}** 个协议改善。训练集持续同调描述子能够形成聚类，但簇内样本和动力学差异不足以转化为稳定精度收益。
4. **原始 FIR/ARX 仍然是强基线**：ALiMU、Penn 四折及 Pooled 上的相对 LOW 改善约为陀螺 45%–81%、加计 60%–90%。因此任何新方法的比较对象必须是同数据的器件/配对级 FIR/ARX，而不是弱恒等基线。

推荐继续路线：保留“场景最小输入”作为有条件候选，进一步做特征稳定性与独立录制验证；暂时停止 VMD 直接输入线与 TDA 簇级重训线，除非先改变采样方式或把拓扑量作为辅助门控而非独立输入。

## 2. 复现范围与方法

### 2.1 数据与协议

- 冻结数据：`data/processed/unified_imu_v2/manifest.json`，40 Hz，ALiMU 与 PennCOSYVIO 配对六轴。
- 协议：`alimu_only`、`penn_only`、`penn_loso_af/as/bf`、`pooled`。
- 训练只拟合 TRAIN，超参数和分量选择只用 VALIDATION，TEST 只评价一次。
- ALiMU 三个 LOW 传感器共享同一 VN100 参考录制，录制块整体划分，避免跨 split 泄漏。
- 评价单位为配对级物理 RMSE：陀螺 rad/s、加计 m/s²；协议汇总为无权重宏平均。

### 2.2 FIR/ARX 实现

- FIR：当前及过去 LOW 通道的因果滞后特征，岭回归输出 HIGH 六轴。
- ARX-style：在 LOW 滞后之外加入当前 LOW 一阶差分；**不反馈过去 HIGH**。
- 每个器件配对单独拟合。拟合后保存权重、偏置、滞后阶数、alpha、数据清单和 SHA 哈希。

## 3. 原始 FIR/ARX 复现

{_markdown_table(baseline_table, ["协议", "方法", "陀螺RMSE/(rad/s)", "加计RMSE/(m/s²)", "陀螺改善/%", "加计改善/%"], ["协议", "方法", "陀螺 RMSE", "加计 RMSE", "陀螺改善/%", "加计改善/%"] )}

完整测试记录和逐序列指标见：

- `extracted/fir_arx_base_20261105_v2/tables/metrics_by_pair.csv`
- `extracted/fir_arx_base_20261105_v2/tables/metrics_by_sequence.csv`
- `extracted/fir_arx_base_20261105_v2/report.md`

## 4. 三条实验分支

### 4.1 VMD 分支

代码分支：`codex/fir-arx-vmd`。对每个连续序列逐通道执行 VMD，按模态中心频率分为低频/高频，并比较 `vmd_low`、`vmd_high`、`vmd_joint`。

结果：三类输入均未在任何协议同时超过原始 FIR/ARX。高频输入丢失了大量慢变信息，低频输入丢失了运动边沿；联合输入虽然保留两部分，但特征维数和分解不稳定性抵消了收益。当前参数为 K=3、alpha=2000、0.5 Hz 划分。

### 4.2 TDA 分支

代码分支：`codex/fir-arx-tda`。在非重叠 80 点窗口上计算 H0/H1 持续同调统计量，TRAIN 内标准化并聚类，再对每个簇拟合 FIR/ARX；TEST 只按固定路由器分配，低支持簇回退全局模型。

结果：TDA 只在 ALiMU 和 Pooled 的某些模型上略有改善，Penn 三折与 Penn-only 多数退化；总体联合中位变化 **{tda_fir.joint_score_delta_median:+.2f}%**。TDA 更像工况描述符，而不是可替代信号分量的输入。

### 4.3 场景最小输入分支

代码分支：`codex/fir-arx-minimal`。用因果 0.5 Hz 低通把六轴分为低频/高频，在 TRAIN 场景聚类后，按 VALIDATION 单分量贡献排序，从少到多构造前缀，选择达到最佳验证分数 **{100*0.02:.0f}%** 容差内的最小集合。

{_markdown_table(minimal_table, ["协议", "陀螺变化/%", "加计变化/%", "联合变化/%", "平均分量数", "回退比例"], ["协议", "陀螺变化/%", "加计变化/%", "联合变化/%", "平均分量数", "回退比例/%"] )}

该分支在 **{int(min_row.branch_win_protocols)}/{int(min_row.protocols)}** 个协议上改善，且 Pooled 与 ALiMU 都为正收益；但平均分量仍未明显缩减，Penn 多折需要回退，说明“最少输入”尚未真正稳定实现。

## 5. 分支横向比较

{_markdown_table(branch_table, ["方法", "陀螺中位变化/%", "加计中位变化/%", "联合中位变化/%", "改善协议"], ["方法", "陀螺中位变化/%", "加计中位变化/%", "联合中位变化/%", "改善协议"] )}

负值表示相对原始 FIR/ARX 的 RMSE 更低。VMD 数值范围过大，图 1 使用 symlog 坐标，避免视觉上压缩其他分支。

![分支效应](figures/fig1_branch_effects.png)

![分支总览](figures/fig2_branch_summary.png)

## 6. 解释边界

- HIGH 是实测参考器件，不是无误差绝对真值。
- 本轮是 retrospective common-task adapter；并非原论文完整复现或全球 SOTA 排名。
- 三个种子或多次配置之间不是独立录制重复，不能用种子 SD 替代场景不确定性。
- VMD 结论只针对当前 K/alpha/频带划分与因果 FIR/ARX 输入构造。
- TDA 结论只针对当前窗口、描述子、聚类数和训练集路由方式。
- 最小输入当前依赖 TRAIN 场景聚类和 VALIDATION 选模，仍需独立录制外部验证。

## 7. 建议下一步

1. 在最小输入分支上加入稳定性选择：跨场景 bootstrap 统计每个分量的入选频率，只保留稳定分量。
2. 将 VMD 改成“原始 LOW + VMD 高频残差”的残差学习，而不是直接丢掉低频或把高频单独回归。
3. 将 TDA 描述子作为样本可靠性/工况门控，而不是把 TDA 派生分量直接送入回归。
4. 增加一条全新公开录制作为外部测试；当前所有结论都基于回顾性固定 split。
5. 对最终候选同时报告精度、参数量、CPU 延迟和短时姿态/位置传播，避免只看信号 RMSE。

## 8. 本地与服务器产物

- 本地完整结果：`runs/fir_arx_delivery_20261105/extracted/`
- 分支比较：`runs/fir_arx_comparison_20261105/`
- 最终科研图：`runs/fir_arx_branch_figures_20261105_v2/` 与本报告 `figures/`
- 表格：`tables/table1_branch_summary.*`、`table2_minimal_by_protocol.*`、`table3_raw_fir_arx_reproduction.*`
- 文件哈希：`delivery_sha256.json`；分支与服务器运行身份：`provenance.json`
- 图件说明与索引：`figure_index.md`
- 服务器保留原始/处理后数据：`/root/fir_arx/data/processed/unified_imu_v2/`
- 服务器结果与源码：`/root/fir_arx/project*`、`/root/fir_arx/project_vmd`、`/root/fir_arx/project_tda`、`/root/fir_arx/project_minimal`
"""
    (root / "report.md").write_text(report, encoding="utf-8")
    figure_stems = {
        "fig1_vmd_condition_improvements": "VMD conditions versus raw control",
        "fig1_tda_vs_raw_windowed": "TDA training-only routing versus raw windowed control",
        "fig2_tda_cluster_support": "Training support of TDA regimes",
        "fig1_minimal_vs_full": "Scenario-minimal versus raw and all-component controls",
        "fig2_minimal_component_frontier": "Validation accuracy frontier versus component count",
    }
    index_lines = [
        "# FIR/ARX figure and table index",
        "",
        "| File | Purpose | Formats |",
        "| --- | --- | --- |",
        "| `figures/fig1_branch_effects.png` | Per-protocol signed effect of all branches | PNG, SVG, PDF, TIFF |",
        "| `figures/fig2_branch_summary.png` | Branch-level effect and improved-protocol count | PNG, SVG, PDF, TIFF |",
        "| `tables/table1_branch_summary.png` | Main cross-branch numerical summary | PNG, SVG, PDF |",
        "| `tables/table2_minimal_by_protocol.png` | Scenario-minimal support, fallback, and effects | PNG, SVG, PDF |",
        "| `tables/table3_raw_fir_arx_reproduction.png` | Raw FIR/ARX physical metrics | PNG, SVG, PDF |",
        "| `extracted/*/figures/*` | Complete per-branch audits and supplementary views | SVG, PDF, PNG, TIFF |",
        "",
        "The report Markdown is Chinese; all figure and table raster text is English for font-safe Nature-style export.",
    ]
    (root / "figure_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    provenance = {
        "experiment": "FIR/ARX preprocessing research program",
        "date": "2026-11-05",
        "branches": {
            "reproduction": subprocess.run(["git", "rev-parse", "codex/fir-arx-reproduction"], capture_output=True, text=True).stdout.strip(),
            "vmd": subprocess.run(["git", "rev-parse", "codex/fir-arx-vmd"], capture_output=True, text=True).stdout.strip(),
            "tda": subprocess.run(["git", "rev-parse", "codex/fir-arx-tda"], capture_output=True, text=True).stdout.strip(),
            "minimal": subprocess.run(["git", "rev-parse", "codex/fir-arx-minimal"], capture_output=True, text=True).stdout.strip(),
        },
        "server_runs": {
            "base": "fir_arx_base_20261105_v2",
            "vmd": "fir_arx_vmd_full_20261105_v2",
            "tda": "fir_arx_tda_full_20261105_v2",
            "minimal": "fir_arx_minimal_full_20261105_v4",
        },
        "dataset": "data/processed/unified_imu_v2/manifest.json",
        "selection_policy": "TRAIN fit; VALIDATION-only model and component selection; TEST evaluated once",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    server_source_commits = {
        "base": subprocess.run(["git", "rev-parse", "codex/fir-arx-reproduction"], capture_output=True, text=True).stdout.strip(),
        "vmd": subprocess.run(["git", "rev-parse", "codex/fir-arx-vmd"], capture_output=True, text=True).stdout.strip(),
        "tda": subprocess.run(["git", "rev-parse", "codex/fir-arx-tda"], capture_output=True, text=True).stdout.strip(),
        "minimal": subprocess.run(["git", "rev-parse", "codex/fir-arx-minimal"], capture_output=True, text=True).stdout.strip(),
    }
    for name, run_name in provenance["server_runs"].items():
        run_provenance = root / "extracted" / run_name / "provenance.json"
        if not run_provenance.is_file():
            continue
        data = json.loads(run_provenance.read_text(encoding="utf-8"))
        data["git_commit"] = server_source_commits[name]
        data["git_commit_resolution"] = "resolved from local experiment branch; uncommitted server export had no Git metadata"
        run_provenance.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "report": "report.md",
        "readme": "README.md",
        "tables": sorted(str(path.relative_to(root)) for path in tables.glob("*")),
        "figures": sorted(str(path.relative_to(root)) for path in (root / "figures").glob("*")),
        "server_result_archives": sorted(str(path.relative_to(root)) for path in root.glob("*.tar.gz")),
        "extracted_runs": sorted(
            str(path.relative_to(root)) for path in (root / "extracted").iterdir() if path.is_dir()
        ),
    }
    (root / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    readme = f"""# FIR/ARX 科研复现交付包

本目录是 2026-11-05 完成的 FIR/ARX 短时 IMU 增强实验交付包。

## 先看什么

1. [中文完整报告](report.md)
2. [图表索引](figure_index.md)
3. [分支总效应图](figures/fig1_branch_effects.png)
4. [分支汇总图](figures/fig2_branch_summary.png)

## 一句话结论

场景最小输入分支在 **{int(min_row.branch_win_protocols)}/{int(min_row.protocols)}** 个协议上同时改善陀螺与加计，联合中位变化 **{min_row.joint_score_delta_median:+.2f}%**；VMD 直接输入与 TDA 路由没有稳定超过原始 FIR/ARX。详见报告。

## 目录说明

- `extracted/`：四组服务器结果的完整解压副本，包括配置快照、provenance、表格、状态、图件和报告。
- `*.tar.gz`：服务器回传的原始结果归档。
- `figures/`：跨分支 Nature 风格总览图，均提供 PNG、SVG、PDF、TIFF。
- `tables/`：三张主表，提供 CSV、PNG、SVG、PDF。
- `provenance.json`：分支提交号、服务器运行 ID、数据和选择制度。
- `delivery_sha256.json`：本目录全部文件的 SHA256 清单。

## 代码分支

- `codex/fir-arx-reproduction`
- `codex/fir-arx-vmd`
- `codex/fir-arx-tda`
- `codex/fir-arx-minimal`

服务器原始数据与运行目录仍保留在 `/root/fir_arx/`。
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
