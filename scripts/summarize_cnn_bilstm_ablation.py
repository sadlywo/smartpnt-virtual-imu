"""Produce a concise Chinese reading guide from the completed ablation tables.

This adds no selection rule: all window choices come from the frozen reporting
script's validation_selected_windows.csv. Paired percentages are averaged only
after computing each same-seed comparison.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CELLS = [("alimu_only", "alimu", "ALiMU共享→ALiMU"),
         ("pooled", "alimu", "联合共享→ALiMU"),
         ("pooled", "penncosyvio", "联合共享→Penn")]
NAMES = dict(cnn_bilstm="CNN-BiLSTM", bilstm_only="仅BiLSTM", cnn_only="仅CNN",
             mlp_only="逐点MLP", cnn_lstm="CNN-LSTM32", cnn_lstm_wide="CNN-LSTM64",
             pointwise_bilstm="逐点CNN-BiLSTM", cnn_bilstm_dense="CNN-BiLSTM整窗监督",
             raw="Raw")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def estimate(row):
    return format(row["mean"], ".6g") + (" ± " + format(row["sd"], ".2g") if row["n"] > 1 else "")


def summarize(run):
    print("READS completed ablation summary, paired deltas and VALID choices; WRITES findings and trace receipt; SKIPS training, predictions and model reselection")
    names = ["config.yaml", "training.complete.json", "summary.csv", "module_paired_deltas.csv",
             "factorial_interactions.csv", "validation_selected_windows.csv", "compute_summary.csv",
             "overlap_summary.csv", "overlap_paired_deltas.csv"]
    before = {name: sha(run / name) for name in names}
    complete = json.loads((run / "training.complete.json").read_text(encoding="utf-8"))
    assert complete["planned"] == complete["completed"] == 138
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    summary = pd.read_csv(run / "summary.csv")
    test = summary[summary.split.eq("test") & summary.aggregation.eq("dataset")]
    delta = pd.read_csv(run / "module_paired_deltas.csv")
    delta = delta[delta.split.eq("test") & delta.aggregation.eq("dataset") & delta.window.eq(320)]
    selected = pd.read_csv(run / "validation_selected_windows.csv")
    compute = pd.read_csv(run / "compute_summary.csv")
    overlap = pd.read_csv(run / "overlap_summary.csv")
    overlap = overlap[overlap.split.eq("test")]

    def value(p, d, variant, window, metric):
        z = test[test.protocol.eq(p) & test.dataset_id.eq(d) & test.variant.eq(variant)
                 & test.window.eq(window) & test.metric.eq(metric)]
        assert len(z) == 1, (p, d, variant, window, metric)
        return z.iloc[0]

    lines = ["# 共享 CNN-BiLSTM 消融：结果导读", "",
             "此前查阅的本地材料中没有严格的共享CNN-BiLSTM窗口／模块消融，只有联合HPO、完整模型比较及TCN的消融。本次完成138项拟合：2种训练范围×3种子×（5种主结构×4个窗口＋3项320点补充控制）。",
             "", "以下误差为TEST分量RMSE（gyro：rad/s；accel：m/s²），越低越好，均值±样本SD来自3个优化种子。窗口在40 Hz下名义对应1/2/4/8秒。所有主实验输入同一组anchor，使用相同中心40点监督与评估。", "",
             "## 完整骨干的窗口长度", "",
             "四种窗口仅改变上下文；不增加主实验训练标签和评估点。此表展示敏感性，正式选窗使用VALID，不按TEST最低值作选择。", ""]
    rows = []
    for p, d, label in CELLS:
        chosen = int(selected[selected.protocol.eq(p) & selected.variant.eq("cnn_bilstm")].selected_window.iloc[0])
        for w in cfg["windows"]:
            rows.append([label, str(w) + ("（VALID选定）" if w == chosen else ""),
                         estimate(value(p, d, "cnn_bilstm", w, "gyro_rmse")),
                         estimate(value(p, d, "cnn_bilstm", w, "accel_rmse"))])
    lines += [table(["训练→测试域", "窗口/点", "陀螺RMSE", "加速度RMSE"], rows), "",
              "## 固定320点的模块与补充控制", "",
              "每格为相对同种子完整CNN-BiLSTM的RMSE变化百分比均值±SD；正数代表移除／改变后变差，负数代表变好。↑列表示3个种子中误差增大的次数。固定配方下的消融没有为每个子网络重新调参。", ""]
    rows = []
    delta_rows = []
    for p, d, label in CELLS:
        for variant in [v for v in cfg["variants"] if v != "cnn_bilstm"] + cfg["supplemental"]:
            cells = []
            for metric in ["gyro_rmse", "accel_rmse"]:
                z = delta[delta.protocol.eq(p) & delta.dataset_id.eq(d) & delta.variant.eq(variant) & delta.metric.eq(metric)]
                assert set(z.seed) == set(cfg["seeds"])
                mean, sd = z.relative_delta_pct.mean(), z.relative_delta_pct.std(ddof=1)
                worse = int((z.delta > 0).sum())
                cells.extend([f"{mean:+.2f} ± {sd:.2f}%", f"{worse}/3"])
                delta_rows.append(dict(protocol=p, dataset_id=d, variant=variant, metric=metric,
                                       delta_pct_mean=mean, delta_pct_sd=sd, worse_seeds=worse, seeds=len(z)))
            rows.append([label, NAMES[variant], *cells])
    lines += [table(["训练→测试域", "变体", "陀螺变化", "↑", "加速度变化", "↑"], rows), "",
              "LSTM64比完整双向模型参数更多（81,414对73,222），用于容量对照；逐点CNN将三个k5卷积改为k1，保留层深、BN及非线性；整窗监督仅改变训练loss支持，不属于纯模块开关。单向LSTM变体仍使用对称CNN，不能当作完整因果模型。纯CNN的推理感受野固定为13点（三层k5、stride1）；其长窗口差异还包含BN训练统计及优化轨迹变化，不能解读为纯CNN直接使用了整段长窗口。", "",
              "## 各主结构按VALID选窗后的TEST表现", "",
              "每个训练范围、每种主结构只选择一个窗口；联合共享按两个数据集等权、设备对等权的VALID归一化分数选窗，不能为AL和Penn各选不同窗口。分数是gyro/Raw与accel/Raw的均值，再按设备对和数据集平衡，不直接相加不同物理单位。", ""]
    rows = []
    selected_rows = []
    for p, d, label in CELLS:
        raw_g = value(p, d, "raw", 0, "gyro_rmse")["mean"]
        raw_a = value(p, d, "raw", 0, "accel_rmse")["mean"]
        rows.append([label, "Raw", "—", f"{raw_g:.6g}", f"{raw_a:.6g}", "—", "—"])
        for variant in cfg["variants"]:
            choice = selected[selected.protocol.eq(p) & selected.variant.eq(variant)].iloc[0]
            w = int(choice.selected_window)
            g, a = value(p, d, variant, w, "gyro_rmse"), value(p, d, variant, w, "accel_rmse")
            score = value(p, d, variant, w, "normalized_score")
            rows.append([label, NAMES[variant], w, estimate(g), estimate(a),
                         f"{100*(1-g['mean']/raw_g):.2f}% / {100*(1-a['mean']/raw_a):.2f}%", estimate(score)])
            selected_rows.append(dict(protocol=p, dataset_id=d, variant=variant, window=w,
                                      gyro_mean=g["mean"], gyro_sd=g["sd"], accel_mean=a["mean"], accel_sd=a["sd"],
                                      normalized_score_mean=score["mean"], normalized_score_sd=score["sd"],
                                      validation_selection_score_mean=choice.score_mean))
    lines += [table(["训练→测试域", "结构", "VALID选窗", "陀螺RMSE", "加速度RMSE", "较Raw改善（陀螺/加速度）", "归一化分数"], rows), "",
              "## 多锚窗输出融合诊断", "",
              "固定完整／整窗监督的320点检查点和相同TEST时间点，把中心单次输出改为覆盖同一点的多锚窗输出平均；不再训练、不重新选择epoch。每窗仍为320点，但8窗的输入并集可达600点，且纳入非中心输出；中心监督模型的这些位置没有直接loss监督。因此不能把差值全部归因为纯平均或方差降低。百分比先逐seed计算再平均。", ""]
    rows = []
    for row in overlap.itertuples():
        label = next(label for p, d, label in CELLS if p == row.protocol and d == row.dataset_id)
        rows.append([label, NAMES[row.variant], "陀螺" if row.metric == "gyro_rmse" else "加速度",
                     f"{row.center_mean:.6g}", f"{row.overlap_mean:.6g}",
                     f"{row.delta_pct_mean:+.2f} ± {row.delta_pct_sd:.2f}%"])
    lines += [table(["训练→测试域", "训练方式", "指标", "中心输出RMSE", "平均输出RMSE", "变化"], rows), "",
              "## 计算成本", "",
              "参数量与每个seed的batch1同步计时中位数，再跨seed求均值±SD。计时受本机并发负载影响；输入包含未来上下文，计算延迟不等于流式等待时间。", ""]
    rows = []
    for p in cfg["protocols"]:
        for variant in cfg["variants"] + cfg["supplemental"]:
            z = compute[compute.protocol.eq(p) & compute.variant.eq(variant) & compute.window.eq(320)].set_index("metric")
            rows.append([p, NAMES[variant], int(z.loc["parameters", "mean"]), estimate(z.loc["batch1_ms_median"]),
                         estimate(z.loc["actual_epochs"])])
    lines += [table(["训练范围", "结构", "参数量", "batch1中位延迟/ms", "实际轮数"], rows), "",
              "## 适用范围与完整证据", "",
              "ALiMU共享模型以BMX160、MPU9250、LSM9DS1为LOW、VN100为HIGH参照；联合模型额外使用Penn的TangoTop→ADIS16488。ALiMU TEST为dyn08/09两次物理录制，Penn TEST为bs一次录制；没有独立静态TEST或未见器件TEST。三个优化种子不增加独立录制数。", "",
              "主实验没有对每个删除模块的子网络再调参；结论限定于固定共享配方。本次复用了历史TEST划分，是后续机制研究。由于共同中心标签的设计不同于历史整窗训练和拼接，绝对RMSE不能直接并入历史模型排名。", "",
              "[完整技术报告](report.md) · [逐设备、逐录制与逐轴汇总](summary.csv) · [逐seed模块配对](module_paired_deltas.csv) · [窗口配对](window_paired_deltas.csv) · [CNN×循环块交互](factorial_interactions.csv) · [独立重算核验](verification.json)", ""]
    outputs = {"findings.md": "\n\n".join(lines)+"\n"}
    for name, text in outputs.items():
        (run / name).write_text(text, encoding="utf-8")
    pd.DataFrame(delta_rows).to_csv(run / "module_delta_digest.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(run / "selected_test_digest.csv", index=False)
    for name, digest in before.items():
        assert sha(run / name) == digest, name
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), selection_changed=False,
                   inputs_sha256=before, source_sha256=sha(Path(__file__)),
                   outputs_sha256={name: sha(run / name) for name in ["findings.md", "module_delta_digest.csv", "selected_test_digest.csv"]})
    (run / "findings_provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("WRITES findings.md, module_delta_digest.csv, selected_test_digest.csv, findings_provenance.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run_dir.resolve())
