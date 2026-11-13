# -*- coding: utf-8 -*-
"""Compare center and overlap averaging at fixed checkpoints and timestamps."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_imu.ablations.data import CommonWindows, SharedSource, LOW, physical_metrics
from virtual_imu.ablations.models import build_ablation_model
from virtual_imu.ablations.training import configure, sha, write_json

LOG = logging.getLogger(__name__)


def stitch_on_common_support(data, normalized):
    """Average physical predictions before selecting the unchanged center rows."""
    if normalized.shape != (len(data), 6, data.max_window):
        raise ValueError("A full prediction is required for each maximum-window anchor")
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite overlap predictions")
    total = np.zeros((len(data.frame), 6), dtype=np.float64)
    counts = np.zeros(len(data.frame), dtype=np.int64)
    low = data.frame[LOW].to_numpy(float)
    for i, start in enumerate(data.starts):
        stop = start + data.max_window
        residual = normalized[i].T.astype(np.float64) * data.stats["residual_std"] + data.stats["residual_mean"]
        total[start:stop] += low[start:stop] + residual
        counts[start:stop] += 1
    chosen = data.target_indices
    if np.any(counts[chosen] == 0):
        raise ValueError("Overlap comparison cannot broaden or fill missing primary support")
    return total[chosen] / counts[chosen, None], counts[chosen]


def evaluate(run):
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    completed = json.loads((run / "training.complete.json").read_text(encoding="utf-8"))
    lock = json.loads((run / "selections.lock.json").read_text(encoding="utf-8"))
    jobs = json.loads((run / "planned_jobs.json").read_text(encoding="utf-8"))
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    if completed["completed"] != len(jobs) or len(lock["checkpoints_sha256"]) != len(jobs) or len(pd.read_csv(run / "compute_cost.csv")) != len(jobs):
        raise ValueError("Complete primary fitting/evaluation is required before this diagnostic")
    selected = [j for j in jobs if j["window"] == 320 and j["variant"] in ["cnn_bilstm", "cnn_bilstm_dense"]]
    for rel, expected in provenance["source_sha256"].items():
        assert sha(ROOT / rel) == expected
    for job in selected:
        if sha(Path(job["directory"]) / "best.pt") != lock["checkpoints_sha256"][job["id"]]:
            raise ValueError("Frozen checkpoint changed before overlap comparison")
    device = configure(cfg)
    amp = cfg["training"]["amp"] and device.type == "cuda"
    metrics, inputs_hashes, outputs_hashes, counts_info = [], {}, {}, {}
    for protocol in cfg["protocols"]:
        source = SharedSource(ROOT, cfg["data"], protocol)
        assert source.dataset.manifest_sha256 == cfg["data"]["manifest_sha256"]
        stats_path = run / "supports" / protocol / "statistics.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        inputs_hashes[str(stats_path)] = sha(stats_path)
        options = {k: cfg["data"][k] for k in ["max_window", "block_size", "stride"]}
        data = {role: CommonWindows(source.load(role), stats=stats, **options) for role in ["validation", "test"]}
        tensors = {role: torch.from_numpy(d.arrays(d.max_window)[0]).to(device) for role, d in data.items()}
        for job in [j for j in selected if j["protocol"] == protocol]:
            checkpoint = Path(job["directory"]) / "best.pt"
            saved = torch.load(checkpoint, map_location=device)
            model, _ = build_ablation_model(job["variant"], job["seed"], cfg["model"])
            model.load_state_dict(saved["model"])
            model = model.to(device).eval()
            inputs_hashes[str(checkpoint)] = sha(checkpoint)
            for role, d in data.items():
                chunks = []
                with torch.inference_mode():
                    for start in range(0, len(d), cfg["evaluation"]["batch_size"]):
                        with torch.autocast(device_type=device.type, enabled=amp):
                            value = model(tensors[role][start:start+cfg["evaluation"]["batch_size"]])
                        chunks.append(value.float().cpu().numpy())
                full = np.concatenate(chunks)
                center = (d.max_window - d.block_size) // 2
                center_path = run / "predictions" / (job["id"] + "__" + role + ".npz")
                with np.load(center_path) as archive:
                    primary = archive["prediction"]
                    assert str(archive["support_sha256"]) == d.support_sha256
                if not np.allclose(d.physical(full[..., center:center+d.block_size]), primary, rtol=1e-7, atol=1e-10):
                    raise ValueError("Fixed checkpoint center output differs from primary evaluation")
                overlap, counts = stitch_on_common_support(d, full)
                out = run / "predictions" / ("overlap__" + job["id"] + "__" + role + ".npz")
                np.savez_compressed(out, prediction=overlap, normalized_windows=full, overlap_counts=counts, support_sha256=np.array(d.support_sha256))
                inputs_hashes[str(center_path)] = sha(center_path)
                outputs_hashes[str(out.relative_to(run))] = sha(out)
                counts_info[protocol + "__" + role] = dict(min=int(counts.min()), max=int(counts.max()), samples=len(counts),
                    input_union_points_min=int(d.max_window + (counts.min()-1)*d.stride),
                    input_union_points_max=int(d.max_window + (counts.max()-1)*d.stride))
                for method, prediction in [("center", primary), ("overlap", overlap)]:
                    metrics.extend(dict(job_id=job["id"], protocol=protocol, variant=job["variant"], window=320, seed=job["seed"],
                                        split=role, aggregation_mode=method, **row) for row in physical_metrics(d.target_frame, prediction))
                LOG.info("WRITES overlap diagnostic %s %s; SKIPS checkpoint reselection", job["id"], role)
    table = pd.DataFrame(metrics)
    table.to_csv(run / "overlap_metrics_per_seed.csv", index=False)
    pairs = []
    datasets = table[table.aggregation.eq("dataset")]
    for key, group in datasets.groupby(["protocol", "split", "dataset_id", "variant", "seed"]):
        for metric in ["gyro_rmse", "accel_rmse"]:
            values = group.set_index("aggregation_mode")[metric]
            pairs.append(dict(zip(["protocol", "split", "dataset_id", "variant", "seed"], key), metric=metric,
                              center=values["center"], overlap=values["overlap"], delta_pct=100 * (values["overlap"] / values["center"] - 1)))
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(run / "overlap_paired_deltas.csv", index=False)
    summary = pairs.groupby(["protocol", "split", "dataset_id", "variant", "metric"], as_index=False).agg(
        center_mean=("center", "mean"), center_sd=("center", "std"), overlap_mean=("overlap", "mean"), overlap_sd=("overlap", "std"),
        delta_pct_mean=("delta_pct", "mean"), delta_pct_sd=("delta_pct", "std"), seeds=("seed", "nunique"))
    summary.to_csv(run / "overlap_summary.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from virtual_imu.viz.style import apply_paper_style, save_figure
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    cells = [("alimu_only", "alimu"), ("pooled", "alimu"), ("pooled", "penncosyvio")]
    for ax, metric, title in zip(axes, ["gyro_rmse", "accel_rmse"], ["Gyro", "Accel"]):
        for index, (variant, label) in enumerate([("cnn_bilstm", "Center-trained"), ("cnn_bilstm_dense", "Dense-trained")]):
            rows = [summary[summary.protocol.eq(p) & summary.dataset_id.eq(d) & summary.split.eq("test") & summary.variant.eq(variant) & summary.metric.eq(metric)].iloc[0] for p, d in cells]
            ax.bar(np.arange(3) + (index-.5)*.32, [r.delta_pct_mean for r in rows], width=.3,
                   yerr=[r.delta_pct_sd for r in rows], capsize=2, color=["#1f77b4", "#2a9d8f"][index], label=label)
        ax.axhline(0, color="#1f2933", linewidth=.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(["AL-only / AL", "Joint / AL", "Joint / Penn"], rotation=20, ha="right")
        ax.set_ylabel(title + " RMSE change after overlap fusion [%]")
        ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_figure(fig, run / "figures", "overlap_aggregation_diagnostic")
    text = "# 多锚窗输出融合诊断\n\n固定已选中的320点检查点和中心40点测试支持，将单次中心输出改为覆盖相同时间点的多锚窗输出平均。不新增训练，也不依据此诊断重新选择epoch或模型。每个锚窗仍为320点，但stride40的8窗融合可将单点的输入并集扩展到600点，并纳入非中心输出；中心监督模型的非中心位置没有直接loss监督。因此该诊断同时包含上下文组合与输出位置变化，不能分离纯平均和上下文的净收益。误差仍为分量RMSE，向量RMSE=√3倍。下表变化为逐seed百分比的均值±样本SD；负数改善。重叠预测相互相关，不作为额外独立样本。\n\n"
    text += "| 训练协议/测试域 | 训练方式 | 量 | 中心RMSE | 平均后RMSE | 变化%（均值±SD） |\n| --- | --- | --- | --- | --- | --- |\n"
    for row in summary[summary.split.eq("test")].itertuples():
        text += "| " + row.protocol + "/" + row.dataset_id + " | " + row.variant + " | " + row.metric + " | " + format(row.center_mean, ".7g") + " | " + format(row.overlap_mean, ".7g") + " | " + format(row.delta_pct_mean, ".3f") + " ± " + format(row.delta_pct_sd, ".3f") + " |\n"
    text += "\n这项诊断衡量固定检查点、相同目标时间点上的多锚窗融合表现。它不复现历史整窗VALID选择，也不把不同训练配方或不同支持的绝对误差混排。实际覆盖次数和输入并集范围见 overlap_diagnostic_receipt.json，完整逐seed、逐设备与逐录制数值见 overlap_metrics_per_seed.csv。\n\n![多锚窗融合诊断](" + (run / "figures/overlap_aggregation_diagnostic.png").as_posix() + ")\n"
    (run / "overlap_aggregation.md").write_text(text, encoding="utf-8")
    for path in [run / "overlap_metrics_per_seed.csv", run / "overlap_paired_deltas.csv", run / "overlap_summary.csv", run / "overlap_aggregation.md", *paths]:
        outputs_hashes[str(path.relative_to(run))] = sha(path)
    write_json(run / "overlap_diagnostic_receipt.json", dict(status="complete", checkpoint_selection_changed=False,
               inputs_sha256=inputs_hashes, outputs_sha256=outputs_hashes, source_sha256=sha(__file__), supports=counts_info,
               trained_models=len(selected), added_training_fits=0, prediction_files=2*len(selected)))
    LOG.info("WRITES overlap tables, report and one figure set; SKIPS additional training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cnn_bilstm_shared_ablation_20261106")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("READS locked completed checkpoints and shared inputs; WRITES overlap diagnostic; SKIPS training and model selection")
    evaluate(args.run_dir.resolve())
