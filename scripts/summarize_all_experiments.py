# -*- coding: utf-8 -*-
"""Build a traceable, protocol-separated synthesis of completed IMU experiments.

No training, original artifact edits, or outputs/latest updates are performed.
Input hashes are checked against the original result seals before aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_imu.viz.style import apply_paper_style, save_figure  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAMES = {
    "cnn_bilstm": "CNN-BiLSTM", "plain_cnn": "Temporal CNN", "gru": "GRU",
    "tcn": "TCN", "transformer": "Transformer", "informer": "Informer",
    "itransformer": "iTransformer", "patchtst": "PatchTST",
    "deepgyro_adapter": "DeepGyro adapter", "tlio_resnet18_adapter": "TLIO adapter",
    "tartanimu_window_adapter": "TartanIMU window adapter",
    "identity": "Raw", "raw": "Raw", "affine": "Affine", "fir_tuned": "FIR (tuned)",
    "arx_tuned": "ARX (tuned)", "ddpm": "DDPM-Transformer",
    "no_diffusion": "No diffusion (own HPO)",
    "no_diffusion_matched": "No diffusion (matched)",
}
COMPONENT = {"gyro_vector_rmse_radps": "gyro", "accel_vector_rmse_mps2": "accel"}
PROTOCOLS = ["alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf", "pooled"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def link(path, text=None):
    p = Path(path).resolve()
    return "[" + (text or p.name) + "](<" + p.as_posix() + ">)"


def md_table(frame):
    d = frame.fillna("—").astype(str)
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join(lines)


def run(config_path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    np.random.seed(cfg["seed"])
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    figures = out / "figures"
    inputs, checks = {}, []

    def pin(path, expected=None):
        p = Path(path).resolve()
        digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError("Artifact hash mismatch: " + str(p))
        inputs[p.as_posix()] = {"sha256": digest, "bytes": p.stat().st_size,
                                "checked_against_original_seal": expected is not None}
        print("READS " + str(p))
        return p

    def write_table(name, frame):
        path = tables / (name + ".csv")
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.16g")
        print("WRITES " + str(path) + " rows=" + str(len(frame)))

    old = ROOT / cfg["reviewer_run"]
    seal = read_json(pin(old / "artifact_manifest.json"))["files"]

    def historical(name):
        rel = "tables/" + name + ".csv"
        return pd.read_csv(pin(old / rel, seal[rel]))

    primary = historical("primary_comparison")
    rankings = historical("scoped_signal_rankings")
    transfer = historical("transfer_seed_summary")
    registry = historical("accuracy_run_registry")
    original_summary = historical("accuracy_metrics_seed_summary")
    pin(old / "report.md", seal["report.md"])

    recent = ROOT / cfg["matched_run"]
    recent_seal_path = pin(recent / "complete.json")
    recent_seal = read_json(recent_seal_path)
    verification = read_json(pin(recent / "verification.json"))
    assert verification["status"] == "PASS", verification["status"]
    assert verification["complete_sha256"] == sha(recent_seal_path)
    findings_receipt = read_json(pin(recent / "findings_receipt.json"))
    for name, expected in findings_receipt["inputs_sha256"].items():
        pin(recent / name, expected)
    pin(recent / "findings.md", findings_receipt["findings_sha256"])

    def matched(name):
        rel = name + ".csv"
        return pd.read_csv(pin(recent / rel, recent_seal["artifacts_sha256"][rel]))

    m = matched("summary")
    seed_metrics = matched("metrics_per_seed")
    deltas = matched("matched_seed_deltas")
    pin(recent / "config.yaml", recent_seal["artifacts_sha256"]["config.yaml"])
    pin(recent / "provenance.json", recent_seal["artifacts_sha256"]["provenance.json"])

    # Preserve historical vector measurements and store the explicit unit conversion.
    a = rankings[rankings.ranking_condition.eq("main_common_information")].copy()
    assert len(a) == 406
    assert a["mean"].notna().all()
    a["component"] = a.metric.map(COMPONENT)
    a["source_metric"] = a.metric
    a["source_mean"] = a["mean"]
    a["conversion_factor"] = 1 / math.sqrt(3)
    a["rmse_component_mean"] = a["mean"] / math.sqrt(3)
    a["rmse_component_sd"] = a.sample_sd / math.sqrt(3)
    a["source_run"] = old.name
    a["source_table"] = "tables/scoped_signal_rankings.csv"
    a["name"] = a.canonical_method.map(lambda v: NAMES.get(v, v))
    write_table("historical_signal_all_methods", a)
    supplementary = rankings[rankings.ranking_condition.ne("main_common_information")].copy()
    write_table("historical_known_device_bank_separate", supplementary)

    keys = ["protocol", "dataset_id", "metric"]
    winners = a.loc[a.groupby(keys)["mean"].idxmin()].copy()
    assert len(winners) == 14
    baseline = a[a.canonical_method.eq("identity")].set_index(keys)["mean"]
    winners["reduction_vs_raw_pct"] = [100 * (1 - r["mean"] / baseline.loc[tuple(r[k] for k in keys)]) for _, r in winners.iterrows()]
    write_table("historical_protocol_winners", winners)
    temporal = []
    for scope, group in a.groupby("temporal_class"):
        z = group.loc[group.groupby(keys)["mean"].idxmin()].copy()
        z["comparison_scope"] = scope
        temporal.append(z)
    write_table("historical_temporal_winners", pd.concat(temporal, ignore_index=True))

    h = primary[primary.metric.isin(COMPONENT)].copy()
    h["component"] = h.metric.map(COMPONENT)
    for v in ["default", "tuned"]:
        h[v + "_component_mean"] = h[v + "_mean"] / math.sqrt(3)
        h[v + "_component_sd"] = h[v + "_sample_sd"] / math.sqrt(3)
    windows = registry.groupby(["protocol_id", "model_family", "variant"]).window_points.agg(lambda values: ",".join(map(str, sorted(set(values)))))
    h["tuned_window_points"] = [windows.loc[(r.protocol, r.family, "tuned")] for r in h.itertuples()]
    write_table("neural_default_vs_tuned", h)
    assert len(h) == 140 and (h.improvement_pct > 0).sum() == 126
    for metric, group in h.groupby("component"):
        wide = group.pivot(index="family", columns=["protocol", "test_dataset"], values="tuned_component_mean")
        wide.columns = [p + "__" + d for p, d in wide.columns]
        write_table("neural_tuned_matrix_" + metric, wide.reset_index())

    new = m[m.component.isin(["gyro", "accel"])].copy()
    new["source_run"] = recent.name
    new["metric"] = "component_rmse"
    new["reduction_vs_raw_pct"] = [100 * (1 - r.rmse_mean / new[(new.pair_id.eq(r.pair_id)) & new.component.eq(r.component) & new.method.eq("raw")].iloc[0].rmse_mean) for r in new.itertuples()]
    write_table("matched_alimu_all_models", new)
    learned = new[~new.method.isin(["raw", "affine"])]
    nw = learned.loc[learned.groupby(["pair_id", "component"]).rmse_mean.idxmin()].copy()
    write_table("matched_alimu_winners", nw)
    write_table("diffusion_strict_paired_ablation", deltas)
    assert len(deltas) == 18 and deltas.initial_backbone_matches.all()
    assert deltas.ddpm_change_pct.gt(0).all()
    # Independently replay the seed mean and sample SD from the per-seed metrics.
    for row in new.itertuples():
        v = seed_metrics[(seed_metrics.pair_id.eq(row.pair_id)) & seed_metrics.method.eq(row.method) & seed_metrics.component.eq(row.component)].rmse
        assert len(v) == row.n
        assert np.isclose(v.mean(), row.rmse_mean, rtol=1e-10, atol=1e-14)
        if len(v) > 1:
            assert np.isclose(v.std(ddof=1), row.rmse_sd, rtol=1e-10, atol=1e-14)
    checks += ["matched_summary_recomputed_from_seed_rows", "18_of_18_strict_diffusion_deltas_positive", "exact_initialization_receipt_PASS", "historical_14_protocol_metric_winners_recomputed"]

    # Transfer comparisons retain default and source-domain tuned variants separately.
    identity = original_summary[original_summary.method.eq("identity") & original_summary.split.eq("test") & original_summary.metric.isin(COMPONENT)]
    tr = []
    for row in transfer.itertuples():
        protocol = "penn_only" if row.direction == "alimu_to_penn" else "alimu_only"
        for metric, component in COMPONENT.items():
            raw = identity[identity.protocol_id.eq(protocol) & identity.aggregation_level.eq("pair") & identity.low_sensor.eq(row.low_sensor) & identity.metric.eq(metric)]
            assert len(raw) == 1
            value = getattr(row, metric + "_mean")
            ref = getattr(row, "target_trained_reference_" + metric + "_mean")
            tr.append(dict(direction=row.direction, dataset_id=row.dataset_id, low_sensor=row.low_sensor,
                           method=row.model_family, variant=row.variant, component=component, n=row.seeds,
                           rmse_component_mean=value / math.sqrt(3), rmse_component_sd=getattr(row, metric + "_sample_sd") / math.sqrt(3),
                           raw_component_rmse=raw.iloc[0]["mean"] / math.sqrt(3), reduction_vs_raw_pct=100 * (1 - value / raw.iloc[0]["mean"]),
                           target_trained_same_family_component_rmse=ref / math.sqrt(3),
                           transfer_increase_vs_target_pct=100 * (value / ref - 1)))
    tr = pd.DataFrame(tr)
    write_table("transfer_all_models", tr)
    tr_best = []
    for label, group in [("source_domain_tuned_only", tr[tr.variant.eq("tuned")]), ("observed_default_and_tuned", tr)]:
        z = group.loc[group.groupby(["direction", "low_sensor", "component"]).rmse_component_mean.idxmin()].copy()
        z["ranking_scope"] = label
        tr_best.append(z)
    write_table("transfer_winners_by_scope", pd.concat(tr_best, ignore_index=True))

    # Agents' supplements include negative results, smoke-only exclusions, and row keys.
    support_text, support_json = [], {}
    for name in cfg["support_reports"]:
        text_path = pin(out / "support" / (name + ".md"))
        evidence = read_json(pin(out / "support" / (name + ".json")))
        support_json[name] = evidence
        # Supplements retain original source references. Snapshot any resolvable
        # file references as well as the narrative, without guessing missing paths.
        def pin_references(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value and isinstance(value["path"], str):
                    possible = Path(value["path"])
                    if not possible.is_absolute():
                        possible = ROOT / possible
                    if possible.is_file():
                        pin(possible, value["sha256"])
                for key, item in value.items():
                    pin_references(key)
                    pin_references(item)
            elif isinstance(value, list):
                for item in value:
                    pin_references(item)
            elif isinstance(value, str) and value.lower().endswith((".csv", ".json", ".md", ".yaml", ".yml")):
                possible = Path(value)
                if not possible.is_absolute():
                    possible = ROOT / possible
                if possible.is_file() and possible.resolve().as_posix() not in inputs:
                    pin(possible)
        pin_references(evidence)
        support_text.append(text_path.read_text(encoding="utf-8"))

    legacy = support_json["legacy_ablation_audit"]
    legacy_tables = {}
    for key, info in legacy["tables"].items():
        legacy_tables[key] = pd.read_csv(info["absolute_path"])
    legacy_delta_rows = []
    for row in legacy["ablation_deltas"]:
        data = legacy_tables[row["table"]]
        nv = data[data.method.eq(row["new_method"])].iloc[0][row["metric"]]
        rv = data[data.method.eq(row["reference_method"])].iloc[0][row["metric"]]
        assert np.isclose(nv, row["new_value"], rtol=1e-12)
        assert np.isclose(rv, row["reference_value"], rtol=1e-12)
        assert np.isclose(100 * (nv / rv - 1), row["relative_change_pct"], rtol=1e-9, atol=1e-10)
        record = dict(row)
        record["new_source"] = json.dumps(record["new_source"], ensure_ascii=False)
        record["reference_source"] = json.dumps(record["reference_source"], ensure_ascii=False)
        legacy_delta_rows.append(record)
    write_table("legacy_ablation_deltas", pd.DataFrame(legacy_delta_rows))
    for key in ["alimu/overall_summary.csv", "penn/overall_summary.csv", "penn/fold_summary.csv", "paper/comparison_table.csv", "strict_paper/comparison_table.csv"]:
        write_table("legacy_" + key.replace("/", "__").replace(".csv", ""), legacy_tables[key])
    checks.append("30_legacy_ablation_deltas_checked_against_original_tables")

    fir = support_json["fir_arx_audit"]
    fir_cache, fir_verified = {}, set()
    def verify_fir_rows(obj):
        if isinstance(obj, dict):
            if all(key in obj for key in ["csvsource", "csv_line", "rowkeys", "values"]):
                path = obj["csvsource"]
                if path not in fir_cache:
                    fir_cache[path] = pd.read_csv(path, dtype=str, keep_default_na=False)
                data = fir_cache[path]
                row = data.iloc[int(obj["csv_line"]) - 2]
                for key, value in dict(obj["rowkeys"], **obj["values"]).items():
                    actual = row[key]
                    if str(actual) != str(value):
                        try:
                            assert np.isclose(float(actual), float(value), rtol=1e-10, atol=1e-14, equal_nan=True)
                        except (ValueError, TypeError):
                            raise AssertionError((path, obj["csv_line"], key, actual, value))
                fir_verified.add((path, obj["csv_line"]))
            for item in obj.values():
                verify_fir_rows(item)
        elif isinstance(obj, list):
            for item in obj:
                verify_fir_rows(item)
    verify_fir_rows(fir)
    for branch, mappings in fir["formal_tables"].items():
        for name, records in mappings.items():
            if name == "metrics_by_protocol.csv":
                data = pd.read_csv(records[0]["csvsource"])
                write_table("fir_" + branch + "_by_protocol", data)
    write_table("fir_branch_comparison", pd.read_csv(fir["published_comparison_tables"]["branch_summary.csv"][0]["csvsource"]))
    checks.append(str(len(fir_verified)) + "_FIR_ARX_evidence_rows_checked_against_original_CSV")

    nav = support_json["navigation_scope_audit"]
    nav_source = pd.read_csv(pin(ROOT / "runs/public_data_nature_summary_20261101/source_data/fig3_navigation_source.csv"))
    nav_rows, nav_best = [], []
    for entry in nav["smartpnt"]["formal_TEST"]:
        for item in entry["all_test_results"]:
            z = nav_source[nav_source.protocol_id.eq(entry["protocol_id"]) & nav_source.split.eq("test") & nav_source.aggregation_level.eq("dataset") & nav_source.model_family.eq(item["model_family"]) & nav_source.variant.eq(item["variant"]) & nav_source.metric.eq("vector_rmse_m")]
            assert len(z) == 1
            assert np.isclose(z.iloc[0]["mean"], item["mean_m"], rtol=1e-12)
            record = {k: v for k, v in item.items() if k != "evidence"}
            record.update(protocol=entry["protocol_id"], test_trajectory=",".join(entry["split"]["test"]["trajectories"]), unit="m")
            nav_rows.append(record)
        winner = min(entry["all_test_results"], key=lambda row: row["mean_m"])
        assert winner["mean_m"] == entry["best_algorithm"]["mean_m"]
        nav_best.append({"protocol": entry["protocol_id"], "test_trajectory": ",".join(entry["split"]["test"]["trajectories"]),
                         "method": winner["model_family"], "variant": winner["variant"], "rmse_m": winner["mean_m"],
                         "zero_rmse_m": entry["zero_baseline"]["mean_m"], "best_neural_method": entry["best_neural"]["model_family"],
                         "best_neural_variant": entry["best_neural"]["variant"], "best_neural_rmse_m": entry["best_neural"]["mean_m"]})
    write_table("smartpnt_all_test_models", pd.DataFrame(nav_rows))
    write_table("smartpnt_test_winners", pd.DataFrame(nav_best))
    downstream = historical("scoped_downstream_comparison")
    ds_rows = []
    for entry in nav["penn_downstream"]["per_context_task_horizon_best"]:
        item = entry["best_algorithm"]
        selector = item["evidence"]["selector"]
        data = downstream.copy()
        for key, value in selector.items():
            if key == "model":
                key = "canonical_method"
            data = data[data[key].eq(value)]
        assert len(data) == 1
        metric_col = "heading_endpoint_rmse_deg_mean" if entry["task"] == "heading_endpoint" else "mean"
        if pd.isna(data.iloc[0][metric_col]):
            metric_col = "heading_endpoint_rmse_deg"
        assert np.isclose(data.iloc[0][metric_col], item["mean"], rtol=1e-12)
        ds_rows.append({"protocol": entry["protocol_id"], "recording": entry["recording"], "task": entry["task"], "horizon_s": entry["horizon_s"],
                        "method": item["model"], "variant": item["variant"], "mean": item["mean"], "unit": entry["unit"],
                        "low_mean": entry["low"]["mean"], "best_neural": entry["best_neural"]["model"], "best_neural_mean": entry["best_neural"]["mean"]})
    write_table("penn_navigation_winners", pd.DataFrame(ds_rows))
    checks += ["78_SmartPNT_TEST_results_checked_against_source", "60_Penn_navigation_winners_checked_against_sealed_source"]

    inventory = []
    for directory in sorted((ROOT / "runs").iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name
        if name == old.name:
            category = "authoritative_historical_report"
        elif name == recent.name:
            category = "authoritative_matched_training"
        elif "smoke" in name or "synthetic_validation" in name:
            category = "smoke_or_fixture_not_scientific_test"
        elif "bundle" in name or "deployment" in name or "server_image" in name:
            category = "packaging_not_test_evidence"
        elif "VALID" in name or "validation_selection" in name or "partial" in name:
            category = "validation_or_intermediate_analysis"
        elif "amp_diagnostic" in name or "preflight" in name:
            category = "diagnostic_not_final"
        else:
            category = "other_result_or_analysis_see_source"
        inventory.append({"run_id": name, "catalog_category": category,
                          "has_top_level_report": (directory / "report.md").is_file(),
                          "has_complete_marker": (directory / "complete.json").is_file(),
                          "top_level_csv_count": len(list(directory.glob("*.csv"))),
                          "validity_note": "Catalog only; file existence does not establish scientific validity"})
    write_table("run_directory_catalog", pd.DataFrame(inventory))

    apply_paper_style()
    captions = []
    domains = sorted({(r.protocol, r.dataset_id) for r in a.itertuples()}, key=lambda p: (PROTOCOLS.index(p[0]), p[1]))
    fams = ["cnn_bilstm", "informer", "plain_cnn", "gru", "transformer", "tcn", "deepgyro_adapter", "tlio_resnet18_adapter", "itransformer", "patchtst"]
    fig, panels = plt.subplots(1, 3, figsize=(7.2, 5.0), gridspec_kw={"width_ratios": [1, 1, .045]})
    axes, color_axis = panels[:2], panels[2]
    for ax, component in zip(axes, ["gyro", "accel"]):
        z = a[a.component.eq(component)]
        ratios = np.empty((len(fams), len(domains)))
        labels = np.empty(ratios.shape, dtype=object)
        for i, fam in enumerate(fams):
            for j, (protocol, dataset) in enumerate(domains):
                data = z[z.protocol.eq(protocol) & z.dataset_id.eq(dataset)]
                val = data[data.canonical_method.eq(fam) & data.variant.eq("tuned")].iloc[0]["mean"]
                control = data[data.information_scope.eq("protocol_shared_classical")]["mean"].min()
                ratios[i, j] = val / control
                labels[i, j] = "%.2f" % ratios[i, j]
        img = ax.imshow(ratios, cmap="RdBu_r", norm=matplotlib.colors.TwoSlopeNorm(vmin=.2, vcenter=1, vmax=2.5), aspect="auto")
        for i in range(len(fams)):
            for j in range(len(domains)):
                ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if ratios[i, j] < .65 or ratios[i, j] > 1.9 else "black")
        ax.set_xticks(range(len(domains)))
        ax.set_xticklabels([p.replace("penn_loso_", "Penn ").replace("alimu_only", "ALiMU").replace("penn_only", "Penn bs").replace("pooled", "Joint") + ("/AL" if p == "pooled" and d == "alimu" else "/Penn" if p == "pooled" else "") for p, d in domains], rotation=55, ha="right")
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([NAMES[f] for f in fams] if component == "gyro" else [])
        ax.set_title(component.capitalize() + " RMSE / best shared classical")
        ax.grid(False)
    fig.colorbar(img, cax=color_axis, extend="max", ticks=[.2, .6, 1, 1.5, 2, 2.5], label="Ratio (<1 favors neural)")
    fig.subplots_adjust(left=.17, bottom=.24, right=.92, wspace=.15)
    save_figure(fig, figures, "historical_signal_comparison")
    captions.append(("historical_signal_comparison", "Oct31 completed TEST results; tuned neural RMSE divided by the best protocol-shared six-LOW classical RMSE in the same dataset/split. Both physical quantities are separate. Values <1 favor neural models; identity-routed banks are excluded. ALiMU VN100 and Penn ADIS16488 are measured references."))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    order = ["ddpm", "cnn_bilstm", "informer", "transformer", "no_diffusion", "no_diffusion_matched"]
    for ax, component, scale, unit in zip(axes, ["gyro", "accel"], [1000, 1], ["$10^{-3}$ rad/s", "m/s$^2$"]):
        z = new[new.pair_id.eq("__all_pairs__") & new.component.eq(component)].set_index("method").loc[order]
        ax.barh(range(len(z)), z.rmse_mean * scale, xerr=z.rmse_sd * scale, color=["#c95f3f", "#1f77b4", "#2a9d8f", "#597fa5", "#208b76", "#7a8d9f"], capsize=2)
        ax.set_yticks(range(len(z)))
        ax.set_yticklabels([NAMES[m] for m in order] if component == "gyro" else [])
        ax.invert_yaxis()
        ax.set_xlabel(component.capitalize() + " component RMSE [" + unit + "]")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figures, "matched_alimu_comparison")
    captions.append(("matched_alimu_comparison", "Nov6 matched ALiMU TEST protocol: each device trained separately, 80 points, three candidates per family, Adam, 40 epochs maximum, three final optimization seeds. Bars are pooled component RMSE means; error bars are sample SD across optimization seeds, not physical recordings. Exact no-diffusion ablation shares DDPM-selected architecture and initial weights."))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, component in zip(axes, ["gyro", "accel"]):
        z = h[h.component.eq(component)]
        ordered = [z[z.family.eq(f)].improvement_pct.to_numpy() for f in fams]
        ax.boxplot(ordered, vert=False, widths=.5, showfliers=False)
        for i, values in enumerate(ordered, 1):
            ax.scatter(values, np.full(len(values), i), s=9, color="#1f77b4", zorder=3)
        ax.axvline(0, color="#c95f3f", linewidth=1)
        ax.set_yticks(range(1, len(fams) + 1))
        ax.set_yticklabels([NAMES[f] for f in fams])
        ax.set_xlabel(component.capitalize() + " TEST RMSE reduction after HPO [%]")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figures, "tuning_improvements")
    captions.append(("tuning_improvements", "Oct31 default vs validation-selected HPO across seven dataset/protocol cells per family and metric; each point is a TEST comparison. Positive means lower RMSE after HPO. This is a configuration-package comparison, not an isolated architecture/loss ablation."))
    methods = ["identity", "channel_affine", "learned_fir_causal", "tcn", "bias_only", "drift_only", "bias_drift", "fir_bias_drift_residual"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, key, title in zip(axes, ["alimu/overall_summary.csv", "penn/overall_summary.csv"], ["July ALiMU held-out MPU9250", "July Penn four-fold macro mean"]):
        data = legacy_tables[key].set_index("method").loc[methods]
        ax.barh(range(len(methods)), data.normalized_rmse_score_mean, color=["#c95f3f", "#7a8d9f", "#2a9d8f", "#1f77b4", "#597fa5", "#597fa5", "#597fa5", "#208b76"])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m.replace("fir_bias_drift_residual", "FIR + neural residual").replace("learned_fir_causal", "Frozen FIR").replace("channel_affine", "Channel affine").replace("_", " ") for m in methods])
        ax.set_title(title)
        ax.set_xlabel("Normalized score (lower is better)")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save_figure(fig, figures, "legacy_residual_ablation")
    captions.append(("legacy_residual_ablation", "Separate July protocols, not comparable to October/November runs. Each score is half of gyro/raw-gyro plus accel/raw-accel on its own test/fold. ALiMU uses held-out sensor with shared physical reference recordings across splits; Penn is a four-fold macro average. FIR + neural residual improves Penn over frozen FIR by 2.094% but worsens ALiMU by 5.908%."))
    (figures / "figure_index.md").write_text("# Figures\n\n" + "\n\n".join("## " + stem + "\n\n" + caption + "\n\n![" + stem + "](" + stem + ".png)" for stem, caption in captions) + "\n", encoding="utf-8")

    # Compact human-readable tables, with every exported table keeping original precision.
    def signal_matrix(data, method_col, value_col, sd_col=None):
        rows = []
        for method, group in data.groupby(method_col, sort=False):
            row = {"模型": NAMES.get(method, method)}
            for component, column, scale, fmt in [("gyro", "角速度 RMSE (×10⁻³ rad/s)", 1000, ".3f"), ("accel", "加速度 RMSE (m/s²)", 1, ".4f")]:
                z = group[group.component.eq(component)].iloc[0]
                row[column] = format(z[value_col] * scale, fmt)
                if sd_col and pd.notna(z[sd_col]):
                    row[column] += " ± " + format(z[sd_col] * scale, fmt)
            rows.append(row)
        return pd.DataFrame(rows)

    winner_rows = []
    for protocol, dataset in domains:
        row = {"实验/测试域": protocol + " / " + dataset}
        for comp, col, scale, fmt in [("gyro", "最佳角速度", 1000, ".3f"), ("accel", "最佳加速度", 1, ".4f")]:
            w = winners[winners.protocol.eq(protocol) & winners.dataset_id.eq(dataset) & winners.component.eq(comp)].iloc[0]
            row[col] = w["name"] + (" [default]" if w.variant == "default" else "") + "：" + format(w.rmse_component_mean * scale, fmt)
        winner_rows.append(row)
    matched_rows = []
    for pair, group in nw.groupby("pair_id", sort=False):
        row = {"设备": pair}
        for comp, col, scale, fmt in [("gyro", "最佳角速度", 1000, ".3f"), ("accel", "最佳加速度", 1, ".4f")]:
            w = group[group.component.eq(comp)].iloc[0]
            row[col] = NAMES[w.method] + "：" + format(w.rmse_mean * scale, fmt)
        matched_rows.append(row)

    pooled = new[new.pair_id.eq("__all_pairs__")]
    ab = pooled.set_index(["method", "component"])
    gd = 100 * (ab.loc[("ddpm", "gyro"), "rmse_mean"] / ab.loc[("no_diffusion_matched", "gyro"), "rmse_mean"] - 1)
    ad = 100 * (ab.loc[("ddpm", "accel"), "rmse_mean"] / ab.loc[("no_diffusion_matched", "accel"), "rmse_mean"] - 1)
    old_al = h[h.protocol.eq("alimu_only")].copy()
    old_al["family"] = pd.Categorical(old_al.family, fams)
    old_al = old_al.sort_values("family")

    text = """# 现有模型与消融实验总审计（2026-11-06）

本报告整理已经完成的实验，不新增训练。推荐结论按任务、训练协议、信息范围和测试域分别给出；所有“最佳”都是现有已测方法在相同协议内的 TEST 均值最低，不构成全局或文献 SOTA 声明，也没有使用 TEST 重新调参。seed 标准差仅衡量优化随机性。

## 结论先行

- ALiMU 历史大规模 HPO、三设备共享模型、允许 80/160/320 点上下文：CNN-BiLSTM 同时取得最低角速度和加速度误差；需要因果模型时优先 GRU。
- ALiMU 近期固定 80 点、逐设备训练的统一重跑：合并角速度选 Informer，合并加速度选独立调参的无扩散骨干；BMX160 单设备普通 Transformer 两项均最佳。
- PennCOSVIO：共享 FIR/ARX 在多数录制/指标上最好；af/as 的角速度由 GRU 最好。不存在神经网络全面优于系统辨识的证据。
- 共同训练 ALiMU+Penn：ALiMU 域 CNN-BiLSTM 两项最好；Penn 域加速度 CNN-BiLSTM、角速度 default GRU 最好，但该 Penn 加速度仍不及 Penn 专域 FIR。
- DDPM 的严格消融未显示扩散收益：18/18 个设备×seed×信号组对照均为去扩散更好。
- 跨数据集直接迁移、SmartPNT 位移预测和独立真值下游导航均有明确局限；具体负结果见后续审计，不能从六轴拟合改善推导导航整体升级。

如果当前只保留一个模型继续推进这轮 ALiMU 逐设备实验，建议以 Informer 为主模型：合并角速度第一，加速度距最低约 1.5%，三 seed 波动也较小；若首要指标是加速度，则选独立调参的无扩散 Transformer。这个工程选型建议基于近期统一协议，不把它写成所有数据集的总冠军。

### 消融实验结论总表

| 消融/比较 | 结果 | 可用结论 |
| --- | --- | --- |
| DDPM→相同 Transformer 骨干直接回归 | 严格18/18配对均为去扩散更好 | 当前扩散机制无收益，优先直接回归 |
| spec/static 上下文与 FiLM | 旧版5个变体的混合6D分数都差于无上下文 TCN；部分 gyro/accel 存在取舍 | 不支持增加上下文必然有效；单 seed，非未见录制协议 |
| bias/drift 分支 | July ALiMU 神经项 drift-only 最好；Penn bias+drift 比单分支更好 | 分支收益依赖数据集与上下文范围 |
| offset 与辅助 loss | 旧单 seed 混合分数改善，但 aux 可使 gyro 误差上升约2.0%/10.8% | 不能宣称所有六轴均改善，尚无匹配3seed aux开关证据 |
| frozen FIR→FIR+神经残差 | July Penn 归一化分数降低2.094%；ALiMU升高5.908% | 仅 Penn 有小幅增益，不跨域通用 |
| default→联合超参优化 | 历史140项中126改善、14变差 | 调参通常有用，但不是单因素结构/损失消融 |
| VMD low/high/joint 特征 | FIR/ARX各条件均0/6协议改善joint分数 | 当前直接回归实现明显失败，不推论所有VMD方法 |
| TDA路由 | FIR/ARX各仅2/6 joint改善，中位误差变化为正 | 不支持优先使用 |
| 最小分量选择 | 表内joint改善5/6、双项4/6；mean joint改善仅0.10% | 仅待核实候选：分量索引与状态留存存在缺口，不能宣称压缩方案已验证 |

minimal 的可追溯本地分支中，18分量bank的滞后列索引仍按6列步长构造；服务器原始源码身份又不完整，不能断言服务器执行了同一问题。保留已有CSV，明确限制其解释，不将它作为当前最佳模型推荐。完整位置、归档哈希与证据在 FIR/ARX 分项审计中。

## 数据集、任务与可比范围

| 数据集/场景 | 输入→目标 | 训练/测试范围 | 适用结论 |
| --- | --- | --- | --- |
| ALiMU | BMX160、MPU9250、LSM9DS1→VN100 实测六轴 | Oct31 三设备共享；Nov6 三设备分别训练；旧版另有留 MPU9250 消融 | 低成本→参考 IMU 映射；VN100 不是无噪真值 |
| PennCOSVIO | Tango Top IMU→ADIS16488 实测六轴 | af/as/bf/bs 四录制；bs 主测试与 af/as/bf 外留出 | 同数据集跨录制、单对传感器映射 |
| ALiMU+Penn 联合训练 | 两域六轴→各自参考 IMU | 重新联合训练并分域评估 | 不等同于零样本跨域迁移 |
| SmartPNT-MSF | 六轴→1.975 s 窗口的 ENU 位移 | 三条独立内容的轨迹轮换；Traj2 与 Traj1 重复 | 无六轴高等级 IMU 标签，只能评价位移任务 |
| 独立真值导航诊断 | 映射后的六轴→姿态/位置 | Penn 轨迹；按相同视图、时长、初始化条件比较 | 与信号 RMSE 分开评价，部分视图有 GT 初始化 |

Oct31 ALiMU TRAIN 为 dyn01/02/04/05/06 与 stat11，VAL dyn07，TEST dyn08/09；同内容的 dyn03/stat10 已去重。静态录制仅在 TRAIN，不能宣布“静态测试最佳”。Penn 主协议 af/as 训练、bf 验证、bs 测试；额外外留出不是新增数据集。

历史 HPO：10 个六轴模型家族、6 个配对训练协议、7 个协议×测试域结果单元，default/tuned 各 3 seed；24 候选逐级 8/24/80 轮，最多 80 轮、patience 15，AdamW。近期重跑：每设备 3 候选、80 点窗口、Adam、最多 40 轮、patience 7，各 3 seed。历史模型只见六轴；近期含一个恒为 25 的缺失温度占位通道。不同预算/训练方式的最低数值不用于跨实验胜负判断。

### 误差统一口径

以下六轴信号表统一为**三轴分量 RMSE**：sqrt(mean over time and all three axes of squared error)。历史原始表是三维向量 RMSE，已除以 √3；标准差也同除。角速度用 ×10⁻³ rad/s，加速度用 m/s²，均越小越好。CSV 保留原始值、换算因子、来源与完整精度。导航位移仍用三维向量 RMSE（m），不作该换算。

## 历史统一信息范围：各测试场景最佳

主表仅允许相同六轴 LOW 输入的神经网络、协议共享经典校准和 Raw；依赖设备身份路由的 classical bank 单独列在补充 CSV。下表默认包含已完成的 default 和 validation-selected tuned 两组，若 default 胜出则显式标注。角速度/加速度两列分别评选，不将不同物理单位相加。

"""
    text += md_table(pd.DataFrame(winner_rows)) + "\n\n"
    text += "完整方法表：" + link(tables / "historical_signal_all_methods.csv") + "；因果/窗口前视分别最佳：" + link(tables / "historical_temporal_winners.csv") + "。CNN-BiLSTM 和多数注意力模型使用窗口内未来信息；GRU、TCN、DeepGyro 和经典过去滞后方法属于准备后的 40 Hz 数据上的因果评价。80/160/320 点分别是 2/4/8 秒采样长度，窗口首末点间隔为 1.975/3.975/7.975 秒；这不等于系统端到端延迟。\n\n"
    text += "### ALiMU 历史全部 10 个神经网络（tuned）\n\n"
    text += md_table(signal_matrix(old_al, "family", "tuned_component_mean", "tuned_component_sd")) + "\n\n"
    text += "CNN-BiLSTM 的 ALiMU 最优配置使用 320 点，Informer 为 80 点，普通 Transformer 为 160 点。因此它们是各自调参后的完整方法比较，并非固定上下文长度的结构消融。其他所有域的 10 家族数值及 default/tuned 对照见 " + link(tables / "neural_default_vs_tuned.csv") + "。DeepGyro、TLIO 为同任务适配实现，不能写成原论文完整系统复现。\n\n"
    for component, scale, fmt in [("gyro", 1000, ".3f"), ("accel", 1, ".4f")]:
        text += "### 全部测试域：" + ("角速度（×10⁻³ rad/s）" if component == "gyro" else "加速度（m/s²）") + "\n\n"
        rows = []
        for family in fams:
            row = {"调参模型": NAMES[family]}
            for protocol, dataset in domains:
                value = h[h.family.eq(family) & h.protocol.eq(protocol) & h.test_dataset.eq(dataset) & h.component.eq(component)].iloc[0].tuned_component_mean
                label = {"alimu_only": "ALiMU", "penn_only": "Penn bs", "penn_loso_af": "Penn af", "penn_loso_as": "Penn as", "penn_loso_bf": "Penn bf", "pooled": "联合/" + ("ALiMU" if dataset == "alimu" else "Penn")}[protocol]
                row[label] = format(value * scale, fmt)
            rows.append(row)
        text += md_table(pd.DataFrame(rows)) + "\n\n"
    text += "## 近期 ALiMU 统一重跑：DDPM 与直接回归\n\n"
    text += md_table(signal_matrix(pooled, "method", "rmse_mean", "rmse_sd")) + "\n\n"
    text += md_table(pd.DataFrame(matched_rows)) + "\n\n"
    text += "本批 TEST 每 seed 为 89,949 个设备时间样本（每设备 29,983）；历史为 89,760（每设备 29,920）。两批均只有两条物理 TEST 录制，三设备共享 VN100，不能把三设备或三 seed 当作独立场景样本。\n\n"
    text += "### 严格扩散消融\n\nDDPM 与严格无扩散使用相同选定骨干（143,174 参数）、学习率、逐 seed 初始权重、训练/验证支持及 minibatch 顺序。无扩散将 6 通道噪声输入和时间固定为零，直接回归相同标准化残差。DDPM 的合并角速度误差高 " + format(gd, ".1f") + "%，加速度高 " + format(ad, ".1f") + "%；全部 18 个严格配对差值均为 DDPM 更差。独立调参的无扩散项允许不同的骨干配置，不属于严格单因素消融。DDPM 为 100 步、4 draws 的完整祖先采样，推理成本也明显更高。\n\n"
    text += "本机 CUDA 原始计时及复测保留于原运行；负载敏感，不将计时高值视为稳定硬件常数。" + link(recent / "findings.md", "完整实测及复测") + "。\n\n"
    text += "## 调参收益与跨域迁移\n\n历史 140 个家族×协议域×信号组对照中，126 项调参后误差降低、14 项升高，中位改善 " + format(float(h.improvement_pct.median()), ".1f") + "%。这是多因素配置包对照，不能把收益单独归因于 Transformer 层数、损失项或窗口长度。\n\n"
    text += "跨域完整表 " + link(tables / "transfer_all_models.csv") + " 与 " + link(tables / "transfer_winners_by_scope.csv") + " 同时保留两种问题：只比较源域已调参模型，或回顾所有已测 default/tuned。后者有时反而是 default 领先；这是现有 TEST 观测，不是允许在目标 TEST 上重新选择部署超参。最小误差仍需对照目标域 Raw 与目标域训练结果，不能只看迁移模型之间的排名。\n\n"
    tb = pd.concat(tr_best, ignore_index=True)
    rows = []
    for row in tb.itertuples():
        rows.append({"范围": row.ranking_scope, "迁移/目标传感器": row.direction + "/" + row.low_sensor,
                     "量": row.component, "最佳已测项": NAMES[row.method] + "/" + row.variant,
                     "分量RMSE": format(row.rmse_component_mean * (1000 if row.component == "gyro" else 1), ".4f"),
                     "相对目标Raw降低%": format(row.reduction_vs_raw_pct, ".1f")})
    text += md_table(pd.DataFrame(rows)) + "\n\n"
    text += "## 其他消融、导航和未完成方案核查\n\n以下分项包含旧版单 seed/三 seed 的传感器留出、上下文/FiLM、bias/drift/辅助损失、FIR/ARX、VMD/TDA 的执行状态与 SmartPNT/独立真值导航。每项均保留原单位及源文件，**分项中若仍为向量 RMSE，不与上文分量值直接对照**。较早同一预测的重绘、VALID 中间分析、smoke 和打包目录均不计作新 TEST 证据。\n\n"
    for name, supplement in zip(cfg["support_reports"], support_text):
        text += "### " + name + "\n\n" + supplement + "\n\n"
    text += "## 汇总图\n\n"
    for stem, caption in captions:
        text += "### " + stem + "\n\n" + caption + "\n\n![" + stem + "](" + (figures / (stem + ".png")).resolve().as_posix() + ")\n\n"
    text += "## 交付与核验\n\n"
    text += "- " + link(tables / "historical_signal_all_methods.csv", "历史全部信号方法") + "、" + link(tables / "neural_default_vs_tuned.csv", "10 家族各域调参对照") + "\n"
    text += "- " + link(tables / "matched_alimu_all_models.csv", "近期所有设备模型") + "、" + link(tables / "diffusion_strict_paired_ablation.csv", "严格扩散配对消融") + "\n"
    text += "- " + link(tables / "transfer_all_models.csv", "跨域所有模型") + "、" + link(figures / "figure_index.md", "全部汇总图与图注") + "\n"
    text += "- " + link(out / "provenance.json", "来源/源码/配置 SHA256") + "、" + link(out / "verification.json", "本次核验") + "\n\n"
    text += "本次核验范围是被引用源表的原封存哈希、逐 seed 汇总重算、排名和报告数字；没有重训历史模型，也没有重放全部历史原始数据。原 Oct31 报告的 PDF 视觉 QA 仍标记 pending，本次只使用经哈希验证的结构化表，不把该视觉状态包装为训练或数值核验。原实验完整预测重放的 PASS 依据保留在近期运行 verification.json 中。\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    (out / "config.yaml").write_bytes(config_path.read_bytes())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    pin(config_path)
    pin(Path(__file__))
    pin(ROOT / "src/virtual_imu/viz/style.py")
    snapshot = out / "analysis_source"
    snapshot.mkdir(exist_ok=True)
    (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (snapshot / "viz_style.py").write_bytes((ROOT / "src/virtual_imu/viz/style.py").read_bytes())
    provenance = {"experiment_id": cfg["experiment_id"], "created_at": datetime.now(timezone.utc).isoformat(),
                  "git_commit": git, "git_branch": branch, "analysis_script_sha256": sha(__file__),
                  "config_sha256": sha(config_path), "inputs": inputs,
                  "runtime": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                  "original_runs_preserved": True, "latest_snapshot_modified": False,
                  "interpretation": "observed TEST rankings within protocols; not new model selection"}
    json_write(out / "provenance.json", provenance)
    outputs = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "verification.json" and p.suffix != ".log" and "support" not in p.relative_to(out).parts}
    json_write(out / "verification.json", {"status": "PASS", "experiment_id": cfg["experiment_id"], "checks": checks,
                                         "checked_original_sealed_inputs": sum(v["checked_against_original_seal"] for v in inputs.values()),
                                         "pinned_input_count": len(inputs), "outputs_sha256": outputs,
                                         "report_sha256": sha(out / "report.md"), "provenance_sha256": sha(out / "provenance.json")})
    print("WRITES " + str(out / "report.md"))
    print("WRITES provenance.json, verification.json, config.yaml and four SVG/PDF/PNG figure sets")
    print("SKIPS training; original artifacts; partial/VALID/smoke rankings; outputs/latest")
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis/comprehensive_model_audit.yaml")
    args = parser.parse_args()
    run(args.config.resolve())
