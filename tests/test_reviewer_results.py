"""Saved physical metrics, honest seed SD, deterministic baselines and ranks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.eval.formal_evaluation import HIGH, LOW
from virtual_imu.eval.reviewer_evaluation import _pair_metrics
from virtual_imu.models import make_model
from virtual_imu.training.reviewer_hpo import run_reviewer_hpo

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/aggregate_reviewer_results.py"
METHODS = ["model", "identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear", "ridge_tuned", "fir_tuned", "arx_tuned"]


def module():
    spec = importlib.util.spec_from_file_location("reviewer_results_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path: Path, *, varying_classical=False, fail_first=False):
    manifest = tmp_path / "manifest.json"
    records = [{"sequence_id": f"{split}_segment", "original_sequence_id": f"recording_{split}", "physical_group_id": f"alimu/{split}", "dataset_id": "alimu", "low_sensor": "BMX160", "high_sensor": "VN100"} for split in ("validation", "test")]
    manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
    split_root = tmp_path / "splits" / "alimu_only"
    split_root.mkdir(parents=True)
    for name in ("split_manifest.json", "strict_manifest.json"):
        (split_root / name).write_text("{}", encoding="utf-8")
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump({"data": {"dataset_manifest": str(manifest), "window_size": 80}, "model": {"type": "gru"}, "training": {}}), encoding="utf-8")
    config = {"seed": 20261024, "confirmation_seed": 20261025, "seeds": [20261010, 20261011, 20261012], "comparison": {"default_refits": True}, "device": "cpu", "tasks": {"paired": str(source)}, "protocol_split_root": str(split_root.parent), "protocols": [{"id": "alimu_only", "task": "paired"}], "candidates": [{"id": "gru", "tasks": ["paired"], "model": {"type": "gru", "hidden_size": 4, "num_layers": 1, "dropout": 0.0}, "search_space": {"model.hidden_size": {"choice": [8]}}}], "search": {"trials": 2, "rung_epochs": [1], "rung_keep": [2], "common_space": {"training.lr": {"choice": [0.002]}}, "paired_space": {"data.window_size": {"choice": [160]}}}}
    calls = []

    def train(task, actual, frozen):
        calls.append(actual)
        if fail_first and len(calls) == 1:
            raise MemoryError("visible failed source search fit")
        directory = Path(actual["output"]["runs_dir"]) / "fit"
        directory.mkdir(parents=True)
        model = make_model(actual["model"])
        score = 1 / actual["model"]["hidden_size"]
        (directory / "config.yaml").write_text(yaml.safe_dump(actual), encoding="utf-8")
        (directory / "metrics.csv").write_text(f"epoch,train_loss,val_loss,val_grouped_score\n1,{score},{score},{score}\n", encoding="utf-8")
        torch.save({"config": actual, "model_config": actual["model"], "model_state_dict": model.state_dict(), "best_epoch": 1, "best_metric": score, "checkpoint_metric": "val_grouped_score"}, directory / "best.ckpt")
        inputs = {}
        for key, name in (("dataset_manifest", "data_manifest.json"), ("split_manifest", "split_manifest.json"), ("strict_manifest", "strict_manifest.json")):
            path = Path(actual["data"][key])
            shutil.copyfile(path, directory / name)
            inputs[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (directory / "provenance.json").write_text(json.dumps({"unified_dataset": {"inputs": inputs}}), encoding="utf-8")
        return directory

    def evaluate(task, directory, actual):
        folder = directory / "evaluation"
        folder.mkdir()
        rows = []
        for split in ("validation", "test"):
            frame = pd.DataFrame(np.zeros((320, 6)), columns=LOW)
            frame[HIGH] = 1.0
            frame["timestamp_s"] = np.arange(320) / 40
            frame["sequence_id"] = f"{split}_segment"
            frame["dataset_id"], frame["low_sensor"], frame["high_sensor"] = "alimu", "BMX160", "VN100"
            for index, method in enumerate(METHODS):
                if method == "model":
                    error = (actual["seed"] - 20261009) / (20 if actual["reviewer_protocol"]["variant"] == "tuned" else 10)
                    prediction = np.ones((320, 6)) + error
                elif method == "identity":
                    prediction = np.zeros((320, 6))
                else:
                    extra = actual["seed"] * 1e-4 if varying_classical and method == "bias_mean" else 0
                    prediction = np.ones((320, 6)) + index * 0.01 + extra
                for axis_index, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
                    frame[f"{method}_{axis}"] = prediction[:, axis_index]
                metrics = _pair_metrics(prediction, frame[HIGH].to_numpy())
                row = {"split": split, "dataset_id": "alimu", "low_sensor": "BMX160", "high_sensor": "VN100", "method": method, **{key: value for key, value in metrics.items() if np.isscalar(value)}}
                for axis_index, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
                    for key, label in (("axis_rmse", "rmse"), ("axis_mae", "mae"), ("axis_mean_error", "mean_error"), ("axis_error_std", "error_std")):
                        row[f"{axis}_{label}_{'radps' if axis_index < 3 else 'mps2'}"] = metrics[key][axis_index]
                rows.append(row)
            frame.to_parquet(folder / f"{split}_predictions.parquet", index=False)
        pd.DataFrame(rows).to_csv(folder / "paired_metrics.csv", index=False)
        report = {"training_run": str(directory), "task": task, "reviewer_protocol": actual["reviewer_protocol"], "seed": actual["seed"]}
        (folder / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    return run_reviewer_hpo(config, root=tmp_path, train_backend=train, evaluate_backend=evaluate)


def test_physical_metrics_keep_gyro_accel_units_and_navigation_window_error():
    aggregate = module()
    paired = aggregate.physical_metrics("paired", np.array([[1, 0, 0, 0, 2, 0], [1, 0, 0, 0, 2, 0]]), np.zeros((2, 6)))
    assert paired["gyro_vector_rmse_radps"] == 1
    assert paired["accel_vector_rmse_mps2"] == 2
    assert paired["gyro_vector_mae_radps"] == 1 and paired["accel_bias_norm_mps2"] == 2
    navigation = aggregate.physical_metrics("navigation", np.array([[1, 2, 2], [1, 2, 2]]), np.zeros((2, 3)))
    assert navigation["vector_rmse_m"] == 3 and navigation["vector_mae_m"] == 3
    assert navigation["horizontal_rmse_m"] == pytest.approx(np.sqrt(5))
    assert navigation["bias_norm_m"] == 3 and navigation["east_error_std_m"] == 0


def test_saved_artifacts_aggregate_seed_sd_classical_n1_matched_deltas_and_failures(tmp_path):
    aggregate = module()
    root = fixture(tmp_path, fail_first=True)
    out = tmp_path / "summary"
    report = aggregate.aggregate_results(root, out, allow_subset=True)
    assert report["runs"] == 6 and report["failed_fit_jobs"] == 1
    summary = pd.read_csv(out / "metrics_seed_summary.csv")
    chosen = summary.loc[summary.split.eq("test") & summary.aggregation_level.eq("pair") & summary.metric.eq("gyro_vector_rmse_radps")]
    neural = chosen.loc[chosen.variant.eq("default")].iloc[0]
    assert neural["mean"] == pytest.approx(np.sqrt(3) * 0.2)
    assert neural.sample_sd == pytest.approx(np.sqrt(3) * 0.1)
    assert neural.n == 3 and neural.optimization_seed_count == 3
    classical = chosen.loc[chosen.variant.eq("classical")]
    assert classical.n.eq(1).all() and classical.sample_sd.isna().all()
    assert classical.optimization_seed_count.eq(0).all()
    paired = pd.read_csv(out / "matched_seed_delta_summary.csv")
    delta = paired.loc[paired.split.eq("test") & paired.aggregation_level.eq("pair") & paired.metric.eq("gyro_vector_rmse_radps")].iloc[0]
    assert delta.delta_mean == pytest.approx(-np.sqrt(3) * 0.1) and delta.paired_seed_count == 3
    relative = pd.read_csv(out / "relative_identity_seed_summary.csv")
    relative_model = relative.loc[relative.split.eq("test") & relative.variant.eq("default") & relative.metric.eq("gyro_vector_rmse_radps")].iloc[0]
    assert relative_model.mean_percent == pytest.approx(20) and relative_model.sample_sd_percent == pytest.approx(10)
    assert relative_model.mean_improvement_percent_vs_identity == pytest.approx(80)
    ledger = pd.read_csv(out / "all_fit_jobs.csv")
    assert ledger.status.eq("failed").sum() == 1
    assert ledger.elapsed_seconds.notna().all()
    dedup = json.loads((out / "classical_deduplication.json").read_text(encoding="utf-8"))
    assert len(dedup) == 18 and all(len(entry["source_job_ids"]) == 6 for entry in dedup)
    assert aggregate.aggregate_results(root, out, allow_subset=True, resume=True) == report
    table = out / "metrics_seed_summary.csv"
    original_table = table.read_bytes()
    table.write_bytes(b"changed output table")
    with pytest.raises(ValueError, match="output artifact changed"):
        aggregate.aggregate_results(root, out, allow_subset=True, resume=True)
    table.write_bytes(original_table)
    saved = next(Path(job["run_dir"]) / "evaluation/test_predictions.parquet" for job in json.loads((root / "pipeline.json").read_text(encoding="utf-8"))["jobs"].values() if job["phase"] == "final_tuned")
    saved.write_bytes(b"changed after aggregation")
    with pytest.raises(ValueError, match="source artifact changed"):
        aggregate.aggregate_results(root, out, allow_subset=True, resume=True)


def test_repeated_classical_predictions_are_not_deduplicated_if_they_differ(tmp_path):
    aggregate = module()
    root = fixture(tmp_path, varying_classical=True)
    with pytest.raises(ValueError, match="Repeated classical"):
        aggregate.aggregate_results(root, tmp_path / "summary", allow_subset=True)
    assert not (tmp_path / "summary").exists()


def test_rank_correlations_separate_pooled_slices_and_reversed_standalone_training_domains():
    aggregate = module()
    rows = []
    for protocol, dataset, values in (("alimu_only", "alimu", (1.0, 2.0)), ("penn_only", "penncosyvio", (2.0, 1.0)), ("pooled", "alimu", (1.0, 2.0)), ("pooled", "penncosyvio", (2.0, 1.0))):
        for family, value in zip(("cnn", "gru"), values, strict=True):
            rows.append({"task": "paired", "protocol_id": protocol, "training_scope": aggregate.training_scope(protocol), "split": "test", "aggregation_level": "dataset", "dataset_id": dataset, "low_sensor": "__all_devices__", "high_sensor": "__all_devices__", "variant": "tuned", "metric": "gyro_vector_rmse_radps", "unit": "rad/s", "method": "model", "model_family": family, "mean": value})
    ranks, correlations = aggregate.ranking_tables(pd.DataFrame(rows), ["gyro_vector_rmse_radps"])
    assert ranks.loc[ranks.protocol_id.eq("alimu_only") & ranks.model_family.eq("cnn"), "rank"].item() == 1
    reversed_domains = correlations.loc[correlations.comparison.eq("different_training_and_evaluation_datasets")]
    assert len(reversed_domains) == 1
    assert reversed_domains.spearman_rho.iloc[0] == pytest.approx(-1)
    assert reversed_domains.kendall_tau_b.iloc[0] == pytest.approx(-1)
    assert "pooled_training_evaluation_slices" in set(correlations.comparison)
    assert "same_target_different_training_scope" in set(correlations.comparison)
    assert not any("pvalue" in name for name in correlations.columns)


def test_production_gate_rejects_small_or_unfinished_core(tmp_path):
    aggregate = module()
    root = fixture(tmp_path)
    with pytest.raises(ValueError, match=r"93|558"):
        aggregate.aggregate_results(root, tmp_path / "summary")
    path = root / "pipeline.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"] = "searching"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="completed nonsmoke"):
        aggregate.aggregate_results(root, tmp_path / "summary", allow_subset=True)


def test_navigation_saved_schema_scores_each_continuous_segment_without_stitching():
    aggregate = module()
    parts, records = [], []
    for sequence, error in (("seg_a", 1.0), ("seg_b", 3.0)):
        offset = 20.0 if sequence == "seg_b" else 0.0
        frame = pd.DataFrame({"sequence_id": [sequence] * 2, "original_sequence_id": ["Traj1"] * 2, "start_index": [0, 40], "end_index": [79, 119], "start_time_s": [offset, offset + 1.0], "end_time_s": [offset + 1.975, offset + 2.975]})
        for axis in aggregate.NAV_AXES:
            frame[f"true_{axis}"] = 0.0
        for method, offset in (("model", error), ("zero", 0.0), ("train_mean", 2.0), ("ridge", 0.5), ("ridge_tuned", 0.25)):
            for index, axis in enumerate(aggregate.NAV_AXES):
                frame[f"{'pred' if method == 'model' else method}_{axis}"] = offset if index == 0 else 0.0
        parts.append(frame)
        records.append({"sequence_id": sequence, "original_sequence_id": "Traj1", "physical_group_id": "smartpnt_msf/Traj1", "dataset_id": "smartpnt_msf", "low_sensor": "imu", "high_sensor": "navigation_groundtruth"})
    frame = pd.concat(parts, ignore_index=True)
    saved_rows = []
    from virtual_imu.eval.formal_navigation import navigation_error_metrics

    for method in aggregate.NAV_METHODS:
        prefix = "pred" if method == "model" else method
        prediction = frame[[f"{prefix}_{axis}" for axis in aggregate.NAV_AXES]].to_numpy()
        target = frame[[f"true_{axis}" for axis in aggregate.NAV_AXES]].to_numpy()
        values = navigation_error_metrics(prediction, target)
        error = prediction - target
        values.update(vector_mae_m=float(np.linalg.norm(error, axis=1).mean()), bias_norm_m=float(np.linalg.norm(error.mean(axis=0))), axis_error_std_m=error.std(axis=0).tolist())
        row = {"split": "test", "method": method, **{key: value for key, value in values.items() if np.isscalar(value)}}
        for index, axis in enumerate(("east", "north", "up")):
            for key, label in (("axis_rmse_m", "rmse"), ("axis_mae_m", "mae"), ("axis_bias_m", "bias"), ("axis_error_std_m", "error_std")):
                row[f"{axis}_{label}_m"] = values[key][index]
        saved_rows.append(row)
    entry = {"protocol_id": "navigation_rotation_0", "model_family": "gru", "variant": "tuned", "seed": 20261010, "job_id": "saved_nav"}
    rows = aggregate.prediction_metrics(frame, pd.DataFrame(saved_rows), entry, "navigation", "test", {"records": records}, aggregate.ClassicalAudit(), {}, aggregate.DEFAULT_CONFIG)
    metrics = pd.DataFrame(rows)
    selected = metrics.loc[metrics.method.eq("model") & metrics.metric.eq("vector_rmse_m")]
    assert sorted(selected.loc[selected.aggregation_level.eq("sequence"), "value"]) == [1.0, 3.0]
    assert selected.loc[selected.aggregation_level.eq("dataset"), "value"].item() == pytest.approx(np.sqrt(5))
    assert set(metrics.unit) == {"m"}


def test_external_auxiliary_aggregation_needs_only_gate_companion_and_frozen_src(tmp_path):
    root = fixture(tmp_path)
    auxiliary = tmp_path / "auxiliary" / "scripts"
    auxiliary.mkdir(parents=True)
    copied = auxiliary / SCRIPT.name
    shutil.copyfile(SCRIPT, copied)
    shutil.copyfile(PROJECT / "scripts/evaluate_reviewer_transfer.py", auxiliary / "evaluate_reviewer_transfer.py")
    code = (
        "import importlib.util,json,pathlib;"
        f"spec=importlib.util.spec_from_file_location('external_summary',{str(copied)!r});"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        f"result=module.aggregate_results(pathlib.Path({str(root)!r}),pathlib.Path({str(tmp_path / 'external_summary')!r}),allow_subset=True);"
        "print(json.dumps(result))"
    )
    completed = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, env={**os.environ, "PYTHONPATH": str(PROJECT / "src")}, capture_output=True, text=True, encoding="utf-8", check=True)
    assert json.loads(completed.stdout.strip().splitlines()[-1])["runs"] == 6
    assert not (auxiliary.parent / "src").exists()
    assert not (auxiliary.parent / "pyproject.toml").exists()


def _frozen_source_fixture(project: Path):
    package = project / "src" / "virtual_imu"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("PACKAGE_VERSION = 1\n", encoding="utf-8")
    (package / "metrics.py").write_text("METRIC_VERSION = 1\n", encoding="utf-8")
    # Windows-style source identities are intentional even when this test runs
    # on another host; executing relative identities must remain POSIX paths.
    hashes = {str(path).replace("/", "\\"): hashlib.sha256(path.read_bytes()).hexdigest() for path in package.glob("*.py")}
    return {"status": "complete", "smoke": False, "contract": {"source_sha256": hashes}}


@pytest.mark.parametrize("mutation", ["changed", "added", "removed"])
def test_portable_source_gate_accepts_healthy_windows_identity_and_checks_full_roster(tmp_path, mutation):
    aggregate = module()
    state = _frozen_source_fixture(tmp_path)
    healthy = aggregate._processing_sources(tmp_path, state)
    assert healthy["strict_roster_verified"] is True
    assert set(healthy["matched_trained_source_sha256"]) == {"src/virtual_imu/__init__.py", "src/virtual_imu/metrics.py"}
    assert healthy["matched_trained_source_sha256"] == healthy["executed_source_sha256"]
    package = tmp_path / "src" / "virtual_imu"
    if mutation == "changed":
        (package / "metrics.py").write_text("METRIC_VERSION = 2\n", encoding="utf-8")
    elif mutation == "added":
        (package / "new_metric.py").write_text("NEW_METRIC = 1\n", encoding="utf-8")
    else:
        (package / "metrics.py").unlink()
    with pytest.raises(ValueError, match=r"source|rosters"):
        aggregate._processing_sources(tmp_path, state)


def test_first_aggregate_rejects_mutated_frozen_source_before_prediction_access(tmp_path, monkeypatch):
    aggregate = module()
    project = tmp_path / "project"
    state = _frozen_source_fixture(project)
    root = tmp_path / "completed_core"
    root.mkdir()
    (root / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
    (project / "src/virtual_imu/metrics.py").write_text("METRIC_VERSION = 99\n", encoding="utf-8")
    monkeypatch.setattr(aggregate, "_transfer_api", lambda: SimpleNamespace(verify_pipeline=lambda *args, **kwargs: []))
    monkeypatch.setattr(aggregate.Path, "cwd", classmethod(lambda cls: project))
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("Predictions were accessed before frozen-source validation"))
    with pytest.raises(ValueError, match="Executed source differs"):
        aggregate.aggregate_results(root, tmp_path / "first_summary")
    assert not (tmp_path / "first_summary").exists()
    assert not list(tmp_path.glob(".first_summary.*.partial"))


def test_relative_rmse_ratio_and_improvement_percent_have_opposite_clear_signs():
    aggregate = module()
    rows = []
    for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2"):
        for method, family, value in (("identity", "identity", 2.0), ("model", "better", 1.0), ("model", "worse", 3.0)):
            rows.append({"task": "paired", "aggregation_level": "pair", "metric": metric, "protocol_id": "alimu_only", "training_scope": "alimu_only", "split": "test", "dataset_id": "alimu", "low_sensor": "BMX160", "high_sensor": "VN100", "method": method, "model_family": family, "variant": "classical" if method == "identity" else "tuned", "seed": None if method == "identity" else 20261010, "value": value})
    rows.append({**rows[-1], "metric": "gx_mean_error_radps", "value": -100.0})
    result = aggregate.relative_identity(pd.DataFrame(rows), 1e-12)
    assert set(result.metric) == {"gyro_vector_rmse_radps", "accel_vector_rmse_mps2"}
    for family, ratio, improvement in (("identity", 100.0, 0.0), ("better", 50.0, 50.0), ("worse", 150.0, -50.0)):
        selected = result.loc[result.model_family.eq(family)]
        assert len(selected) == 2
        assert selected.relative_rmse_percent.eq(ratio).all()
        assert selected.improvement_percent_vs_identity.eq(improvement).all()
