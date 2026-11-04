"""Actual CPU checkpoints, fixed-seed deduplication and latency resume integrity."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.models import make_model
from virtual_imu.training.reviewer_hpo import run_reviewer_hpo

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "benchmark_reviewer_latency.py"


def _module():
    spec = importlib.util.spec_from_file_location("reviewer_latency_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed_pipeline(tmp_path: Path) -> Path:
    source = tmp_path / "paired.yaml"
    source.write_text(yaml.safe_dump({"data": {"window_size": 80}, "model": {"type": "gru"}, "training": {}}), encoding="utf-8")
    config = {
        "seed": 20261024, "confirmation_seed": 20261025,
        "seeds": [20261010, 20261011, 20261012], "comparison": {"default_refits": True},
        "device": "cpu", "tasks": {"paired": str(source)},
        "implementation_files": [str(PROJECT / "src/virtual_imu/models/__init__.py"), str(PROJECT / "src/virtual_imu/models/gru.py")],
        "protocols": [{"id": "alimu_only", "task": "paired"}],
        "candidates": [{"id": "gru", "tasks": ["paired"], "model": {"type": "gru", "hidden_size": 4, "num_layers": 1, "dropout": 0.0}, "search_space": {"model.hidden_size": {"choice": [8]}}}],
        "search": {"trials": 2, "rung_epochs": [1], "rung_keep": [2], "common_space": {"training.lr": {"choice": [0.002]}}, "paired_space": {"data.window_size": {"choice": [160]}}},
    }

    def train(task: str, actual: dict, frozen: Path) -> Path:
        directory = Path(actual["output"]["runs_dir"]) / "fit"
        directory.mkdir(parents=True)
        torch.manual_seed(actual["seed"])
        model = make_model(actual["model"])
        score = 1 / actual["model"]["hidden_size"]
        (directory / "config.yaml").write_text(yaml.safe_dump(actual), encoding="utf-8")
        (directory / "provenance.json").write_text('{"fit_split":"train"}', encoding="utf-8")
        (directory / "metrics.csv").write_text(f"epoch,train_loss,val_loss,val_grouped_score\n1,{score},{score},{score}\n", encoding="utf-8")
        torch.save({"config": actual, "model_config": actual["model"], "model_state_dict": model.state_dict(), "best_epoch": 1, "best_metric": score, "checkpoint_metric": "val_grouped_score"}, directory / "best.ckpt")
        return directory

    def evaluate(task: str, directory: Path, actual: dict) -> dict:
        folder = directory / "evaluation"
        folder.mkdir()
        pd.DataFrame({"prediction": [0.1, 0.2]}).to_parquet(folder / "test_predictions.parquet", index=False)
        report = {"training_run": str(directory), "task": task, "reviewer_protocol": copy.deepcopy(actual["reviewer_protocol"]), "seed": actual["seed"], "test_rmse": 0.1}
        (folder / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    return run_reviewer_hpo(config, root=tmp_path, train_backend=train, evaluate_backend=evaluate)


def _run(module, root, out, **kwargs):
    return module.run_latency(root, out, device="cpu", warmup=1, iterations=2, repeats=2, expected_cells=1, expected_evaluations=6, **kwargs)


def test_actual_cpu_measurement_inventories_every_checkpoint_and_deduplicates_shapes(tmp_path):
    module = _module()
    root = _completed_pipeline(tmp_path)
    out = tmp_path / "latency"
    report = _run(module, root, out)
    assert report["status"] == "complete" and report["entries"] == 6
    assert report["unique_architectures"] == 2
    inventory = json.loads((out / "checkpoint_inventory.json").read_text(encoding="utf-8"))
    assert len(inventory["entries"]) == 6
    assert {entry["window_size"] for entry in inventory["entries"]} == {80, 160}
    assert len({entry["parameters"] for entry in inventory["entries"]}) == 2
    assert all(entry["checkpoint_state_schema_sha256"] and entry["frozen_config_sha256"] for entry in inventory["entries"])
    representatives = {entry["timing_representative_entry_id"] for entry in inventory["entries"]}
    assert {entry["seed"] for entry in inventory["entries"] if entry["entry_id"] in representatives} == {20261010}
    assert {record["window_size"] for record in report["records"]} == {80, 160}
    assert all(record["batch_size"] == 1 and record["wall"]["count"] == 4 for record in report["records"])
    assert all(record["wall"]["p50_ms"] >= 0 and record["wall"]["p95_ms"] >= record["wall"]["p50_ms"] and record["loop_throughput_windows_per_second"] > 0 for record in report["records"])
    assert all(record["cuda_event"] is None for record in report["records"])
    assert not report["matches_historical_default_budget"]
    assert len(pd.read_csv(out / "architecture_mapping.csv")) == 6
    assert len(pd.read_csv(out / "latency_raw_timings.csv")) == 8


def test_completed_resume_verifies_hashes_and_never_retimes(tmp_path, monkeypatch):
    module = _module()
    root = _completed_pipeline(tmp_path)
    out = tmp_path / "latency"
    first = _run(module, root, out)
    api = module._measurement_api()

    def forbidden_measurement(*args):
        raise AssertionError("Completed architecture was measured again")

    monkeypatch.setattr(api, "_benchmark_record", forbidden_measurement)
    monkeypatch.setattr(module, "_measurement_api", lambda: api)
    second = _run(module, root, out, resume=True)
    assert second == first
    path = next((out / "architectures").glob("*/measurements.json"))
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        _run(module, root, out, resume=True)


@pytest.mark.parametrize("artifact", ["evaluation/test_predictions.parquet", "best.ckpt"])
def test_mutated_core_prediction_or_checkpoint_prevents_timing(tmp_path, artifact):
    module = _module()
    root = _completed_pipeline(tmp_path)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    job = next(job for job in state["jobs"].values() if job["phase"] == "final_tuned")
    path = Path(job["run_dir"]) / artifact
    path.write_bytes(b"changed evaluated predictions")
    with pytest.raises(ValueError, match="SHA256"):
        _run(module, root, tmp_path / "latency")
    assert not (tmp_path / "latency").exists()


def test_partial_timing_is_preserved_and_cannot_be_implicitly_replaced(tmp_path, monkeypatch):
    module = _module()
    root = _completed_pipeline(tmp_path)
    out = tmp_path / "latency"
    api = module._measurement_api()

    def fail_measurement(*args):
        raise RuntimeError("interrupted actual timing")

    monkeypatch.setattr(api, "_benchmark_record", fail_measurement)
    monkeypatch.setattr(module, "_measurement_api", lambda: api)
    with pytest.raises(RuntimeError, match="interrupted actual timing"):
        _run(module, root, out)
    before = (out / "latency_state.json").read_bytes()
    with pytest.raises(RuntimeError, match="Partial timing is preserved"):
        _run(module, root, out, resume=True)
    assert (out / "latency_state.json").read_bytes() == before
    assert not (out / "latency.json").exists()


def test_cli_requires_full_93_cell_core_and_indexed_cuda(tmp_path):
    module = _module()
    root = _completed_pipeline(tmp_path)
    with pytest.raises(ValueError, match="every selected cell"):
        module.main(["--pipeline-root", str(root), "--out", str(tmp_path / "latency"), "--device", "cpu"])
    with pytest.raises(SystemExit):
        module.main(["--pipeline-root", str(root), "--out", str(tmp_path / "latency"), "--device", "cuda"])


def test_unfinished_core_never_creates_latency_output(tmp_path):
    module = _module()
    root = _completed_pipeline(tmp_path)
    path = root / "pipeline.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["status"] = "searching"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="completed nonsmoke core"):
        _run(module, root, tmp_path / "latency")
    assert not (tmp_path / "latency").exists()


def test_external_auxiliary_location_uses_frozen_src_and_only_companion_script(tmp_path):
    root = _completed_pipeline(tmp_path)
    auxiliary = tmp_path / "external_auxiliary" / "scripts"
    auxiliary.mkdir(parents=True)
    copied = auxiliary / SCRIPT.name
    shutil.copyfile(SCRIPT, copied)
    shutil.copyfile(PROJECT / "scripts/benchmark_forward_latency.py", auxiliary / "benchmark_forward_latency.py")
    code = (
        "import importlib.util,json,pathlib;"
        f"spec=importlib.util.spec_from_file_location('aux_latency',{str(copied)!r});"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        f"audit=module.validate_reviewer_pipeline(pathlib.Path({str(root)!r}),expected_cells=1,expected_evaluations=6);"
        "api=module._measurement_api();"
        "print(json.dumps({'entries':len(audit['entries']),'measurement_script':api.__file__}))"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, env={**os.environ, "PYTHONPATH": str(PROJECT / "src")}, capture_output=True, text=True, encoding="utf-8", check=True)
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["entries"] == 6
    assert Path(report["measurement_script"]).parent == auxiliary
    assert not (auxiliary.parent / "src").exists()
    assert not (auxiliary.parent / "pyproject.toml").exists()
