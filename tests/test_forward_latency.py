"""Latency statistics, exact run identity mapping, and a real CPU checkpoint."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from virtual_imu.models import make_model

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_forward_latency.py"


def _module():
    spec = importlib.util.spec_from_file_location("forward_latency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_latency_statistics_use_linear_percentiles_and_sample_sd():
    result = _module().latency_statistics([1.0, 2.0, 3.0, 4.0])
    assert result["mean_ms"] == 2.5
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(3.85)
    assert result["sample_sd_ms"] == pytest.approx((5 / 3) ** 0.5)
    assert result["count"] == 4
    with pytest.raises(ValueError):
        _module().latency_statistics([float("nan")])


def test_pipeline_mapping_uses_exact_current_and_previous_pipeline_prefixes(tmp_path):
    module = _module()
    root = tmp_path / "new"
    old = tmp_path / "old"
    root.mkdir()
    old.mkdir()
    entries = [
        {
            "task": "paired",
            "candidate_id": "new",
            "seed": 1,
            "status": "evaluated",
            "config": "/server/new/configs/paired.yaml",
            "run_dir": "/server/new/experiments/run1",
        },
        {
            "task": "paired",
            "candidate_id": "old",
            "seed": 1,
            "status": "evaluated",
            "reused": True,
            "config": "/server/old/configs/paired.yaml",
            "run_dir": "/server/old/experiments/run2",
            "source_pipeline": "/server/old/pipeline.json",
        },
    ]
    for folder, name in [(root, "run1"), (old, "run2")]:
        (folder / "experiments" / name).mkdir(parents=True)
        (folder / "experiments" / name / "best.ckpt").touch()
    (root / "pipeline.json").write_text(json.dumps({"status": "complete", "runs": entries}))
    records = module.discover_pipeline(root, old)
    assert Path(records[0]["checkpoint"]) == root / "experiments/run1/best.ckpt"
    assert Path(records[1]["checkpoint"]) == old / "experiments/run2/best.ckpt"
    entries[0]["run_dir"] = "/other/experiments/run1"
    (root / "pipeline.json").write_text(json.dumps({"status": "complete", "runs": entries}))
    with pytest.raises(ValueError, match="prefix"):
        module.discover_pipeline(root, old)


def test_cpu_latency_cli_uses_real_checkpoint_and_preserves_artifacts(tmp_path):
    module = _module()
    config = {
        "seed": 9,
        "data": {"window_size": 8},
        "model": {
            "type": "gru",
            "in_channels": 6,
            "out_channels": 3,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
        },
    }
    model = make_model(config["model"])
    checkpoint = tmp_path / "best.ckpt"
    torch.save(
        {"config": config, "model_config": config["model"], "model_state_dict": model.state_dict()},
        checkpoint,
    )
    paths = tmp_path / "selected.json"
    paths.write_text(json.dumps([["navigation", "gru", 9, str(checkpoint)]]))
    out = tmp_path / "runs" / "latency"
    argv = [
        "--checkpoint-paths",
        str(paths),
        "--out",
        str(out),
        "--device",
        "cpu",
        "--warmup",
        "1",
        "--iterations",
        "2",
        "--repeats",
        "2",
        "--batch-sizes",
        "1",
    ]
    assert module.main(argv) == 0
    report = json.loads((out / "latency.json").read_text(encoding="utf-8"))
    record = report["records"][0]
    assert report["status"] == "complete"
    assert record["parameters"] == sum(p.numel() for p in model.parameters())
    assert record["parameter_bytes"] == sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    assert record["output_shape"] == [1, 3, 8]
    assert record["config_snapshot"] == config
    assert record["wall"]["count"] == 4
    assert record["cuda_event"] is None
    assert len(record["raw_wall_ms"]) == 2
    assert all(len(repeat) == 2 for repeat in record["raw_wall_ms"])
    assert record["loop_throughput_windows_per_second"] > 0
    assert (out / "latency_per_seed.csv").is_file()
    assert len((out / "latency_raw_timings.csv").read_text(encoding="utf-8").splitlines()) == 5
    assert (out / "latency_seed_summary.csv").is_file()
    before = (out / "latency.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.main(argv)
    assert (out / "latency.json").read_bytes() == before


def test_checkpoint_seed_mismatch_fails_without_fabricating_measurement(tmp_path):
    module = _module()
    config = {
        "seed": 9,
        "data": {"window_size": 8},
        "model": {
            "type": "gru",
            "in_channels": 6,
            "out_channels": 3,
            "hidden_size": 4,
            "num_layers": 1,
        },
    }
    checkpoint = tmp_path / "best.ckpt"
    torch.save(
        {
            "config": config,
            "model_config": config["model"],
            "model_state_dict": make_model(config["model"]).state_dict(),
        },
        checkpoint,
    )
    paths = tmp_path / "selected.json"
    paths.write_text(
        json.dumps(
            [
                {
                    "task": "navigation",
                    "candidate_id": "gru",
                    "seed": 10,
                    "checkpoint": str(checkpoint),
                }
            ]
        )
    )
    out = tmp_path / "runs" / "bad_identity"
    with pytest.raises(ValueError, match="identity"):
        module.main(
            [
                "--checkpoint-paths",
                str(paths),
                "--out",
                str(out),
                "--device",
                "cpu",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--repeats",
                "1",
            ]
        )
    assert not (out / "latency.json").exists()
    assert not list((out / "records").iterdir())
    assert json.loads((out / "failure.json").read_text(encoding="utf-8"))["completed_records"] == 0
