import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from virtual_imu.training import formal_pipeline as pipeline


@pytest.fixture(autouse=True)
def restore_torch_runtime_after_pipeline_test():
    flags = (torch.are_deterministic_algorithms_enabled(), torch.get_num_threads(),
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.set_num_threads(flags[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[2:4]
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[4:6]
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def _config(tmp_path):
    source = tmp_path / "paired.yaml"
    source.write_text(
        yaml.safe_dump({"experiment_name": "paired", "training": {"epochs": 60}, "output": {}})
    )
    return {
        "seeds": [11, 12, 13],
        "tasks": {"paired": str(source)},
        "device": "cpu",
        "runs_dir": str(tmp_path / "runs"),
    }


def test_pipeline_freezes_each_seed_and_isolates_smoke_from_formal(tmp_path, monkeypatch):
    seen = []

    def fake_train(config, config_path=None):
        seen.append(config)
        run = Path(config["output"]["runs_dir"]) / config["experiment_name"]
        run.mkdir(parents=True)
        return run

    monkeypatch.setattr(
        pipeline, "_train", lambda task, config, config_path: fake_train(config, config_path)
    )
    monkeypatch.setattr(pipeline, "_evaluate", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "aggregate_results", lambda *a: {})
    root = pipeline.run_formal_pipeline(_config(tmp_path), smoke=True)
    assert [item["seed"] for item in seen] == [11]
    assert seen[0]["training"]["epochs"] == 1
    assert "smoke" in seen[0]["experiment_name"]
    manifest = json.loads((root / "pipeline.json").read_text())
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 1
    assert (root / "configs" / "paired_seed11.yaml").is_file()
    seen.clear()
    root2 = pipeline.run_formal_pipeline(_config(tmp_path))
    assert root != root2
    assert [item["seed"] for item in seen] == [11, 12, 13]
    assert all(item["training"]["epochs"] == 60 for item in seen)


def test_pipeline_records_failure_before_raising_and_stops_later_seeds(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def fail(*args):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(pipeline, "_train", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        pipeline.run_formal_pipeline(config)
    manifests = list((tmp_path / "runs").glob("*/pipeline.json"))
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert manifest["active_seed"] == 11
    assert manifest["runs"] == []


@pytest.mark.parametrize("seeds", [[], [1, 1], [True]])
def test_pipeline_rejects_invalid_seed_matrix(tmp_path, seeds):
    config = _config(tmp_path)
    config["seeds"] = seeds
    with pytest.raises(ValueError, match="seeds"):
        pipeline.run_formal_pipeline(config)
