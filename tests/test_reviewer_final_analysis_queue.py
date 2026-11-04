"""Final analysis is serialized after the complete, verified timing workflow."""

import importlib.util
import json
from pathlib import Path

import pytest


def coordinator():
    path = Path(__file__).resolve().parents[1] / "scripts/server/run_reviewer_final_analysis.py"
    spec = importlib.util.spec_from_file_location("reviewer_final_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preceding_stage_gate_rejects_failed_partial_and_wrong_source():
    module = coordinator()
    assert not module.preceding_ready({"status": "running"}, "expected")
    with pytest.raises(RuntimeError, match="failed"):
        module.preceding_ready({"status": "failed"}, "expected")
    with pytest.raises(ValueError, match="identity changed"):
        module.preceding_ready(
            {"status": "complete", "contract": {"aux_manifest_sha256": "other"}}, "expected"
        )
    with pytest.raises(ValueError, match="four finished"):
        module.preceding_ready(
            {"status": "complete", "contract": {"aux_manifest_sha256": "expected"}, "stages": {}},
            "expected",
        )


def test_serialized_commands_and_hash_verified_resume(tmp_path, monkeypatch):
    module = coordinator()
    helper = module.helpers()
    monkeypatch.setattr(helper, "core_ready", lambda core, workflow: True)
    monkeypatch.setattr(module, "helpers", lambda: helper)
    project, aux, prior_aux, output = [
        tmp_path / name for name in ("project", "aux", "prior_aux", "output")
    ]
    pipeline = project / "runs/core"
    pipeline.mkdir(parents=True)
    (pipeline / "pipeline.json").write_text('{"status":"complete"}', encoding="utf-8")
    for directory in (aux, prior_aux):
        directory.mkdir()
        (directory / "aux_manifest.json").write_text('{"files":{}}', encoding="utf-8")
    prior_root = prior_aux / "runs/prior"
    prior_root.mkdir(parents=True)
    stages = {}
    for name in ("transfer", "downstream", "latency_gpu", "latency_cpu"):
        destination = prior_root / name
        destination.mkdir()
        (destination / "metrics.json").write_text("{}", encoding="utf-8")
        stages[name] = {"status": "complete", "artifact_sha256": helper.inventory(destination)}
    manifest_sha = helper.sha(prior_aux / "aux_manifest.json")
    prior_state = prior_root / "postprocessing_state.json"
    prior_state.write_text(
        json.dumps(
            {
                "status": "complete",
                "contract": {"aux": str(prior_aux), "aux_manifest_sha256": manifest_sha},
                "core_pipeline_sha256": helper.sha(pipeline / "pipeline.json"),
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        assert str(aux / "deps") in kwargs["env"]["PYTHONPATH"]
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        destination = Path(command[command.index("--out") + 1])
        destination.mkdir()
        (destination / "metrics.json").write_text("{}", encoding="utf-8")
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    argv = [
        "--project-root",
        str(project),
        "--pipeline-root",
        str(pipeline),
        "--aux-root",
        str(aux),
        "--preceding-state",
        str(prior_state),
        "--preceding-manifest-sha256",
        manifest_sha,
        "--output-root",
        str(output),
    ]
    module.main(argv)
    assert len(calls) == 2
    module.main([*argv, "--resume"])
    assert len(calls) == 2
    (prior_root / "latency_gpu/metrics.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Preceding output checksum"):
        module.main([*argv, "--resume"])
