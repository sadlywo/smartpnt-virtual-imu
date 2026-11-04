"""The server auxiliary queue must not mistake partial training for completion."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def coordinator():
    path = Path(__file__).resolve().parents[1] / "scripts/server/run_reviewer_postprocessing.py"
    spec = importlib.util.spec_from_file_location("reviewer_post_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_pipeline():
    cells = {f"protocol/model_{i}": {"selected": {}} for i in range(93)}
    jobs = {}
    for cell in cells:
        for variant in ("default", "tuned"):
            for seed in (20261010, 20261011, 20261012):
                job = {
                    "cell": cell,
                    "phase": "final_" + variant,
                    "seed": seed,
                    "status": "complete",
                    "evaluation_status": "complete",
                }
                jobs[f"{cell}_{variant}_{seed}"] = job
    return {"status": "complete", "smoke": False, "cells": cells, "jobs": jobs}


def test_requires_exact_full_identity_matrix_and_parent_success():
    module = coordinator()
    pipeline = full_pipeline()
    assert not module.core_ready(pipeline, "")
    assert not module.core_ready(pipeline, "REVIEWER_CORE_PASSED\n")
    assert not module.core_ready(pipeline, "WORKFLOW_EXIT=0\n")
    assert module.core_ready(pipeline, "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n")
    duplicate = copy.deepcopy(pipeline)
    first, second = list(duplicate["jobs"])[:2]
    duplicate["jobs"][second] = duplicate["jobs"][first].copy()
    with pytest.raises(ValueError, match="identity matrix"):
        module.core_ready(duplicate, "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n")


def test_running_or_failed_core_never_starts_postprocessing():
    module = coordinator()
    assert not module.core_ready({"status": "searching"}, "WORKFLOW_STARTED\n")
    with pytest.raises(RuntimeError, match="failed"):
        module.core_ready({"status": "failed"}, "")
    with pytest.raises(RuntimeError, match="did not succeed"):
        module.core_ready({"status": "searching"}, "WORKFLOW_EXIT=1\n")


def test_missing_evaluation_and_smoke_markers_are_rejected():
    module = coordinator()
    pipeline = full_pipeline()
    next(iter(pipeline["jobs"].values()))["evaluation_status"] = "running"
    with pytest.raises(ValueError, match="identity matrix"):
        module.core_ready(pipeline, "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n")


def test_queue_launches_only_after_success_and_verifies_completed_resume(tmp_path, monkeypatch):
    module = coordinator()
    project, aux, output = [tmp_path / name for name in ("project", "auxiliary", "result")]
    pipeline = project / "runs/exact_core"
    pipeline.mkdir(parents=True)
    (pipeline / "pipeline.json").write_text(json.dumps(full_pipeline()), encoding="utf-8")
    (project / "runs/reviewer_workflow_status.log").write_text(
        "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n", encoding="utf-8"
    )
    aux.mkdir()
    (aux / "aux_manifest.json").write_text('{"files": {}}', encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        assert kwargs["check"]
        assert kwargs["cwd"] == project
        assert kwargs["env"]["PYTHONPATH"] == str(project / "src")
        destination = Path(command[command.index("--out") + 1])
        destination.mkdir()
        (destination / "results.json").write_text('{"status":"complete"}', encoding="utf-8")
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    argv = [
        "--project-root",
        str(project),
        "--pipeline-root",
        str(pipeline),
        "--aux-root",
        str(aux),
        "--output-root",
        str(output),
    ]
    module.main(argv)
    assert len(calls) == 4
    assert json.loads((output / "postprocessing_state.json").read_text())["status"] == "complete"
    module.main([*argv, "--resume"])
    assert len(calls) == 4
    (output / "transfer/results.json").write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts changed"):
        module.main([*argv, "--resume"])
    pipeline = full_pipeline()
    pipeline["smoke"] = True
    with pytest.raises(ValueError, match="identity matrix"):
        module.core_ready(pipeline, "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n")
