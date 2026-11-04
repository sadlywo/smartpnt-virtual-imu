"""Premature export and wrong-run completion markers must be rejected."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/server/queue_reviewer_export.py"
    spec = importlib.util.spec_from_file_location("reviewer_export_queue", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_queue_requires_parent_markers_and_both_finished_analysis_stages():
    api = module()
    pipeline = Path("/exact/pipeline")
    core = {"status": "complete"}
    analysis = {
        "status": "complete",
        "contract": {"pipeline": str(pipeline)},
        "stages": {
            "accuracy_tables": {"status": "complete"},
            "classical_latency": {"status": "complete"},
        },
    }
    workflow = "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n"
    assert not api.ready(core, analysis, workflow, "", pipeline)
    assert not api.ready(
        core, analysis, "REVIEWER_CORE_PASSED\n", "REVIEWER_FINAL_ANALYSIS_PASSED\n", pipeline
    )
    assert api.ready(core, analysis, workflow, "REVIEWER_FINAL_ANALYSIS_PASSED\n", pipeline)
    analysis["stages"].pop("classical_latency")
    with pytest.raises(ValueError, match="both required"):
        api.ready(core, analysis, workflow, "REVIEWER_FINAL_ANALYSIS_PASSED\n", pipeline)


def test_running_failed_and_other_run_are_never_export_ready():
    api = module()
    pipeline = Path("/exact/pipeline")
    assert not api.ready(
        {"status": "searching"}, {"status": "waiting_for_preceding_auxiliary"}, "", "", pipeline
    )
    with pytest.raises(RuntimeError, match="failed"):
        api.ready(
            {"status": "failed"}, {"status": "waiting_for_preceding_auxiliary"}, "", "", pipeline
        )
    with pytest.raises(RuntimeError, match="unsuccessfully"):
        api.ready({"status": "searching"}, {}, "WORKFLOW_EXIT=1\n", "", pipeline)
    with pytest.raises(ValueError, match="different exact core"):
        api.ready(
            {"status": "complete"},
            {"status": "complete", "contract": {"pipeline": "/another/pipeline"}},
            "",
            "",
            pipeline,
        )
