"""The export waiter must not bypass the fourth scientific prerequisite."""

import importlib.util
import json
from pathlib import Path

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / "scripts/server/queue_reviewer_shared_export.py"
    assert path.is_file(), "Shared-export waiter has not been implemented"
    spec = importlib.util.spec_from_file_location("shared_export_queue_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete(pipeline):
    core = {"status": "complete"}
    analysis = {
        "status": "complete",
        "contract": {"pipeline": str(pipeline)},
        "stages": {
            name: {"status": "complete"} for name in ("accuracy_tables", "classical_latency")
        },
    }
    shared = {
        "status": "complete",
        "analysis_profile": "reviewer_shared_classical_v1",
        "contract": {"pipeline": str(pipeline)},
        "stages": {
            name: {"status": "complete"}
            for name in ("fit_select", "signal_evaluation", "downstream", "cpu_latency")
        },
    }
    return core, analysis, shared


def test_shared_waiter_requires_its_own_terminal_workflow_and_exact_identity(tmp_path):
    module = api()
    pipeline = tmp_path / "exact"
    core, analysis, shared = complete(pipeline)
    workflow = "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n"
    analysis_log = "REVIEWER_FINAL_ANALYSIS_PASSED\n"
    shared_log = "REVIEWER_SHARED_CONTROLS_PASSED\n"
    assert not module.ready(
        core, analysis, {"status": "waiting_for_aux2"}, workflow, analysis_log, "", pipeline
    )
    assert not module.ready(core, analysis, shared, workflow, analysis_log, "", pipeline)
    assert module.ready(core, analysis, shared, workflow, analysis_log, shared_log, pipeline)
    shared["contract"]["pipeline"] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="different"):
        module.ready(core, analysis, shared, workflow, analysis_log, shared_log, pipeline)


def test_failure_or_incomplete_stage_is_preserved_instead_of_triggering_export(tmp_path):
    module = api()
    pipeline = tmp_path / "exact"
    core, analysis, shared = complete(pipeline)
    shared["status"] = "failed"
    with pytest.raises(RuntimeError, match="failed"):
        module.ready(core, analysis, shared, "", "", "", pipeline)
    shared["status"] = "complete"
    shared["stages"]["cpu_latency"]["status"] = "running"
    with pytest.raises(ValueError, match="stages"):
        module.ready(
            core,
            analysis,
            shared,
            "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n",
            "REVIEWER_FINAL_ANALYSIS_PASSED\n",
            "REVIEWER_SHARED_CONTROLS_PASSED\n",
            pipeline,
        )


def test_tool_manifest_cannot_omit_executing_source_or_adopt_mutated_config(tmp_path):
    module = api()
    config = tmp_path / "configs/export.yaml"
    config.parent.mkdir()
    config.write_text("{}\n")
    manifest = tmp_path / "aux_manifest.json"
    manifest.write_text(json.dumps({"files": {"configs/export.yaml": module.old.sha(config)}}))
    with pytest.raises(ValueError, match="source roster"):
        module.verify_tools(tmp_path, module.old.sha(manifest), config)
    source_root = Path(__file__).resolve().parents[1]
    roster = {}
    for name in (
        "queue_reviewer_shared_export.py",
        "export_reviewer_shared_results.py",
        "queue_reviewer_export.py",
        "export_reviewer_results.py",
    ):
        relative = "scripts/server/" + name
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source_root / relative).read_bytes())
        roster[relative] = module.old.sha(target)
    manifest.write_text(
        json.dumps({"files": {**roster, "configs/export.yaml": module.old.sha(config)}})
    )
    expected = module.old.sha(manifest)
    assert module.verify_tools(tmp_path, expected, config)["files"]
    config.write_text("[]\n")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_tools(tmp_path, expected, config)
