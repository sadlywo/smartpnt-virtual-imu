"""Positive queue invocation and completed-analysis identity binding."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def setup(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/server/queue_reviewer_export.py"
    spec = importlib.util.spec_from_file_location("export_queue_integration", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project, aux1, aux2, tools, output = [
        tmp_path / name for name in ("core", "aux1", "aux2", "tools", "export")
    ]
    pipeline = project / "runs/exact"
    pipeline.mkdir(parents=True)
    aux1.mkdir()
    aux2.mkdir()
    tools.mkdir()
    (pipeline / "pipeline.json").write_text('{"status":"complete"}', encoding="utf-8")
    (project / "runs/reviewer_workflow_status.log").write_text(
        "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n", encoding="utf-8"
    )
    (aux2 / "final_analysis_console.log").write_text(
        "REVIEWER_FINAL_ANALYSIS_PASSED\n", encoding="utf-8"
    )
    config = tools / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    (tools / "aux_manifest.json").write_text(
        json.dumps({"files": {"config.yaml": module.sha(config)}}), encoding="utf-8"
    )
    analysis = aux2 / "final_analysis_state.json"
    analysis.write_text(
        json.dumps(
            {
                "status": "complete",
                "contract": {"pipeline": str(pipeline)},
                "core_pipeline_sha256": module.sha(pipeline / "pipeline.json"),
                "stages": {
                    "accuracy_tables": {"status": "complete"},
                    "classical_latency": {"status": "complete"},
                },
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--project-root",
        str(project),
        "--pipeline-root",
        str(pipeline),
        "--aux1-root",
        str(aux1),
        "--aux2-root",
        str(aux2),
        "--analysis-state",
        str(analysis),
        "--tools-root",
        str(tools),
        "--config",
        str(config),
        "--out",
        str(output),
    ]
    calls = []

    def exporter(command, **kwargs):
        assert kwargs["cwd"] == project and kwargs["check"]
        assert command[command.index("--pipeline-root") + 1] == str(pipeline)
        assert command[command.index("--aux2-root") + 1] == str(aux2)
        output.mkdir(exist_ok=True)
        (output / "export_manifest.json").write_text('{"status":"complete"}', encoding="utf-8")
        calls.append(command)

    monkeypatch.setattr(module.subprocess, "run", exporter)
    return module, argv, tools, pipeline, calls


def test_completed_chain_calls_exporter_with_exact_roots_and_explicit_resume(tmp_path, monkeypatch):
    module, argv, tools, _, calls = setup(tmp_path, monkeypatch)
    module.main(argv)
    assert len(calls) == 1 and "--resume" not in calls[0]
    saved = json.loads((tools / "export_queue_state.json").read_text())
    assert saved["status"] == "complete"
    assert len(saved["export_manifest_sha256"]) == hashlib.sha256().digest_size * 2
    module.main([*argv, "--resume"])
    assert len(calls) == 2 and calls[1][-1] == "--resume"


def test_changed_core_after_analysis_never_invokes_exporter(tmp_path, monkeypatch):
    module, argv, tools, pipeline, calls = setup(tmp_path, monkeypatch)
    (pipeline / "pipeline.json").write_text(
        '{"status":"complete","different":true}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different core states"):
        module.main(argv)
    assert calls == []
    assert json.loads((tools / "export_queue_state.json").read_text())["status"] == "failed"
