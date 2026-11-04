"""Wait for the exact completed final analysis, then invoke the audited exporter.

Waiting is read-only. No training process is started, stopped, or resumed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ready(core, analysis, workflow, analysis_console, pipeline):
    if core.get("status") == "failed" or analysis.get("status") == "failed":
        raise RuntimeError("A prerequisite workflow failed; result export remains blocked")
    exits = [line for line in workflow.splitlines() if line.startswith("WORKFLOW_EXIT=")]
    if exits and exits[-1] != "WORKFLOW_EXIT=0":
        raise RuntimeError("The core workflow exited unsuccessfully")
    if core.get("status") != "complete" or analysis.get("status") != "complete":
        return False
    if analysis.get("contract", {}).get("pipeline") != str(pipeline):
        raise ValueError("Final analysis belongs to a different exact core run")
    if set(analysis.get("stages", {})) != {"accuracy_tables", "classical_latency"} or any(
        stage.get("status") != "complete" for stage in analysis["stages"].values()
    ):
        raise ValueError("Final analysis completion does not cover both required stages")
    return bool(
        exits
        and exits[-1] == "WORKFLOW_EXIT=0"
        and "REVIEWER_CORE_PASSED" in workflow.splitlines()
        and "REVIEWER_FINAL_ANALYSIS_PASSED" in analysis_console.splitlines()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--aux1-root", required=True, type=Path)
    parser.add_argument("--aux2-root", required=True, type=Path)
    parser.add_argument("--analysis-state", required=True, type=Path)
    parser.add_argument("--tools-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    project, pipeline, tools_root, out = [
        p.resolve() for p in (args.project_root, args.pipeline_root, args.tools_root, args.out)
    ]
    print(f"READS: core {pipeline} and final analysis {args.analysis_state}", flush=True)
    print(
        f"WRITES: isolated export queue state/logs and audited multipart export {out}", flush=True
    )
    print(
        "SKIPS: training mutations, premature evaluation/export, network and credentials",
        flush=True,
    )
    manifest_path = tools_root / "aux_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = {
        "project": str(project),
        "pipeline": str(pipeline),
        "tools": str(tools_root),
        "out": str(out),
        "analysis_state": str(args.analysis_state.resolve()),
        "tools_manifest_sha256": sha(manifest_path),
        "config_sha256": sha(args.config),
    }

    def verify_tools():
        if (
            sha(manifest_path) != contract["tools_manifest_sha256"]
            or sha(args.config) != contract["config_sha256"]
        ):
            raise ValueError("Export queue source/config identity changed")
        for name, expected in manifest["files"].items():
            path = (tools_root / name).resolve()
            if not path.is_relative_to(tools_root) or not path.is_file() or sha(path) != expected:
                raise ValueError(f"Export tool source checksum differs: {name}")

    state_path = tools_root / "export_queue_state.json"
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("Existing export queue requires explicit --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Export queue resume identity changed")
    else:
        state = {"contract": contract, "status": "waiting_for_final_analysis"}
        write_json(state_path, state)
    verify_tools()
    try:
        while True:
            core = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
            analysis = (
                json.loads(args.analysis_state.read_text(encoding="utf-8"))
                if args.analysis_state.exists()
                else {"status": "not_started"}
            )
            workflow_file = project / "runs/reviewer_workflow_status.log"
            workflow = workflow_file.read_text(encoding="utf-8") if workflow_file.exists() else ""
            console_file = args.aux2_root / "final_analysis_console.log"
            console = console_file.read_text(encoding="utf-8") if console_file.exists() else ""
            if ready(core, analysis, workflow, console, pipeline):
                break
            time.sleep(30)
        verify_tools()
        core_sha = sha(pipeline / "pipeline.json")
        if (
            core_sha != analysis["core_pipeline_sha256"]
            or state.get("core_pipeline_sha256", core_sha) != core_sha
        ):
            raise ValueError("Export queue and completed analysis refer to different core states")
        state.update(
            status="exporting",
            core_pipeline_sha256=core_sha,
            started_utc=datetime.now(UTC).isoformat(),
        )
        write_json(state_path, state)
        command = [
            str(project / ".venv/bin/python"),
            "-B",
            str(tools_root / "scripts/server/export_reviewer_results.py"),
            "--project-root",
            str(project),
            "--pipeline-root",
            str(pipeline),
            "--aux1-root",
            str(args.aux1_root.resolve()),
            "--aux2-root",
            str(args.aux2_root.resolve()),
            "--config",
            str(args.config.resolve()),
            "--out",
            str(out),
        ]
        if out.exists():
            command.append("--resume")
        with (tools_root / "export_console.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        transport = out / "export_manifest.json"
        actual = json.loads(transport.read_text(encoding="utf-8"))
        if actual.get("status") != "complete":
            raise ValueError("Exporter returned without a complete transport manifest")
        state.update(
            status="complete",
            export_manifest_sha256=sha(transport),
            completed_utc=datetime.now(UTC).isoformat(),
        )
        write_json(state_path, state)
        print("REVIEWER_EXPORT_QUEUE_PASSED", flush=True)
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
