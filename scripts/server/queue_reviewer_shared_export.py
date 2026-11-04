"""Wait read-only for the shared-classical fourth project, then export once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

PROFILE = "reviewer_shared_classical_v1"
_old_path = Path(__file__).with_name("queue_reviewer_export.py")
_old_bytes = _old_path.read_bytes()
if (
    hashlib.sha256(_old_bytes).hexdigest()
    != "1def14d21dfa994773b22a1482ada0d37dda339f9ee60a93478640750a4eeec8"
):
    raise ValueError("Frozen original export-waiter source checksum differs")
_spec = importlib.util.spec_from_file_location("shared_export_original_waiter", _old_path)
old = importlib.util.module_from_spec(_spec)
exec(compile(_old_bytes, str(_old_path), "exec"), old.__dict__)


def _terminal(text, marker):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines and lines[-1] == marker)


def ready(core, analysis, shared, workflow, analysis_console, shared_console, pipeline):
    if shared.get("status") == "failed":
        raise RuntimeError(
            "Shared-classical prerequisite failed; preserve evidence without exporting"
        )
    prior_ready = old.ready(core, analysis, workflow, analysis_console, pipeline)
    if not prior_ready or shared.get("status") != "complete":
        return False
    if shared.get("analysis_profile") != PROFILE or shared.get("contract", {}).get(
        "pipeline"
    ) != str(pipeline):
        raise ValueError("Shared workflow belongs to a different profile or exact core run")
    if set(shared.get("stages", {})) != {
        "fit_select",
        "signal_evaluation",
        "downstream",
        "cpu_latency",
    } or any(stage.get("status") != "complete" for stage in shared["stages"].values()):
        raise ValueError("Shared workflow completion does not cover all four stages")
    return _terminal(analysis_console, "REVIEWER_FINAL_ANALYSIS_PASSED") and _terminal(
        shared_console, "REVIEWER_SHARED_CONTROLS_PASSED"
    )


def verify_tools(root, expected_manifest_sha256, config, expected_config_sha256=None):
    manifest_path = root / "aux_manifest.json"
    if old.sha(manifest_path) != expected_manifest_sha256:
        raise ValueError("Pinned export-tools source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Export-tools source manifest is empty")
    required = {
        "scripts/server/" + name
        for name in (
            "queue_reviewer_shared_export.py",
            "export_reviewer_shared_results.py",
            "queue_reviewer_export.py",
            "export_reviewer_results.py",
        )
    }
    if not required <= set(manifest["files"]) or manifest["files"].get(
        "scripts/server/queue_reviewer_shared_export.py"
    ) != old.sha(Path(__file__)):
        raise ValueError("Export-tools source roster omits or changes executing queue source")
    for name, expected in manifest["files"].items():
        parts = PurePosixPath(name).parts
        if (
            not parts
            or name.startswith("/")
            or "\\" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ValueError("Unsafe export-tools manifest path")
        path = root.joinpath(*parts)
        for parent in (path, *path.parents):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Symlink/reparse export tool or ancestor is forbidden")
        if not path.is_file() or old.sha(path) != expected:
            raise ValueError(f"Export-tool source checksum differs: {name}")
    if (
        not config.is_relative_to(root)
        or config.relative_to(root).as_posix() not in manifest["files"]
    ):
        raise ValueError("Export config is not part of the pinned tools manifest")
    if expected_config_sha256 and old.sha(config) != expected_config_sha256:
        raise ValueError("Export queue configuration changed")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "pipeline-root",
        "aux1-root",
        "aux2-root",
        "aux3-root",
        "analysis-state",
        "shared-state",
        "tools-root",
        "config",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tools-manifest-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, Path(os.path.abspath(value)))
    project, pipeline, tools, out = args.project_root, args.pipeline_root, args.tools_root, args.out
    print(f"READS: exact core and final/shared workflow metadata for {pipeline}", flush=True)
    print(f"WRITES: isolated queue state/logs and four-project export under {out}", flush=True)
    print(
        "SKIPS: training changes, precompletion export, numerical TEST reads while waiting, network and authentication",
        flush=True,
    )
    verify_tools(tools, args.tools_manifest_sha256, args.config)
    manifest_path = tools / "aux_manifest.json"
    contract = {
        "analysis_profile": PROFILE,
        **{key: str(value) for key, value in vars(args).items() if isinstance(value, Path)},
        "project": str(project),
        "pipeline": str(pipeline),
        "tools": str(tools),
        "tools_manifest_sha256": args.tools_manifest_sha256,
        "config_sha256": old.sha(args.config),
    }
    state_path = tools / "export_queue_state.json"
    if state_path.exists():
        if not args.resume:
            raise FileExistsError("Existing export queue requires explicit exact-contract --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("contract") != contract:
            raise ValueError("Export queue resume contract changed")
        if state.get("status") not in {"waiting_for_shared_controls", "complete"}:
            raise RuntimeError("Interrupted/failed export is preserved; no automatic fresh attempt")
        if state.get("status") == "complete" and not out.exists():
            raise ValueError("Previously completed export is missing; no automatic replacement")
    else:
        if out.exists():
            raise FileExistsError("Unowned existing export is never silently adopted")
        state = {
            "contract": contract,
            "status": "waiting_for_shared_controls",
            "created_utc": datetime.now(UTC).isoformat(),
        }
        old.write_json(state_path, state)

    def read(path):
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"status": "not_started"}
        )

    def text(path):
        return path.read_text(encoding="utf-8") if path.exists() else ""

    try:
        while True:
            verify_tools(tools, args.tools_manifest_sha256, args.config, contract["config_sha256"])
            core, analysis, shared = (
                read(path)
                for path in (pipeline / "pipeline.json", args.analysis_state, args.shared_state)
            )
            if ready(
                core,
                analysis,
                shared,
                text(project / "runs/reviewer_workflow_status.log"),
                text(args.aux2_root / "final_analysis_console.log"),
                text(args.aux3_root / "shared_controls_console.log"),
                pipeline,
            ):
                break
            if state.get("status") == "complete":
                raise ValueError("Previously complete export prerequisites are no longer terminal")
            time.sleep(30)
        core_sha = old.sha(pipeline / "pipeline.json")
        if (
            any(value.get("core_pipeline_sha256") != core_sha for value in (analysis, shared))
            or state.get("core_pipeline_sha256", core_sha) != core_sha
        ):
            raise ValueError("Completed workflow pipeline fingerprints differ")
        state.update(
            status="exporting",
            core_pipeline_sha256=core_sha,
            started_utc=state.get("started_utc", datetime.now(UTC).isoformat()),
        )
        old.write_json(state_path, state)
        command = [
            str(project / ".venv/bin/python"),
            "-B",
            str(tools / "scripts/server/export_reviewer_shared_results.py"),
        ]
        for key in (
            "project_root",
            "pipeline_root",
            "aux1_root",
            "aux2_root",
            "aux3_root",
            "config",
            "out",
        ):
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        if out.exists():
            command.append("--resume")
        with (tools / "export_console.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
        transport = out / "export_manifest.json"
        result = read(transport)
        if result.get("status") != "complete" or result.get("analysis_profile") != PROFILE:
            raise ValueError("Exporter returned without a complete four-project transport manifest")
        verify_tools(tools, args.tools_manifest_sha256, args.config, contract["config_sha256"])
        if old.sha(manifest_path) != contract["tools_manifest_sha256"]:
            raise ValueError("Export source manifest changed during export")
        state.update(
            status="complete",
            export_manifest_sha256=old.sha(transport),
            completed_utc=state.get("completed_utc", datetime.now(UTC).isoformat()),
        )
        old.write_json(state_path, state)
        print("REVIEWER_SHARED_EXPORT_QUEUE_PASSED", flush=True)
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        old.write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
