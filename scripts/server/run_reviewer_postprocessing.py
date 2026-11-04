"""Wait for the exact frozen reviewer workflow, then run auxiliary evaluations.

This coordinator never trains or alters the core checkout. A failed stage stays
failed; restarting requires an explicit --resume and preserves existing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def core_ready(pipeline, workflow):
    """Require every identity and the parent's successful terminal marker."""
    if pipeline.get("status") == "failed":
        raise RuntimeError("Core pipeline failed; postprocessing is blocked")
    exits = [line for line in workflow.splitlines() if line.startswith("WORKFLOW_EXIT=")]
    if exits and exits[-1] != "WORKFLOW_EXIT=0":
        raise RuntimeError(f"Core workflow did not succeed: {exits[-1]}")
    if pipeline.get("status") != "complete":
        return False
    cells = pipeline["cells"]
    jobs = [j for j in pipeline["jobs"].values() if j["phase"].startswith("final_")]
    identities = [(j["cell"], j["phase"], j["seed"]) for j in jobs]
    expected = {
        (cell, "final_" + variant, seed)
        for cell in cells
        for variant in ("default", "tuned")
        for seed in (20261010, 20261011, 20261012)
    }
    if (
        pipeline.get("smoke")
        or len(cells) != 93
        or any("selected" not in cell for cell in cells.values())
        or len(jobs) != 558
        or len(set(identities)) != 558
        or set(identities) != expected
        or any(j["status"] != "complete" or j.get("evaluation_status") != "complete" for j in jobs)
    ):
        raise ValueError(
            "Core complete marker disagrees with the 93-cell/558-final identity matrix"
        )
    return bool(
        exits and exits[-1] == "WORKFLOW_EXIT=0" and "REVIEWER_CORE_PASSED" in workflow.splitlines()
    )


def inventory(directory):
    return {
        path.relative_to(directory).as_posix(): sha(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    project, pipeline, aux, out = [
        p.resolve()
        for p in (args.project_root, args.pipeline_root, args.aux_root, args.output_root)
    ]
    print(
        f"READS: exact pipeline {pipeline}, workflow terminal status, auxiliary SHA inventory",
        flush=True,
    )
    print(f"WRITES: postprocessing state/logs and independent results under {out}", flush=True)
    print(
        "SKIPS: training, source edits, test-driven selection, authentication changes", flush=True
    )
    manifest_path = aux / "aux_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def verify_sources():
        if sha(manifest_path) != contract["aux_manifest_sha256"]:
            raise ValueError("Auxiliary manifest changed")
        for name, expected in manifest["files"].items():
            path = (aux / name).resolve()
            if not path.is_relative_to(aux) or not path.is_file() or sha(path) != expected:
                raise ValueError(f"Auxiliary source checksum differs: {name}")

    contract = {
        "project": str(project),
        "pipeline": str(pipeline),
        "aux": str(aux),
        "aux_manifest_sha256": sha(manifest_path),
    }
    state_path = out / "postprocessing_state.json"
    if out.exists():
        if not args.resume:
            raise FileExistsError("Postprocessing output exists; explicit --resume required")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Postprocessing resume contract changed")
    else:
        out.mkdir(parents=True)
        state = {"contract": contract, "status": "waiting_for_core", "stages": {}}
        write_json(state_path, state)
    verify_sources()
    try:
        last_status = None
        while True:
            core = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
            workflow_path = project / "runs/reviewer_workflow_status.log"
            workflow = workflow_path.read_text(encoding="utf-8") if workflow_path.exists() else ""
            if core_ready(core, workflow):
                break
            observed = (
                core.get("status"),
                core.get("active_job"),
                tuple(sorted(core.get("counts", {}).items())),
            )
            if observed != last_status:
                print(
                    f"WAITING: {core.get('status')}; active={core.get('active_job')}; counts={core.get('counts')}",
                    flush=True,
                )
                last_status = observed
            time.sleep(30)
        verify_sources()
        core_sha = sha(pipeline / "pipeline.json")
        if state.get("core_pipeline_sha256", core_sha) != core_sha:
            raise ValueError("Completed core pipeline changed during auxiliary resume")
        state.update(status="running", core_pipeline_sha256=core_sha)
        write_json(state_path, state)
        python = str(project / ".venv/bin/python")
        script_root = aux / "scripts"
        common = ["--pipeline-root", str(pipeline)]
        stages = [
            (
                "transfer",
                "evaluate_reviewer_transfer.py",
                [
                    *common,
                    "--protocol-index",
                    str(project / "runs/reviewer_protocol_20261024/splits/protocol_index.json"),
                    "--device",
                    "cuda",
                ],
            ),
            (
                "downstream",
                "evaluate_reviewer_downstream.py",
                [*common, "--config", str(project / "configs/downstream_penn.yaml")],
            ),
            ("latency_gpu", "benchmark_reviewer_latency.py", [*common, "--device", "cuda:0"]),
            ("latency_cpu", "benchmark_reviewer_latency.py", [*common, "--device", "cpu"]),
        ]
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(project / "src"),
            MPLBACKEND="Agg",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        for name, script, flags in stages:
            verify_sources()
            destination = out / name
            previous = state["stages"].get(name, {})
            if previous.get("status") == "complete":
                if inventory(destination) != previous["artifact_sha256"]:
                    raise ValueError(f"Completed auxiliary artifacts changed: {name}")
                print(f"SKIPS: SHA-verified completed auxiliary stage {name}", flush=True)
                continue
            command = [python, "-B", str(script_root / script), *flags, "--out", str(destination)]
            if destination.exists():
                command.append("--resume")
            state["stages"][name] = {
                "status": "running",
                "command": command,
                "started_utc": datetime.now(UTC).isoformat(),
            }
            write_json(state_path, state)
            with (out / f"{name}_console.log").open("a", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=project,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            state["stages"][name].update(
                status="complete",
                completed_utc=datetime.now(UTC).isoformat(),
                artifact_sha256=inventory(destination),
            )
            write_json(state_path, state)
        state.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
        write_json(state_path, state)
        print("REVIEWER_AUXILIARY_PASSED", flush=True)
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
