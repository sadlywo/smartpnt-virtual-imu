"""Serialize final accuracy aggregation and classical CPU timing after aux1.

The preceding transfer/downstream/neural timing stages must finish and pass
their saved checksum inventories before this process reads final predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def helpers():
    path = Path(__file__).with_name("run_reviewer_postprocessing.py")
    spec = importlib.util.spec_from_file_location("reviewer_queue_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preceding_ready(state, expected_manifest_sha256):
    if state.get("status") == "failed":
        raise RuntimeError("Preceding auxiliary workflow failed; final analysis is blocked")
    if state.get("status") != "complete":
        return False
    if state["contract"]["aux_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Preceding auxiliary source identity changed")
    expected = {"transfer", "downstream", "latency_gpu", "latency_cpu"}
    if set(state["stages"]) != expected or any(
        value.get("status") != "complete" for value in state["stages"].values()
    ):
        raise ValueError("Preceding completion marker lacks all four finished stages")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--aux-root", type=Path, required=True)
    parser.add_argument("--preceding-state", type=Path, required=True)
    parser.add_argument("--preceding-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    helper = helpers()
    project, pipeline, aux, out = [
        path.resolve()
        for path in (args.project_root, args.pipeline_root, args.aux_root, args.output_root)
    ]
    print(
        f"READS: exact completed core {pipeline} and preceding state {args.preceding_state}",
        flush=True,
    )
    print(
        f"WRITES: independent final aggregation and classical CPU timings under {out}", flush=True
    )
    print(
        "SKIPS: training, candidate changes, concurrent neural/classical timing, test selection",
        flush=True,
    )
    manifest_path = aux / "aux_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = {
        "project": str(project),
        "pipeline": str(pipeline),
        "aux": str(aux),
        "aux_manifest_sha256": helper.sha(manifest_path),
        "preceding_state": str(args.preceding_state.resolve()),
        "preceding_manifest_sha256": args.preceding_manifest_sha256,
    }

    def verify_sources():
        if helper.sha(manifest_path) != contract["aux_manifest_sha256"]:
            raise ValueError("Final auxiliary manifest changed")
        for name, expected in manifest["files"].items():
            path = (aux / name).resolve()
            if not path.is_relative_to(aux) or not path.is_file() or helper.sha(path) != expected:
                raise ValueError(f"Final auxiliary source checksum differs: {name}")

    state_path = out / "final_analysis_state.json"
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing final analysis requires explicit --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Final analysis resume contract changed")
    else:
        out.mkdir(parents=True)
        state = {"contract": contract, "status": "waiting_for_preceding_auxiliary", "stages": {}}
        helper.write_json(state_path, state)
    verify_sources()
    try:
        while True:
            core = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
            workflow_file = project / "runs/reviewer_workflow_status.log"
            workflow = workflow_file.read_text(encoding="utf-8") if workflow_file.exists() else ""
            core_ok = helper.core_ready(core, workflow)
            prior = (
                json.loads(args.preceding_state.read_text(encoding="utf-8"))
                if args.preceding_state.exists()
                else {"status": "not_started"}
            )
            if preceding_ready(prior, args.preceding_manifest_sha256) and core_ok:
                break
            time.sleep(30)
        preceding_root = args.preceding_state.parent
        prior_aux = Path(prior["contract"]["aux"])
        if helper.sha(prior_aux / "aux_manifest.json") != args.preceding_manifest_sha256:
            raise ValueError("Preceding source manifest checksum changed")
        for name, stage in prior["stages"].items():
            if helper.inventory(preceding_root / name) != stage["artifact_sha256"]:
                raise ValueError(f"Preceding output checksum differs: {name}")
        core_sha = helper.sha(pipeline / "pipeline.json")
        if (
            core_sha != prior["core_pipeline_sha256"]
            or state.get("core_pipeline_sha256", core_sha) != core_sha
        ):
            raise ValueError(
                "Final and preceding analysis refer to different completed core states"
            )
        state.update(status="running", core_pipeline_sha256=core_sha)
        helper.write_json(state_path, state)
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=os.pathsep.join([str(aux / "deps"), str(project / "src")]),
            MPLBACKEND="Agg",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        for name, script, config in (
            ("accuracy_tables", "aggregate_reviewer_results.py", "reviewer_result_summary.yaml"),
            (
                "classical_latency",
                "benchmark_reviewer_classical_latency.py",
                "reviewer_classical_latency.yaml",
            ),
        ):
            verify_sources()
            destination = out / name
            previous = state["stages"].get(name, {})
            if previous.get("status") == "complete":
                if helper.inventory(destination) != previous["artifact_sha256"]:
                    raise ValueError(f"Completed final analysis artifacts changed: {name}")
                print(f"SKIPS: SHA-verified completed final stage {name}", flush=True)
                continue
            command = [
                str(project / ".venv/bin/python"),
                "-B",
                str(aux / "scripts" / script),
                "--pipeline-root",
                str(pipeline),
                "--config",
                str(aux / "configs" / config),
                "--out",
                str(destination),
            ]
            if destination.exists():
                command.append("--resume")
            state["stages"][name] = {
                "status": "running",
                "command": command,
                "started_utc": datetime.now(UTC).isoformat(),
            }
            helper.write_json(state_path, state)
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
                artifact_sha256=helper.inventory(destination),
                completed_utc=datetime.now(UTC).isoformat(),
            )
            helper.write_json(state_path, state)
        state.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
        helper.write_json(state_path, state)
        print("REVIEWER_FINAL_ANALYSIS_PASSED", flush=True)
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed_utc=datetime.now(UTC).isoformat(),
        )
        helper.write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
