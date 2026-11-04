"""Queue tests use metadata/files created only below an explicit SYNTHETIC run."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "runs/reviewer_shared_runtime_validation_20261025/SYNTHETIC"


def module():
    path = ROOT / "scripts/server/run_reviewer_shared_controls.py"
    spec = importlib.util.spec_from_file_location("shared_queue_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def scenario(monkeypatch):
    tool = module()
    base = SYNTHETIC / uuid.uuid4().hex
    project = base / "core"
    pipeline = project / "runs" / tool.CORE_RUN_ID
    pipeline.mkdir(parents=True)
    source = project / "scripts/frozen.py"
    source.parent.mkdir()
    source.write_text("# SYNTHETIC frozen source\n", encoding="utf-8")
    source_map = {"scripts/frozen.py": tool.sha(source)}
    fingerprint = tool.bundle_fingerprint(source_map)
    monkeypatch.setattr(tool, "SOURCE_FINGERPRINT", fingerprint)
    write(
        project / "bundle_manifest.json",
        {
            "entries": [
                {"path": name, "sha256": checksum, "kind": "source"}
                for name, checksum in source_map.items()
            ],
            "source_fingerprint": fingerprint,
            "dataset_manifest_sha256": tool.DATASET_SHA256,
        },
    )
    cells, jobs = {}, {}
    for i in range(93):
        cell = f"protocol/cell{i}"
        selection = pipeline / "selections" / (cell.replace("/", "__") + ".json")
        write(selection, {"selected": "SYNTHETIC"})
        cells[cell] = {"selected": {}, "selection_sha256": tool.sha(selection)}
        for variant in ("default", "tuned"):
            for seed in tool.FINAL_SEEDS:
                key = f"{i}_{variant}_{seed}"
                jobs[key] = {
                    "id": key,
                    "cell": cell,
                    "phase": "final_" + variant,
                    "seed": seed,
                    "status": "complete",
                    "evaluation_status": "complete",
                }
    core = {
        "status": "complete",
        "smoke": False,
        "selection_uses_test": False,
        "final_seeds": list(tool.FINAL_SEEDS),
        "final_variants": ["default", "tuned"],
        "cells": cells,
        "jobs": jobs,
        "contract": {
            "source_sha256": source_map,
            "config": {"dataset_manifest_sha256": tool.DATASET_SHA256},
        },
    }
    write(pipeline / "pipeline.json", core)
    (pipeline / "matrix_config.yaml").write_text(
        "dataset_manifest_sha256: " + tool.DATASET_SHA256 + "\n", encoding="utf-8"
    )
    workflow = project / "runs/reviewer_workflow_status.log"
    workflow.write_text("REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n", encoding="utf-8")
    paths, manifests = {}, {}
    for label, stages in tool.PRIOR_STAGES.items():
        aux = base / label
        aux.mkdir()
        (aux / "source.py").write_text("# SYNTHETIC\n", encoding="utf-8")
        write(aux / "aux_manifest.json", {"files": {"source.py": tool.sha(aux / "source.py")}})
        manifests[label] = tool.sha(aux / "aux_manifest.json")
        paths[label] = aux / "run" / "state.json"
        write(
            paths[label],
            {
                "status": "complete",
                "core_pipeline_sha256": tool.sha(pipeline / "pipeline.json"),
                "contract": {
                    "project": str(project.resolve()),
                    "pipeline": str(pipeline.resolve()),
                    "aux": str(aux.resolve()),
                    "aux_manifest_sha256": manifests[label],
                },
                "stages": {name: {"status": "complete"} for name in stages},
            },
        )
    monkeypatch.setattr(tool, "AUX1_MANIFEST_SHA256", manifests["aux1"])
    monkeypatch.setattr(tool, "AUX2_MANIFEST_SHA256", manifests["aux2"])
    return tool, project, pipeline, paths, core


def gate(case):
    tool, project, pipeline, paths, _ = case
    return tool.metadata_gate(project, pipeline, paths["aux1"], paths["aux2"])


def test_exact_metadata_gate_waits_without_decoding_any_prediction(scenario):
    _tool, _, pipeline, _, core = scenario
    assert gate(scenario)["ready"]
    core["status"] = "running"
    write(pipeline / "pipeline.json", core)
    assert gate(scenario)["ready"] is False
    assert not list(pipeline.rglob("*prediction*"))


@pytest.mark.parametrize(
    "mutation",
    [
        "fake_counts",
        "duplicate_final",
        "wrong_source",
        "wrong_run",
        "missing_marker",
        "aux_failed",
        "wrong_aux_hash",
    ],
)
def test_gate_rejects_false_completion_and_wrong_identities(scenario, mutation):
    tool, project, pipeline, paths, core = scenario
    if mutation == "fake_counts":
        core["jobs"].pop(next(iter(core["jobs"])))
        core["counts"] = {"finals": 558}
        write(pipeline / "pipeline.json", core)
    elif mutation == "duplicate_final":
        values = list(core["jobs"].values())
        values[1]["seed"] = values[0]["seed"]
        write(pipeline / "pipeline.json", core)
    elif mutation == "wrong_source":
        (project / "scripts/frozen.py").write_text("mutated", encoding="utf-8")
    elif mutation == "wrong_run":
        with pytest.raises(ValueError, match="run"):
            tool.metadata_gate(
                project, pipeline.parent / "another_run", paths["aux1"], paths["aux2"]
            )
        return
    elif mutation == "missing_marker":
        (project / "runs/reviewer_workflow_status.log").write_text(
            "WORKFLOW_EXIT=0\n", encoding="utf-8"
        )
    else:
        prior = tool.read_json(paths["aux2"])
        if mutation == "aux_failed":
            prior["status"] = "failed"
        else:
            prior["contract"]["aux_manifest_sha256"] = "0" * 64
        write(paths["aux2"], prior)
    with pytest.raises((ValueError, RuntimeError)):
        gate(scenario)


@pytest.mark.parametrize("status", ["starting", "parent_closing", "running"])
def test_startup_parent_closing_are_not_terminal(scenario, status):
    tool, _, _, paths, _ = scenario
    prior = tool.read_json(paths["aux2"])
    prior["status"] = status
    write(paths["aux2"], prior)
    assert gate(scenario)["ready"] is False


def test_child_stage_dispatch_is_sequential_and_freeze_checked_before_signal(scenario):
    tool, project, pipeline, paths, _ = scenario
    commands = tool.stage_commands(
        project, pipeline, project, paths["aux1"], paths["aux2"], project / "newrun"
    )
    assert [name for name, _ in commands] == [
        "fit_select",
        "signal_evaluation",
        "downstream",
        "cpu_latency",
    ]
    for _, command in commands:
        assert command[command.index("--out") + 1].endswith("newrun")
        assert "--project-root" in command and "--pipeline-root" in command
    assert "--stage" in commands[0][1] and commands[0][1][-1] == "fit_select"
    assert commands[1][1][-1] == "signal_evaluation"
    assert "evaluate_reviewer_shared_downstream.py" in " ".join(commands[2][1])
    with pytest.raises((FileNotFoundError, ValueError)):
        tool.verify_global_freeze(project / "newrun")


def test_completed_stage_hash_mutation_rejected_and_partial_preserved(scenario):
    tool, project, _, _, _ = scenario
    out = project / "newrun"
    stage = out / "fit_select"
    stage.mkdir(parents=True)
    (stage / "value.txt").write_text("SYNTHETIC", encoding="utf-8")
    write(stage / "stage_summary.json", {"status": "complete", "stage": "fit_select", "tables": {}})
    write(
        stage / "artifact_manifest.json",
        {
            "root_kind": "aux3_run",
            "status": "complete",
            "files": {
                "fit_select/value.txt": tool.sha(stage / "value.txt"),
                "fit_select/stage_summary.json": tool.sha(stage / "stage_summary.json"),
            },
        },
    )
    assert tool.verify_stage(out, "fit_select")["status"] == "complete"
    (stage / "value.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        tool.verify_stage(out, "fit_select")
    partial = out / "signal_evaluation.partial.SYNTHETIC"
    partial.mkdir()
    (partial / "error.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial"):
        tool.reject_partial_restart(out, "signal_evaluation")
    assert (partial / "error.json").exists()


def test_public_cli_has_no_payload_subset_or_synthetic_bypass():
    options = {option for action in module().parser()._actions for option in action.option_strings}
    assert not options & {"--synthetic", "--subset", "--allow-subset", "--smoke", "--skip-gate"}


def test_real_terminal_log_requirement_is_enforced(scenario):
    tool, project, pipeline, paths, _ = scenario
    for label, marker, filename in (
        ("aux1", "REVIEWER_AUXILIARY_PASSED", "postprocessing_console.log"),
        ("aux2", "REVIEWER_FINAL_ANALYSIS_PASSED", "final_analysis_console.log"),
    ):
        aux = Path(tool.read_json(paths[label])["contract"]["aux"])
        (aux / filename).write_text(marker + "\n", encoding="utf-8")
    assert tool.metadata_gate(
        project, pipeline, paths["aux1"], paths["aux2"], require_aux_terminal=True
    )["ready"]
    aux = Path(tool.read_json(paths["aux2"])["contract"]["aux"])
    (aux / "final_analysis_console.log").write_text("parent_closing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        tool.metadata_gate(
            project, pipeline, paths["aux1"], paths["aux2"], require_aux_terminal=True
        )


def test_queue_runs_children_in_order_freezes_before_audit_and_resume_checks_source(
    scenario, monkeypatch
):
    from types import SimpleNamespace

    tool, project, pipeline, paths, _ = scenario
    aux = project.parent / "aux3"
    aux.mkdir()
    source = aux / "queue_source.py"
    source.write_text("# SYNTHETIC\n", encoding="utf-8")
    write(aux / "aux_manifest.json", {"files": {source.name: tool.sha(source)}})
    args = SimpleNamespace(
        project_root=project,
        pipeline_root=pipeline,
        aux_root=aux,
        aux1_state=paths["aux1"],
        preceding_state=paths["aux2"],
        out=aux / "run",
        aux_manifest_sha256=tool.sha(aux / "aux_manifest.json"),
        dependency_root=[],
        poll_seconds=1,
        resume=False,
    )
    frozen_gate = gate(scenario)
    monkeypatch.setattr(tool, "metadata_gate", lambda *args, **kwargs: frozen_gate)
    events = []

    def freeze(out):
        assert "fit_select" in events
        events.append("FREEZE")
        return "SYNTHETIC_selection_sha"

    monkeypatch.setattr(tool, "verify_global_freeze", freeze)

    def run_child(command, **kwargs):
        assert kwargs["shell"] is False and kwargs["cwd"] == project
        assert kwargs["env"]["PYTHONPATH"].split(tool.os.pathsep)[:2] == [
            str(project / "src"),
            str(aux / "scripts"),
        ]
        name = next(
            name
            for name, cmd in tool.stage_commands(
                project, pipeline, aux, paths["aux1"], paths["aux2"], args.out
            )
            if cmd == command
        )
        if name != "fit_select":
            assert "FREEZE" in events
        events.append(name)
        stage = args.out / name
        stage.mkdir()
        write(stage / "stage_summary.json", {"status": "complete", "stage": name, "counts": {}})
        write(
            stage / "artifact_manifest.json",
            {"files": {name + "/stage_summary.json": tool.sha(stage / "stage_summary.json")}},
        )

    monkeypatch.setattr(tool.subprocess, "run", run_child)
    monkeypatch.setattr(
        tool, "verify_stage", lambda out, name: tool.read_json(out / name / "stage_summary.json")
    )
    monkeypatch.setattr(
        tool,
        "validate_complete_artifacts",
        lambda *args, **kwargs: {
            "status": "complete",
            "counts": {"SYNTHETIC": 1},
            "global_selection_set_sha256": "SYNTHETIC_selection_sha",
        },
    )
    state = tool.run(args)
    assert state["status"] == "complete"
    assert [event for event in events if event != "FREEZE"] == list(tool.STAGES)
    before = (args.out / "shared_controls_state.json").read_bytes()
    args.resume = True
    source.write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source checksum"):
        tool.run(args)
    assert (args.out / "shared_controls_state.json").read_bytes() == before


def test_unsafe_and_case_colliding_source_paths_rejected(scenario):
    tool, project, *_ = scenario
    for member in (
        "../outside",
        "scripts//frozen.py",
        "scripts/FROZEN.py",
        "E:/outside",
        "scripts\\frozen.py",
    ):
        with pytest.raises(ValueError):
            tool.safe_path(project, member)
    with pytest.raises(ValueError, match="collide"):
        tool._members({"scripts/a.py": "a", "scripts/A.py": "b"})


def synthetic_saved_predictions(protocol):
    import numpy as np
    import pandas as pd

    from reviewer_shared_classical import AXES, HIGH, LOW, METHODS

    data = pd.DataFrame(np.zeros((880, 6)), columns=LOW)
    data[HIGH] = 0.0
    data["timestamp_s"] = np.arange(len(data)) / 40
    data["source_timestamp_s"] = data.timestamp_s + 100.0
    data["available_timestamp_s"] = data.timestamp_s + 0.1
    data["sequence_id"] = "SYNTHETIC_supported_stream"
    data["original_sequence_id"] = {
        "penn_loso_af": "af",
        "penn_loso_as": "as",
        "penn_loso_bf": "bf",
    }.get(protocol, "bs")
    data["dataset_id"], data["low_sensor"], data["high_sensor"] = "penncosyvio", "low", "high"
    data["calibration_id"] = "penncosyvio_published_body_extrinsics_v1"
    data["lever_arm_compensated"], data["sample_rate_hz"] = True, 40.0
    for method in METHODS:
        for axis in AXES:
            data[f"{method}_{axis}"] = 0.0
    return data


def complete_synthetic_run(*, materialize_header=False):
    """Create stored fixture arrays only: no solver, neural inference or real payload."""
    import sys

    import numpy as np
    import pandas as pd
    import yaml

    sys.path.insert(0, str(ROOT / "scripts"))
    from benchmark_reviewer_shared_classical_latency import DEFAULTS, array_sha, parameter_tables
    from evaluate_reviewer_shared_classical import metric_rows, source_contract
    from reviewer_shared_classical import (
        AXES,
        SEARCH,
        _state,
        digest,
        finish_stage,
        macro_breakdown,
        normalize_bank_pair,
        write_once,
    )

    tool = module()
    fixture_root = SYNTHETIC / uuid.uuid4().hex
    out = fixture_root / "aux3/runs/SYNTHETIC_shared_controls"
    out.mkdir(parents=True)
    pipeline = fixture_root / "core" / "runs" / tool.CORE_RUN_ID
    project = pipeline.parent.parent
    # These are inert source audit fixtures, never an import path or executable package.
    source_files = {}
    for name in (
        "eval/formal_evaluation.py",
        "eval/baselines.py",
        "eval/reviewer_evaluation.py",
        "io/training_split_manifest.py",
        "training/paired_validation.py",
        "eval/downstream_attitude.py",
    ):
        relative = "src/virtual_imu/" + name
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        source_files[relative] = tool.sha(destination)
    tool.SOURCE_FINGERPRINT = tool.bundle_fingerprint(source_files)
    dataset = project / "metadata/SYNTHETIC_dataset_manifest.json"
    write(dataset, {"fixture": "SYNTHETIC metadata only; no dataset payload"})
    tool.DATASET_SHA256 = tool.sha(dataset)
    bundle_path = project / "bundle_manifest.json"
    write(
        bundle_path,
        {
            "source_fingerprint": tool.SOURCE_FINGERPRINT,
            "dataset_manifest_sha256": tool.DATASET_SHA256,
            "entries": [
                {"path": name, "kind": "source", "sha256": checksum}
                for name, checksum in source_files.items()
            ],
        },
    )
    shared_config = yaml.safe_load(
        (ROOT / "configs/reviewer_shared_classical.yaml").read_text(encoding="utf-8")
    )
    shared_config.update(
        expected_source_fingerprint=tool.SOURCE_FINGERPRINT,
        expected_dataset_manifest_sha256=tool.DATASET_SHA256,
    )
    config_path = project / "metadata/SYNTHETIC_shared_config.yaml"
    config_path.write_text(yaml.safe_dump(shared_config), encoding="utf-8")
    manifests = {}
    for protocol in tool.PROTOCOLS:
        references = {"dataset_manifest": {"path": str(dataset), "sha256": tool.sha(dataset)}}
        for role in ("split_manifest", "strict_manifest", "assignment_manifest"):
            path = project / "metadata" / protocol / (role + ".json")
            write(path, {"fixture": "SYNTHETIC", "protocol": protocol, "role": role})
            references[role] = {"path": str(path), "sha256": tool.sha(path)}
        manifests[protocol] = references
    cells, jobs = {}, {}
    for protocol in tool.PROTOCOLS:
        for family in range(10):
            cell = f"{protocol}/SYNTHETIC_family_{family}"
            cells[cell] = {"selected": {}}
    for protocol in ("nav_a", "nav_b", "nav_c"):
        for family in range(11):
            cells[f"{protocol}/SYNTHETIC_family_{family}"] = {"selected": {}}
    for cell in cells:
        for variant in ("default", "tuned"):
            for seed in tool.FINAL_SEEDS:
                identifier = cell.replace("/", "__") + f"__{variant}__{seed}"
                jobs[identifier] = {
                    "id": identifier,
                    "cell": cell,
                    "phase": "final_" + variant,
                    "seed": seed,
                    "status": "complete",
                    "evaluation_status": "complete",
                }
    bank_saved, bank_paths, bank_references = {}, {}, []
    pair_identity = {"dataset_id": "penncosyvio", "low_sensor": "low", "high_sensor": "high"}
    for protocol in tool.PROTOCOLS:
        fixed = {
            "bias": [0.0] * 6,
            "affine": np.zeros((7, 6)).tolist(),
            "ridge_affine": np.zeros((7, 6)).tolist(),
            "system_id": {
                "fir_taps": 9,
                "fir_weights": np.zeros((54, 6)).tolist(),
                "fir_bias": [0.0] * 6,
                "arx_taps": 12,
                "arx_weights": np.zeros((78, 6)).tolist(),
                "arx_bias": [0.0] * 6,
                "alpha": 0.01,
            },
        }
        plain = {"weights": np.zeros((6, 6)).tolist(), "bias": [0.0] * 6, "alpha": 1e-6}
        pair = {
            "identity": pair_identity,
            "fixed": fixed,
            "ridge_tuned": plain,
            "fir_tuned": {**plain, "taps": 1, "include_delta": False},
            "arx_tuned": {
                **plain,
                "taps": 3,
                "include_delta": True,
                "weights": np.zeros((24, 6)).tolist(),
            },
        }
        for method, taps_grid in (
            ("ridge_tuned", [None]),
            ("fir_tuned", SEARCH["fir_taps"]),
            ("arx_tuned", SEARCH["arx_taps"]),
        ):
            pair[method] = {
                **pair[method],
                "validation_score": 0.0,
                "history": [
                    {"alpha": alpha, "taps": taps, "validation_score": 0.0}
                    for taps in taps_grid
                    for alpha in SEARCH["ridge_alpha"]
                ],
            }
        bank_saved[protocol] = {"pairs": [pair]}
        path = (
            pipeline
            / "experiments"
            / ("SYNTHETIC_" + protocol)
            / "evaluation/baseline_train_state.json"
        )
        write(path, bank_saved[protocol])
        prediction_hashes = {}
        for role in ("validation", "test"):
            prediction_path = path.parent / f"{role}_predictions.parquet"
            synthetic_saved_predictions(protocol).to_parquet(prediction_path, index=False)
            prediction_hashes[str(prediction_path)] = tool.sha(prediction_path)
        bank_paths[protocol] = path
        for job in jobs.values():
            if job["cell"].split("/", 1)[0] == protocol:
                job.update(
                    run_dir=str(path.parent.parent),
                    evaluation_artifact_sha256={str(path): tool.sha(path), **prediction_hashes},
                )
                bank_references.append(
                    {
                        "protocol": protocol,
                        "job_id": job["id"],
                        "state_path": str(path),
                        "sha256": tool.sha(path),
                    }
                )
    core = {
        "status": "complete",
        "smoke": False,
        "selection_uses_test": False,
        "final_seeds": list(tool.FINAL_SEEDS),
        "final_variants": ["default", "tuned"],
        "cells": cells,
        "jobs": jobs,
        "contract": {
            "source_sha256": source_files,
            "config": {"dataset_manifest_sha256": tool.DATASET_SHA256},
        },
    }
    write(pipeline / "pipeline.json", core)
    prior_paths, prior_checksums = {}, {}
    for label, state_name in (
        ("aux1", "postprocessing_state.json"),
        ("aux2", "final_analysis_state.json"),
    ):
        prior_project = fixture_root / label
        source = prior_project / "scripts/SYNTHETIC_source.py"
        source.parent.mkdir(parents=True)
        source.write_text("# SYNTHETIC metadata-only prior source\n", encoding="utf-8")
        source_manifest = prior_project / "aux_manifest.json"
        write(source_manifest, {"files": {"scripts/SYNTHETIC_source.py": tool.sha(source)}})
        prior_paths[label] = prior_project / "runs" / ("SYNTHETIC_" + label) / state_name
        write(
            prior_paths[label],
            {
                "status": "complete",
                "core_pipeline_sha256": tool.sha(pipeline / "pipeline.json"),
                "contract": {
                    "project": str(project),
                    "pipeline": str(pipeline),
                    "aux": str(prior_project),
                    "aux_manifest_sha256": tool.sha(source_manifest),
                },
                "stages": {
                    stage: {"status": "complete", "artifact_sha256": {}}
                    for stage in tool.PRIOR_STAGES[label]
                },
            },
        )
        prior_checksums[label] = tool.sha(prior_paths[label])
    contract = {
        "pipeline_root": str(pipeline),
        "pipeline_sha256": tool.sha(pipeline / "pipeline.json"),
        "aux1_state_sha256": prior_checksums["aux1"],
        "aux2_state_sha256": prior_checksums["aux2"],
        "aux1_state_path": str(prior_paths["aux1"]),
        "aux2_state_path": str(prior_paths["aux2"]),
        "config_path": str(config_path),
        "config_sha256": tool.sha(config_path),
        "config": shared_config,
        "source_sha256": source_contract(project),
        "manifests": manifests,
        "frozen_bundle_identity": {
            "path": str(bundle_path),
            "manifest_sha256": tool.sha(bundle_path),
            "source_fingerprint": tool.SOURCE_FINGERPRINT,
            "source_file_count": len(source_files),
        },
    }
    fit_contract_sha = digest(contract)
    staging = out / "SYNTHETIC_fit_staging"
    staging.mkdir()
    write_once(staging / "contract.json", contract)
    selections, controls, candidates, state_rows, totals, all_shared = {}, [], [], [], [], {}
    for protocol in tool.PROTOCOLS:
        states = {}
        for method in tool.METHODS:
            taps = {"learned_fir_causal": 9, "arx_linear": 12, "fir_tuned": 1, "arx_tuned": 3}.get(
                method
            )
            delta = method in {"arx_linear", "arx_tuned"}
            alpha = 1e-6 if method.endswith("_tuned") else None
            if method == "identity":
                state = _state(protocol, method, kind="identity")
            elif method == "bias_mean":
                state = _state(protocol, method, kind="bias", intercept=np.zeros(6))
            else:
                state = _state(
                    protocol,
                    method,
                    weights=np.zeros((6 * ((taps or 1) + int(delta)), 6)),
                    intercept=np.zeros(6),
                    taps=taps,
                    delta=delta,
                    alpha=alpha,
                    arithmetic="design_matmul"
                    if method in {"affine", "ridge_affine"}
                    else "weights_bias",
                )
            states[method] = state
            path = staging / protocol / f"{method}.state.json"
            write_once(path, state)
            common = {
                "protocol": protocol,
                "canonical_method": method,
                "information_scope": "protocol_shared_classical",
                "variant": "shared",
                "n": 1,
                "sd": None,
            }
            controls.append(
                {
                    **common,
                    "state_path": f"fit_select/{protocol}/{method}.state.json",
                    "state_sha256": tool.sha(path),
                    "state_content_sha256": digest(state),
                    "numeric_sha256": state["numeric_sha256"],
                    "coefficient_count": state["coefficient_count"],
                    "coefficient_bytes": state["coefficient_bytes"],
                }
            )
            for scope, variant, pair in (
                ("protocol_shared_classical", "shared", "{}"),
                ("known_device_bank", "bank", '{"SYNTHETIC":"pair"}'),
            ):
                row = {
                    **common,
                    "information_scope": scope,
                    "variant": variant,
                    "pair_identity_json": pair,
                    "coefficient_count": state["coefficient_count"],
                    "coefficient_bytes": state["coefficient_bytes"],
                    "normalizer_count": 0,
                    "normalizer_bytes": 0,
                }
                state_rows.append(row)
                totals.append(
                    {
                        **row,
                        "bank_coefficient_count": row["coefficient_count"],
                        "bank_coefficient_bytes": row["coefficient_bytes"],
                        "pair_count": 1,
                    }
                )
        history = []
        breakdown = macro_breakdown(np.zeros((880, 6)), synthetic_saved_predictions(protocol))
        for method, taps_grid in (
            ("ridge_tuned", [None]),
            ("fir_tuned", SEARCH["fir_taps"]),
            ("arx_tuned", SEARCH["arx_taps"]),
        ):
            for taps in taps_grid:
                for alpha in SEARCH["ridge_alpha"]:
                    history.append(
                        {
                            "protocol": protocol,
                            "information_scope": "protocol_shared_classical",
                            "canonical_method": method,
                            "variant": "shared",
                            "taps": taps,
                            "alpha": alpha,
                            "order": len(history),
                            "finite": True,
                            "error": None,
                            "validation_score": 0.0,
                            "include_delta": method == "arx_tuned",
                            "elapsed_s": 0.0,
                            "breakdown": breakdown,
                        }
                    )
        candidates.extend(
            {
                **{key: value for key, value in item.items() if key != "breakdown"},
                "breakdown_json": json.dumps(item["breakdown"], sort_keys=True),
            }
            for item in history
        )
        selection = {
            "protocol": protocol,
            "fit_role": "train",
            "selection_role": "validation",
            "synthetic_fixture": False,
            "states": states,
            "train_row_sha256": digest([protocol, "SYNTHETIC_train_order"]),
            "validation_row_sha256": digest([protocol, "SYNTHETIC_validation_order"]),
            "contract_sha256": fit_contract_sha,
            "source_sha256": contract["source_sha256"],
            "manifests": manifests[protocol],
            "state_sha256": {method: digest(state) for method, state in states.items()},
            "search": SEARCH,
            "tie_rule": "strict_smaller_first_in_grid_order",
            "candidate_history": history,
            "selections": {
                method: {"validation_score": 0.0}
                for method in ("ridge_tuned", "fir_tuned", "arx_tuned")
            },
        }
        path = staging / protocol / "selection.json"
        all_shared[protocol] = states
        write_once(path, selection)
        selections[protocol] = {
            "path": f"fit_select/{protocol}/selection.json",
            "sha256": tool.sha(path),
        }
    write_once(
        staging / "selection_set.json",
        {
            "status": "frozen_before_any_test_load",
            "protocols": selections,
            "contract_sha256": fit_contract_sha,
        },
    )
    selection_sha = tool.sha(staging / "selection_set.json")

    def finish(stage, staging, tables, stage_contract=None):
        if stage_contract is not None:
            write_once(staging / "contract.json", stage_contract)
        return finish_stage(
            staging,
            out / stage,
            {
                "stage": stage,
                "contract_sha256": digest(stage_contract)
                if stage_contract is not None
                else fit_contract_sha,
                "selection_set_sha256": selection_sha,
            },
            {role: pd.DataFrame(rows) for role, rows in tables.items()},
        )

    finish(
        "fit_select",
        staging,
        {
            "control_registry": controls,
            "candidate_history": candidates,
            "shared_state_inventory": controls,
            "method_scope": controls,
        },
    )
    staging = out / "SYNTHETIC_signal_staging"
    staging.mkdir()
    signals, metrics, neural = [], [], []
    for protocol in tool.PROTOCOLS:
        for split in ("validation", "test"):
            data = synthetic_saved_predictions(protocol)
            for method in tool.METHODS:
                for axis in AXES:
                    data[f"{method}_{axis}"] = 0.0
                row = {
                    "protocol": protocol,
                    "canonical_method": method,
                    "information_scope": "protocol_shared_classical",
                    "variant": "shared",
                    "split": split,
                    "rows": len(data),
                    "n": 1,
                    "sd": None,
                    "prediction_path": f"signal_evaluation/{protocol}/{split}_predictions.parquet",
                    "selection_set_sha256": selection_sha,
                }
                signals.append(row)
                metrics.extend(
                    metric_rows(
                        protocol,
                        split,
                        method,
                        data[[f"{method}_{axis}" for axis in AXES]].to_numpy(float),
                        data,
                    )
                )
            destination = staging / protocol / f"{split}_predictions.parquet"
            destination.parent.mkdir(exist_ok=True)
            data.to_parquet(destination, index=False)
    for job in jobs.values():
        protocol = job["cell"].split("/", 1)[0]
        if protocol in tool.PROTOCOLS:
            for split in ("validation", "test"):
                neural.append(
                    {
                        "protocol": protocol,
                        "job_id": job["id"],
                        "cell": job["cell"],
                        "phase": job["phase"],
                        "seed": job["seed"],
                        "split": split,
                    }
                )
    auxiliary = {}
    for label, stages in tool.PRIOR_STAGES.items():
        auxiliary[label] = {
            "status": "passed",
            "state_sha256": prior_checksums[label],
            "state_path": str(prior_paths[label]),
            "stages": {
                name: {"status": "passed", "numeric_payload_audit": {"status": "passed"}}
                for name in stages
            },
        }
    auxiliary["aux1"]["stages"]["transfer"].update(records=120, unique_records=120)
    auxiliary["aux1"]["stages"]["downstream"].update(
        views=5,
        units={
            protocol: {"status": "complete"}
            for protocol in tool.PROTOCOLS
            if protocol != "alimu_only"
        },
    )
    for name in ("latency_cpu", "latency_gpu"):
        auxiliary["aux1"]["stages"][name]["entries"] = 558
    auxiliary["aux2"]["stages"]["accuracy_tables"]["runs"] = 558
    auxiliary["aux2"]["stages"]["classical_latency"]["final_run_mappings"] = 558
    from reviewer_shared_classical import PHYSICAL
    from virtual_imu.eval.downstream_attitude import (
        evaluate_attitude_stream,
        evaluate_strapdown_stream,
    )

    eval_frame = pd.read_parquet(staging / "penn_only/test_predictions.parquet")
    for method in tool.METHODS:
        for axis in AXES:
            eval_frame[f"{axis}_{method}"] = 0.0
    truth = pd.DataFrame({"timestamp_s": np.arange(-8, 901) / 40 + 100.0})
    truth[[f"r{i}{j}" for i in range(3) for j in range(3)]] = np.tile(
        np.eye(3).ravel(), (len(truth), 1)
    )
    truth[["px_m", "py_m", "pz_m"]] = 0.0
    propagation_args = {
        "prefixes": tool.METHODS,
        "window_samples": [80, 200, 400, 800],
        "rate_hz": 40.0,
        "max_gt_gap_s": 0.1,
        "gt_timestamp_origin_s": 100.0,
    }
    prototype = {
        "attitude": evaluate_attitude_stream(eval_frame, truth, **propagation_args),
        "strapdown": evaluate_strapdown_stream(eval_frame, truth, **propagation_args),
    }
    original_downstream = prior_paths["aux1"].parent / "downstream"
    original_downstream.mkdir(parents=True)
    old_config = original_downstream / "config_snapshot.yaml"
    old_config.write_bytes((ROOT / "configs/downstream_penn.yaml").read_bytes())
    gt_path = original_downstream / "SYNTHETIC_independent_gt.parquet"
    truth.to_parquet(gt_path, index=False)
    downstream_summaries, downstream_windows, downstream_segments, support_proofs, old_units = (
        [],
        [],
        [],
        [],
        {},
    )
    for protocol, recording in (
        ("penn_loso_af", "af"),
        ("penn_loso_as", "as"),
        ("penn_loso_bf", "bf"),
        ("penn_only", "bs"),
        ("pooled", "bs"),
    ):
        old = original_downstream / "units" / protocol
        old.mkdir(parents=True)
        pd.read_parquet(staging / protocol / "test_predictions.parquet")[PHYSICAL].to_parquet(
            old / "evaluated_signals.parquet", index=False
        )
        metadata = {
            "independent_gt_path": str(gt_path),
            "independent_gt_sha256": tool.sha(gt_path),
            "support": {task: values["support"] for task, values in prototype.items()},
            "propagation_protocol": {
                task: values["protocol"] for task, values in prototype.items()
            },
        }
        write(old / "metrics.json", metadata)
        original_windows, original_segments = [], []
        for task, values in prototype.items():
            original_windows.extend(
                {**row, "task": task}
                for row in values["window_details"]
                if row["method"] == "identity"
            )
            original_segments.extend(
                {**row, "task": task}
                for row in values["full_recordings"]
                if row["method"] == "identity"
            )
        write(old / "window_details.json", original_windows)
        write(old / "full_segments.json", original_segments)
        write(
            old / "artifact_manifest.json",
            {
                "status": "complete",
                "contract_sha256": "SYNTHETIC_original_contract",
                "files": {
                    path.relative_to(old).as_posix(): tool.sha(path)
                    for path in old.iterdir()
                    if path.is_file()
                },
            },
        )
        old_units[protocol] = tool.sha(old / "artifact_manifest.json")
        for task, values in prototype.items():
            for section, key in (
                ("summary", "windows"),
                ("windows", "window_details"),
                ("segments", "full_recordings"),
            ):
                for scientific in values[key]:
                    method = scientific["method"]
                    base = {
                        "protocol": protocol,
                        "recording": recording,
                        "information_scope": "protocol_shared_classical",
                        "canonical_method": method,
                        "variant": "shared",
                        "task": task,
                        "n": 1,
                        "sd": None,
                        "identity_alias": method == "identity",
                        "state_content_sha256": digest(all_shared[protocol][method]),
                        **{name: value for name, value in scientific.items() if name != "method"},
                    }
                    if section == "summary":
                        base.update(
                            optimizer_runs=1,
                            status="supported" if base["count"] else "unsupported_no_fixed_windows",
                        )
                        downstream_summaries.append(base)
                    elif section == "windows":
                        downstream_windows.append(base)
                    else:
                        downstream_segments.append(base)
            support_proofs.append(
                {
                    "protocol": protocol,
                    "recording": recording,
                    "task": task,
                    "original_unit_manifest_sha256": old_units[protocol],
                    "independent_gt_sha256": tool.sha(gt_path),
                    "support": values["support"],
                    "propagation_protocol": values["protocol"],
                    "identity_alias_verified": True,
                }
            )
    write(
        original_downstream / "complete.json", {"status": "complete", "unit_manifests": old_units}
    )
    auxiliary["aux1"]["stages"]["downstream"].update(
        path=str(original_downstream),
        units={
            protocol: {"status": "complete", "manifest_sha256": checksum}
            for protocol, checksum in old_units.items()
        },
    )
    write_once(
        staging / "original_full_audit.json",
        {
            "selected_cells": 93,
            "final_evaluations": 558,
            "selection_uses_test": False,
            "verified_files": {"SYNTHETIC_source": "SYNTHETIC_sha"},
            "original_auxiliary_validation": auxiliary,
        },
    )
    aliases = []
    for protocol in tool.PROTOCOLS:
        bank = normalize_bank_pair(protocol, bank_saved[protocol]["pairs"][0])
        for method in tool.METHODS:
            common = {
                "protocol": protocol,
                "canonical_method": method,
                "information_scope": "protocol_shared_classical",
                "variant": "shared",
                "n": 1,
                "sd": None,
            }
            if protocol in {"pooled", "alimu_only"}:
                aliases.append(
                    {
                        **common,
                        "alias_status": "common_identity"
                        if method == "identity"
                        else "not_applicable_multi_pair_bank",
                    }
                )
                continue
            shared_state, bank_state = all_shared[protocol][method], bank[method]
            same = shared_state["hyperparameters"] == bank_state["hyperparameters"]
            aliases.append(
                {
                    **common,
                    "alias_status": "exact_alias" if shared_state == bank_state else "not_alias",
                    "shared_state_sha256": digest(shared_state),
                    "bank_state_sha256": digest(bank_state),
                    "same_hyperparameters": same,
                    "rtol": 1e-10,
                    "atol": 1e-12,
                    "shared_validation_score": 0.0,
                    "bank_validation_score": 0.0,
                    "shared_selected_validation_score": 0.0 if method.endswith("_tuned") else None,
                    "bank_selected_validation_score": 0.0 if method.endswith("_tuned") else None,
                    "candidate_score_history_exact": True if method.endswith("_tuned") else None,
                    "candidate_score_history_equivalent": True
                    if method.endswith("_tuned")
                    else None,
                    "max_prediction_absolute_difference": 0.0,
                    "validation_max_prediction_absolute_difference": 0.0,
                    "test_max_prediction_absolute_difference": 0.0,
                    "validation_prediction_exact": True,
                    "test_prediction_exact": True,
                    "validation_prediction_equivalent": True,
                    "test_prediction_equivalent": True,
                    "proof": {
                        **{
                            name: True
                            for name in (
                                "same_protocol",
                                "train_rows_exact",
                                "validation_rows_exact",
                                "grid_order_exact",
                                "tie_rule_exact",
                                "bank_own_selection_valid",
                                "bank_loader_order_verified",
                            )
                        },
                        "bank_cache_state_sha256": digest(bank_saved[protocol]),
                        "shared_train_order_sha256": digest([protocol, "SYNTHETIC_train_order"]),
                        "shared_validation_order_sha256": digest(
                            [protocol, "SYNTHETIC_validation_order"]
                        ),
                    },
                }
            )
    finish(
        "signal_evaluation",
        staging,
        {
            "signal_registry": signals,
            "metrics_long": metrics,
            "neural_support_registry": neural,
            "alias_audit": aliases,
        },
    )
    staging = out / "SYNTHETIC_downstream_staging"
    staging.mkdir()
    write_once(staging / "support_proofs.json", support_proofs)
    write_once(staging / "downstream_windows.json", downstream_windows)
    write_once(staging / "downstream_segments.json", downstream_segments)
    finish(
        "downstream",
        staging,
        {
            "downstream_summary": downstream_summaries,
            "downstream_windows": downstream_windows,
            "downstream_segments": downstream_segments,
        },
        {
            "fit_selection_set_sha256": selection_sha,
            "signal_manifest_sha256": tool.sha(out / "signal_evaluation/artifact_manifest.json"),
            "signal_contract_sha256": fit_contract_sha,
            "aux1_state_sha256": contract["aux1_state_sha256"],
            "aux2_state_sha256": contract["aux2_state_sha256"],
            "propagation_config_sha256": tool.sha(old_config),
            "script_sha256": tool.sha(ROOT / "scripts/evaluate_reviewer_shared_downstream.py"),
            "integrator_sha256": tool.sha(project / "src/virtual_imu/eval/downstream_attitude.py"),
        },
    )
    staging = out / "SYNTHETIC_cpu_staging"
    staging.mkdir()
    workloads = [
        {
            "protocol": protocol,
            "information_scope": scope,
            "canonical_method": method,
            "variant": variant,
            "batch_size_per_pair": batch,
            "window_size": window,
            "pair_count": 1,
            "windows_per_call": batch,
            "p50_ms": 0.01,
            "p95_ms": 0.02,
            "loop_throughput_windows_per_second": 100.0,
        }
        for protocol in tool.PROTOCOLS
        for scope, variant in (
            ("protocol_shared_classical", "shared"),
            ("known_device_bank", "bank"),
        )
        for method in tool.METHODS
        for batch in (1, 128)
        for window in (80, 160, 320)
    ]
    normalized_banks = {
        protocol: normalize_bank_pair(protocol, saved["pairs"][0])
        for protocol, saved in bank_saved.items()
    }
    parameter_records = []
    for protocol in tool.PROTOCOLS:
        for method, state in all_shared[protocol].items():
            path = out / "fit_select" / protocol / f"{method}.state.json"
            parameter_records.append(
                {
                    "protocol": protocol,
                    "information_scope": "protocol_shared_classical",
                    "canonical_method": method,
                    "variant": "shared",
                    "state": state,
                    "state_path": path.relative_to(out).as_posix(),
                    "state_file_sha256": tool.sha(path),
                    "serialized_state_bytes": path.stat().st_size,
                }
            )
        path = bank_paths[protocol]
        for _reference in (item for item in bank_references if item["protocol"] == protocol):
            for method, state in normalized_banks[protocol].items():
                parameter_records.append(
                    {
                        "protocol": protocol,
                        "information_scope": "known_device_bank",
                        "canonical_method": method,
                        "variant": "bank",
                        "state": state,
                        "pair_identity": pair_identity,
                        "state_path": str(path),
                        "state_file_sha256": tool.sha(path),
                        "serialized_state_bytes": path.stat().st_size,
                    }
                )
    state_rows, totals = parameter_tables(parameter_records)
    raw_records, raw_rows, state_timings, schedulers, proofs = {}, [], [], {}, []
    histories = {
        protocol: max(
            max(
                (state["feature_spec"]["taps"] or 1) - 1,
                int(state["feature_spec"]["include_delta"]),
            )
            for state in [*all_shared[protocol].values(), *normalized_banks[protocol].values()]
        )
        for protocol in tool.PROTOCOLS
    }
    pair_key = tool.digest(pair_identity)
    for protocol in tool.PROTOCOLS:
        history = histories[protocol]
        for window in (80, 160, 320):
            pool = np.zeros((128, history + window, 6))
            physical = {
                "split": "train",
                "sequence_id": "SYNTHETIC_train",
                "original_sequence_id": "SYNTHETIC_train",
                "start_index": 40,
                "history_start_index": 40 - history,
                "end_index": 40 + window - 1,
                "physical_row_sha256": tool.digest([protocol, window, "SYNTHETIC_physical"]),
                "input_sha256": array_sha(pool[0]),
            }
            schedulers[f"{protocol}/{window}/{pair_key}"] = {
                "pair_identity": pair_identity,
                "rows": [physical] * 128,
                "physical_train_sha256": tool.digest([protocol, "SYNTHETIC_train"]),
                "pool_sha256": array_sha(pool),
                "batch_input_sha256": {
                    str(batch): array_sha(pool[:batch]) for batch in DEFAULTS["batch_sizes"]
                },
            }
        for method in tool.METHODS:
            for scope in ("protocol_shared_classical", "known_device_bank"):
                proofs.append(
                    {
                        "protocol": protocol,
                        "canonical_method": method,
                        "information_scope": scope,
                        "pair_key": pair_key,
                        "numeric_sha256": (
                            all_shared if scope == "protocol_shared_classical" else normalized_banks
                        )[protocol][method]["numeric_sha256"],
                        "proof": [
                            {
                                "sequence_id": "SYNTHETIC_train",
                                "start_checked": True,
                                "interior_checked": True,
                                "history_samples": history,
                                "max_abs_error": 0.0,
                            }
                        ],
                    }
                )
    for row in workloads:
        protocol, method = row["protocol"], row["canonical_method"]
        fitted = (
            all_shared
            if row["information_scope"] == "protocol_shared_classical"
            else normalized_banks
        )[protocol][method]
        batch, window, history = row["batch_size_per_pair"], row["window_size"], histories[protocol]
        pool = np.zeros((batch, history + window, 6))
        numeric, content, inputs = [fitted["numeric_sha256"]], [digest(fitted)], [array_sha(pool)]
        content_key = digest(
            {"states": numeric, "state_contents": content, "inputs": inputs, "history": history}
        )
        if content_key not in raw_records:
            times, loops = np.linspace(0.01, 0.02, 300), np.asarray([0.01, 0.011, 0.012])
            record = {
                "timing_id": "timing_" + content_key[:24],
                "pair_count": 1,
                "windows_per_call": batch,
                "p50_ms": float(np.percentile(times, 50)),
                "p95_ms": float(np.percentile(times, 95)),
                "call_distribution_sd_ms": float(np.std(times, ddof=1)),
                "loop_throughput_windows_per_second": batch * 300 / float(loops.sum()),
                "warmup": 30,
                "iterations": 100,
                "repeats": 3,
                "numeric_state_sha256s": numeric,
                "state_content_sha256s": content,
                "input_pool_sha256s": inputs,
                "input_pool_sha256": digest(inputs),
                "resident_input_shapes": [list(pool.shape)],
                "history_samples": history,
                "raw_wall_ms": times.tolist(),
                "raw_throughput_loop_seconds": loops.tolist(),
            }
            raw_records[content_key] = record
            for index, value in enumerate(times):
                raw_rows.append(
                    {
                        "timing_id": record["timing_id"],
                        "measurement": "call_wall_ms",
                        "repeat": index // 100,
                        "iteration": index % 100,
                        "value": float(value),
                        "unit": "ms",
                    }
                )
            for repeat, value in enumerate(loops):
                raw_rows.append(
                    {
                        "timing_id": record["timing_id"],
                        "measurement": "independent_throughput_loop",
                        "repeat": repeat,
                        "iteration": None,
                        "value": float(value),
                        "unit": "s",
                    }
                )
        record = raw_records[content_key]
        row.update(
            {
                name: record[name]
                for name in (
                    "timing_id",
                    "p50_ms",
                    "p95_ms",
                    "call_distribution_sd_ms",
                    "loop_throughput_windows_per_second",
                )
            }
        )
        row["input_pool_sha256"] = record["input_pool_sha256"]
        row["scheduler_sha256"] = digest([schedulers[f"{protocol}/{window}/{pair_key}"]])
        row.update(n=1, sd=None, measurement_kind="full_pair_workload", history_samples=history)
        state_timings.append(
            {
                **row,
                "pair_identity_json": json.dumps(pair_identity, sort_keys=True),
                "measurement_kind": "per_state",
            }
        )
    write_once(
        staging / "runtime.json",
        {
            "threadpoolctl_version": "3.6.0",
            "threadpools": [{"num_threads": 1, "internal_api": "SYNTHETIC_stored_fixture"}],
            "workload_quiet_check": {"status": "passed", "scope": "SYNTHETIC stored metadata"},
        },
    )
    write_once(staging / "raw_timing_records.json", raw_records)
    write_once(staging / "bank_source_references.json", bank_references)
    write_once(staging / "train_window_schedulers.json", schedulers)
    write_once(staging / "kernel_proofs.json", proofs)
    write_once(staging / "timing_aliases.json", [])
    finish(
        "cpu_latency",
        staging,
        {
            "state_inventory": state_rows,
            "bank_parameter_totals": totals,
            "latency_summary": state_timings,
            "workload_latency": workloads,
            "latency_raw": raw_rows,
        },
        {
            "fit_contract_sha256": fit_contract_sha,
            "selection_set_sha256": selection_sha,
            "pipeline_sha256": contract["pipeline_sha256"],
            "aux1_state_sha256": contract["aux1_state_sha256"],
            "aux2_state_sha256": contract["aux2_state_sha256"],
            "config": DEFAULTS,
            "config_sha256": tool.sha(ROOT / "configs/reviewer_shared_classical_latency.yaml"),
            "upstream_manifests": {
                name: tool.sha(out / name / "artifact_manifest.json")
                for name in ("fit_select", "signal_evaluation", "downstream")
            },
            "script_sha256": tool.sha(
                ROOT / "scripts/benchmark_reviewer_shared_classical_latency.py"
            ),
            "normalized_state_script_sha256": tool.sha(
                ROOT / "scripts/reviewer_shared_classical.py"
            ),
        },
    )
    aux3_project = out.parent.parent
    source_members = {}
    for relative in (
        "scripts/server/run_reviewer_shared_controls.py",
        "scripts/benchmark_reviewer_shared_classical_latency.py",
        "scripts/reviewer_shared_classical.py",
        "scripts/evaluate_reviewer_shared_classical.py",
        "scripts/evaluate_reviewer_shared_downstream.py",
        "scripts/evaluate_reviewer_downstream.py",
        "scripts/evaluate_reviewer_transfer.py",
        "configs/reviewer_shared_classical_latency.yaml",
    ):
        destination = aux3_project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
        source_members[relative] = tool.sha(destination)
    destination = aux3_project / "configs/reviewer_shared_classical.yaml"
    destination.write_bytes(config_path.read_bytes())
    source_members["configs/reviewer_shared_classical.yaml"] = tool.sha(destination)
    aux3_manifest = aux3_project / "aux_manifest.json"
    write(
        aux3_manifest,
        {
            "schema_version": 1,
            "analysis_profile": tool.PROFILE,
            "base_project": str(project),
            "base_pipeline": str(pipeline.relative_to(project)),
            "files": source_members,
        },
    )
    context = {
        "fixture_kind": "SYNTHETIC",
        "core_project": str(project),
        "pipeline_root": str(pipeline),
        "aux3_project": str(aux3_project),
        "aux3_run": str(out),
        "aux3_manifest_sha256": tool.sha(aux3_manifest),
        "aux1_state": str(prior_paths["aux1"]),
        "aux2_state": str(prior_paths["aux2"]),
        "source_fingerprint": tool.SOURCE_FINGERPRINT,
        "dataset_manifest_sha256": tool.DATASET_SHA256,
    }
    write(fixture_root / "SYNTHETIC_fixture_context.json", context)
    if materialize_header:
        index = tool.validate_complete_artifacts(
            out,
            contract["pipeline_sha256"],
            require_index=False,
            expected_aux_manifest_sha256=tool.sha(aux3_manifest),
        )
        write(out / "shared_controls_artifacts.json", index)
        write(
            out / "shared_controls_state.json",
            {
                "schema_version": 1,
                "analysis_profile": tool.PROFILE,
                "status": "complete",
                "core_pipeline_sha256": contract["pipeline_sha256"],
                "counts": index["counts"],
                "global_selection_set_sha256": selection_sha,
                "shared_controls_artifacts_sha256": tool.sha(
                    out / "shared_controls_artifacts.json"
                ),
                "contract": {
                    "project": str(project),
                    "pipeline": str(pipeline),
                    "aux": str(aux3_project),
                    "sources": source_members,
                    "aux_manifest_sha256": tool.sha(aux3_manifest),
                },
                "stages": {
                    name: {
                        "status": "complete",
                        "artifact_sha256": tool.read_json(out / name / "artifact_manifest.json")[
                            "files"
                        ],
                        "artifact_manifest_sha256": tool.sha(out / name / "artifact_manifest.json"),
                        "stage_summary_sha256": tool.sha(out / name / "stage_summary.json"),
                    }
                    for name in tool.STAGES
                },
            },
        )
        (aux3_project / "shared_controls_console.log").write_text(
            "REVIEWER_SHARED_CONTROLS_PASSED\n", encoding="utf-8"
        )
    return tool, out, contract["pipeline_sha256"]


def test_complete_descriptor_checks_real_keys_arrays_and_is_read_only():
    tool, out, core_sha = complete_synthetic_run()
    before = {str(path): path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()}
    index = tool.validate_complete_artifacts(out, core_sha, require_index=False)
    assert index["counts"] == {
        "shared_protocols": 6,
        "shared_logical_controls": 54,
        "shared_learned_controls": 48,
        "shared_identity_controls": 6,
        "shared_signal_slots": 108,
        "shared_downstream_views": 5,
        "shared_downstream_summary_slots": 360,
        "shared_paired_final_mappings": 360,
    }
    assert index["original"]["neural_transfer_records"] == 120
    assert len(index["files"]) > 80
    assert before == {
        str(path): path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()
    }
    write(out / "shared_controls_artifacts.json", index)
    assert tool.validate_complete_artifacts(out, core_sha)["status"] == "complete"
    core_path = (
        Path(tool.read_json(out / "fit_select/contract.json")["pipeline_root"]) / "pipeline.json"
    )
    core_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pipeline checksum"):
        tool.validate_complete_artifacts(out, core_sha)


def rehash_synthetic_stage(tool, out, stage):
    manifest_path = out / stage / "artifact_manifest.json"
    manifest = tool.read_json(manifest_path)
    manifest["files"] = {name: tool.sha(out / name) for name in manifest["files"]}
    write(manifest_path, manifest)


def test_global_freeze_rejects_detached_actual_fit_contract_before_payload_audit():
    tool, out, _ = complete_synthetic_run()
    path = out / "fit_select/contract.json"
    contract = tool.read_json(path)
    contract["detached_SYNTHETIC_change"] = True
    write(path, contract)
    rehash_synthetic_stage(tool, out, "fit_select")
    with pytest.raises(ValueError, match=r"freeze|frozen"):
        tool.verify_global_freeze(out)


def test_cpu_own_contract_cannot_be_rehashed_away_from_frozen_upstream():
    tool, out, core_sha = complete_synthetic_run()
    path = out / "cpu_latency/contract.json"
    contract = tool.read_json(path)
    contract["fit_contract_sha256"] = "0" * 64
    write(path, contract)
    summary_path = out / "cpu_latency/stage_summary.json"
    summary = tool.read_json(summary_path)
    summary["contract_sha256"] = tool.digest(contract)
    write(summary_path, summary)
    manifest_path = out / "cpu_latency/artifact_manifest.json"
    manifest = tool.read_json(manifest_path)
    manifest["contract_sha256"] = tool.digest(contract)
    write(manifest_path, manifest)
    rehash_synthetic_stage(tool, out, "cpu_latency")
    with pytest.raises(ValueError, match="CPU contract"):
        tool.validate_complete_artifacts(out, core_sha, require_index=False)


def test_current_source_mutation_is_rejected_even_with_unchanged_result_counts():
    tool, out, core_sha = complete_synthetic_run()
    contract = tool.read_json(out / "fit_select/contract.json")
    copied_source = (
        Path(contract["pipeline_root"]).parent.parent / "src/virtual_imu/eval/baselines.py"
    )
    copied_source.write_text("# SYNTHETIC mutation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source roster"):
        tool.validate_complete_artifacts(out, core_sha, require_index=False)


def test_cpu_empty_rosters_missing_samples_and_wrong_raw_statistics_reject():
    import copy
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_reviewer_shared_classical import verify_selection_set

    tool, out, _ = complete_synthetic_run()
    contract = tool.read_json(out / "fit_select/contract.json")
    core = tool.read_json(Path(contract["pipeline_root"]) / "pipeline.json")
    _, selections = verify_selection_set(out)
    tables = {
        role: tool._table(out, descriptor)
        for stage in tool.STAGES
        for role, descriptor in tool.read_json(out / stage / "stage_summary.json")["tables"].items()
    }
    empty = dict(tables)
    empty["state_inventory"] = []
    with pytest.raises(ValueError, match="actual state inventory"):
        tool.validate_cpu_artifacts(out, empty, selections, core, contract)
    missing = dict(tables)
    missing["latency_raw"] = tables["latency_raw"][:-1]
    with pytest.raises(ValueError, match="raw timing CSV"):
        tool.validate_cpu_artifacts(out, missing, selections, core, contract)
    wrong = dict(tables)
    wrong["latency_summary"] = copy.deepcopy(tables["latency_summary"])
    wrong["latency_summary"][0]["p95_ms"] = "123.0"
    with pytest.raises(ValueError, match="Public latency statistic"):
        tool.validate_cpu_artifacts(out, wrong, selections, core, contract)
    raw_path = out / "cpu_latency/raw_timing_records.json"
    raw = tool.read_json(raw_path)
    next(iter(raw.values()))["raw_wall_ms"].pop()
    write(raw_path, raw)
    with pytest.raises(ValueError, match="raw300"):
        tool.validate_cpu_artifacts(out, tables, selections, core, contract)


def test_downstream_missing_errors_and_alias_proof_reject():
    import copy
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_reviewer_shared_classical import verify_selection_set

    tool, out, _ = complete_synthetic_run()
    contract = tool.read_json(out / "fit_select/contract.json")
    core = tool.read_json(Path(contract["pipeline_root"]) / "pipeline.json")
    audit = tool.read_json(out / "signal_evaluation/original_full_audit.json")[
        "original_auxiliary_validation"
    ]
    _, selections = verify_selection_set(out)
    tables = {
        role: tool._table(out, descriptor)
        for stage in tool.STAGES
        for role, descriptor in tool.read_json(out / stage / "stage_summary.json")["tables"].items()
    }
    wrong = dict(tables)
    wrong["downstream_windows"] = copy.deepcopy(tables["downstream_windows"])
    wrong["downstream_windows"][0]["endpoint_deg"] = ""
    with pytest.raises(ValueError, match="numeric value is missing"):
        tool.validate_downstream_and_aliases(out, wrong, selections, core, audit, contract)
    wrong = dict(tables)
    wrong["alias_audit"] = copy.deepcopy(tables["alias_audit"])
    next(row for row in wrong["alias_audit"] if row["protocol"] == "penn_only")["proof"] = "{}"
    with pytest.raises(ValueError, match="Alias provenance"):
        tool.validate_downstream_and_aliases(out, wrong, selections, core, audit, contract)
    proof_path = out / "downstream/support_proofs.json"
    proofs = tool.read_json(proof_path)
    proofs[0]["support"]["common_valid_rows"] -= 1
    write(proof_path, proofs)
    with pytest.raises(ValueError, match="support/GT/initialization"):
        tool.validate_downstream_and_aliases(out, tables, selections, core, audit, contract)


def synthetic_validation_inputs(tool, out):
    from evaluate_reviewer_shared_classical import verify_selection_set

    contract = tool.read_json(out / "fit_select/contract.json")
    core = tool.read_json(Path(contract["pipeline_root"]) / "pipeline.json")
    _, selections = verify_selection_set(out)
    tables = {
        role: tool._table(out, descriptor)
        for stage in tool.STAGES
        for role, descriptor in tool.read_json(out / stage / "stage_summary.json")["tables"].items()
    }
    audit = tool.read_json(out / "signal_evaluation/original_full_audit.json")[
        "original_auxiliary_validation"
    ]
    return contract, core, selections, tables, audit


def rewrite_synthetic_public_table(tool, out, role, records):
    """Keep file descriptors and dependency hashes coherent around an invalid table."""
    import csv

    for stage in tool.STAGES:
        summary_path = out / stage / "stage_summary.json"
        summary = tool.read_json(summary_path)
        if role not in summary["tables"]:
            continue
        descriptor = summary["tables"][role]
        path = out / descriptor["path"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=descriptor["columns"])
            writer.writeheader()
            writer.writerows(records)
        descriptor.update(sha256=tool.sha(path), rows=len(records))
        write(summary_path, summary)
        rehash_synthetic_stage(tool, out, stage)
        break
    else:
        raise AssertionError("Missing fixture public role")
    for stage in ("downstream", "cpu_latency"):
        contract_path = out / stage / "contract.json"
        contract = tool.read_json(contract_path)
        if stage == "downstream":
            contract["signal_manifest_sha256"] = tool.sha(
                out / "signal_evaluation/artifact_manifest.json"
            )
        else:
            contract["upstream_manifests"] = {
                name: tool.sha(out / name / "artifact_manifest.json")
                for name in ("fit_select", "signal_evaluation", "downstream")
            }
        write(contract_path, contract)
        summary_path = out / stage / "stage_summary.json"
        summary = tool.read_json(summary_path)
        summary["contract_sha256"] = tool.digest(contract)
        write(summary_path, summary)
        manifest_path = out / stage / "artifact_manifest.json"
        manifest = tool.read_json(manifest_path)
        manifest["contract_sha256"] = tool.digest(contract)
        write(manifest_path, manifest)
        rehash_synthetic_stage(tool, out, stage)


def test_public_cpu_input_and_scheduler_identity_reject_detachment():
    import copy

    tool, out, _ = complete_synthetic_run()
    contract, core, selections, tables, _ = synthetic_validation_inputs(tool, out)
    tool.validate_cpu_artifacts(out, tables, selections, core, contract)
    for role, field in (
        ("latency_summary", "input_pool_sha256"),
        ("workload_latency", "scheduler_sha256"),
    ):
        changed = dict(tables)
        changed[role] = copy.deepcopy(tables[role])
        changed[role][0][field] = "0" * 64
        with pytest.raises(ValueError, match=r"input pool|scheduler"):
            tool.validate_cpu_artifacts(out, changed, selections, core, contract)
        rewrite_synthetic_public_table(tool, out, role, changed[role])
        with pytest.raises(ValueError, match=r"input pool|scheduler"):
            tool.validate_complete_artifacts(out, contract["pipeline_sha256"], require_index=False)
        rewrite_synthetic_public_table(tool, out, role, tables[role])


def test_exact_alias_rejects_contradictory_scores_and_prediction_differences():
    import copy

    tool, out, _ = complete_synthetic_run()
    contract, core, selections, tables, audit = synthetic_validation_inputs(tool, out)
    tool.validate_downstream_and_aliases(out, tables, selections, core, audit, contract)
    changed = dict(tables)
    changed["alias_audit"] = copy.deepcopy(tables["alias_audit"])
    selected = next(
        row
        for row in changed["alias_audit"]
        if row["protocol"] == "penn_only" and row["alias_status"] == "exact_alias"
    )
    selected.update(
        shared_validation_score="123.0",
        bank_validation_score="456.0",
        max_prediction_absolute_difference="123.0",
        validation_max_prediction_absolute_difference="123.0",
        test_max_prediction_absolute_difference="123.0",
    )
    with pytest.raises(ValueError, match=r"Alias.*score|Alias.*difference|exact alias"):
        tool.validate_downstream_and_aliases(out, changed, selections, core, audit, contract)
    rewrite_synthetic_public_table(tool, out, "alias_audit", changed["alias_audit"])
    with pytest.raises(ValueError, match=r"Alias.*score|Alias.*difference|exact alias"):
        tool.validate_complete_artifacts(out, contract["pipeline_sha256"], require_index=False)


def test_numerical_alias_accepts_saved_signed_zero_and_rejects_false_equivalence():
    import copy

    import pandas as pd

    tool, out, _ = complete_synthetic_run()
    contract, core, selections, tables, audit = synthetic_validation_inputs(tool, out)
    protocol, method = "penn_only", "bias_mean"
    changed = dict(tables)
    changed["alias_audit"] = copy.deepcopy(tables["alias_audit"])
    selected = next(
        row
        for row in changed["alias_audit"]
        if row["protocol"] == protocol and row["canonical_method"] == method
    )
    selected["alias_status"] = "numerically_equivalent"
    job = next(job for job in core["jobs"].values() if job["cell"].split("/", 1)[0] == protocol)
    for split in ("validation", "test"):
        path = Path(job["run_dir"]) / "evaluation" / f"{split}_predictions.parquet"
        saved = pd.read_parquet(path)
        columns = [f"{method}_{axis}" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]
        saved[columns] = -saved[columns].to_numpy(float)
        saved.to_parquet(path, index=False)
        for original_job in core["jobs"].values():
            if original_job["cell"].split("/", 1)[0] == protocol:
                original_job["evaluation_artifact_sha256"][str(path)] = tool.sha(path)
    tool.validate_downstream_and_aliases(out, changed, selections, core, audit, contract)
    selected["validation_max_prediction_absolute_difference"] = "123.0"
    with pytest.raises(ValueError, match="Alias prediction difference"):
        tool.validate_downstream_and_aliases(out, changed, selections, core, audit, contract)


def test_candidate_history_csv_cannot_detach_from_its_frozen_scientific_record():
    import copy

    tool, out, core_sha = complete_synthetic_run()
    _, _, selections, tables, _ = synthetic_validation_inputs(tool, out)
    tool.validate_complete_artifacts(out, core_sha, require_index=False)
    changed = copy.deepcopy(tables["candidate_history"])
    changed[0]["validation_score"] = "123.0"
    rewrite_synthetic_public_table(tool, out, "candidate_history", changed)
    with pytest.raises(ValueError, match=r"Candidate.*frozen"):
        tool.validate_complete_artifacts(out, core_sha, require_index=False)
    for field, value in (
        ("alpha", "123.0"),
        ("include_delta", "True"),
        ("finite", "False"),
        ("error", "fabricated"),
    ):
        altered = copy.deepcopy(tables["candidate_history"])
        altered[0][field] = value
        with pytest.raises(ValueError, match=r"Candidate.*frozen"):
            tool.validate_candidate_history(altered, selections)
    altered = copy.deepcopy(tables["candidate_history"])
    breakdown = json.loads(altered[0]["breakdown_json"])
    breakdown["pairs"][0]["score"] = 123.0
    altered[0]["breakdown_json"] = json.dumps(breakdown)
    with pytest.raises(ValueError, match=r"Candidate.*frozen"):
        tool.validate_candidate_history(altered, selections)


def test_public_metric_value_cannot_detach_from_saved_prediction_arrays():
    import copy

    tool, out, core_sha = complete_synthetic_run()
    _, _, _, tables, _ = synthetic_validation_inputs(tool, out)
    tool.validate_complete_artifacts(out, core_sha, require_index=False)
    changed = copy.deepcopy(tables["metrics_long"])
    changed[0]["value"] = "123.0"
    rewrite_synthetic_public_table(tool, out, "metrics_long", changed)
    with pytest.raises(ValueError, match=r"Metric.*saved prediction"):
        tool.validate_complete_artifacts(out, core_sha, require_index=False)
