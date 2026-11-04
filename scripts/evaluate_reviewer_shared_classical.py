"""Freeze all six shared selections before audited VAL/TEST signal evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reviewer_shared_classical import (
    AXES,
    HIGH,
    IDENTITY,
    LOW,
    METHODS,
    PHYSICAL,
    PROFILE,
    PROTOCOLS,
    SCOPE,
    SEARCH,
    alias_audit,
    digest,
    finish_stage,
    fit_protocol,
    frame_sha,
    macro_breakdown,
    normalize_bank_pair,
    physical_support,
    predict_state,
    read_json,
    sha,
    validate_streams,
    verify_artifact_manifest,
    verify_state,
    write_once,
)
from virtual_imu.eval.reviewer_evaluation import _pair_metrics
from virtual_imu.io.multi_imu_splits import validate_no_split_leakage
from virtual_imu.io.training_bundle import portable_relative_path
from virtual_imu.io.training_split_manifest import (
    _apply_row_filter,
    _require_training_eligible_rows,
    _strict_record_index,
    _validate_strict_match,
)


def original_helpers():
    path = Path(__file__).with_name("evaluate_reviewer_downstream.py")
    spec = importlib.util.spec_from_file_location("shared_original_downstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(value, project: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def source_contract(project: Path) -> dict:
    paths = [
        Path(__file__),
        Path(__file__).with_name("reviewer_shared_classical.py"),
        Path(__file__).with_name("evaluate_reviewer_transfer.py"),
        Path(__file__).with_name("evaluate_reviewer_downstream.py"),
    ]
    paths += [
        project / "src/virtual_imu" / name
        for name in (
            "eval/formal_evaluation.py",
            "eval/baselines.py",
            "eval/reviewer_evaluation.py",
            "io/training_split_manifest.py",
            "training/paired_validation.py",
        )
    ]
    return {str(path.resolve()): sha(path) for path in paths}


def verify_frozen_bundle_sources(project: Path, config: dict) -> dict:
    """Read bundle metadata and source bytes only; skip all data/build payloads."""
    path = project / "bundle_manifest.json"
    manifest = read_json(path)
    names, folded, source_hashes = set(), set(), {}
    for entry in manifest["entries"]:
        name = str(portable_relative_path(entry["path"]))
        if name in names or name.casefold() in folded:
            raise ValueError("Duplicate/case-conflicting frozen bundle inventory path")
        names.add(name)
        folded.add(name.casefold())
        if entry["kind"] != "source":
            continue
        source = (project / name).resolve()
        if (
            not source.is_relative_to(project.resolve())
            or source.stat().st_size != entry["size_bytes"]
            or sha(source) != entry["sha256"]
        ):
            raise ValueError("Frozen bundle source checksum/size changed")
        source_hashes[name] = entry["sha256"]
    fingerprint = hashlib.sha256(
        (json.dumps(source_hashes, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    if (
        not source_hashes
        or fingerprint != manifest.get("source_fingerprint")
        or fingerprint != config["expected_source_fingerprint"]
        or manifest.get("dataset_manifest_sha256") != config["expected_dataset_manifest_sha256"]
    ):
        raise ValueError("Frozen bundle source/data fingerprint differs")
    print(
        "SKIPS: frozen bundle data/build_provenance payload bytes until requested split roles",
        flush=True,
    )
    return {
        "path": str(path),
        "manifest_sha256": sha(path),
        "source_fingerprint": fingerprint,
        "source_file_count": len(source_hashes),
        "git_commit": manifest.get("git_commit"),
    }


def metadata_gate(
    pipeline_root: Path, project: Path, aux1_path: Path, aux2_path: Path, config: dict
) -> tuple[dict, list[dict], dict]:
    """S1 gate: metadata/config/manifests only; no prediction/metric payloads."""
    state = read_json(pipeline_root / "pipeline.json")
    finals = [
        job for job in state.get("jobs", {}).values() if job.get("phase", "").startswith("final_")
    ]
    cells = state.get("cells", {})
    if (
        state.get("status") != "complete"
        or state.get("smoke") is not False
        or state.get("selection_uses_test") is not False
        or len(cells) != 93
        or len(finals) != 558
        or any(
            job.get("status") != "complete" or job.get("evaluation_status") != "complete"
            for job in finals
        )
    ):
        raise ValueError(
            "Formal shared controls require complete nonsmoke validation-only 93/558 core"
        )
    if state.get("final_seeds") != [20261010, 20261011, 20261012] or set(
        state.get("final_variants", [])
    ) != {"default", "tuned"}:
        raise ValueError("Frozen final seeds/variants differ")
    expected = {
        (cell, "final_" + variant, seed)
        for cell in cells
        for variant in state["final_variants"]
        for seed in state["final_seeds"]
    }
    actual = [(job["cell"], job["phase"], job["seed"]) for job in finals]
    if len(set(actual)) != 558 or set(actual) != expected:
        raise ValueError("Frozen final identities differ")
    workflow = project / "runs/reviewer_workflow_status.log"
    lines = workflow.read_text(encoding="utf-8").splitlines()
    exits = [line for line in lines if line.startswith("WORKFLOW_EXIT=")]
    if "REVIEWER_CORE_PASSED" not in lines or not exits or exits[-1] != "WORKFLOW_EXIT=0":
        raise ValueError("Successful original core terminal marker required")
    helper = original_helpers()
    checked = {}
    helper._checked_inventory(state["contract"]["source_sha256"], pipeline_root, project, checked)
    # The original HPO input inventory consists exactly of dataset and derived
    # split/strict/assignment JSON metadata. Reject payload entries at S1.
    input_metadata = state.get("input_artifact_sha256", {})
    if not input_metadata or any(
        Path(str(name).replace("\\", "/")).name
        not in {
            "manifest.json",
            "split_manifest.json",
            "strict_manifest.json",
            "assignment_manifest.json",
        }
        for name in input_metadata
    ):
        raise ValueError("Initial core input inventory must contain manifest metadata only")
    helper._checked_inventory(input_metadata, pipeline_root, project, checked)
    for identifier, cell in cells.items():
        selection = pipeline_root / "selections" / (identifier.replace("/", "__") + ".json")
        if "selected" not in cell or sha(selection) != cell.get("selection_sha256"):
            raise ValueError("Original selection identity/checksum differs")
    if (
        yaml.safe_load((pipeline_root / "matrix_config.yaml").read_text(encoding="utf-8"))
        != state["contract"]["config"]
    ):
        raise ValueError("Frozen matrix configuration differs")
    dataset_sha = state["contract"]["config"].get("dataset_manifest_sha256")
    if dataset_sha != config["expected_dataset_manifest_sha256"]:
        raise ValueError("Canonical dataset source identity differs")
    bundle_identity = verify_frozen_bundle_sources(project, config)
    prior_metadata = {}
    for label, path, expected_stages in (
        ("aux1", aux1_path, {"transfer", "downstream", "latency_gpu", "latency_cpu"}),
        ("aux2", aux2_path, {"accuracy_tables", "classical_latency"}),
    ):
        prior = read_json(path)
        if (
            prior.get("status") != "complete"
            or set(prior.get("stages", {})) != expected_stages
            or any(stage.get("status") != "complete" for stage in prior["stages"].values())
        ):
            raise ValueError(f"Original {label} must terminate successfully before shared fitting")
        if prior.get("core_pipeline_sha256") != sha(pipeline_root / "pipeline.json"):
            raise ValueError(f"Original {label} is bound to a different core")
        auxiliary = Path(prior["contract"]["aux"])
        manifest_path = auxiliary / "aux_manifest.json"
        manifest = read_json(manifest_path)
        if sha(manifest_path) != prior["contract"]["aux_manifest_sha256"]:
            raise ValueError(f"Original {label} source manifest changed")
        for name, expected_sha in manifest["files"].items():
            source = (auxiliary / name).resolve()
            if not source.is_relative_to(auxiliary.resolve()) or sha(source) != expected_sha:
                raise ValueError(f"Original {label} source changed")
        prior_metadata[label] = {"path": str(path.resolve()), "sha256": sha(path), "state": prior}
    return (
        state,
        finals,
        {
            "core_pipeline_sha256": sha(pipeline_root / "pipeline.json"),
            "prior": prior_metadata,
            "verified_source_metadata": checked,
            "bundle_identity": bundle_identity,
        },
    )


def protocol_manifests(protocol: str, finals: list[dict], root: Path, project: Path) -> dict:
    helper = original_helpers()
    jobs = sorted(
        [job for job in finals if job["cell"].split("/", 1)[0] == protocol],
        key=lambda row: row["id"],
    )
    if len(jobs) != 60:
        raise ValueError(f"Paired protocol must have all 60 final jobs: {protocol}")
    contracts = []
    for job in jobs:
        path = helper.recorded_path(job["config"], root, project)
        if sha(path) != job["config_sha256"]:
            raise ValueError("Frozen job config changed")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config.get("reviewer_protocol", {}).get("protocol_id") != protocol:
            raise ValueError("Frozen job config protocol differs")
        manifests = {
            name: resolve(config["data"][name], project)
            for name in (
                "dataset_manifest",
                "split_manifest",
                "strict_manifest",
                "assignment_manifest",
            )
        }
        contracts.append(
            {name: {"path": str(value), "sha256": sha(value)} for name, value in manifests.items()}
        )
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Final jobs within a protocol differ in physical split manifests")
    split = read_json(Path(contracts[0]["split_manifest"]["path"]))
    if split.get("fold_id") != protocol or split.get("metadata", {}).get("task") != "paired":
        raise ValueError("Derived split protocol/task identity differs")
    if (
        split.get("canonical_provenance", {}).get("dataset_manifest_sha256")
        != contracts[0]["dataset_manifest"]["sha256"]
    ):
        raise ValueError("Derived split is not bound to the configured canonical dataset")
    return contracts[0]


def load_roles(manifests: dict, roles: tuple[str, ...], project: Path) -> tuple[dict, dict]:
    """Explicit role loader; raw identities are checked before any helper fills IDs."""
    if not roles or set(roles) - {"train", "validation", "test"}:
        raise ValueError("Require explicit role list")
    for descriptor in manifests.values():
        if sha(Path(descriptor["path"])) != descriptor["sha256"]:
            raise ValueError("Frozen split/source metadata changed before role loading")
    split = read_json(Path(manifests["split_manifest"]["path"]))
    strict = read_json(Path(manifests["strict_manifest"]["path"]))
    validate_no_split_leakage(split)
    strict_index = _strict_record_index(strict)
    frames, registries = {}, {}
    print(
        f"SKIPS: unrequested physical split payloads {sorted({'train', 'validation', 'test'} - set(roles))}",
        flush=True,
    )
    for role in roles:
        pieces = []
        for record in split["splits"][role]:
            key = tuple(str(record[name]) for name in ("dataset_id", "pair_id", "sequence_id"))
            strict_record = strict_index.get(key)
            if strict_record is None:
                raise ValueError("Split has no exact strict manifest identity")
            _validate_strict_match(record, strict_record, role, project)
            path = resolve(record["path"], project)
            if sha(path) != record.get("sha256"):
                raise ValueError("Canonical physical payload SHA differs")
            print(f"READS: {role} physical payload {path}", flush=True)
            raw = pd.read_parquet(path)
            if sha(path) != record.get("sha256"):
                raise ValueError("Canonical payload changed while reading requested role")
            _require_training_eligible_rows(raw, path)
            validate_streams(raw, [record])
            if set(raw.sequence_id.astype(str)) != {str(record["sequence_id"])}:
                raise ValueError("Raw source sequence identity differs from manifest")
            raw = _apply_row_filter(raw, record, path)
            # Derived roles replace only the catalog role, never physical identity.
            raw["split"] = role
            pieces.append(raw)
        whole = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        frames[role], registries[role] = physical_support(whole, role, split["splits"][role])
    return frames, registries


def _staging(out: Path, stage: str) -> Path:
    partial = out / ".partial"
    partial.mkdir(parents=True, exist_ok=True)
    attempt = partial / f"{stage}_attempt_{len(list(partial.glob(stage + '_attempt_*'))):04d}"
    attempt.mkdir()
    return attempt


def verify_selection_set(out: Path, *, formal=True) -> tuple[dict, dict]:
    verify_artifact_manifest(out / "fit_select")
    summary = read_json(out / "fit_select/stage_summary.json")
    path = out / "fit_select/selection_set.json"
    selection_set = read_json(path)
    if sha(path) != summary.get("selection_set_sha256") or set(
        selection_set.get("protocols", {})
    ) != set(PROTOCOLS):
        raise ValueError("Global six-protocol selection set is not frozen")
    selections = {}
    for protocol, descriptor in selection_set["protocols"].items():
        source = (out / descriptor["path"]).resolve()
        if not source.is_relative_to(out.resolve()) or sha(source) != descriptor["sha256"]:
            raise ValueError("Frozen protocol selection checksum differs")
        selection = read_json(source)
        if (
            selection["protocol"] != protocol
            or selection["fit_role"] != "train"
            or selection["selection_role"] != "validation"
            or set(selection["states"]) != set(METHODS)
            or (formal and selection.get("synthetic_fixture") is not False)
        ):
            raise ValueError("Frozen formal selection identity/roles/methods differ")
        for state in selection["states"].values():
            verify_state(state)
        for method, state in selection["states"].items():
            if state["protocol"] != protocol or state["canonical_method"] != method:
                raise ValueError("Selected state protocol/method identity differs")
            state_path = out / "fit_select" / protocol / f"{method}.state.json"
            if read_json(state_path) != state:
                raise ValueError("Selected state and catalog predictor file differ")
        expected_grid = SEARCH if formal else selection["search"]
        if (
            selection["search"] != expected_grid
            or selection["tie_rule"] != "strict_smaller_first_in_grid_order"
        ):
            raise ValueError("Frozen selection grid/tie contract differs")
        expected_history = [
            (method, taps, alpha)
            for method, taps_grid in (
                ("ridge_tuned", [None]),
                ("fir_tuned", expected_grid["fir_taps"]),
                ("arx_tuned", expected_grid["arx_taps"]),
            )
            for taps in taps_grid
            for alpha in expected_grid["ridge_alpha"]
        ]
        history = selection["candidate_history"]
        if (
            [(row["canonical_method"], row["taps"], row["alpha"]) for row in history]
            != expected_history
            or [row["order"] for row in history] != list(range(len(history)))
            or any(
                row["finite"] is not True
                or row["error"] is not None
                or not np.isfinite(row["validation_score"])
                for row in history
            )
        ):
            raise ValueError("Frozen candidate grid/order/finite evidence differs")
        for method in ("ridge_tuned", "fir_tuned", "arx_tuned"):
            best = min(
                (row for row in history if row["canonical_method"] == method),
                key=lambda row: row["validation_score"],
            )
            if (
                selection["states"][method]["hyperparameters"]
                != {"alpha": best["alpha"], "taps": best["taps"]}
                or selection["selections"][method]["validation_score"] != best["validation_score"]
            ):
                raise ValueError("Frozen state violates its own strict-first VAL selection")
        if {name: digest(state) for name, state in selection["states"].items()} != selection[
            "state_sha256"
        ]:
            raise ValueError("External selected state hashes differ")
        selections[protocol] = selection
    return selection_set, selections


def fit_select(out: Path, contract: dict, manifests: dict, project: Path, *, resume=False) -> dict:
    contract_sha = digest(contract)
    destination = out / "fit_select"
    if destination.exists():
        if not resume:
            raise FileExistsError("Completed fit_select requires explicit resume")
        verify_artifact_manifest(destination, contract_sha=contract_sha)
        verify_selection_set(out)
        print("SKIPS: checksum-verified complete global fit/selection", flush=True)
        return read_json(destination / "stage_summary.json")
    staging = _staging(out, "fit_select")
    write_once(staging / "contract.json", contract)
    if contract.get("config_path"):
        (staging / "config_snapshot.yaml").write_bytes(Path(contract["config_path"]).read_bytes())
    controls, candidates, inventory, method_scope, streams, selections = [], [], [], [], [], {}
    fixture_flags = []
    for protocol in PROTOCOLS:
        frames, registry = load_roles(manifests[protocol], ("train", "validation"), project)
        for rows in registry.values():
            streams.extend({"protocol": protocol, **row} for row in rows)
        candidate_path = staging / protocol / "candidate_attempts.jsonl"
        candidate_path.parent.mkdir()

        def observer(row, path=candidate_path):
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        selection = fit_protocol(protocol, frames["train"], frames["validation"], observer=observer)
        fixture_flags.append(selection.get("synthetic_fixture", False))
        selection.update(
            contract_sha256=contract_sha,
            manifests=manifests[protocol],
            source_sha256=contract["source_sha256"],
            state_sha256={name: digest(state) for name, state in selection["states"].items()},
        )
        selection_path = staging / protocol / "selection.json"
        write_once(selection_path, selection)
        selections[protocol] = {
            "path": f"fit_select/{protocol}/selection.json",
            "sha256": sha(selection_path),
        }
        for method, state in selection["states"].items():
            state_path = staging / protocol / f"{method}.state.json"
            write_once(state_path, state)
            record = {
                "protocol": protocol,
                "information_scope": SCOPE,
                "canonical_method": method,
                "variant": "shared",
                "state_path": f"fit_select/{protocol}/{method}.state.json",
                "state_sha256": sha(state_path),
                "state_content_sha256": digest(state),
                "numeric_sha256": state["numeric_sha256"],
                "learned_state": method != "identity",
                "identity_alias": method == "identity",
                "n": 1,
                "sd": None,
                "train_row_sha256": selection["train_row_sha256"],
                "validation_row_sha256": selection["validation_row_sha256"],
                "fit_rows": selection["fit_rows"],
                "validation_rows": selection["selection_rows"],
                "alpha": state["hyperparameters"]["alpha"],
                "taps": state["hyperparameters"]["taps"],
                "normalization": "none_raw_si",
                "normalizer_count": 0,
                "coefficient_count": state["coefficient_count"],
                "coefficient_bytes": state["coefficient_bytes"],
                "serialized_state_bytes": state_path.stat().st_size,
            }
            controls.append(record)
            inventory.append(record)
            method_scope.append(
                {
                    **{
                        key: record[key]
                        for key in (
                            "protocol",
                            "information_scope",
                            "canonical_method",
                            "variant",
                            "n",
                            "sd",
                        )
                    },
                    "input_scope": "six_LOW_and_same_stream_LOW_history",
                    "identity_routes_state": False,
                    "display_name": "ARX-style LOW lag/delta"
                    if method.startswith("arx")
                    else method,
                    "coefficient_count_interpretation": "stored_learned_scalars_not_statistical_DoF",
                }
            )
        candidates.extend(
            {
                **{key: value for key, value in row.items() if key != "breakdown"},
                "breakdown_json": json.dumps(row["breakdown"], sort_keys=True),
            }
            for row in selection["candidate_history"]
        )
    write_once(staging / "stream_registry.json", streams)
    write_once(
        staging / "selection_set.json",
        {
            "schema_version": 1,
            "analysis_profile": PROFILE,
            "status": "frozen_before_any_test_load",
            "synthetic_fixture": any(fixture_flags),
            "contract_sha256": contract_sha,
            "protocols": selections,
            "counts": {
                "protocols": 6,
                "logical_controls": 54,
                "learned_states": 48,
                "identity_references": 6,
            },
        },
    )
    summary = {
        "stage": "fit_select",
        "synthetic_fixture": any(fixture_flags),
        "contract_sha256": contract_sha,
        "selection_set_sha256": sha(staging / "selection_set.json"),
        "counts": {
            "protocols": 6,
            "logical_controls": len(controls),
            "learned_states": sum(row["learned_state"] for row in controls),
            "identity_references": sum(row["identity_alias"] for row in controls),
            "candidate_solver_calls": len(candidates),
            "fixed_solver_calls": 24,
            "bias_mean_computations": 6,
            "total_solver_calls": len(candidates) + 24,
            "cache_hits": 0,
        },
        "selection_metric": "equal_dataset_equal_pair_gyro_accel_relative_identity",
        "selection_freeze_barrier": "all_six_before_test",
    }
    return finish_stage(
        staging,
        destination,
        summary,
        {
            "control_registry": pd.DataFrame(controls),
            "candidate_history": pd.DataFrame(candidates),
            "shared_state_inventory": pd.DataFrame(inventory),
            "method_scope": pd.DataFrame(method_scope),
        },
    )


def _audit_numeric_json(value, context=""):
    if isinstance(value, dict):
        for key, child in value.items():
            _audit_numeric_json(child, context + "/" + key)
    elif isinstance(value, list):
        for child in value:
            _audit_numeric_json(child, context)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"Nonfinite original numeric evidence: {context}")


def audit_auxiliary_payloads(directory: Path) -> dict:
    """Audit stored numeric bytes, retaining documented null/unsupported cells."""
    counts = {
        "json_files": 0,
        "csv_files": 0,
        "parquet_files": 0,
        "npy_files": 0,
        "null_numeric_cells": 0,
    }
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            _audit_numeric_json(read_json(path), str(path))
            counts["json_files"] += 1
        elif suffix == ".csv":
            print(f"READS: original saved CSV finite evidence {path}", flush=True)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.reader(stream):
                    if any(
                        cell.strip().lower()
                        in {
                            "nan",
                            "+nan",
                            "-nan",
                            "inf",
                            "+inf",
                            "-inf",
                            "infinity",
                            "+infinity",
                            "-infinity",
                        }
                        for cell in row
                    ):
                        raise ValueError("Original CSV contains literal nonfinite numeric payload")
            try:
                table = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                table = pd.DataFrame()
            numeric = table.select_dtypes(include="number")
            counts["null_numeric_cells"] += int(numeric.isna().sum().sum())
            if np.isinf(numeric.to_numpy(float)).any():
                raise ValueError("Original CSV contains infinite numeric payload")
            counts["csv_files"] += 1
        elif suffix == ".parquet":
            print(f"READS: original saved Parquet finite evidence {path}", flush=True)
            table = pd.read_parquet(path)
            numeric = table.select_dtypes(include="number")
            counts["null_numeric_cells"] += int(numeric.isna().sum().sum())
            if np.isinf(numeric.to_numpy(float)).any():
                raise ValueError("Original Parquet contains infinite numeric payload")
            required = [
                column
                for column in numeric
                if column in {*LOW, *HIGH, "timestamp_s"}
                or column.startswith(("pred_", "true_", "model_"))
                or any(column == f"{method}_{axis}" for method in METHODS for axis in AXES)
            ]
            if required and not np.isfinite(table[required].to_numpy(float)).all():
                raise ValueError("Original required prediction/target payload is nonfinite")
            counts["parquet_files"] += 1
        elif suffix == ".npy":
            print(f"READS: original timing TRAIN pool finite evidence {path}", flush=True)
            values = np.load(path, allow_pickle=False)
            if not np.isfinite(values).all():
                raise ValueError("Original timing input pool is nonfinite")
            counts["npy_files"] += 1
    return {
        "status": "passed",
        **counts,
        "null_policy": "original blank/unsupported/optional numeric cells retained; literal NaN/Inf and required prediction/target nonfinites rejected",
    }


def audit_timing_measurements(records: list[dict]) -> None:
    if not records:
        raise ValueError("Original timing report contains no actual measurements")
    for record in records:
        if any(
            record.get(key) != expected
            for key, expected in {"warmup": 30, "iterations": 100, "repeats": 3}.items()
        ):
            raise ValueError("Original per-record timing budget differs")
        values = np.asarray(record["raw_wall_ms"], float)
        loops = np.asarray(record["raw_throughput_loop_seconds"], float)
        if (
            values.shape != (3, 100)
            or loops.shape != (3,)
            or not np.isfinite(values).all()
            or not np.isfinite(loops).all()
            or np.any(values <= 0)
            or np.any(loops <= 0)
        ):
            raise ValueError("Original timing raw loops/measurements are incomplete/nonfinite")
        observed = record["wall"]
        expected = {
            "count": values.size,
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.quantile(values, 0.5)),
            "p95_ms": float(np.quantile(values, 0.95)),
        }
        if any(
            not np.isclose(observed[key], value, rtol=1e-10, atol=1e-12)
            for key, value in expected.items()
        ):
            raise ValueError("Original timing statistics do not reproduce raw samples")
        throughput = record["batch_size"] * 300 / loops.sum()
        if not np.isclose(
            record["loop_throughput_windows_per_second"], throughput, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Original timing throughput does not reproduce actual loop")


def audit_auxiliary_profiles(gate: dict, root: Path) -> dict:
    """Actual original stage coverage and frozen timing budgets, after S2 barrier."""
    core_sha = sha(root / "pipeline.json")
    core = read_json(root / "pipeline.json")
    final_ids = {job["id"] for job in core["jobs"].values() if job["phase"].startswith("final_")}
    if len(final_ids) != 558:
        raise ValueError("Original core final identity roster differs")
    results = {}
    for label, prior in gate["prior"].items():
        prior_root = Path(prior["path"]).parent
        stages = {}
        for name in prior["state"]["stages"]:
            directory = prior_root / name
            finite = audit_auxiliary_payloads(directory)
            item = {"status": "passed", "path": str(directory), "numeric_payload_audit": finite}
            if name in {"transfer", "downstream", "classical_latency"}:
                contract = read_json(directory / "contract.json")
                if (
                    contract.get("pipeline_sha256") != core_sha
                    or Path(contract.get("pipeline", contract.get("pipeline_path", ""))).resolve()
                    != root.resolve()
                ):
                    raise ValueError("Original auxiliary stage refers to a different core")
            if name == "transfer":
                report = read_json(directory / "transfer_metrics.json")
                records = report.get("source_model_records", [])
                tuples = [
                    (row["direction"], row["model_family"], row["variant"], row["seed"])
                    for row in records
                ]
                families = {
                    candidate["id"]
                    for candidate in core["contract"]["config"]["candidates"]
                    if "paired" in candidate["tasks"]
                }
                expected_tuples = {
                    (direction, family, variant, seed)
                    for direction in ("alimu_to_penn", "penn_to_alimu")
                    for family in families
                    for variant in ("default", "tuned")
                    for seed in (20261010, 20261011, 20261012)
                }
                if (
                    report.get("status") != "complete"
                    or contract.get("nonfinal_debug_subset")
                    or len(tuples) != 120
                    or set(tuples) != expected_tuples
                    or report.get("records") != len(tuples)
                    or contract.get("expected_records") != len(tuples)
                ):
                    raise ValueError("Original transfer requires actual120 unique formal records")
                item.update(
                    records=len(records),
                    unique_records=len(set(tuples)),
                    identity_fields=["direction", "model_family", "variant", "seed"],
                )
            elif name == "downstream":
                report = read_json(directory / "complete.json")
                units = report.get("unit_manifests", {})
                expected = {"penn_loso_af", "penn_loso_as", "penn_loso_bf", "penn_only", "pooled"}
                if report.get("status") != "complete" or set(units) != expected:
                    raise ValueError("Original downstream must retain all5 exact views")
                unit_status = {}
                for protocol, expected_sha in units.items():
                    path = directory / "units" / protocol / "artifact_manifest.json"
                    if sha(path) != expected_sha or read_json(path).get("status") != "complete":
                        raise ValueError("Original downstream unit completion changed")
                    metadata = read_json(path.parent / "metrics.json")
                    expected_recording = {
                        "penn_loso_af": "af",
                        "penn_loso_as": "as",
                        "penn_loso_bf": "bf",
                        "penn_only": "bs",
                        "pooled": "bs",
                    }[protocol]
                    if (
                        metadata.get("protocol") != protocol
                        or metadata.get("recording") != expected_recording
                        or metadata.get("neural_optimizer_runs") != 60
                        or metadata.get("unique_classical_methods") != 9
                        or metadata.get("method_variants_total") != 30
                        or metadata.get("stream_count") != 70
                        or len(metadata.get("model_records", [])) != 60
                    ):
                        raise ValueError(
                            "Original downstream actual method/stream/recording coverage differs"
                        )
                    unit_status[protocol] = {
                        "status": "complete",
                        "manifest_sha256": expected_sha,
                        "recording": expected_recording,
                        "neural_optimizer_runs": len(metadata["model_records"]),
                        "method_variants": metadata["method_variants_total"],
                        "streams": metadata["stream_count"],
                    }
                item.update(views=len(unit_status), units=unit_status)
            elif name in {"latency_cpu", "latency_gpu"}:
                report = read_json(directory / "latency.json")
                saved = read_json(directory / "latency_state.json")
                settings = report.get("timing_protocol", {})
                expected = {
                    "warmup": 30,
                    "iterations": 100,
                    "repeats": 3,
                    "input_seed": 20261024,
                    "batch_sizes": [1] if name == "latency_cpu" else [1, 128],
                    "cpu_threads": 1 if name == "latency_cpu" else 4,
                    "end_to_end": False,
                }
                if (
                    report.get("status") != "complete"
                    or saved.get("status") != "complete"
                    or report.get("entries") != 558
                    or report.get("source_pipeline_sha256") != core_sha
                    or any(settings.get(key) != value for key, value in expected.items())
                    or any(
                        group.get("status") != "complete"
                        for group in saved.get("groups", {}).values()
                    )
                ):
                    raise ValueError("Original neural latency coverage/frozen profile differs")
                if (name == "latency_cpu" and settings.get("device") != "cpu") or (
                    name == "latency_gpu" and not str(settings.get("device", "")).startswith("cuda")
                ):
                    raise ValueError("Original neural latency device differs")
                mapping = read_json(directory / "architecture_mapping.json")
                if (
                    len(mapping) != 558
                    or {row["entry_id"] for row in mapping} != final_ids
                    or not saved.get("groups")
                ):
                    raise ValueError("Original neural latency actual final mappings differ")
                audit_timing_measurements(report.get("records", []))
                item.update(
                    entries=len(mapping),
                    measured_records=len(report.get("records", [])),
                    timing_protocol=settings,
                )
            elif name == "accuracy_tables":
                report = read_json(directory / "summary.json")
                if (
                    report.get("status") != "complete"
                    or report.get("runs") != 558
                    or report.get("source_pipeline_sha256") != core_sha
                    or Path(report.get("pipeline_root", "")).resolve() != root.resolve()
                ):
                    raise ValueError("Original accuracy table coverage/core differs")
                registry = pd.read_csv(directory / "run_registry.csv")
                if len(registry) != 558 or set(registry.job_id) != final_ids:
                    raise ValueError("Original accuracy actual final job registry differs")
                item.update(runs=len(registry))
            elif name == "classical_latency":
                report = read_json(directory / "latency.json")
                expected = {
                    "seed": 20261024,
                    "warmup": 30,
                    "iterations": 100,
                    "repeats": 3,
                    "batch_sizes": [1, 128],
                    "paired_window_sizes": [80, 160, 320],
                    "navigation_window_sizes": [80],
                    "cpu_threads": 1,
                }
                environment = report.get("environment", {})
                pools = environment.get("native_thread_pools", [])
                if (
                    report.get("status") != "complete"
                    or report.get("final_run_mappings") != 558
                    or contract.get("timing") != expected
                    or environment.get("threadpoolctl_version") != "3.6.0"
                    or not pools
                    or any(pool.get("num_threads") != 1 for pool in pools)
                ):
                    raise ValueError(
                        "Original classical latency coverage/BLAS/frozen profile differs"
                    )
                mapping = pd.read_csv(directory / "final_run_mapping.csv")
                if len(mapping) != 558 or set(mapping.job_id) != final_ids:
                    raise ValueError("Original classical latency actual final mappings differ")
                audit_timing_measurements(report.get("records", []))
                item.update(
                    final_run_mappings=len(mapping),
                    measured_records=len(report.get("records", [])),
                    timing_protocol=contract["timing"],
                    native_thread_pools=pools,
                )
            else:
                raise ValueError("Unexpected original auxiliary stage")
            stages[name] = item
        results[label] = {
            "status": "passed",
            "state_path": prior["path"],
            "state_sha256": prior["sha256"],
            "stages": stages,
        }
    return results


def full_original_audit(state, root, project, gate):
    """S2 audit is callable only by signal stage after verify_selection_set."""
    helper = original_helpers()
    finals, evidence = helper.verify_complete_pipeline(state, root, project)
    for prior in gate["prior"].values():
        prior_state = prior["state"]
        prior_root = Path(prior["path"]).parent
        for name, stage in prior_state["stages"].items():
            directory = prior_root / name
            actual = {
                path.relative_to(directory).as_posix(): sha(path)
                for path in sorted(directory.rglob("*"))
                if path.is_file()
            }
            if actual != stage.get("artifact_sha256"):
                raise ValueError(f"Original auxiliary full output inventory changed: {name}")
    # Validate every original final numerical payload, including navigation,
    # without loading checkpoints or running any neural inference.
    for job in finals:
        directory = helper.recorded_path(job["run_dir"], root, project) / "evaluation"
        _audit_numeric_json(read_json(directory / "metrics.json"))
        for path in directory.glob("*predictions.parquet"):
            print(f"READS: post-freeze original finite prediction audit {path}", flush=True)
            frame = pd.read_parquet(path)
            numeric = frame.select_dtypes(include="number")
            if numeric.empty or not np.isfinite(numeric.to_numpy(float)).all():
                raise ValueError("Original prediction numeric payload is nonfinite")
    evidence["original_auxiliary_validation"] = audit_auxiliary_profiles(gate, root)
    return evidence


def support_match(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Compare original order and every physical field; recover only missing clock."""
    columns = [
        column
        for column in expected
        if column in PHYSICAL
        or column.startswith("source_")
        or column in {"pair_id", "segment_id", "reference_recording_id", "recording_id"}
    ]
    missing = set(columns) - set(actual)
    if missing - {"source_timestamp_s"}:
        raise ValueError(f"Neural output omits physical support columns: {sorted(missing)}")
    if len(actual) != len(expected):
        raise ValueError("Neural/shared physical support row count differs")
    available = [column for column in columns if column in actual]
    try:
        pd.testing.assert_frame_equal(
            expected[available].reset_index(drop=True),
            actual[available].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError("Neural/shared original physical order/columns differ") from error
    # Clock recovery is allowed only after exact original keys, LOW/HIGH and
    # metadata match. The canonical manifest-bound clock is the sole source.
    return {
        "rows": len(expected),
        "physical_sha256": frame_sha(expected, columns),
        "columns": columns,
        "missing_clock_recovered_from_exact_canonical_rows": bool(missing),
    }


def metric_rows(protocol, role, method, values, frame):
    result = []
    base = {
        "protocol": protocol,
        "information_scope": SCOPE,
        "canonical_method": method,
        "variant": "shared",
        "split": role,
        "n": 1,
        "sd": None,
    }
    for granularity, keys in (
        ("dataset", ["dataset_id"]),
        ("pair", IDENTITY),
        ("sequence", [*IDENTITY, "sequence_id"]),
    ):
        for identity, positions in frame.groupby(keys, sort=True).indices.items():
            identity = identity if isinstance(identity, tuple) else (identity,)
            group = frame.iloc[positions]
            measured = _pair_metrics(values[positions], group[HIGH].to_numpy(float))
            original = _pair_metrics(group[LOW].to_numpy(float), group[HIGH].to_numpy(float))
            for metric, value in measured.items():
                if isinstance(value, list):
                    entries = [
                        (metric + "_" + axis, number)
                        for axis, number in zip(AXES, value, strict=True)
                    ]
                else:
                    entries = [(metric, value)]
                for name, number in entries:
                    result.append(
                        {
                            **base,
                            "metric_scope": granularity,
                            **dict(zip(keys, identity, strict=True)),
                            "metric": name,
                            "value": number,
                        }
                    )
            for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2"):
                result.append(
                    {
                        **base,
                        "metric_scope": granularity,
                        **dict(zip(keys, identity, strict=True)),
                        "metric": metric + "_relative_identity",
                        "value": measured[metric] / max(original[metric], 1e-12),
                    }
                )
    result.append(
        {
            **base,
            "metric_scope": "protocol_macro",
            "metric": "grouped_relative_score",
            "value": macro_breakdown(values, frame)["score"],
        }
    )
    return result


def _bank_proof(protocol, saved, selection, manifests, frames):
    """Read the exact original cache reference, never search basename/latest."""
    contract = saved["contract"]
    payload = contract["train_validation"]
    columns = [*IDENTITY, "sequence_id", "timestamp_s", *LOW, *HIGH]

    def original_digest(frame):
        ordered = frame.sort_values(["sequence_id", "timestamp_s"], kind="stable")
        return hashlib.sha256(
            pd.util.hash_pandas_object(ordered[columns], index=False).to_numpy().tobytes()
        ).hexdigest()

    if (
        contract["protocol_id"] != protocol
        or contract["task"] != "paired"
        or any(
            contract["manifests"][key]["sha256"] != value["sha256"]
            for key, value in manifests.items()
        )
    ):
        raise ValueError("Original bank cache protocol/source manifest contract differs")
    if (
        payload["train_sha256"] != original_digest(frames["train"])
        or payload["validation_sha256"] != original_digest(frames["validation"])
        or payload["train_rows"] != len(frames["train"])
        or payload["validation_rows"] != len(frames["validation"])
    ):
        raise ValueError("Original bank TRAIN/VAL physical row support differs")
    if any(contract["search"][key] != SEARCH[key] for key in SEARCH):
        raise ValueError("Original bank grid/order differs")
    for role in ("train", "validation"):
        original_order = pd.concat(
            [
                group.sort_values("timestamp_s").reset_index(drop=True)
                for _, group in frames[role].groupby("sequence_id", sort=False)
            ],
            ignore_index=True,
        )
        if frame_sha(original_order, columns) != frame_sha(frames[role], columns):
            raise ValueError("Shared support differs from the original bank loader row order")
    return {
        "same_protocol": True,
        "train_rows_exact": True,
        "validation_rows_exact": True,
        "grid_order_exact": True,
        "tie_rule_exact": True,
        "bank_own_selection_valid": True,
        "bank_loader_order_verified": True,
        "bank_cache_state_sha256": saved["state_sha256"],
        "shared_train_order_sha256": selection["train_row_sha256"],
        "shared_validation_order_sha256": selection["validation_row_sha256"],
    }


def _validate_bank_selection(pair):
    for method in ("ridge_tuned", "fir_tuned", "arx_tuned"):
        selected = pair[method]
        expected = [
            (taps, alpha)
            for taps in (
                [None]
                if method == "ridge_tuned"
                else SEARCH["fir_taps" if method == "fir_tuned" else "arx_taps"]
            )
            for alpha in SEARCH["ridge_alpha"]
        ]
        history = selected["history"]
        if [(row.get("taps"), row["alpha"]) for row in history] != expected or any(
            not np.isfinite(row["validation_score"]) for row in history
        ):
            raise ValueError("Bank candidate grid/finite history differs")
        best = min(history, key=lambda row: row["validation_score"])
        if (
            selected["alpha"] != best["alpha"]
            or selected.get("taps") != best.get("taps")
            or selected["validation_score"] != best["validation_score"]
        ):
            raise ValueError("Bank violates its own strict-first minimum selection contract")


def signal_evaluation(
    out, contract, manifests, state, finals, gate, root, project, *, resume=False
):
    selection_set, selections = verify_selection_set(out)
    contract_sha = digest(contract)
    if selection_set["contract_sha256"] != contract_sha:
        raise ValueError("Current source/input/config contract differs from frozen selection")
    destination = out / "signal_evaluation"
    if destination.exists():
        if not resume:
            raise FileExistsError("Completed signal stage requires resume")
        verify_artifact_manifest(destination, contract_sha=contract_sha)
        return read_json(destination / "stage_summary.json")
    audit = full_original_audit(state, root, project, gate)
    staging = _staging(out, "signal_evaluation")
    write_once(staging / "original_full_audit.json", audit)
    helper = original_helpers()
    metrics, signals, neural, aliases = [], [], [], []
    for protocol in PROTOCOLS:
        frames, registry = load_roles(manifests[protocol], ("train", "validation", "test"), project)
        selection = selections[protocol]
        if (
            frame_sha(frames["train"]) != selection["train_row_sha256"]
            or frame_sha(frames["validation"]) != selection["validation_row_sha256"]
        ):
            raise ValueError("Post-freeze TRAIN/VAL row identity changed")
        jobs = [job for job in finals if job["cell"].split("/", 1)[0] == protocol]
        bank_state, bank_cache = None, None
        original_bank_predictions = {}
        for job in jobs:
            directory = helper.recorded_path(job["run_dir"], root, project) / "evaluation"
            saved = read_json(directory / "baseline_train_state.json")
            if bank_state is not None and saved != bank_state:
                raise ValueError("Known-device bank differs across duplicated neural job mappings")
            bank_state = saved
            provenance = read_json(directory / "provenance.json")
            cache = provenance["classical_cache"]
            cache_path = helper.recorded_path(cache["path"], root, project)
            if sha(cache_path) != cache["sha256"]:
                raise ValueError("Original bank cache source SHA differs")
            bank_cache = read_json(cache_path)
            if (
                digest(bank_cache["states"]) != bank_cache["state_sha256"]
                or bank_cache["states"] != saved
            ):
                raise ValueError("Original bank cache state identity differs")
            for role in ("validation", "test"):
                path = directory / f"{role}_predictions.parquet"
                print(f"READS: exact all-family/variant/seed support {path}", flush=True)
                actual = pd.read_parquet(path)
                proof = support_match(frames[role], actual)
                bank_columns = [f"{method}_{axis}" for method in METHODS for axis in AXES]
                if any(column not in actual for column in bank_columns):
                    raise ValueError(
                        "Original known-device predictions omit a declared classical method"
                    )
                if role in original_bank_predictions:
                    np.testing.assert_array_equal(
                        original_bank_predictions[role], actual[bank_columns].to_numpy(float)
                    )
                else:
                    original_bank_predictions[role] = actual[bank_columns].to_numpy(float)
                neural.append(
                    {
                        "protocol": protocol,
                        "job_id": job["id"],
                        "cell": job["cell"],
                        "phase": job["phase"],
                        "seed": job["seed"],
                        "split": role,
                        "prediction_path": str(path),
                        "prediction_sha256": sha(path),
                        **{key: value for key, value in proof.items() if key != "columns"},
                        "physical_columns_json": json.dumps(proof["columns"]),
                    }
                )
        proof = _bank_proof(protocol, bank_cache, selection, manifests[protocol], frames)
        single = protocol in {"penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf"}
        if single and (
            len(bank_state["pairs"]) != 1 or len(frames["train"][IDENTITY].drop_duplicates()) != 1
        ):
            raise ValueError("Penn-only alias protocol is not actually single-pair")
        if single:
            pair = bank_state["pairs"][0]
            _validate_bank_selection(pair)
            normalized = normalize_bank_pair(protocol, pair)
            for method_index, method in enumerate(METHODS):
                audit_row = alias_audit(
                    selection["states"][method],
                    normalized[method],
                    frames["validation"],
                    contract_proof=proof,
                )
                if method.endswith("_tuned"):
                    left_history = [
                        row
                        for row in selection["candidate_history"]
                        if row["canonical_method"] == method
                    ]
                    right_history = pair[method]["history"]
                    left_scores = [row["validation_score"] for row in left_history]
                    right_scores = [row["validation_score"] for row in right_history]
                    history_exact = left_scores == right_scores
                    history_equivalent = len(left_scores) == len(right_scores) and bool(
                        np.allclose(left_scores, right_scores, rtol=1e-10, atol=1e-12)
                    )
                    audit_row.update(
                        candidate_score_history_exact=history_exact,
                        candidate_score_history_equivalent=history_equivalent,
                        shared_selected_validation_score=selection["selections"][method][
                            "validation_score"
                        ],
                        bank_selected_validation_score=pair[method]["validation_score"],
                    )
                    if not history_exact and audit_row["alias_status"] == "exact_alias":
                        audit_row["alias_status"] = (
                            "numerically_equivalent" if history_equivalent else "not_alias"
                        )
                    if not history_equivalent:
                        audit_row["alias_status"] = "not_alias"
                prediction_status = []
                for role in ("validation", "test"):
                    original_values = original_bank_predictions[role][
                        :, method_index * 6 : (method_index + 1) * 6
                    ]
                    replay = predict_state(normalized[method], frames[role])
                    if not np.allclose(original_values, replay, rtol=1e-10, atol=1e-12):
                        raise ValueError(
                            "Original bank saved predictions differ from their own frozen state"
                        )
                    shared_values = predict_state(selection["states"][method], frames[role])
                    exact = np.array_equal(shared_values, original_values)
                    equivalent = np.allclose(shared_values, original_values, rtol=1e-10, atol=1e-12)
                    prediction_status.append((exact, equivalent))
                    audit_row[role + "_prediction_exact"] = bool(exact)
                    audit_row[role + "_prediction_equivalent"] = bool(equivalent)
                    audit_row[role + "_max_prediction_absolute_difference"] = float(
                        np.max(np.abs(shared_values - original_values))
                    )
                if audit_row["alias_status"] == "exact_alias" and not all(
                    item[0] for item in prediction_status
                ):
                    audit_row["alias_status"] = (
                        "numerically_equivalent"
                        if all(item[1] for item in prediction_status)
                        else "not_alias"
                    )
                elif audit_row["alias_status"] == "numerically_equivalent" and not all(
                    item[1] for item in prediction_status
                ):
                    audit_row["alias_status"] = "not_alias"
                aliases.append(audit_row)
        else:
            for method in METHODS:
                aliases.append(
                    {
                        "protocol": protocol,
                        "information_scope": SCOPE,
                        "canonical_method": method,
                        "variant": "shared",
                        "alias_status": "common_identity"
                        if method == "identity"
                        else "not_applicable_multi_pair_bank",
                        "n": 1,
                        "sd": None,
                    }
                )
        for role in ("validation", "test"):
            frame = frames[role]
            table = frame.copy()
            for method in METHODS:
                predictor = selection["states"][method]
                values = predict_state(predictor, frame)
                for axis, value in zip(AXES, values.T, strict=True):
                    table[f"{method}_{axis}"] = value
                metrics.extend(metric_rows(protocol, role, method, values, frame))
                signals.append(
                    {
                        "protocol": protocol,
                        "information_scope": SCOPE,
                        "canonical_method": method,
                        "variant": "shared",
                        "split": role,
                        "rows": len(frame),
                        "row_sha256": frame_sha(frame),
                        "state_sha256": digest(predictor),
                        "selection_set_sha256": sha(out / "fit_select/selection_set.json"),
                        "prediction_path": f"signal_evaluation/{protocol}/{role}_predictions.parquet",
                        "n": 1,
                        "sd": None,
                        "identity_alias": method == "identity",
                    }
                )
            directory = staging / protocol
            directory.mkdir(exist_ok=True)
            table.to_parquet(directory / f"{role}_predictions.parquet", index=False)
            write_once(directory / f"{role}_streams.json", registry[role])
    if len(neural) != 720 or len(signals) != 108:
        raise ValueError("Logical neural/support/shared split coverage differs from 360/720/108")
    return finish_stage(
        staging,
        destination,
        {
            "stage": "signal_evaluation",
            "synthetic_fixture": any(
                selection.get("synthetic_fixture", False) for selection in selections.values()
            ),
            "contract_sha256": contract_sha,
            "selection_set_sha256": sha(out / "fit_select/selection_set.json"),
            "counts": {
                "logical_signal_slots": len(signals),
                "paired_job_mappings": len({row["job_id"] for row in neural}),
                "neural_split_support_references": len(neural),
                "alias_audits": len(aliases),
            },
        },
        {
            "metrics_long": pd.DataFrame(metrics),
            "signal_registry": pd.DataFrame(signals),
            "neural_support_registry": pd.DataFrame(neural),
            "alias_audit": pd.DataFrame(aliases),
        },
    )


def run(args):
    print(
        f"READS: exact core {args.pipeline_root}, original stage metadata, frozen manifests/config; WRITES: {args.out}/{args.stage}; SKIPS: neural inference, training, TEST selection",
        flush=True,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if (
        config.get("analysis_profile") != PROFILE
        or config.get("protocols") != list(PROTOCOLS)
        or config.get("search") != SEARCH
        or config.get("synthetic_fixture", False)
    ):
        raise ValueError("Formal shared config/profile/protocols/exact grids required")
    project, root, out = (
        args.project_root.resolve(),
        args.pipeline_root.resolve(),
        args.out.resolve(),
    )
    state, finals, gate = metadata_gate(
        root, project, args.aux1_state, args.preceding_state, config
    )
    manifests = {
        protocol: protocol_manifests(protocol, finals, root, project) for protocol in PROTOCOLS
    }
    contract = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "pipeline_root": str(root),
        "pipeline_sha256": sha(root / "pipeline.json"),
        "config_sha256": sha(args.config),
        "config_path": str(args.config.resolve()),
        "config": config,
        "source_sha256": source_contract(project),
        "manifests": manifests,
        "aux1_state_sha256": sha(args.aux1_state),
        "aux2_state_sha256": sha(args.preceding_state),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "information_scope": SCOPE,
        "frozen_bundle_identity": gate["bundle_identity"],
    }
    if args.stage == "fit_select":
        return fit_select(out, contract, manifests, project, resume=args.resume)
    return signal_evaluation(
        out, contract, manifests, state, finals, gate, root, project, resume=args.resume
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "pipeline-root", "aux1-state", "preceding-state", "out", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage", choices=("fit_select", "signal_evaluation"), required=True)
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "selection_set_sha256": result["selection_set_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
