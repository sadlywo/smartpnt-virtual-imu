"""Metadata-first, fail-closed coordinator for the independent shared controls.

No fitting or numeric payload audit occurs while the exact original workflow is
active. All six protocol selections freeze before signal_evaluation can audit
original payloads or load TEST. Every child runs sequentially, without a shell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

PROFILE = "reviewer_shared_classical_v1"
CORE_RUN_ID = "reviewer_hpo_20261024_122113_893148"
SOURCE_FINGERPRINT = "b6a2b7a3f3bab8ce16d4f1225175cf19415736d4afe2a398fa2bee2415e019ec"
DATASET_SHA256 = "5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260"
AUX1_MANIFEST_SHA256 = "91be9fda2d9a48320636789fc5dd6715e631503982b9969219aa93d5959b5edf"
AUX2_MANIFEST_SHA256 = "152cf3458de541426386672357cad8a876211a975a088e3f1609322238623e11"
FINAL_SEEDS = (20261010, 20261011, 20261012)
PROTOCOLS = ("pooled", "alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf")
METHODS = (
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "ridge_tuned",
    "fir_tuned",
    "arx_tuned",
)
PRIOR_STAGES = {
    "aux1": {"transfer", "downstream", "latency_gpu", "latency_cpu"},
    "aux2": {"accuracy_tables", "classical_latency"},
}
STAGES = ("fit_select", "signal_evaluation", "downstream", "cpu_latency")
REQUIRED_ROLES = {
    "control_registry",
    "candidate_history",
    "shared_state_inventory",
    "method_scope",
    "signal_registry",
    "neural_support_registry",
    "metrics_long",
    "alias_audit",
    "downstream_summary",
    "downstream_windows",
    "downstream_segments",
    "state_inventory",
    "bank_parameter_totals",
    "latency_summary",
    "latency_raw",
    "workload_latency",
}


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    print(f"READS: {path}", flush=True)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(f"WRITES: {path}", flush=True)


def write_once(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"WRITES: {path}", flush=True)


def safe_path(root, name):
    """Portable/case-safe manifest member, with no traversal, drive or symlink."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Manifest path must be a portable relative path")
    parts = PurePosixPath(name).parts
    if (
        PurePosixPath(name).is_absolute()
        or PurePosixPath(name).as_posix() != name
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("Unsafe manifest path")
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError("Manifest path escapes its explicit root")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Manifest symbolic links are not portable")
        if current.parent.exists() and part not in {item.name for item in current.parent.iterdir()}:
            raise ValueError(f"Manifest path is missing or differs in case: {name}")
    return path


def _members(files):
    if (
        not isinstance(files, dict)
        or not files
        or len({name.casefold() for name in files}) != len(files)
    ):
        raise ValueError("Manifest members are empty or collide after case folding")
    return files


def bundle_fingerprint(source_map):
    # This is the bundle's indented-map aggregate, not pipeline.contract.source_sha256.
    payload = json.dumps(source_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_bundle_sources(project):
    path = project / "bundle_manifest.json"
    manifest = read_json(path)
    sources = {}
    for entry in manifest["entries"]:
        if entry["kind"] != "source":
            continue  # No data entry is opened or hashed before the global freeze.
        name = entry["path"]
        if name in sources:
            raise ValueError("Duplicate bundle source path")
        source = safe_path(project, name)
        if sha(source) != entry["sha256"]:
            raise ValueError(f"Frozen bundle source checksum differs: {name}")
        sources[name] = entry["sha256"]
    _members(sources)
    actual = bundle_fingerprint(sources)
    if actual != SOURCE_FINGERPRINT or manifest.get("source_fingerprint") != actual:
        raise ValueError("Frozen bundle aggregate source fingerprint differs")
    if manifest.get("dataset_manifest_sha256") != DATASET_SHA256:
        raise ValueError("Frozen bundle dataset manifest identity differs")
    return {
        "manifest_path": str(path),
        "manifest_sha256": sha(path),
        "source_fingerprint": actual,
        "source_map": sources,
        "dataset_manifest_sha256": DATASET_SHA256,
    }


def verify_source_manifest(aux, expected):
    path = aux / "aux_manifest.json"
    if sha(path) != expected:
        raise ValueError("Auxiliary source manifest checksum differs")
    manifest = read_json(path)
    for name, checksum in _members(manifest["files"]).items():
        source = safe_path(aux, name)
        # Numeric runtime data is not a source manifest member.
        if source.suffix.lower() not in {
            ".py",
            ".sh",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".toml",
            ".whl",
            ".txt",
        } and source.name not in {"LICENSE", "METADATA", "RECORD", "WHEEL"}:
            raise ValueError("Auxiliary manifest contains an unexpected non-source payload")
        if sha(source) != checksum:
            raise ValueError(f"Auxiliary source checksum differs: {name}")
    return {"path": str(path), "sha256": expected, "files": manifest["files"]}


def _narrow_sources(project, source_map, bundle_sources):
    if not source_map:
        raise ValueError("Missing original pipeline source map")
    for name, checksum in source_map.items():
        source = Path(name)
        if source.is_absolute():
            if not source.resolve().is_relative_to(project):
                raise ValueError("Original source path is outside the exact core project")
            relative = source.resolve().relative_to(project).as_posix()
        else:
            relative = str(name).replace("\\", "/")
        if (
            bundle_sources.get(relative) != checksum
            or sha(safe_path(project, relative)) != checksum
        ):
            raise ValueError("Original narrower pipeline source map differs from frozen bundle")


def _core_ready(core, workflow):
    if core.get("status") == "failed":
        raise RuntimeError("Original core failed; shared controls blocked")
    lines = [line.strip() for line in workflow.splitlines() if line.strip()]
    exits = [line for line in lines if line.startswith("WORKFLOW_EXIT=")]
    if exits and exits[-1] != "WORKFLOW_EXIT=0":
        raise RuntimeError("Original workflow exited unsuccessfully")
    if core.get("status") != "complete":
        return False
    cells = core.get("cells", {})
    finals = [
        job for job in core.get("jobs", {}).values() if job.get("phase", "").startswith("final_")
    ]
    expected = {
        (cell, "final_" + variant, seed)
        for cell in cells
        for variant in ("default", "tuned")
        for seed in FINAL_SEEDS
    }
    observed = [(job["cell"], job["phase"], job["seed"]) for job in finals]
    if (
        core.get("smoke") is not False
        or core.get("selection_uses_test") is not False
        or core.get("final_seeds") != list(FINAL_SEEDS)
        or set(core.get("final_variants", [])) != {"default", "tuned"}
        or len(cells) != 93
        or any("selected" not in value for value in cells.values())
        or len(finals) != 558
        or len(set(observed)) != 558
        or set(observed) != expected
        or any(
            job.get("status") != "complete" or job.get("evaluation_status") != "complete"
            for job in finals
        )
    ):
        raise ValueError(
            "Original complete metadata lacks exact 93 selected/558 unique evaluated identities"
        )
    if (
        "REVIEWER_CORE_PASSED" not in lines
        or not exits
        or exits[-1] != "WORKFLOW_EXIT=0"
        or not lines
        or lines[-1].strip() != "WORKFLOW_EXIT=0"
        or lines.index("REVIEWER_CORE_PASSED") >= len(lines) - 1
    ):
        raise ValueError("Original core successful terminal markers are missing")
    return True


def _terminal(path, marker):
    print(f"READS: terminal marker metadata {path}", flush=True)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[-1] != marker:
        raise ValueError(f"Original terminal marker missing: {marker}")


def metadata_gate(project, pipeline, aux1_path, aux2_path, *, require_aux_terminal=False):
    """Inspect metadata/source/config only. Never read/hash old numeric outputs."""
    project, pipeline = Path(project).resolve(), Path(pipeline).resolve()
    if pipeline.name != CORE_RUN_ID or pipeline.parent != project / "runs":
        raise ValueError("Shared queue requires the exact original run path")
    bundle = verify_bundle_sources(project)
    core = read_json(pipeline / "pipeline.json")
    _narrow_sources(
        project, core.get("contract", {}).get("source_sha256", {}), bundle["source_map"]
    )
    if core.get("contract", {}).get("config", {}).get("dataset_manifest_sha256") != DATASET_SHA256:
        raise ValueError("Pipeline canonical dataset manifest identity differs")
    workflow_path = project / "runs/reviewer_workflow_status.log"
    workflow = workflow_path.read_text(encoding="utf-8") if workflow_path.exists() else ""
    ready = _core_ready(core, workflow)
    if ready:
        for cell, record in core["cells"].items():
            selection = safe_path(pipeline, "selections/" + cell.replace("/", "__") + ".json")
            if sha(selection) != record.get("selection_sha256"):
                raise ValueError("Original selection metadata checksum differs")
        # Parse YAML only after original core completion. It is configuration, not metrics.
        import yaml

        if (
            yaml.safe_load((pipeline / "matrix_config.yaml").read_text(encoding="utf-8"))
            != core["contract"]["config"]
        ):
            raise ValueError("Original matrix configuration differs from pipeline contract")
    priors = {}
    for label, state_path, expected_sha in (
        ("aux1", Path(aux1_path).resolve(), AUX1_MANIFEST_SHA256),
        ("aux2", Path(aux2_path).resolve(), AUX2_MANIFEST_SHA256),
    ):
        if not state_path.exists():
            ready = False
            priors[label] = {"status": "not_started", "path": str(state_path)}
            continue
        prior = read_json(state_path)
        if prior.get("status") == "failed":
            raise RuntimeError(f"Original {label} failed; shared controls blocked")
        if prior.get("status") != "complete":
            ready = False
            priors[label] = {"status": prior.get("status"), "path": str(state_path)}
            continue
        contract = prior["contract"]
        if (
            Path(contract["project"]).resolve() != project
            or Path(contract["pipeline"]).resolve() != pipeline
            or contract.get("aux_manifest_sha256") != expected_sha
        ):
            raise ValueError(f"Original {label} source/run contract differs")
        if set(prior.get("stages", {})) != PRIOR_STAGES[label] or any(
            item.get("status") != "complete" for item in prior["stages"].values()
        ):
            raise ValueError(f"Original {label} complete state lacks required stages")
        if core.get("status") == "complete" and prior.get("core_pipeline_sha256") != sha(
            pipeline / "pipeline.json"
        ):
            raise ValueError(f"Original {label} references a different core pipeline checksum")
        aux = Path(contract["aux"]).resolve()
        source_manifest = verify_source_manifest(aux, expected_sha)
        if require_aux_terminal:
            filename, marker = (
                ("postprocessing_console.log", "REVIEWER_AUXILIARY_PASSED")
                if label == "aux1"
                else ("final_analysis_console.log", "REVIEWER_FINAL_ANALYSIS_PASSED")
            )
            _terminal(aux / filename, marker)
        priors[label] = {
            "status": "complete",
            "path": str(state_path),
            "sha256": sha(state_path),
            "source_manifest": source_manifest,
        }
    return {
        "ready": ready,
        "audit_scope": "metadata_first_no_numeric_payload_audit",
        "core_pipeline_sha256": sha(pipeline / "pipeline.json") if ready else None,
        "core_pipeline_path": str(pipeline / "pipeline.json"),
        "bundle": bundle,
        "priors": priors,
    }


def verify_stage(out, name):
    directory = Path(out) / name
    manifest = read_json(directory / "artifact_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("root_kind") != "aux3_run":
        raise ValueError("Stage manifest is not a complete aux3 inventory")
    files = _members(manifest["files"])
    actual = {
        path.relative_to(out).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != directory / "artifact_manifest.json"
    }
    if actual != set(files):
        raise ValueError("Completed stage file inventory differs")
    for member, checksum in files.items():
        path = safe_path(out, member)
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != checksum:
            raise ValueError(f"Completed stage checksum changed: {member}")
    summary = read_json(directory / "stage_summary.json")
    if summary.get("status") != "complete" or summary.get("stage") != name:
        raise ValueError("Stage summary does not match its completed directory")
    if manifest.get("contract_sha256") is not None and manifest.get(
        "contract_sha256"
    ) != summary.get("contract_sha256"):
        raise ValueError("Stage summary/manifest contracts differ")
    return summary


def verify_global_freeze(out):
    summary = verify_stage(out, "fit_select")
    path = Path(out) / "fit_select/selection_set.json"
    selection_set = read_json(path)
    fit_contract = read_json(Path(out) / "fit_select/contract.json")
    fit_sha = digest(fit_contract)
    if (
        sha(path) != summary.get("selection_set_sha256")
        or selection_set.get("status") != "frozen_before_any_test_load"
        or set(selection_set.get("protocols", {})) != set(PROTOCOLS)
        or selection_set.get("contract_sha256") != fit_sha
        or summary.get("contract_sha256") != fit_sha
    ):
        raise ValueError("All six protocol selections must freeze globally before TEST audit/load")
    for protocol, descriptor in selection_set["protocols"].items():
        path = safe_path(out, descriptor["path"])
        if sha(path) != descriptor["sha256"]:
            raise ValueError("Frozen selection checksum changed")
        selection = read_json(path)
        states = selection.get("states", {})
        if (
            selection.get("protocol") != protocol
            or selection.get("fit_role") != "train"
            or selection.get("selection_role") != "validation"
            or selection.get("synthetic_fixture") is not False
            or set(states) != set(METHODS)
            or selection.get("state_sha256")
            != {method: digest(state) for method, state in states.items()}
            or selection.get("contract_sha256") != fit_sha
            or selection.get("source_sha256") != fit_contract.get("source_sha256")
            or selection.get("manifests") != fit_contract.get("manifests", {}).get(protocol)
        ):
            raise ValueError("Frozen formal selection states/roles differ")
    return sha(Path(out) / "fit_select/selection_set.json")


def reject_partial_restart(out, stage):
    paths = [
        path
        for path in Path(out).rglob("*")
        if path.is_dir()
        and (
            (stage in path.name and "partial" in path.name)
            or (path.parent.name in {"partial", ".partial"} and path.name.startswith(stage))
        )
        and any(path.rglob("*"))
    ]
    if paths:
        raise RuntimeError(f"Preserved partial evidence prevents silent stage restart: {paths}")


def stage_commands(project, pipeline, aux, aux1, aux2, out):
    common = [
        "--project-root",
        str(project),
        "--pipeline-root",
        str(pipeline),
        "--aux1-state",
        str(aux1),
        "--preceding-state",
        str(aux2),
        "--out",
        str(out),
    ]
    scripts, config = aux / "scripts", aux / "configs/reviewer_shared_classical.yaml"
    result = []
    for stage in ("fit_select", "signal_evaluation"):
        result.append(
            (
                stage,
                [
                    sys.executable,
                    "-B",
                    str(scripts / "evaluate_reviewer_shared_classical.py"),
                    *common,
                    "--config",
                    str(config),
                    "--stage",
                    stage,
                ],
            )
        )
    result.append(
        (
            "downstream",
            [
                sys.executable,
                "-B",
                str(scripts / "evaluate_reviewer_shared_downstream.py"),
                *common,
                "--config",
                str(config),
            ],
        )
    )
    result.append(
        (
            "cpu_latency",
            [
                sys.executable,
                "-B",
                str(scripts / "benchmark_reviewer_shared_classical_latency.py"),
                *common,
                "--config",
                str(aux / "configs/reviewer_shared_classical_latency.yaml"),
                "--shared-config",
                str(config),
            ],
        )
    )
    return result


def _table(out, descriptor):
    path = safe_path(out, descriptor["path"])
    if sha(path) != descriptor["sha256"]:
        raise ValueError("Public table checksum differs")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != descriptor["columns"]:
            raise ValueError("Public table columns differ")
        rows = list(reader)
    if len(rows) != descriptor["rows"]:
        raise ValueError("Public table row count differs")
    for row in rows:
        for value in row.values():
            if value and value.strip().lower() in {
                "inf",
                "-inf",
                "+inf",
                "infinity",
                "-infinity",
                "nan",
            }:
                raise ValueError("Nonfinite serialized public table value")
    return rows


def _unique(rows, fields, expected, label):
    values = [tuple(row[field] for field in fields) for row in rows]
    if len(values) != len(set(values)) or set(values) != expected:
        raise ValueError(f"{label} actual tuple coverage differs")


def _scientific_json_equal(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_scientific_json_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _scientific_json_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        )
    return actual == expected


def _scientific_csv_equal(cell, expected, *, exact_number=False):
    if expected is None:
        return cell == ""
    if isinstance(expected, bool):
        return cell == str(expected)
    if isinstance(expected, (int, float)):
        try:
            actual = float(cell)
        except (TypeError, ValueError):
            return False
        return (
            math.isfinite(actual)
            and math.isfinite(expected)
            and (
                actual == expected
                if exact_number
                else math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            )
        )
    return cell == str(expected)


def validate_candidate_history(rows, selections):
    fields = ("protocol", "canonical_method", "order")
    expected = {
        (protocol, item["canonical_method"], str(item["order"])): item
        for protocol, selection in selections.items()
        for item in selection["candidate_history"]
    }
    _unique(rows, fields, set(expected), "actual frozen candidate history")
    required = {
        "protocol",
        "information_scope",
        "canonical_method",
        "variant",
        "order",
        "alpha",
        "taps",
        "include_delta",
        "validation_score",
        "finite",
        "error",
        "elapsed_s",
        "breakdown",
    }
    for row in rows:
        frozen = expected[tuple(row[name] for name in fields)]
        if not required <= set(frozen):
            raise ValueError("Candidate frozen scientific history fields are incomplete")
        flat = {name: value for name, value in frozen.items() if name != "breakdown"}
        if set(row) != {*flat, "breakdown_json"}:
            raise ValueError("Candidate CSV scientific columns differ from frozen history JSON")
        for name, value in flat.items():
            if not _scientific_csv_equal(
                row[name], value, exact_number=name in {"order", "alpha", "taps"}
            ):
                raise ValueError(f"Candidate CSV field differs from frozen history JSON: {name}")
        try:
            breakdown = json.loads(row["breakdown_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Candidate CSV macro breakdown is not its frozen JSON") from error
        if not _scientific_json_equal(breakdown, frozen["breakdown"]):
            raise ValueError("Candidate CSV macro breakdown differs from frozen history JSON")


def validate_metric_rows(rows, expected):
    fields = (
        "protocol",
        "information_scope",
        "canonical_method",
        "variant",
        "split",
        "metric_scope",
        "dataset_id",
        "low_sensor",
        "high_sensor",
        "sequence_id",
        "metric",
    )

    def key(row):
        return tuple(str(row.get(name, "")) for name in fields)

    expected_by_key = {key(row): row for row in expected}
    if len(expected_by_key) != len(expected):
        raise ValueError("Metric saved prediction reconstruction has duplicate scientific keys")
    actual_keys = [key(row) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(expected_by_key):
        raise ValueError("Metric scientific tuple coverage differs from saved prediction arrays")
    columns = set().union(*(row.keys() for row in expected))
    for row in rows:
        wanted = expected_by_key[key(row)]
        if set(row) != columns or any(
            not _scientific_csv_equal(row[name], wanted.get(name)) for name in columns
        ):
            raise ValueError("Metric scientific field/value differs from saved prediction arrays")


def verify_contract_chain(out, summaries, manifests, fit_contract, core):
    """Bind common fit identity and the downstream/CPU dependency DAG exactly."""
    # This validator is also compiled by the separate exporter. Resolve its
    # pinned sibling helpers before the first import in a fresh process.
    scripts_root = str(Path(__file__).resolve().parents[1])
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    import yaml

    from evaluate_reviewer_shared_classical import source_contract

    fit_sha = digest(fit_contract)
    selection_set = read_json(out / "fit_select/selection_set.json")
    if selection_set.get("contract_sha256") != fit_sha:
        raise ValueError("Global selection set contract differs from actual fit contract")
    for name in ("fit_select", "signal_evaluation"):
        if (
            summaries[name].get("contract_sha256") != fit_sha
            or read_json(out / name / "artifact_manifest.json").get("contract_sha256") != fit_sha
        ):
            raise ValueError("Fit/signal stage contract differs from the one frozen fit identity")
    for descriptor in selection_set["protocols"].values():
        selection = read_json(safe_path(out, descriptor["path"]))
        protocol = selection["protocol"]
        if (
            selection.get("contract_sha256") != fit_sha
            or selection.get("source_sha256") != fit_contract.get("source_sha256")
            or selection.get("manifests") != fit_contract.get("manifests", {}).get(protocol)
        ):
            raise ValueError(
                "Per-protocol selection is detached from the global fit/source/role contract"
            )
    config_path = Path(fit_contract["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        sha(config_path) != fit_contract.get("config_sha256")
        or config != fit_contract.get("config")
        or config.get("analysis_profile") != PROFILE
        or config.get("expected_source_fingerprint") != SOURCE_FINGERPRINT
        or config.get("expected_dataset_manifest_sha256") != DATASET_SHA256
    ):
        raise ValueError("Actual frozen shared configuration differs from fit contract")
    project = Path(fit_contract["pipeline_root"]).resolve().parent.parent
    if source_contract(project) != fit_contract.get("source_sha256"):
        raise ValueError("Actual executed source roster/hash differs from fit contract")
    bundle = verify_bundle_sources(project)
    identity = fit_contract.get("frozen_bundle_identity", {})
    if (
        identity.get("path") != bundle["manifest_path"]
        or identity.get("manifest_sha256") != bundle["manifest_sha256"]
        or identity.get("source_fingerprint") != SOURCE_FINGERPRINT
        or identity.get("source_file_count") != len(bundle["source_map"])
        or core.get("contract", {}).get("config", {}).get("dataset_manifest_sha256")
        != DATASET_SHA256
    ):
        raise ValueError("Actual original source/data identity is detached from shared fit")
    role_manifests = fit_contract.get("manifests", {})
    if set(role_manifests) != set(PROTOCOLS):
        raise ValueError("Missing exact six-protocol role manifest contracts")
    for protocol, references in role_manifests.items():
        if set(references) != {
            "dataset_manifest",
            "split_manifest",
            "strict_manifest",
            "assignment_manifest",
        }:
            raise ValueError("Incomplete protocol role manifest roster")
        for role, reference in references.items():
            path = Path(reference["path"])
            path = path if path.is_absolute() else project / path
            if sha(path) != reference["sha256"] or (
                role == "dataset_manifest" and reference["sha256"] != DATASET_SHA256
            ):
                raise ValueError(
                    f"Actual frozen protocol role/source manifest differs: {protocol}/{role}"
                )
    selection_sha = sha(out / "fit_select/selection_set.json")
    downstream = read_json(out / "downstream/contract.json")
    cpu = read_json(out / "cpu_latency/contract.json")
    for stage, contract in (("downstream", downstream), ("cpu_latency", cpu)):
        contract_sha = digest(contract)
        if (
            summaries[stage].get("contract_sha256") != contract_sha
            or read_json(out / stage / "artifact_manifest.json").get("contract_sha256")
            != contract_sha
        ):
            raise ValueError("Downstream/CPU summary does not hash its actual stage contract")
        if (
            contract.get("aux1_state_sha256") != fit_contract["aux1_state_sha256"]
            or contract.get("aux2_state_sha256") != fit_contract["aux2_state_sha256"]
        ):
            raise ValueError("Downstream/CPU stage is bound to different original auxiliary states")
    if (
        downstream.get("fit_selection_set_sha256") != selection_sha
        or downstream.get("signal_manifest_sha256") != manifests["signal_evaluation"]["sha256"]
        or downstream.get("signal_contract_sha256") != fit_sha
        or downstream.get("script_sha256")
        != sha(Path(__file__).resolve().parents[1] / "evaluate_reviewer_shared_downstream.py")
        or downstream.get("integrator_sha256")
        != sha(project / "src/virtual_imu/eval/downstream_attitude.py")
    ):
        raise ValueError(
            "Downstream contract does not bind exact frozen fit/signal/source dependencies"
        )
    expected_upstream = {
        name: manifests[name]["sha256"]
        for name in ("fit_select", "signal_evaluation", "downstream")
    }
    if (
        cpu.get("fit_contract_sha256") != fit_sha
        or cpu.get("selection_set_sha256") != selection_sha
        or cpu.get("pipeline_sha256") != fit_contract["pipeline_sha256"]
        or cpu.get("upstream_manifests") != expected_upstream
        or cpu.get("script_sha256")
        != sha(
            Path(__file__).resolve().parents[1] / "benchmark_reviewer_shared_classical_latency.py"
        )
        or cpu.get("normalized_state_script_sha256")
        != sha(Path(__file__).resolve().parents[1] / "reviewer_shared_classical.py")
    ):
        raise ValueError("CPU contract does not bind exact frozen fit/source/upstream manifests")
    from benchmark_reviewer_shared_classical_latency import DEFAULTS

    latency_config = (
        Path(__file__).resolve().parents[2] / "configs/reviewer_shared_classical_latency.yaml"
    )
    if cpu.get("config") != DEFAULTS or cpu.get("config_sha256") != sha(latency_config):
        raise ValueError("CPU timing budget/config differs from its frozen source configuration")
    return fit_sha


def validate_cpu_artifacts(out, rows, selections, core, fit_contract):
    """Reconstruct deployed states and rederive every reported timing statistic."""
    import numpy as np

    from benchmark_reviewer_shared_classical_latency import DEFAULTS, parameter_tables
    from evaluate_reviewer_shared_classical import original_helpers
    from reviewer_shared_classical import normalize_bank_pair

    helper = original_helpers()
    pipeline = Path(fit_contract["pipeline_root"])
    project = pipeline.parent.parent
    finals = [
        job
        for job in core["jobs"].values()
        if job["phase"].startswith("final_") and job["cell"].split("/", 1)[0] in PROTOCOLS
    ]
    banks, references, records = {}, [], []
    for job in finals:
        protocol = job["cell"].split("/", 1)[0]
        path = (
            helper.recorded_path(job["run_dir"], pipeline, project)
            / "evaluation/baseline_train_state.json"
        )
        checksum = sha(path)
        declared = {
            str(helper.recorded_path(name, pipeline, project).resolve()): value
            for name, value in job.get("evaluation_artifact_sha256", {}).items()
        }
        if declared.get(str(path.resolve())) != checksum:
            raise ValueError(
                "Bank inventory source is detached from frozen original final artifact"
            )
        saved = read_json(path)
        if protocol in banks and banks[protocol] != saved:
            raise ValueError("Bank state differs across repeated final mappings")
        banks[protocol] = saved
        references.append(
            {"protocol": protocol, "job_id": job["id"], "state_path": str(path), "sha256": checksum}
        )
        for pair in saved["pairs"]:
            for method, state in normalize_bank_pair(protocol, pair).items():
                records.append(
                    {
                        "protocol": protocol,
                        "information_scope": "known_device_bank",
                        "canonical_method": method,
                        "variant": "bank",
                        "pair_identity": pair["identity"],
                        "state": state,
                        "state_path": str(path),
                        "state_file_sha256": checksum,
                        "serialized_state_bytes": path.stat().st_size,
                    }
                )
    observed_references = read_json(out / "cpu_latency/bank_source_references.json")

    def key(row):
        return row["protocol"], row["job_id"]

    if sorted(references, key=key) != sorted(observed_references, key=key):
        raise ValueError("Actual360 original bank source mappings differ")
    normalized = {}
    for protocol in PROTOCOLS:
        normalized[protocol] = {
            digest(pair["identity"]): {
                "identity": pair["identity"],
                "states": normalize_bank_pair(protocol, pair),
            }
            for pair in banks[protocol]["pairs"]
        }
        if len(normalized[protocol]) != len(banks[protocol]["pairs"]) or not normalized[protocol]:
            raise ValueError("Duplicate or empty original bank physical pair roster")
        for method, state in selections[protocol]["states"].items():
            path = out / "fit_select" / protocol / f"{method}.state.json"
            records.append(
                {
                    "protocol": protocol,
                    "information_scope": "protocol_shared_classical",
                    "canonical_method": method,
                    "variant": "shared",
                    "state": state,
                    "state_path": path.relative_to(out).as_posix(),
                    "state_file_sha256": sha(path),
                    "serialized_state_bytes": path.stat().st_size,
                }
            )
    expected_inventory, expected_totals = parameter_tables(records)

    def compare_table(actual, expected, fields, label):
        expected_by_key = {tuple(str(row[name]) for name in fields): row for row in expected}
        _unique(actual, fields, set(expected_by_key), label)
        for row in actual:
            expected_row = expected_by_key[tuple(row[name] for name in fields)]
            for name, value in expected_row.items():
                if value is None:
                    if row.get(name) not in {None, ""}:
                        raise ValueError(f"{label} null field differs: {name}")
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if float(row[name]) != float(value):
                        raise ValueError(f"{label} actual numeric field differs: {name}")
                elif str(value) != row.get(name):
                    raise ValueError(f"{label} actual field differs: {name}")

    group_fields = ("protocol", "information_scope", "canonical_method", "variant")
    compare_table(
        rows["state_inventory"],
        expected_inventory,
        (*group_fields, "pair_identity_json", "numeric_sha256"),
        "actual state inventory",
    )
    compare_table(
        rows["bank_parameter_totals"], expected_totals, group_fields, "actual bank totals"
    )
    runtime = read_json(out / "cpu_latency/runtime.json")
    if (
        runtime.get("threadpoolctl_version") != "3.6.0"
        or not runtime.get("threadpools")
        or any(pool.get("num_threads") != 1 for pool in runtime["threadpools"])
        or runtime.get("workload_quiet_check", {}).get("status") != "passed"
    ):
        raise ValueError("Actual CPU runtime/BLAS/workload verification is incomplete")
    raw = read_json(out / "cpu_latency/raw_timing_records.json")
    if not raw:
        raise ValueError("Raw timing record inventory is empty")
    by_id = {}
    expected_raw_rows = {}
    for content_sha, record in raw.items():
        identity = {
            "states": record.get("numeric_state_sha256s"),
            "state_contents": record.get("state_content_sha256s"),
            "inputs": record.get("input_pool_sha256s"),
            "history": record.get("history_samples"),
        }
        if (
            digest(identity) != content_sha
            or record.get("timing_id") != "timing_" + content_sha[:24]
        ):
            raise ValueError("Raw timing state/input content identity differs")
        identifier = record["timing_id"]
        if identifier in by_id or record.get("input_pool_sha256") != digest(
            record["input_pool_sha256s"]
        ):
            raise ValueError("Duplicate or detached raw timing input pool identity")
        times = np.asarray(record.get("raw_wall_ms"), dtype=np.float64)
        loops = np.asarray(record.get("raw_throughput_loop_seconds"), dtype=np.float64)
        if (
            times.shape != (300,)
            or loops.shape != (3,)
            or not np.isfinite(times).all()
            or not np.isfinite(loops).all()
            or np.any(times <= 0)
            or np.any(loops <= 0)
            or any(
                record.get(name) != DEFAULTS[name] for name in ("warmup", "iterations", "repeats")
            )
        ):
            raise ValueError("Actual raw300 call/3 loop timings or frozen budget differ")
        shapes = record.get("resident_input_shapes", [])
        windows = sum(shape[0] for shape in shapes)
        if (
            len(shapes) != record.get("pair_count")
            or windows != record.get("windows_per_call")
            or any(len(shape) != 3 or shape[2] != 6 for shape in shapes)
            or len(record.get("numeric_state_sha256s", [])) != len(shapes)
            or len(record.get("state_content_sha256s", [])) != len(shapes)
        ):
            raise ValueError("Raw workload actual pair/window/state shapes differ")
        calculated = {
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "call_distribution_sd_ms": float(np.std(times, ddof=1)),
            "loop_throughput_windows_per_second": windows * 300 / float(loops.sum()),
        }
        for name, value in calculated.items():
            if not math.isclose(float(record[name]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Reported CPU statistic differs from actual stored raw timings")
        by_id[identifier] = record
        for index, value in enumerate(times):
            expected_raw_rows[(identifier, "call_wall_ms", str(index // 100), str(index % 100))] = (
                float(value),
                "ms",
            )
        for repeat, value in enumerate(loops):
            expected_raw_rows[(identifier, "independent_throughput_loop", str(repeat), "")] = (
                float(value),
                "s",
            )
    raw_fields = ("timing_id", "measurement", "repeat", "iteration")
    for row in rows["latency_raw"]:
        row["repeat"] = str(int(float(row["repeat"])))
        row["iteration"] = "" if row["iteration"] == "" else str(int(float(row["iteration"])))
    _unique(
        rows["latency_raw"], raw_fields, set(expected_raw_rows), "actual raw timing CSV samples"
    )
    for row in rows["latency_raw"]:
        value, unit = expected_raw_rows[tuple(row[name] for name in raw_fields)]
        if row["unit"] != unit or not math.isclose(
            float(row["value"]), value, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("Raw timing CSV differs from actual JSON call/loop sample")
    schedulers = read_json(out / "cpu_latency/train_window_schedulers.json")
    expected_scheduler_keys = {
        f"{protocol}/{window}/{pair}"
        for protocol in PROTOCOLS
        for window in (80, 160, 320)
        for pair in normalized[protocol]
    }
    if set(schedulers) != expected_scheduler_keys:
        raise ValueError("Frozen same-pair TRAIN scheduler coverage differs")
    histories = {}
    for protocol in PROTOCOLS:
        states = [
            *selections[protocol]["states"].values(),
            *[state for pair in normalized[protocol].values() for state in pair["states"].values()],
        ]
        histories[protocol] = max(
            max(
                (state["feature_spec"]["taps"] or 1) - 1,
                int(state["feature_spec"]["include_delta"]),
            )
            for state in states
        )
    for name, scheduler in schedulers.items():
        protocol, window, pair = name.split("/")
        physical_rows = scheduler.get("rows", [])
        if (
            scheduler.get("pair_identity") != normalized[protocol][pair]["identity"]
            or len(physical_rows) != 128
            or set(scheduler.get("batch_input_sha256", {}))
            != {str(batch) for batch in DEFAULTS["batch_sizes"]}
            or scheduler.get("pool_sha256") != scheduler["batch_input_sha256"].get("128")
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
                for value in scheduler["batch_input_sha256"].values()
            )
        ):
            raise ValueError("TRAIN scheduler actual physical pair or128 input windows differ")
        for row in physical_rows:
            if (
                row.get("split") != "train"
                or row["start_index"] - row["history_start_index"] != histories[protocol]
                or row["end_index"] - row["start_index"] + 1 != int(window)
                or not row.get("sequence_id")
                or not row.get("physical_row_sha256")
                or not row.get("input_sha256")
            ):
                raise ValueError("TRAIN same-stream resident history/window row proof differs")
    expected_state_keys = set()

    def pair_text(identity):
        return json.dumps(identity, sort_keys=True)

    for protocol in PROTOCOLS:
        pairs = sorted(normalized[protocol])
        for method in METHODS:
            for batch in (1, 128):
                for window in (80, 160, 320):
                    expected_state_keys.add(
                        (
                            protocol,
                            "protocol_shared_classical",
                            method,
                            "shared",
                            pair_text(normalized[protocol][pairs[0]]["identity"]),
                            str(batch),
                            str(window),
                        )
                    )
                    for pair in pairs:
                        expected_state_keys.add(
                            (
                                protocol,
                                "known_device_bank",
                                method,
                                "bank",
                                pair_text(normalized[protocol][pair]["identity"]),
                                str(batch),
                                str(window),
                            )
                        )
    _unique(
        rows["latency_summary"],
        (*group_fields, "pair_identity_json", "batch_size_per_pair", "window_size"),
        expected_state_keys,
        "all actual per-state B/W timings",
    )
    used = set()
    matched_inputs = {}
    for role in ("latency_summary", "workload_latency"):
        for row in rows[role]:
            identifier = row.get("timing_id")
            if identifier not in by_id:
                raise ValueError("Public timing row lacks an actual raw measurement")
            record = by_id[identifier]
            used.add(identifier)
            protocol, method = row["protocol"], row["canonical_method"]
            pairs = sorted(normalized[protocol])
            if role == "latency_summary":
                pairs = [
                    next(
                        pair
                        for pair in pairs
                        if pair_text(normalized[protocol][pair]["identity"])
                        == row["pair_identity_json"]
                    )
                ]
            states = (
                [selections[protocol]["states"][method]] * len(pairs)
                if row["information_scope"] == "protocol_shared_classical"
                else [normalized[protocol][pair]["states"][method] for pair in pairs]
            )
            batch, window = int(row["batch_size_per_pair"]), int(row["window_size"])
            expected_inputs = [
                schedulers[f"{protocol}/{window}/{pair}"]["batch_input_sha256"][str(batch)]
                for pair in pairs
            ]
            if (
                record["input_pool_sha256s"] != expected_inputs
                or row.get("input_pool_sha256") != digest(expected_inputs)
                or row.get("input_pool_sha256") != record["input_pool_sha256"]
            ):
                raise ValueError(
                    "Public/raw input pool SHA differs from actual frozen pair/B/W scheduler"
                )
            if role == "workload_latency":
                scheduler_sha = digest(
                    [schedulers[f"{protocol}/{window}/{pair}"] for pair in pairs]
                )
                if row.get("scheduler_sha256") != scheduler_sha:
                    raise ValueError(
                        "Public workload scheduler SHA differs from its ordered frozen protocol/pair roster"
                    )
                schedule_key = protocol, method, batch, window
                previous_inputs = matched_inputs.setdefault(schedule_key, expected_inputs)
                if previous_inputs != record["input_pool_sha256s"]:
                    raise ValueError(
                        "Shared/bank P-by-B workloads use different frozen input pools"
                    )
            expected_shapes = [[batch, histories[protocol] + window, 6]] * len(pairs)
            if (
                record["numeric_state_sha256s"] != [state["numeric_sha256"] for state in states]
                or record["state_content_sha256s"] != [digest(state) for state in states]
                or record["resident_input_shapes"] != expected_shapes
                or record["history_samples"] != histories[protocol]
                or int(row["pair_count"]) != len(pairs)
                or int(row["windows_per_call"]) != len(pairs) * batch
            ):
                raise ValueError(
                    "Public state/workload timing is detached from actual arrays/history/scheduler"
                )
            for name in (
                "p50_ms",
                "p95_ms",
                "call_distribution_sd_ms",
                "loop_throughput_windows_per_second",
            ):
                if not math.isclose(
                    float(row[name]), float(record[name]), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "Public latency statistic differs from actual raw timing record"
                    )
    if used != set(by_id):
        raise ValueError("Unreferenced or missing actual raw timing records")
    proofs = read_json(out / "cpu_latency/kernel_proofs.json")
    expected_proofs = {
        (protocol, scope, method, pair)
        for protocol in PROTOCOLS
        for scope in ("protocol_shared_classical", "known_device_bank")
        for method in METHODS
        for pair in normalized[protocol]
    }
    proof_keys = [
        (row["protocol"], row["information_scope"], row["canonical_method"], row["pair_key"])
        for row in proofs
    ]
    if len(proof_keys) != len(set(proof_keys)) or set(proof_keys) != expected_proofs:
        raise ValueError("Actual full predictor start/interior kernel proof coverage differs")
    for row in proofs:
        state = (
            selections[row["protocol"]]["states"][row["canonical_method"]]
            if row["information_scope"] == "protocol_shared_classical"
            else normalized[row["protocol"]][row["pair_key"]]["states"][row["canonical_method"]]
        )
        if row["numeric_sha256"] != state["numeric_sha256"] or not row.get("proof"):
            raise ValueError("Kernel proof is detached from the actual normalized state")
        for proof in row["proof"]:
            if (
                proof.get("start_checked") is not True
                or proof.get("interior_checked") is not True
                or not math.isfinite(float(proof["max_abs_error"]))
            ):
                raise ValueError("Kernel start/interior numeric proof is incomplete")
    return {
        "actual_shared_states": 54,
        "actual_bank_pair_states": len(expected_inventory) - 54,
        "actual_timing_records": len(by_id),
        "raw_call_samples": 300 * len(by_id),
        "raw_loop_samples": 3 * len(by_id),
    }


def validate_downstream_and_aliases(out, rows, selections, core, original_aux, fit_contract):
    """Verify saved errors/summary math and exact original support/alias evidence."""
    import ast

    import numpy as np

    from evaluate_reviewer_shared_classical import original_helpers
    from reviewer_shared_classical import AXES, macro_breakdown, normalize_bank_pair

    helper = original_helpers()

    def numeric(value, *, nullable=False):
        if value is None or value == "":
            if nullable:
                return None
            raise ValueError("Required scientific numeric value is missing")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Scientific reported numeric value is nonfinite")
        return result

    def equal_number(actual, expected):
        number = numeric(actual, nullable=expected is None)
        if expected is None:
            return number is None
        return number is not None and math.isclose(
            number, float(expected), rel_tol=1e-10, abs_tol=1e-12
        )

    windows = read_json(out / "downstream/downstream_windows.json")
    segments = read_json(out / "downstream/downstream_segments.json")
    for role, payload in (("downstream_windows", windows), ("downstream_segments", segments)):
        if len(rows[role]) != len(payload):
            raise ValueError("Scientific long CSV/JSON row counts differ")
        for csv_row, record in zip(rows[role], payload, strict=True):
            for name, value in record.items():
                cell = csv_row.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not equal_number(cell, value):
                        raise ValueError(
                            "Scientific long CSV numeric errors differ from stored JSON"
                        )
                elif value is None:
                    if cell not in {None, ""}:
                        raise ValueError("Scientific long CSV unsupported value differs")
                elif str(value) != cell:
                    raise ValueError("Scientific long CSV metadata differs from stored JSON")
            if csv_row.get("n") not in {"1", "1.0"} or csv_row.get("sd") not in {None, ""}:
                raise ValueError("Downstream deterministic long rows must keep n=1 and blank SD")
    proofs = read_json(out / "downstream/support_proofs.json")
    views = {
        "penn_loso_af": "af",
        "penn_loso_as": "as",
        "penn_loso_bf": "bf",
        "penn_only": "bs",
        "pooled": "bs",
    }
    proof_keys = [(row["protocol"], row["task"]) for row in proofs]
    if (
        len(proof_keys) != 10
        or len(set(proof_keys)) != 10
        or set(proof_keys)
        != {(protocol, task) for protocol in views for task in ("attitude", "strapdown")}
    ):
        raise ValueError("Actual five-view/two-task independent GT support proofs differ")
    original_root = Path(original_aux["aux1"]["stages"]["downstream"]["path"])
    downstream_contract = read_json(out / "downstream/contract.json")
    if downstream_contract.get("propagation_config_sha256") != sha(
        original_root / "config_snapshot.yaml"
    ):
        raise ValueError("Downstream original propagation configuration hash changed")
    import pandas as pd

    from evaluate_reviewer_shared_downstream import exact_physical_frame

    old_completion = read_json(original_root / "complete.json")
    if set(old_completion.get("unit_manifests", {})) != set(views):
        raise ValueError("Original actual five downstream source units differ")
    for protocol, recording in views.items():
        old = original_root / "units" / protocol
        expected_sha = old_completion["unit_manifests"][protocol]
        if sha(old / "artifact_manifest.json") != expected_sha:
            raise ValueError("Original downstream support unit manifest changed")
        manifest = read_json(old / "artifact_manifest.json")
        helper.verify_completed(old, manifest["contract_sha256"])
        metadata = read_json(old / "metrics.json")
        exact_physical_frame(
            pd.read_parquet(out / "signal_evaluation" / protocol / "test_predictions.parquet"),
            pd.read_parquet(old / "evaluated_signals.parquet"),
            recording,
        )
        references = {
            "windows": read_json(old / "window_details.json"),
            "segments": read_json(old / "full_segments.json"),
        }
        for task in ("attitude", "strapdown"):
            proof = next(
                row for row in proofs if row["protocol"] == protocol and row["task"] == task
            )
            support = proof.get("support", {})
            original_support = metadata["support"][task]
            if (
                proof.get("recording") != recording
                or proof.get("identity_alias_verified") is not True
                or proof.get("original_unit_manifest_sha256") != expected_sha
                or proof.get("independent_gt_sha256") != metadata["independent_gt_sha256"]
                or sha(Path(metadata["independent_gt_path"])) != metadata["independent_gt_sha256"]
                or proof.get("propagation_protocol") != metadata["propagation_protocol"][task]
                or {key: value for key, value in support.items() if key != "common_methods"}
                != {
                    key: value for key, value in original_support.items() if key != "common_methods"
                }
                or support.get("common_methods") != list(METHODS)
            ):
                raise ValueError(
                    "Shared scientific support/GT/initialization proof differs from original source"
                )
            propagation = proof["propagation_protocol"]
            if (
                propagation.get("ground_truth") != "independent_optical_Penn_AprilTag_rig_pose"
                or propagation.get("ground_truth_during_propagation") is not False
                or propagation.get("rate_hz") != 40.0
            ):
                raise ValueError("Frozen independent GT propagation contract differs")
        for section, payload in (("windows", windows), ("segments", segments)):
            original_identity = [row for row in references[section] if row["method"] == "identity"]
            names = ("task", "sequence_id", "segment", "start_s", "end_s") + (
                ("samples",) if section == "windows" else ()
            )
            expected_keys = [tuple(row[name] for name in names) for row in original_identity]
            for method in METHODS:
                measured = [
                    row
                    for row in payload
                    if row["protocol"] == protocol and row["canonical_method"] == method
                ]
                if [tuple(row[name] for name in names) for row in measured] != expected_keys:
                    raise ValueError(
                        "Shared physical window/segment keys differ from original identity support"
                    )
                for row in measured:
                    if row["recording"] != recording or row.get("state_content_sha256") != digest(
                        selections[protocol]["states"][method]
                    ):
                        raise ValueError(
                            "Downstream scientific row references a different recording/state"
                        )
                    required = (
                        (
                            ("endpoint_deg", "sum_squared_deg", "tilt_endpoint_deg")
                            if section == "windows"
                            else ("endpoint_deg", "rmse_deg", "tilt_rmse_deg")
                        )
                        if row["task"] == "attitude"
                        else (
                            "position_endpoint_m",
                            "horizontal_endpoint_m",
                            "position_rmse_m",
                            "position_sum_squared_m",
                        )
                    )
                    if any(numeric(row[name]) < 0 for name in required):
                        raise ValueError("Scientific magnitude/error cannot be negative")
                    if row["task"] == "strapdown" and (
                        np.asarray(row["initial_velocity_mps"]).shape != (3,)
                        or not np.isfinite(row["initial_velocity_mps"]).all()
                    ):
                        raise ValueError("Frozen strapdown initial velocity evidence differs")
                identity = [row for row in measured if method == "identity"]
                for original, measured_row in zip(
                    original_identity, identity, strict=bool(identity)
                ):
                    for name, value in original.items():
                        if name == "method":
                            continue
                        other = measured_row[name]
                        if isinstance(value, (int, float, list)):
                            if not np.allclose(value, other, rtol=1e-10, atol=1e-12):
                                raise ValueError(
                                    "Shared/original identity downstream numerical alias differs"
                                )
                        elif value != other:
                            raise ValueError("Shared/original identity downstream metadata differs")
    for summary in rows["downstream_summary"]:
        protocol, method, task, size = (
            summary["protocol"],
            summary["canonical_method"],
            summary["task"],
            int(summary["samples"]),
        )
        details = [
            row
            for row in windows
            if row["protocol"] == protocol
            and row["canonical_method"] == method
            and row["task"] == task
            and row["samples"] == size
        ]
        count = len(details)
        if (
            int(summary["count"]) != count
            or summary["status"] != ("supported" if count else "unsupported_no_fixed_windows")
            or not equal_number(summary["horizon_s"], (size - 1) / 40)
        ):
            raise ValueError(
                "Downstream summary count/status/horizon differs from actual window errors"
            )
        if task == "attitude":
            endpoints = np.asarray([row["endpoint_deg"] for row in details])
            heading = np.asarray(
                [
                    row["heading_endpoint_deg"]
                    for row in details
                    if row["heading_endpoint_deg"] is not None
                ]
            )
            calculated = {
                "endpoint_rmse_deg": float(np.sqrt(np.mean(endpoints**2))) if count else None,
                "endpoint_median_deg": float(np.median(endpoints)) if count else None,
                "endpoint_p95_deg": float(np.quantile(endpoints, 0.95)) if count else None,
                "all_samples_rmse_deg": math.sqrt(
                    sum(row["sum_squared_deg"] for row in details) / (size * count)
                )
                if count
                else None,
                "tilt_endpoint_rmse_deg": math.sqrt(
                    sum(row["tilt_endpoint_deg"] ** 2 for row in details) / count
                )
                if count
                else None,
                "heading_endpoint_rmse_deg": float(np.sqrt(np.mean(heading**2)))
                if len(heading)
                else None,
                "heading_valid_windows": len(heading),
            }
        else:
            endpoints = np.asarray([row["position_endpoint_m"] for row in details])
            calculated = {
                "position_endpoint_rmse_m": float(np.sqrt(np.mean(endpoints**2)))
                if count
                else None,
                "horizontal_endpoint_rmse_m": math.sqrt(
                    sum(row["horizontal_endpoint_m"] ** 2 for row in details) / count
                )
                if count
                else None,
                "position_endpoint_median_m": float(np.median(endpoints)) if count else None,
                "position_endpoint_p95_m": float(np.quantile(endpoints, 0.95)) if count else None,
                "all_samples_position_rmse_m": math.sqrt(
                    sum(row["position_sum_squared_m"] for row in details) / (size * count)
                )
                if count
                else None,
            }
        if any(not equal_number(summary[name], value) for name, value in calculated.items()):
            raise ValueError(
                "Downstream reported scientific summary differs from stored window errors"
            )
    pipeline = Path(fit_contract["pipeline_root"])
    project = pipeline.parent.parent
    alias_predictions = {}
    for row in rows["alias_audit"]:
        protocol, method = row["protocol"], row["canonical_method"]
        status = row.get("alias_status")
        if protocol in {"pooled", "alimu_only"}:
            if status != (
                "common_identity" if method == "identity" else "not_applicable_multi_pair_bank"
            ):
                raise ValueError("Multi-pair bank applicability status differs")
            continue
        if status not in {"exact_alias", "numerically_equivalent", "not_alias"}:
            raise ValueError("Single-pair alias status is missing or unsupported")
        job = next(
            job
            for job in core["jobs"].values()
            if job["phase"].startswith("final_") and job["cell"].split("/", 1)[0] == protocol
        )
        bank_saved = read_json(
            helper.recorded_path(job["run_dir"], pipeline, project)
            / "evaluation/baseline_train_state.json"
        )
        if len(bank_saved["pairs"]) != 1:
            raise ValueError("Penn alias source is not actually a single physical pair")
        bank = normalize_bank_pair(protocol, bank_saved["pairs"][0])[method]
        shared = selections[protocol]["states"][method]
        proof = ast.literal_eval(row["proof"])
        required = (
            "same_protocol",
            "train_rows_exact",
            "validation_rows_exact",
            "grid_order_exact",
            "tie_rule_exact",
            "bank_own_selection_valid",
            "bank_loader_order_verified",
        )
        if (
            not isinstance(proof, dict)
            or not all(proof.get(name) is True for name in required)
            or proof.get("bank_cache_state_sha256") != digest(bank_saved)
            or proof.get("shared_train_order_sha256") != selections[protocol]["train_row_sha256"]
            or proof.get("shared_validation_order_sha256")
            != selections[protocol]["validation_row_sha256"]
            or row.get("shared_state_sha256") != digest(shared)
            or row.get("bank_state_sha256") != digest(bank)
            or numeric(row["rtol"]) != 1e-10
            or numeric(row["atol"]) != 1e-12
        ):
            raise ValueError("Alias provenance/state/support/tolerance proof differs")
        same_hp = shared["hyperparameters"] == bank["hyperparameters"]
        if row["same_hyperparameters"] != str(same_hp):
            raise ValueError("Alias hyperparameter equivalence flag differs from actual states")
        for name in (
            "shared_validation_score",
            "bank_validation_score",
            "max_prediction_absolute_difference",
            "validation_max_prediction_absolute_difference",
            "test_max_prediction_absolute_difference",
        ):
            if numeric(row[name]) < 0:
                raise ValueError("Alias reported score or prediction difference cannot be negative")
        if protocol not in alias_predictions:
            pairs = {}
            declared = {
                str(helper.recorded_path(name, pipeline, project).resolve()): checksum
                for name, checksum in job.get("evaluation_artifact_sha256", {}).items()
            }
            for split in ("validation", "test"):
                original_path = (
                    helper.recorded_path(job["run_dir"], pipeline, project)
                    / "evaluation"
                    / f"{split}_predictions.parquet"
                )
                if declared.get(str(original_path.resolve())) != sha(original_path):
                    raise ValueError(
                        "Alias saved original prediction is detached from its frozen final artifact"
                    )
                actual = pd.read_parquet(original_path)
                selected = pd.read_parquet(
                    out / "signal_evaluation" / protocol / f"{split}_predictions.parquet"
                )
                from evaluate_reviewer_shared_classical import support_match

                support_match(selected, actual)
                pairs[split] = selected, actual
            alias_predictions[protocol] = pairs
        measured = {}
        for split, (selected, actual) in alias_predictions[protocol].items():
            columns = [f"{method}_{axis}" for axis in AXES]
            left, right = selected[columns].to_numpy(float), actual[columns].to_numpy(float)
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError("Alias saved shared/original predictions must be finite")
            difference = float(np.max(np.abs(left - right)))
            exact, equivalent = (
                np.array_equal(left, right),
                np.allclose(left, right, rtol=1e-10, atol=1e-12),
            )
            if (
                not equal_number(row[split + "_max_prediction_absolute_difference"], difference)
                or row[split + "_prediction_exact"] != str(bool(exact))
                or row[split + "_prediction_equivalent"] != str(bool(equivalent))
            ):
                raise ValueError(
                    "Alias prediction difference/flags contradict actual saved shared/bank predictions"
                )
            measured[split] = {
                "difference": difference,
                "exact": exact,
                "equivalent": equivalent,
                "max_reference": float(np.max(np.abs(right))),
            }
        selected, actual = alias_predictions[protocol]["validation"]
        columns = [f"{method}_{axis}" for axis in AXES]
        shared_score = macro_breakdown(selected[columns].to_numpy(float), selected)["score"]
        bank_score = macro_breakdown(actual[columns].to_numpy(float), selected)["score"]
        if not equal_number(row["shared_validation_score"], shared_score) or not equal_number(
            row["bank_validation_score"], bank_score
        ):
            raise ValueError("Alias VAL scores contradict actual saved shared/bank predictions")
        if method.endswith("_tuned"):
            bank_fit = bank_saved["pairs"][0][method]
            selected_score = selections[protocol]["selections"][method]["validation_score"]
            if (
                not equal_number(row["shared_selected_validation_score"], selected_score)
                or not equal_number(
                    row["bank_selected_validation_score"], bank_fit["validation_score"]
                )
                or not equal_number(shared_score, selected_score)
                or not equal_number(bank_score, bank_fit["validation_score"])
            ):
                raise ValueError("Alias tuned VAL score is detached from its own frozen selection")
            shared_history = [
                item["validation_score"]
                for item in selections[protocol]["candidate_history"]
                if item["canonical_method"] == method
            ]
            bank_history = [item["validation_score"] for item in bank_fit["history"]]
            history_exact = shared_history == bank_history
            history_equivalent = len(shared_history) == len(bank_history) and bool(
                np.allclose(shared_history, bank_history, rtol=1e-10, atol=1e-12)
            )
            if row["candidate_score_history_exact"] != str(history_exact) or row[
                "candidate_score_history_equivalent"
            ] != str(history_equivalent):
                raise ValueError("Alias candidate score history flags contradict frozen selections")
            if (status == "exact_alias" and not history_exact) or (
                status == "numerically_equivalent" and not history_equivalent
            ):
                raise ValueError("Claimed alias contradicts its actual candidate score history")
        if status == "exact_alias" and (
            numeric(row["shared_validation_score"]) != numeric(row["bank_validation_score"])
            or shared_score != bank_score
            or any(
                numeric(row[name]) != 0
                for name in (
                    "max_prediction_absolute_difference",
                    "validation_max_prediction_absolute_difference",
                    "test_max_prediction_absolute_difference",
                )
            )
        ):
            raise ValueError(
                "Claimed exact alias has unequal VAL scores or nonzero prediction differences"
            )
        if status == "numerically_equivalent" and (
            not np.isclose(shared_score, bank_score, rtol=1e-10, atol=1e-12)
            or not np.isclose(
                numeric(row["shared_validation_score"]),
                numeric(row["bank_validation_score"]),
                rtol=1e-10,
                atol=1e-12,
            )
            or not all(item["equivalent"] for item in measured.values())
            or numeric(row["max_prediction_absolute_difference"])
            > 2 * (1e-12 + 1e-10 * measured["validation"]["max_reference"])
        ):
            raise ValueError(
                "Claimed numerical alias contradicts actual VAL score/prediction tolerance"
            )
        if status != "not_alias":
            if (
                not same_hp
                or np.shape(shared["weights"]) != np.shape(bank["weights"])
                or not np.allclose(shared["weights"], bank["weights"], rtol=1e-10, atol=1e-12)
                or not np.allclose(shared["intercept"], bank["intercept"], rtol=1e-10, atol=1e-12)
                or any(
                    row[split + "_prediction_equivalent"] != "True"
                    for split in ("validation", "test")
                )
            ):
                raise ValueError(
                    "Claimed numerical alias differs from actual coefficient/HP/prediction proof"
                )
        if status == "exact_alias" and (
            shared["numeric_sha256"] != bank["numeric_sha256"]
            or any(row[split + "_prediction_exact"] != "True" for split in ("validation", "test"))
        ):
            raise ValueError(
                "Claimed exact alias differs from actual numerical state or prediction proof"
            )
    return {
        "support_proofs": len(proofs),
        "window_error_rows": len(windows),
        "segment_error_rows": len(segments),
        "alias_rows": len(rows["alias_audit"]),
    }


def validate_complete_artifacts(
    out,
    expected_core_pipeline_sha256=None,
    *,
    require_index=True,
    expected_aux_manifest_sha256=None,
):
    """Read-only final validator for queue/exporter; no mutations or external runs.

    Validates every stage file hash, table descriptor, real key sets and finite
    shared numeric predictions. Call only after the six-protocol freeze barrier.
    The original full audit is performed by signal_evaluation, never by waiting.
    """
    out = Path(out).resolve()
    selection_sha = verify_global_freeze(out)
    summaries, tables, manifests = {}, {}, {}
    for stage in STAGES:
        summary = verify_stage(out, stage)
        if summary.get("schema_version") != 1 or summary.get("analysis_profile") != PROFILE:
            raise ValueError("Shared stage schema/profile differs")
        if summary.get("selection_set_sha256") != selection_sha:
            raise ValueError("Stage does not reference the one global frozen selection set")
        for role, descriptor in summary.get("tables", {}).items():
            if role in tables:
                raise ValueError("Public table role is ambiguous across stages")
            tables[role] = descriptor
        summaries[stage] = summary
        manifests[stage] = {
            "path": f"{stage}/artifact_manifest.json",
            "sha256": sha(out / stage / "artifact_manifest.json"),
            "summary_path": f"{stage}/stage_summary.json",
            "summary_sha256": sha(out / stage / "stage_summary.json"),
        }
    if not REQUIRED_ROLES <= set(tables):
        raise ValueError("Missing named shared-control public table roles")
    rows = {role: _table(out, descriptor) for role, descriptor in tables.items()}
    logical = {
        (protocol, "protocol_shared_classical", method, "shared")
        for protocol in PROTOCOLS
        for method in METHODS
    }
    _unique(
        rows["control_registry"],
        ("protocol", "information_scope", "canonical_method", "variant"),
        logical,
        "54 controls",
    )
    for role in ("shared_state_inventory", "method_scope"):
        _unique(
            rows[role],
            ("protocol", "information_scope", "canonical_method", "variant"),
            logical,
            role,
        )
    signal_keys = {(*key, split) for key in logical for split in ("validation", "test")}
    _unique(
        rows["signal_registry"],
        ("protocol", "information_scope", "canonical_method", "variant", "split"),
        signal_keys,
        "108 signals",
    )
    metric_keys = {
        tuple(
            row[key]
            for key in ("protocol", "information_scope", "canonical_method", "variant", "split")
        )
        for row in rows["metrics_long"]
    }
    if metric_keys != signal_keys or any(
        not math.isfinite(float(row["value"])) for row in rows["metrics_long"]
    ):
        raise ValueError("Finite signal metric coverage differs from all108 logical slots")
    neural = rows["neural_support_registry"]
    if (
        len(neural) != 720
        or len({row["job_id"] for row in neural}) != 360
        or len({(row["job_id"], row["split"]) for row in neural}) != 720
        or any(
            row["protocol"] not in PROTOCOLS or row["split"] not in {"validation", "test"}
            for row in neural
        )
    ):
        raise ValueError("360 paired final mappings/720 split references differ")
    for protocol in PROTOCOLS:
        paired = [row for row in neural if row["protocol"] == protocol]
        if len(paired) != 120 or len({row["job_id"] for row in paired}) != 60:
            raise ValueError(
                "Each paired protocol requires all60 family/variant/seed mappings and both splits"
            )
    downstream = rows["downstream_summary"]
    views = {
        ("penn_loso_af", "af"),
        ("penn_loso_as", "as"),
        ("penn_loso_bf", "bf"),
        ("penn_only", "bs"),
        ("pooled", "bs"),
    }
    tasks = {row["task"] for row in downstream}
    if tasks != {"attitude", "strapdown"}:
        raise ValueError("Downstream must contain exactly both frozen propagation tasks")
    expected_downstream = {
        (protocol, recording, "protocol_shared_classical", method, "shared", task, str(samples))
        for protocol, recording in views
        for method in METHODS
        for task in tasks
        for samples in (80, 200, 400, 800)
    }
    _unique(
        downstream,
        (
            "protocol",
            "recording",
            "information_scope",
            "canonical_method",
            "variant",
            "task",
            "samples",
        ),
        expected_downstream,
        "360 downstream slots",
    )
    alias_keys = {(protocol, method) for protocol in PROTOCOLS for method in METHODS}
    _unique(
        rows["alias_audit"],
        ("protocol", "canonical_method"),
        alias_keys,
        "protocol/method alias audit",
    )
    workload_keys = {
        (protocol, scope, method, variant, str(batch), str(window))
        for protocol in PROTOCOLS
        for scope, variant in (
            ("protocol_shared_classical", "shared"),
            ("known_device_bank", "bank"),
        )
        for method in METHODS
        for batch in (1, 128)
        for window in (80, 160, 320)
    }
    _unique(
        rows["workload_latency"],
        (
            "protocol",
            "information_scope",
            "canonical_method",
            "variant",
            "batch_size_per_pair",
            "window_size",
        ),
        workload_keys,
        "full pair workload latency",
    )
    for role in ("control_registry", "signal_registry", "method_scope", "downstream_summary"):
        if any(
            row.get("n") not in {"1", "1.0"} or row.get("sd") not in {None, ""}
            for row in rows[role]
        ):
            raise ValueError("Deterministic controls must have n=1 and blank SD")
    fit_contract = read_json(out / "fit_select/contract.json")
    if (
        expected_core_pipeline_sha256 is not None
        and fit_contract.get("pipeline_sha256") != expected_core_pipeline_sha256
    ):
        raise ValueError("Final shared artifacts bind a different original core")
    core_path = Path(fit_contract["pipeline_root"]) / "pipeline.json"
    if sha(core_path) != fit_contract["pipeline_sha256"]:
        raise ValueError("Actual frozen original pipeline checksum differs")
    core = read_json(core_path)
    if not _core_ready(core, "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n"):
        raise ValueError("Original exact final matrix is not complete")
    actual_paired = [
        job
        for job in core["jobs"].values()
        if job.get("phase", "").startswith("final_") and job["cell"].split("/", 1)[0] in PROTOCOLS
    ]
    expected_neural = {
        (
            job["cell"].split("/", 1)[0],
            job["id"],
            job["cell"],
            job["phase"],
            str(job["seed"]),
            split,
        )
        for job in actual_paired
        for split in ("validation", "test")
    }
    _unique(
        neural,
        ("protocol", "job_id", "cell", "phase", "seed", "split"),
        expected_neural,
        "actual frozen original paired family/variant/seed mappings",
    )
    fit_contract_sha = verify_contract_chain(out, summaries, manifests, fit_contract, core)
    # Use the producer's strict normalized schema checks and exact selection rule.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import pandas as pd

    from evaluate_reviewer_shared_classical import metric_rows, verify_selection_set
    from reviewer_shared_classical import AXES, HIGH, LOW, verify_state

    _, selections = verify_selection_set(out)
    validate_candidate_history(rows["candidate_history"], selections)
    for row in rows["control_registry"]:
        state = selections[row["protocol"]]["states"][row["canonical_method"]]
        path = safe_path(out, row["state_path"])
        if (
            sha(path) != row["state_sha256"]
            or digest(state) != row["state_content_sha256"]
            or state["numeric_sha256"] != row["numeric_sha256"]
            or state["coefficient_count"] != int(row["coefficient_count"])
            or state["coefficient_bytes"] != int(row["coefficient_bytes"])
        ):
            raise ValueError("Actual control coefficients/file content differ from public registry")
    learned = identity = 0
    for protocol in PROTOCOLS:
        for method in METHODS:
            state = selections[protocol]["states"][method]
            verify_state(state)
            path = safe_path(out, f"fit_select/{protocol}/{method}.state.json")
            if read_json(path) != state:
                raise ValueError("Actual public state file differs from frozen selection")
            learned += state["state_kind"] != "identity"
            identity += state["state_kind"] == "identity"
    paths = {row["prediction_path"] for row in rows["signal_registry"]}
    if len(paths) != 12:
        raise ValueError("Expected six protocols times two shared prediction matrices")
    reconstructed_metrics = []
    for name in paths:
        path = safe_path(out, name)
        frame = pd.read_parquet(path)
        columns = [f"{method}_{axis}" for method in METHODS for axis in AXES]
        if (
            not set(columns + LOW + HIGH) <= set(frame.columns)
            or not np.isfinite(frame[columns + LOW + HIGH].to_numpy(float)).all()
        ):
            raise ValueError("Shared prediction finite matrix/schema differs")
        np.testing.assert_array_equal(
            frame[[f"identity_{axis}" for axis in AXES]].to_numpy(float), frame[LOW].to_numpy(float)
        )
        for row in (row for row in rows["signal_registry"] if row["prediction_path"] == name):
            if int(row["rows"]) != len(frame) or row["selection_set_sha256"] != selection_sha:
                raise ValueError("Shared signal physical row count/freeze binding differs")
            method_columns = [f"{row['canonical_method']}_{axis}" for axis in AXES]
            reconstructed_metrics.extend(
                metric_rows(
                    row["protocol"],
                    row["split"],
                    row["canonical_method"],
                    frame[method_columns].to_numpy(float),
                    frame,
                )
            )
    validate_metric_rows(rows["metrics_long"], reconstructed_metrics)
    audit = read_json(out / "signal_evaluation/original_full_audit.json")
    if (
        audit.get("selected_cells") != 93
        or audit.get("final_evaluations") != 558
        or audit.get("selection_uses_test") is not False
        or not audit.get("verified_files")
    ):
        raise ValueError("Original93/558 full payload audit evidence is incomplete")
    original_aux = audit.get("original_auxiliary_validation", {})
    for label, stages in PRIOR_STAGES.items():
        entry = original_aux.get(label, {})
        if entry.get("status") != "passed" or set(entry.get("stages", {})) != stages:
            raise ValueError("Original auxiliary actual full-audit stage coverage differs")
        for stage in entry["stages"].values():
            if (
                stage.get("status") != "passed"
                or stage.get("numeric_payload_audit", {}).get("status") != "passed"
            ):
                raise ValueError("Original auxiliary numeric payload audit failed")
        if entry.get("state_sha256") != fit_contract.get(
            "aux1_state_sha256" if label == "aux1" else "aux2_state_sha256"
        ):
            raise ValueError("Original auxiliary audit references a different frozen state")
        if sha(Path(entry["state_path"])) != entry["state_sha256"]:
            raise ValueError("Actual original auxiliary state changed after frozen fit/audit")
    transfer = original_aux["aux1"]["stages"]["transfer"]
    old_downstream = original_aux["aux1"]["stages"]["downstream"]
    if (
        transfer.get("records") != 120
        or transfer.get("unique_records") != 120
        or old_downstream.get("views") != 5
        or set(old_downstream.get("units", {})) != {protocol for protocol, _ in views}
        or any(unit.get("status") != "complete" for unit in old_downstream["units"].values())
    ):
        raise ValueError("Original actual120 transfer/5 downstream gates differ")
    if (
        any(
            original_aux["aux1"]["stages"][name].get("entries") != 558
            for name in ("latency_gpu", "latency_cpu")
        )
        or original_aux["aux2"]["stages"]["accuracy_tables"].get("runs") != 558
        or original_aux["aux2"]["stages"]["classical_latency"].get("final_run_mappings") != 558
    ):
        raise ValueError("Original actual neural/accuracy/classical final coverage differs")
    # Stored state arrays, public accounting and full-workload timing remain independently auditable.
    state_rows = rows["state_inventory"]
    for row in state_rows:
        if (
            int(row["coefficient_bytes"]) != int(row["coefficient_count"]) * 8
            or int(row["normalizer_count"]) != 0
            or int(row["normalizer_bytes"]) != 0
        ):
            raise ValueError("Actual float64 state inventory accounting differs")
    for row in rows["bank_parameter_totals"]:
        selected = [
            item
            for item in state_rows
            if all(
                item[key] == row[key]
                for key in ("protocol", "information_scope", "canonical_method", "variant")
            )
        ]
        if (
            int(row["bank_coefficient_count"])
            != sum(int(item["coefficient_count"]) for item in selected)
            or int(row["bank_coefficient_bytes"])
            != sum(int(item["coefficient_bytes"]) for item in selected)
            or int(row["pair_count"]) != len({item["pair_identity_json"] for item in selected})
        ):
            raise ValueError("Bank deployment totals differ from unique effective pair states")
    for role in ("latency_summary", "workload_latency"):
        for row in rows[role]:
            if int(row["windows_per_call"]) != int(row["pair_count"]) * int(
                row["batch_size_per_pair"]
            ) or any(
                not math.isfinite(float(row[key])) or float(row[key]) <= 0
                for key in ("p50_ms", "p95_ms", "loop_throughput_windows_per_second")
            ):
                raise ValueError("Actual full-loop latency/throughput shape or values differ")
    cpu_validation = validate_cpu_artifacts(out, rows, selections, core, fit_contract)
    scientific_validation = validate_downstream_and_aliases(
        out, rows, selections, core, original_aux, fit_contract
    )
    counts = {
        "shared_protocols": len(PROTOCOLS),
        "shared_logical_controls": len(rows["control_registry"]),
        "shared_learned_controls": learned,
        "shared_identity_controls": identity,
        "shared_signal_slots": len(rows["signal_registry"]),
        "shared_downstream_views": len(views),
        "shared_downstream_summary_slots": len(downstream),
        "shared_paired_final_mappings": len({row["job_id"] for row in neural}),
    }
    if learned != 48 or identity != 6:
        raise ValueError("Actual normalized learned/identity state counts differ")
    result = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "profile": PROFILE,
        "status": "complete",
        "counts": counts,
        "global_selection_set_sha256": selection_sha,
        "fit_contract_sha256": fit_contract_sha,
        "cpu_validation": cpu_validation,
        "scientific_validation": scientific_validation,
        "stage_manifests": manifests,
        "tables": tables,
        "files": {
            f"{stage}/artifact_manifest.json": value["sha256"] for stage, value in manifests.items()
        },
        "original": {
            "core_pipeline_sha256": fit_contract["pipeline_sha256"],
            "selected_cells": audit["selected_cells"],
            "evaluated_final_records": audit["final_evaluations"],
            "neural_transfer_records": transfer["unique_records"],
            "downstream_views": old_downstream["views"],
            "original_full_audit_path": "signal_evaluation/original_full_audit.json",
            "original_full_audit_sha256": sha(out / "signal_evaluation/original_full_audit.json"),
            "original_audit_evidence_sha256": digest(audit),
        },
        "inventory_references": {
            "shared_states": tables["shared_state_inventory"],
            "all_effective_states": tables["state_inventory"],
            "bank_totals": tables["bank_parameter_totals"],
        },
    }
    for stage in STAGES:
        result["files"].update(read_json(out / stage / "artifact_manifest.json")["files"])
    if expected_aux_manifest_sha256 is not None:
        result["aux_manifest_sha256"] = expected_aux_manifest_sha256
    if require_index:
        index = read_json(out / "shared_controls_artifacts.json")
        if index != result:
            raise ValueError("Final shared public index differs from verified actual artifacts")
    return result


def run(args):
    project, pipeline, aux, out = [
        path.resolve() for path in (args.project_root, args.pipeline_root, args.aux_root, args.out)
    ]
    print(
        f"READS: exact original source/status/config/manifest metadata; WRITES: {out}; SKIPS: all fitting/timing until aux2 succeeds, prefreeze numeric payloads",
        flush=True,
    )
    source_manifest = verify_source_manifest(aux, args.aux_manifest_sha256)
    contract = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "project": str(project),
        "pipeline": str(pipeline),
        "aux": str(aux),
        "out": str(out),
        "aux_manifest_sha256": args.aux_manifest_sha256,
        "aux1_state": str(args.aux1_state.resolve()),
        "preceding_state": str(args.preceding_state.resolve()),
        "source_fingerprint": SOURCE_FINGERPRINT,
        "dataset_manifest_sha256": DATASET_SHA256,
        "aux1_manifest_sha256": AUX1_MANIFEST_SHA256,
        "aux2_manifest_sha256": AUX2_MANIFEST_SHA256,
        "dependency_roots": [str(path.resolve()) for path in args.dependency_root],
        "python": str(Path(sys.executable).resolve()),
        "sources": source_manifest["files"],
    }
    state_path = out / "shared_controls_state.json"
    if out.exists():
        if not args.resume:
            raise FileExistsError("Shared output exists; explicit exact-contract --resume required")
        state = read_json(state_path)
        if state.get("contract") != contract:
            raise ValueError("Shared queue resume contract/source identity changed")
        if state.get("status") == "failed":
            raise RuntimeError("Failed queue evidence is preserved; no silent restart")
        if any(item.get("status") != "complete" for item in state.get("stages", {}).values()):
            raise RuntimeError("Interrupted/partial stage evidence is preserved; no silent restart")
    else:
        out.mkdir(parents=True)
        state = {
            "schema_version": 1,
            "analysis_profile": PROFILE,
            "contract": contract,
            "status": "waiting_for_aux2",
            "created_utc": now(),
            "stages": {},
        }
        write_state(state_path, state)
    try:
        previous = None
        while True:
            verify_source_manifest(aux, args.aux_manifest_sha256)
            gate = metadata_gate(
                project, pipeline, args.aux1_state, args.preceding_state, require_aux_terminal=True
            )
            if gate["ready"]:
                break
            compact = {label: item["status"] for label, item in gate["priors"].items()}
            if compact != previous:
                print(
                    f"WAITING: original workflows {compact}; no fit, timing or numeric audit",
                    flush=True,
                )
                previous = compact
            time.sleep(args.poll_seconds)
        if (
            state.get("core_pipeline_sha256", gate["core_pipeline_sha256"])
            != gate["core_pipeline_sha256"]
        ):
            raise ValueError("Original completed pipeline changed across resume")
        if state.get("original_gate", gate) != gate:
            raise ValueError("Original completed source/state metadata changed across resume")
        state.update(core_pipeline_sha256=gate["core_pipeline_sha256"], original_gate=gate)
        aux2_project = Path(read_json(args.preceding_state)["contract"]["aux"]).resolve()
        allowed_dependencies = {aux / "deps", aux2_project / "deps"}
        for dependency in args.dependency_root:
            directory = dependency.resolve()
            if directory not in allowed_dependencies or not directory.is_dir():
                raise ValueError(
                    "Dependency root must be an explicit frozen aux3 or original aux2 deps directory"
                )
            owner = aux if directory.is_relative_to(aux) else aux2_project
            manifest = read_json(owner / "aux_manifest.json")
            for path in directory.rglob("*.py"):
                relative = path.relative_to(owner).as_posix()
                if path.is_symlink() or manifest["files"].get(relative) != sha(path):
                    raise ValueError(
                        "Executed dependency Python source is absent from its pinned manifest"
                    )
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=os.pathsep.join(
                [
                    str(project / "src"),
                    str(aux / "scripts"),
                    *[str(path.resolve()) for path in args.dependency_root],
                ]
            ),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            PYTHONIOENCODING="utf-8",
            MPLBACKEND="Agg",
        )
        for name, command in stage_commands(
            project, pipeline, aux, args.aux1_state.resolve(), args.preceding_state.resolve(), out
        ):
            verify_source_manifest(aux, args.aux_manifest_sha256)
            current_gate = metadata_gate(
                project, pipeline, args.aux1_state, args.preceding_state, require_aux_terminal=True
            )
            if current_gate != gate:
                raise ValueError(
                    "Original successful workflow metadata changed during shared stages"
                )
            if name != "fit_select":
                selection_sha = verify_global_freeze(out)
                if state.get("global_selection_set_sha256", selection_sha) != selection_sha:
                    raise ValueError("Global selected states changed after freeze")
                state["global_selection_set_sha256"] = selection_sha
            previous_stage = state["stages"].get(name)
            if previous_stage:
                summary = verify_stage(out, name)
                if (
                    sha(out / name / "artifact_manifest.json")
                    != previous_stage["artifact_manifest_sha256"]
                    or sha(out / name / "stage_summary.json")
                    != previous_stage["stage_summary_sha256"]
                ):
                    raise ValueError("Completed stage external hashes changed across resume")
                print(f"SKIPS: exact SHA-verified completed shared stage {name}", flush=True)
                continue
            if (out / name).exists():
                raise RuntimeError("Unrecorded completed stage cannot be silently adopted")
            reject_partial_restart(out, name)
            state["status"] = name
            state["stages"][name] = {"status": "running", "command": command, "started_utc": now()}
            write_state(state_path, state)
            print(f"COMMAND: {json.dumps(command, ensure_ascii=False)}", flush=True)
            with (out / f"{name}_console.log").open("x", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=project,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    shell=False,
                )
            summary = verify_stage(out, name)
            state["stages"][name].update(
                status="complete",
                completed_utc=now(),
                artifact_manifest_sha256=sha(out / name / "artifact_manifest.json"),
                stage_summary_sha256=sha(out / name / "stage_summary.json"),
                artifact_sha256=read_json(out / name / "artifact_manifest.json")["files"],
                counts=summary.get("counts", {}),
            )
            write_state(state_path, state)
        verify_source_manifest(aux, args.aux_manifest_sha256)
        if (
            metadata_gate(
                project, pipeline, args.aux1_state, args.preceding_state, require_aux_terminal=True
            )
            != gate
        ):
            raise ValueError(
                "Original successful workflow metadata changed before final completion"
            )
        index = validate_complete_artifacts(
            out,
            gate["core_pipeline_sha256"],
            require_index=False,
            expected_aux_manifest_sha256=args.aux_manifest_sha256,
        )
        index_path = out / "shared_controls_artifacts.json"
        if index_path.exists():
            if read_json(index_path) != index:
                raise ValueError("Completed public index changed")
        else:
            write_once(index_path, index)
        state.update(
            status="complete",
            completed_utc=state.get("completed_utc", now()),
            shared_controls_artifacts_sha256=sha(index_path),
            counts=index["counts"],
            global_selection_set_sha256=index["global_selection_set_sha256"],
        )
        write_state(state_path, state)
        print("REVIEWER_SHARED_CONTROLS_PASSED", flush=True)
        return state
    except Exception as error:
        state.update(
            status="failed", failed_utc=now(), error_type=type(error).__name__, error=str(error)
        )
        write_state(state_path, state)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "pipeline-root",
        "aux-root",
        "aux1-state",
        "preceding-state",
        "out",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--aux-manifest-sha256", required=True)
    result.add_argument("--dependency-root", type=Path, action="append", default=[])
    result.add_argument("--poll-seconds", type=int, choices=range(1, 61), default=30)
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
