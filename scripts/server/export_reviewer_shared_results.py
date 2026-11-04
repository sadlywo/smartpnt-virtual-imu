"""Four-project export transport for the shared-classical comparison profile.

Reuses the byte-audited original exporter without changing its frozen source.
No training, inference, authentication, network, or source deletion occurs here.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import stat
import tarfile
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROFILE = "reviewer_shared_classical_v1"
BASE_SOURCE_SHA256 = "c08e8b2b528f144be22dfde51efd694cfe2258a80c095aa011fdfd49b50c2690"
_base_path = Path(__file__).with_name("export_reviewer_results.py")
_base_bytes = _base_path.read_bytes()
if hashlib.sha256(_base_bytes).hexdigest() != BASE_SOURCE_SHA256:
    raise ValueError("Frozen original exporter source checksum differs")
_base_spec = importlib.util.spec_from_file_location("reviewer_export_frozen_transport", _base_path)
base = importlib.util.module_from_spec(_base_spec)
exec(compile(_base_bytes, str(_base_path), "exec"), base.__dict__)


def validate_roots(roots: dict[str, Path], out: Path) -> None:
    if set(roots) != {"core_project", "aux1_project", "aux2_project", "aux3_project"}:
        raise ValueError("Require all four independent project roots")
    values = list(roots.values())
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(values)
        for b in values[i + 1 :]
    ):
        raise ValueError("All four source projects must be independent")
    if any(out.is_relative_to(root) or root.is_relative_to(out) for root in values):
        raise ValueError("Export output must be independent of every source project")
    for root in roots.values():
        for parent in (root, *root.parents):
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Symlink/reparse source root or ancestor is forbidden")


def verify_export(out: Path, *, verify_sources=True) -> dict:
    """Verify the stream with the old transport, then the fourth source root."""
    result = base.verify_export(out, verify_sources=False)
    inner = json.loads((out / base.INNER_PATH).read_text(encoding="utf-8"))
    if result.get("analysis_profile") != PROFILE or inner.get("analysis_profile") != PROFILE:
        raise ValueError("Completed export lacks the shared-classical analysis profile")
    roots = {
        key: Path(value) for key, value in inner["source_roots"].items() if key.endswith("_project")
    }
    if set(roots) != {"core_project", "aux1_project", "aux2_project", "aux3_project"}:
        raise ValueError("Completed export lacks one of four original project identities")
    if verify_sources:
        validate_roots(roots, out.resolve())
        for member, expected in inner["files"].items():
            metadata = inner["file_metadata"][member]
            if metadata["source_path"] is None:
                if (
                    not member.startswith("metadata/processing/")
                    and member != "metadata/completion_audit.json"
                ):
                    raise ValueError("Only generated processing evidence may lack a source path")
                continue
            source = Path(metadata["source_path"])
            owners = [root for root in roots.values() if source.is_relative_to(root)]
            if (
                len(owners) != 1
                or base._safe_source(owners[0], source).stat().st_size != metadata["size_bytes"]
                or base._sha(source) != expected
            ):
                raise ValueError(f"Export source changed: {source}")
    return result


def write_package(
    roots,
    run_roots,
    out,
    files,
    generated,
    audit,
    *,
    validation,
    support,
    settings,
    synthetic,
    part_bytes=base.PART_BYTES,
    config_path=None,
    excluded=None,
):
    """Stream a previously audited inventory; the public CLI supplies all gates."""
    validate_roots(roots, out)
    if out.exists():
        raise FileExistsError("Export output already exists; verify it instead of overwriting")
    if part_bytes < 1 or not 0 <= settings["compression_level"] <= 9:
        raise ValueError("Invalid compression level or part budget")
    if set(run_roots) != {"core_pipeline", "aux1", "aux2", "aux3"}:
        raise ValueError("Four exact run identities are required")
    for key, run in run_roots.items():
        owner = roots["core_project" if key == "core_pipeline" else key + "_project"]
        if not run.is_relative_to(owner / "runs"):
            raise ValueError("A run identity escapes its source project runs directory")
    import yaml

    self_path = Path(__file__).resolve()
    self_bytes = self_path.read_bytes()
    exporter_sha = hashlib.sha256(self_bytes).hexdigest()
    orchestration = {
        name: self_path.with_name(name).read_bytes()
        for name in ("queue_reviewer_shared_export.py", "queue_reviewer_export.py")
    }
    orchestration_sha = {
        name: hashlib.sha256(data).hexdigest() for name, data in orchestration.items()
    }
    snapshot = yaml.safe_dump(settings, allow_unicode=True, sort_keys=True).encode("utf-8")
    processing = {
        "exporter_source_path": str(self_path),
        "exporter_sha256": exporter_sha,
        "base_exporter_sha256": BASE_SOURCE_SHA256,
        "orchestration_source_sha256": orchestration_sha,
        "effective_config_sha256": hashlib.sha256(snapshot).hexdigest(),
        "config_source_path": str(config_path.resolve()) if config_path else None,
        "config_source_sha256": base._sha(config_path) if config_path else None,
        "source_prefix_mappings": "Four independent original experiment project roots; generated members have source_path null.",
    }
    generated = {
        **generated,
        **{"metadata/processing/" + name: data for name, data in orchestration.items()},
        "metadata/processing/export_reviewer_shared_results.py": self_bytes,
        "metadata/processing/export_reviewer_results.py": _base_bytes,
        "metadata/processing/export_config.yaml": snapshot,
        "metadata/processing/provenance.json": base._json_bytes(processing),
        "metadata/completion_audit.json": base._json_bytes(validation),
    }
    files = dict(files)
    for member, data in generated.items():
        if member in files:
            raise ValueError("Generated member collides with original source inventory")
        files[member] = {
            "source_path": None,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "role": "export_processing_source_config_or_completion_audit",
        }
    files = dict(sorted(files.items()))
    for member in files:
        base.portable(member)
    if len({member.casefold() for member in files}) != len(files):
        raise ValueError("Archive members collide on case-insensitive filesystems")
    mirrors = {
        "core_project": "server/core_project",
        **{key: "server/" + key.removesuffix("_project") for key in roots if key != "core_project"},
    }
    source_roots = {key: str(value) for key, value in {**roots, **run_roots}.items()}
    for key, run in run_roots.items():
        owner_key = "core_project" if key == "core_pipeline" else key + "_project"
        mirrors[key] = mirrors[owner_key] + "/" + run.relative_to(roots[owner_key]).as_posix()
    inner = {
        "schema_version": 1,
        "status": "complete",
        "analysis_profile": PROFILE,
        "synthetic": synthetic,
        "created_utc": datetime.now(UTC).isoformat(),
        "roots": mirrors,
        "source_roots": source_roots,
        "path_mappings": [
            {"source_prefix": str(root), "mirror_prefix": mirrors[key]}
            for key, root in roots.items()
        ],
        "files": {member: entry["sha256"] for member, entry in files.items()},
        "file_metadata": {
            member: {key: value for key, value in entry.items() if key != "sha256"}
            for member, entry in files.items()
        },
        "validation": {"analysis_profile": PROFILE, "audit_path": "metadata/completion_audit.json"},
        "external_inputs": support,
        "excluded": excluded or [],
        "scope": "Frozen core, both original auxiliaries, and shared-classical auxiliary; original bulk corpus remains an explicitly fingerprinted external input.",
    }
    inner_data = base._json_bytes(inner)
    attempt = out.with_name(out.name + ".partial." + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    writer = base.PartWriter(attempt / "parts", part_bytes)
    try:
        (attempt / "metadata").mkdir()
        (attempt / base.INNER_PATH).write_bytes(inner_data)
        with gzip.GzipFile(
            fileobj=writer,
            mode="wb",
            compresslevel=settings["compression_level"],
            mtime=0,
            filename="",
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for member, data in [
                    (base.INNER_PATH, inner_data),
                    *((member, None) for member in files),
                ]:
                    entry = files.get(member)
                    data = generated.get(member, data)
                    info = tarfile.TarInfo(member)
                    info.mode, info.mtime = 0o644, 0
                    if data is not None:
                        info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
                        continue
                    source = Path(entry["source_path"])
                    before = audit.inspect(source)
                    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    with os.fdopen(os.open(source, flags), "rb") as stream:
                        if base._signature(os.fstat(stream.fileno())) != before["signature"]:
                            raise ValueError("Source changed before tar streaming")
                        reader = base.CheckedReader(stream)
                        info.size = entry["size_bytes"]
                        archive.addfile(info, reader)
                        if (
                            reader.count != entry["size_bytes"]
                            or reader.digest.hexdigest() != entry["sha256"]
                            or stream.read(1)
                            or base._signature(os.fstat(stream.fileno())) != before["signature"]
                        ):
                            raise ValueError("Source changed during tar streaming")
        writer.finish()
        audit.assert_unchanged()
        if (
            base._sha(self_path) != exporter_sha
            or base._sha(_base_path) != BASE_SOURCE_SHA256
            or (config_path and base._sha(config_path) != processing["config_source_sha256"])
        ):
            raise ValueError("Export processing source or configuration changed")
        if any(
            base._sha(self_path.with_name(name)) != expected
            for name, expected in orchestration_sha.items()
        ):
            raise ValueError("Export orchestration source changed during streaming")
        outer = {
            "schema_version": 1,
            "status": "complete",
            "analysis_profile": PROFILE,
            "archive": {
                "format": "tar.gz",
                "sha256": writer.global_hash.hexdigest(),
                "size_bytes": writer.total,
            },
            "inner_manifest": {
                "path": base.INNER_PATH,
                "sha256": hashlib.sha256(inner_data).hexdigest(),
            },
            "parts": writer.parts,
            "source_roots": source_roots,
            "files": len(files),
            "source_bytes": sum(entry["size_bytes"] for entry in files.values()),
            "config": settings,
            "exporter_sha256": exporter_sha,
            "processing_provenance": processing,
        }
        base._write_json(attempt / "export_manifest.json", outer)
        verify_export(attempt, verify_sources=False)
        attempt.rename(out)
        return outer
    except Exception as error:
        writer.abort()
        base._write_json(
            attempt / "failure.json",
            {
                "status": "partial_failed_attempt_preserved",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "completed_parts": writer.parts,
            },
        )
        raise


def _shared_header(
    aux, project, pipeline, aux1, aux2, core_sha, settings, audit, *, _validator=None
):
    run, state = base._find_aux_run(aux, "shared_controls_state.json", project, pipeline, audit)
    manifest_path = aux / "aux_manifest.json"
    manifest_sha = settings.get("shared_aux_manifest_sha256")
    if not isinstance(manifest_sha, str):
        raise ValueError("Shared auxiliary source manifest SHA must be pinned in export config")
    audit.check(manifest_path, manifest_sha)
    manifest = audit.json(manifest_path)
    contract = state.get("contract", {})
    if (
        state.get("analysis_profile") != PROFILE
        or state.get("status") != "complete"
        or state.get("core_pipeline_sha256") != core_sha
        or contract.get("analysis_profile") != PROFILE
        or contract.get("out") != str(run)
        or contract.get("aux_manifest_sha256") != manifest_sha
    ):
        raise ValueError("Shared auxiliary is not a complete exact-source workflow")
    if (
        Path(manifest.get("base_project", "")).resolve() != project
        or (project / manifest.get("base_pipeline", "")).resolve() != pipeline
        or contract.get("sources") != manifest.get("files")
    ):
        raise ValueError("Shared auxiliary source/core identity differs")
    if (
        contract.get("source_fingerprint") != settings.get("source_fingerprint")
        or contract.get("dataset_manifest_sha256") != settings["dataset_manifest_sha256"]
    ):
        raise ValueError("Shared auxiliary original source/data fingerprints differ")
    base._file_map(aux, manifest["files"], audit)
    base._terminal(aux / settings["aux3_console_log"], "REVIEWER_SHARED_CONTROLS_PASSED", audit)
    expected_stages = {"fit_select", "signal_evaluation", "downstream", "cpu_latency"}
    if set(state.get("stages", {})) != expected_stages or any(
        item.get("status") != "complete" for item in state["stages"].values()
    ):
        raise ValueError("Shared auxiliary does not cover all four completed stages")
    validator_path = aux / "scripts/server/run_reviewer_shared_controls.py"
    if validator_path.relative_to(aux).as_posix() not in manifest["files"]:
        raise ValueError("Shared validator is absent from pinned auxiliary source manifest")
    spec = importlib.util.spec_from_file_location(
        "reviewer_shared_export_verified_validator", validator_path
    )
    module = importlib.util.module_from_spec(spec)
    data = validator_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != audit.inspect(validator_path)["sha256"]:
        raise ValueError("Shared validator changed before compilation")
    exec(compile(data, str(validator_path), "exec"), module.__dict__)
    # Its first operation verifies the complete six-protocol freeze barrier.
    # The private test callback uses the same strict validator with synthetic
    # source/data fingerprints. export_results and the CLI always use the
    # checksum-verified module loaded above, with the frozen real fingerprints.
    validator = module.validate_complete_artifacts if _validator is None else _validator
    index = validator(run, core_sha, expected_aux_manifest_sha256=manifest_sha)
    audit.check(run / "shared_controls_artifacts.json", state["shared_controls_artifacts_sha256"])
    if (
        index.get("analysis_profile") != PROFILE
        or index.get("status") != "complete"
        or state.get("counts") != index.get("counts")
        or state.get("global_selection_set_sha256") != index.get("global_selection_set_sha256")
    ):
        raise ValueError("Shared public index and completed workflow state disagree")
    fit_contract = audit.json(run / "fit_select/contract.json")
    if (
        fit_contract.get("pipeline_root") != str(pipeline)
        or fit_contract.get("pipeline_sha256") != core_sha
    ):
        raise ValueError("Shared fit contract binds a different exact core")
    base._file_map(aux, fit_contract["source_sha256"], audit, absolute=True)
    config_path = Path(fit_contract["config_path"])
    audit.check(config_path, fit_contract["config_sha256"])
    import yaml

    if yaml.safe_load(config_path.read_text(encoding="utf-8")) != fit_contract["config"]:
        raise ValueError("Shared fitted configuration content differs")
    original_audit = audit.json(run / "signal_evaluation/original_full_audit.json")
    for stage in expected_stages:
        stage_manifest = audit.json(run / stage / "artifact_manifest.json")
        if stage_manifest.get("root_kind") != "aux3_run" or stage_manifest.get("files") != state[
            "stages"
        ][stage].get("artifact_sha256"):
            raise ValueError("Shared stage final inventory differs from workflow state")
        base._file_map(run, stage_manifest["files"], audit)
    for key, root, name in (
        ("aux1_state", aux1, "postprocessing_state.json"),
        ("preceding_state", aux2, "final_analysis_state.json"),
    ):
        prior_run, _ = base._find_aux_run(root, name, project, pipeline, audit)
        if contract.get(key) != str(prior_run / name):
            raise ValueError("Shared auxiliary preceding workflow run differs")
        expected = contract.get(
            "aux1_manifest_sha256" if key == "aux1_state" else "aux2_manifest_sha256"
        )
        audit.check(root / "aux_manifest.json", expected)
        label = "aux1" if key == "aux1_state" else "aux2"
        current_state_sha = audit.inspect(prior_run / name)["sha256"]
        if (
            fit_contract.get(label + "_state_sha256") != current_state_sha
            or original_audit.get("original_auxiliary_validation", {})
            .get(label, {})
            .get("state_sha256")
            != current_state_sha
        ):
            raise ValueError(
                "Shared fit and numeric audit do not bind the current original auxiliary state"
            )
    return run, state, index


def export_results(
    project_root,
    pipeline_root,
    aux1_root,
    aux2_root,
    aux3_root,
    out,
    *,
    config=None,
    resume=False,
    _expected_counts=(93, 558),
    _part_bytes=base.PART_BYTES,
    _config_path=None,
):
    settings = {
        **base.DEFAULTS,
        "aux3_console_log": "shared_controls_console.log",
        "source_fingerprint": "b6a2b7a3f3bab8ce16d4f1225175cf19415736d4afe2a398fa2bee2415e019ec",
        **(config or {}),
    }
    roots = {
        key: Path(os.path.abspath(path))
        for key, path in zip(
            ("core_project", "aux1_project", "aux2_project", "aux3_project"),
            (project_root, aux1_root, aux2_root, aux3_root),
            strict=True,
        )
    }
    out = Path(os.path.abspath(out))
    validate_roots(roots, out)
    project, aux1, aux2, aux3 = (
        roots[key] for key in ("core_project", "aux1_project", "aux2_project", "aux3_project")
    )
    pipeline = Path(os.path.abspath(pipeline_root))
    if not pipeline.is_relative_to(project / "runs"):
        raise ValueError("Core pipeline must remain under the original project runs directory")
    print(
        f"READS: frozen core {pipeline} and all three completed exact-source auxiliaries",
        flush=True,
    )
    print(f"WRITES: independent four-project multipart export {out}", flush=True)
    print(
        "SKIPS: training, inference, network/authentication, deletion, original bulk corpus and incomplete shared controls",
        flush=True,
    )
    audit = base.SourceAudit(roots)
    state = base._core_header(project, pipeline, settings, audit, _expected_counts)
    core_sha = audit.inspect(pipeline / "pipeline.json")["sha256"]
    # Verify shared freeze before any original result payload inventory/read.
    aux3_run, aux3_state, index = _shared_header(
        aux3, project, pipeline, aux1, aux2, core_sha, settings, audit
    )
    aux1_run, aux1_state = base._aux_header(
        "aux1", aux1, project, pipeline, core_sha, settings, audit, _expected_counts[1]
    )
    aux2_run, aux2_state = base._aux_header(
        "aux2", aux2, project, pipeline, core_sha, settings, audit, _expected_counts[1]
    )
    if (
        Path(aux2_state["contract"].get("preceding_state", "")).resolve()
        != aux1_run / "postprocessing_state.json"
        or aux2_state["contract"].get("preceding_manifest_sha256")
        != audit.inspect(aux1 / "aux_manifest.json")["sha256"]
    ):
        raise ValueError("Original auxiliary chain does not bind the verified preceding run")
    if out.exists():
        if not resume:
            raise FileExistsError("Existing export requires explicit --resume verification")
        existing = verify_export(out)
        if existing["config"] != settings or existing.get("exporter_sha256") != base._sha(
            Path(__file__)
        ):
            raise ValueError("Completed export source/config identity changed")
        return existing
    if resume:
        raise RuntimeError(
            "No completed export exists to resume; partial attempts remain preserved"
        )
    validation = base._validate_completed_core(project, pipeline, state, audit)
    support = base._validate_support(project, settings, audit)
    files, skipped = base._collect(project, pipeline, aux1, aux2, settings, audit)
    for source in base._walk(aux3, aux3, skipped):
        member = "server/aux3/" + source.relative_to(aux3).as_posix()
        record = audit.inspect(source)
        files[member] = {
            "source_path": str(source),
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
            "role": "shared_auxiliary_source_config_state_results_and_logs",
        }
    completed = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "status": "passed",
        "synthetic": _expected_counts != (93, 558),
        "counts": {
            "selected_cells": validation["cells"],
            "final_evaluated": validation["validated_final_evaluations"],
            "transfer_models": aux1_state["export_validation"]["transfer"]["count"],
            "downstream_views": aux1_state["export_validation"]["downstream"]["count"],
            **index["counts"],
            "shared_paired_support_refs": 720,
        },
        "source_pipeline": str(pipeline / "pipeline.json"),
        "source_pipeline_sha256": core_sha,
        "checkpoint_checks": [
            {
                "job_id": job["job_id"],
                "phase": job["phase"],
                "source_path": job["checkpoint"]["path"],
                "sha256": job["checkpoint"]["sha256"],
                "finite": True,
                "tensor_count": job["checkpoint"]["model_tensors"],
                "tensor_elements": job["checkpoint"]["tensor_elements"],
            }
            for job in validation["jobs"]
            if job["status"] == "passed"
        ],
        "prediction_checks": [
            {
                "job_id": job["job_id"],
                "split": split,
                "source_path": record["path"],
                "sha256": record["sha256"],
                "finite": record["numeric_finite"],
                "rows": record["rows"],
                "columns": record["total_columns"],
                "checked_numeric_columns": record["checked_numeric_columns"],
                "checked_identity_columns": record["checked_identity_columns"],
                "identity_nonnull": record["identity_nonnull"],
            }
            for job in validation["jobs"]
            for split, record in job.get("evaluation", {}).get("predictions", {}).items()
        ],
        "selection_checks": validation["selections"],
        "fit_checks": validation["jobs"],
        "auxiliary_checks": {
            "aux1": aux1_state["export_validation"],
            "aux2": aux2_state["export_validation"],
            "aux3": index,
        },
        "shared_controls_index": {
            "source_path": str(aux3_run / "shared_controls_artifacts.json"),
            "sha256": aux3_state["shared_controls_artifacts_sha256"],
        },
        "marker_checks": [
            {
                "source_path": str(path),
                "sha256": audit.inspect(path)["sha256"],
                "status": "passed",
                "required_marker": marker,
            }
            for path, marker in (
                (
                    project / settings["core_workflow_log"],
                    "REVIEWER_CORE_PASSED + final WORKFLOW_EXIT=0",
                ),
                (aux1 / settings["aux1_console_log"], "REVIEWER_AUXILIARY_PASSED"),
                (aux2 / settings["aux2_console_log"], "REVIEWER_FINAL_ANALYSIS_PASSED"),
                (aux3 / settings["aux3_console_log"], "REVIEWER_SHARED_CONTROLS_PASSED"),
            )
        ],
        "source_roster_check": {
            "status": "passed",
            "scope": "Frozen original core and both original auxiliaries remain validated by the pinned original exporter; shared auxiliary source manifest, all four stage inventories and actual identity/finite/selection coverage additionally validated.",
        },
        "processing_source_sha256": base._sha(Path(__file__)),
    }
    return write_package(
        roots,
        {"core_pipeline": pipeline, "aux1": aux1_run, "aux2": aux2_run, "aux3": aux3_run},
        out,
        files,
        {},
        audit,
        validation=completed,
        support=support,
        settings=settings,
        synthetic=_expected_counts != (93, 558),
        part_bytes=_part_bytes,
        config_path=_config_path,
        excluded=skipped,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "pipeline-root",
        "aux1-root",
        "aux2-root",
        "aux3-root",
        "config",
        "out",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    export_results(
        args.project_root,
        args.pipeline_root,
        args.aux1_root,
        args.aux2_root,
        args.aux3_root,
        args.out,
        config=config,
        resume=args.resume,
        _config_path=args.config,
    )
    print("REVIEWER_SHARED_EXPORT_PASSED", flush=True)


if __name__ == "__main__":
    main()
