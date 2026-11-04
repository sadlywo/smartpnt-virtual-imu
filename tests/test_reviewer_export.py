"""Real tiny checkpoint/Parquet workflow; no actual experiment export."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exporter():
    return load(PROJECT / "scripts/server/export_reviewer_results.py", "export_test")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")


def inventory(directory):
    return {path.relative_to(directory).as_posix(): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def finished_fixture(base):
    """Six real final fits; genuine one-failed-search ledger and hashed sources."""
    core, aux1, aux2 = (base / name for name in ("core", "aux1", "aux2"))
    core.mkdir(parents=True)
    helper = load(PROJECT / "tests/test_reviewer_results.py", "saved_results_fixture")
    pipeline = helper.fixture(core, fail_first=True)
    source = core / "src/virtual_imu/training/reviewer_hpo.py"
    source.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT / "src/virtual_imu/training/reviewer_hpo.py", source)
    state = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
    # Fixture callback predates formal report split wrapper; bind its saved rows.
    for job in state["jobs"].values():
        if not job["phase"].startswith("final_"):
            continue
        path = pipeline / "evaluations" / f"{job['id']}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["metrics"]["splits"] = {"validation": {}, "test": {}}
        write(path, report)
        job["evaluation_sha256"] = sha(path)
        job["evaluation_artifact_sha256"][str(path)] = sha(path)
    write(pipeline / "pipeline.json", state)
    (core / "runs/reviewer_workflow_status.log").write_text("REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n", encoding="utf-8")
    # Metadata included; opaque original bulk bytes verified but omitted.
    dataset = core / "data/processed/unified_imu_v2/manifest.json"
    write(dataset, {"records": []})
    bulk = dataset.parent / "sequences/original.parquet"
    bulk.parent.mkdir()
    bulk.write_bytes(os.urandom(4096))
    gt_trees = ["data/raw/penncosyvio_ground_truth", "data/processed/penncosyvio_ground_truth"]
    raw = core / gt_trees[0] / "af/pose.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("0 1 2 3\n", encoding="utf-8")
    processed = core / gt_trees[1] / "af.parquet"
    processed.parent.mkdir(parents=True)
    processed.write_bytes(b"tiny independent optical support fixture")
    records = [{"path": raw.relative_to(core).as_posix(), "sha256": sha(raw), "processed_path": processed.relative_to(core).as_posix(), "processed_sha256": sha(processed)}]
    for tree in gt_trees:
        write(core / tree / "manifest.json", {"records": records})
    write(core / "bundle_manifest.json", {"dataset_manifest_sha256": sha(dataset), "entries": [{"path": path.relative_to(core).as_posix(), "sha256": sha(path), "size_bytes": path.stat().st_size} for path in (source, dataset, bulk)]})
    core_sha = sha(pipeline / "pipeline.json")
    for number, aux in ((1, aux1), (2, aux2)):
        aux.mkdir()
        (aux / "scripts").mkdir()
        (aux / "scripts/helper.py").write_text("# immutable auxiliary source\n", encoding="utf-8")
        (aux / "deps.bin").write_bytes(os.urandom(25000))
        (aux / "auth.json").write_text("EXCLUDED TEST SECRET", encoding="utf-8")
        (aux / ".venv").mkdir()
        (aux / ".venv/runtime.bin").write_bytes(b"EXCLUDED RUNTIME")
        manifest = {"base_project": str(core), "base_pipeline": pipeline.relative_to(core).as_posix(), "files": {"scripts/helper.py": sha(aux / "scripts/helper.py"), "deps.bin": sha(aux / "deps.bin")}}
        write(aux / "aux_manifest.json", manifest)
        contract = {"project": str(core), "pipeline": str(pipeline), "aux": str(aux), "aux_manifest_sha256": sha(aux / "aux_manifest.json")}
        if number == 2:
            contract.update(preceding_state=str(aux1 / "runs/postprocess/postprocessing_state.json"), preceding_manifest_sha256=sha(aux1 / "aux_manifest.json"))
        output = aux / "runs" / ("postprocess" if number == 1 else "analysis")
        stages = {}
        names = ("transfer", "downstream", "latency_gpu", "latency_cpu") if number == 1 else ("accuracy_tables", "classical_latency")
        for name in names:
            directory = output / name
            write(directory / "contract.json", {"pipeline": str(pipeline), "pipeline_sha256": core_sha, "expected_records": 2, "nonfinal_debug_subset": False})
            if name == "transfer":
                write(directory / "transfer_metrics.json", {"status": "complete", "records": 2, "source_model_records": [{"direction": direction, "model_family": "gru", "variant": "tuned", "seed": 20261010} for direction in ("alimu_to_penn", "penn_to_alimu")]})
            elif name == "downstream":
                units = {}
                for unit in ("penn_loso_af", "penn_loso_as", "penn_loso_bf", "penn_only", "pooled", "overview"):
                    folder = directory / ("overview" if unit == "overview" else "units/" + unit)
                    write(folder / "metrics.json", {"finite_error": 1.0})
                    write(folder / "artifact_manifest.json", {"files": inventory(folder)})
                    units[unit] = sha(folder / "artifact_manifest.json")
                overview = units.pop("overview")
                write(directory / "complete.json", {"status": "complete", "unit_manifests": units, "overview_manifest_sha256": overview})
            elif name.startswith("latency_"):
                write(directory / "latency.json", {"status": "complete", "source_pipeline": str(pipeline), "source_pipeline_sha256": core_sha, "entries": 6})
                write(directory / "latency_state.json", {"status": "complete", "groups": {"fixed_architecture": {"status": "complete"}}, "final_artifact_sha256": {"latency.json": sha(directory / "latency.json")}})
            elif name == "accuracy_tables":
                write(directory / "summary.json", {"status": "complete", "pipeline_root": str(pipeline), "source_pipeline_sha256": core_sha, "runs": 6})
            else:
                write(directory / "latency.json", {"status": "complete", "final_run_mappings": 6})
            if not name.startswith("latency_"):
                write(directory / "artifact_manifest.json", {"files": inventory(directory)})
            stages[name] = {"status": "complete", "artifact_sha256": inventory(directory)}
        write(output / ("postprocessing_state.json" if number == 1 else "final_analysis_state.json"), {"status": "complete", "contract": contract, "core_pipeline_sha256": core_sha, "stages": stages})
        (aux / ("postprocessing_console.log" if number == 1 else "final_analysis_console.log")).write_text("REVIEWER_AUXILIARY_PASSED\n" if number == 1 else "REVIEWER_FINAL_ANALYSIS_PASSED\n", encoding="utf-8")
    return core, pipeline, aux1, aux2, {"dataset_manifest_sha256": sha(dataset)}


def run_export(base, *, out=None):
    module = exporter()
    roots = finished_fixture(base)
    destination = out or base / "export"
    report = module.export_results(*roots[:4], destination, config=roots[4], _expected_counts=(1, 6), _part_bytes=4096)
    return module, roots, destination, report


def test_real_stream_crosses_parts_and_safe_reassembly_matches_every_hash(tmp_path):
    module, roots, out, report = run_export(tmp_path)
    assert len(report["parts"]) > 5
    assert all(part["size_bytes"] == 4096 for part in report["parts"][:-1])
    compressed = b"".join((out / part["path"]).read_bytes() for part in report["parts"])
    assert hashlib.sha256(compressed).hexdigest() == report["archive"]["sha256"]
    inner_data = (out / "metadata/package_manifest.json").read_bytes()
    inner = json.loads(inner_data)
    assert inner["synthetic"] and len(inner["path_mappings"]) == 3
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as archive:
        members = archive.getmembers()
        assert members[0].name == "metadata/package_manifest.json"
        assert archive.extractfile(members[0]).read() == inner_data
        seen = set()
        for member in members:
            assert member.isfile() and not member.issym() and not member.islnk()
            safe = module.portable(member.name)
            assert member.name.casefold() not in seen
            seen.add(member.name.casefold())
            path = mirror.joinpath(*safe.parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                shutil.copyfileobj(archive.extractfile(member), target)
            if member.name != "metadata/package_manifest.json":
                assert sha(path) == inner["files"][member.name]
                assert path.stat().st_size == inner["file_metadata"][member.name]["size_bytes"]
    helper = load(PROJECT / "scripts/reviewer_artifact_package.py", "export_package_helper")
    package = helper.load_package(mirror, allow_synthetic=True)
    assert package.root_path("core_pipeline").name == roots[1].name
    audit = package.read_json("metadata/completion_audit.json")
    assert audit["counts"] == {"selected_cells": 1, "final_evaluated": 6, "transfer_models": 2, "downstream_views": 5}
    assert len(audit["prediction_checks"]) == 12 and all(row["finite"] and row["rows"] == 320 for row in audit["prediction_checks"])
    assert all(row["finite"] and row["tensor_count"] > 0 for row in audit["checkpoint_checks"])
    assert any(row["status"] == "skipped_as_failed" for row in audit["fit_checks"])
    assert not any("auth.json" in name or ".venv" in name or "sequences/original.parquet" in name for name in inner["files"])
    assert {"metadata/processing/export_reviewer_results.py", "metadata/processing/export_config.yaml", "metadata/processing/provenance.json"} <= set(inner["files"])
    times = {part["path"]: (out / part["path"]).stat().st_mtime_ns for part in report["parts"]}
    assert module.export_results(*roots[:4], out, config=roots[4], resume=True, _expected_counts=(1, 6)) == report
    assert times == {part["path"]: (out / part["path"]).stat().st_mtime_ns for part in report["parts"]}
    part = out / report["parts"][0]["path"]
    original = part.read_bytes()
    part.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="part SHA"):
        module.export_results(*roots[:4], out, config=roots[4], resume=True, _expected_counts=(1, 6))
    part.write_bytes(original)
    source = roots[2] / "scripts/helper.py"
    source.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 differs"):
        module.export_results(*roots[:4], out, config=roots[4], resume=True, _expected_counts=(1, 6))


def test_production_gate_and_missing_terminal_marker_reject_state_alone(tmp_path):
    module, roots = exporter(), finished_fixture(tmp_path)
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="complete selected/final"):
        module.export_results(*roots[:4], out, config=roots[4])
    (roots[0] / "runs/reviewer_workflow_status.log").write_text("status complete\nWORKFLOW_EXIT=0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal success"):
        module.export_results(*roots[:4], out, config=roots[4], _expected_counts=(1, 6))
    assert not out.exists() and not list(tmp_path.glob("export.partial.*"))


def test_partial_attempt_is_preserved_and_never_overwritten_or_silently_resumed(tmp_path, monkeypatch):
    module, roots = exporter(), finished_fixture(tmp_path)
    out = tmp_path / "export"
    original = module.PartWriter.write
    calls = 0

    def failing(self, data):
        nonlocal calls
        calls += 1
        result = original(self, data)
        if self.total > 10000:
            raise OSError("injected disk interruption")
        return result

    monkeypatch.setattr(module.PartWriter, "write", failing)
    with pytest.raises(OSError, match="disk interruption"):
        module.export_results(*roots[:4], out, config=roots[4], _expected_counts=(1, 6), _part_bytes=4096)
    attempts = list(tmp_path.glob("export.partial.*"))
    assert len(attempts) == 1 and not out.exists()
    snapshot = inventory(attempts[0])
    assert json.loads((attempts[0] / "failure.json").read_text(encoding="utf-8"))["status"] == "partial_failed_attempt_preserved"
    with pytest.raises(RuntimeError, match="partial attempts are preserved"):
        module.export_results(*roots[:4], out, config=roots[4], resume=True, _expected_counts=(1, 6))
    assert snapshot == inventory(attempts[0])
    monkeypatch.setattr(module.PartWriter, "write", original)
    module.export_results(*roots[:4], out, config=roots[4], _expected_counts=(1, 6), _part_bytes=4096)
    assert snapshot == inventory(attempts[0])


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a\\b", "a/../b", "a//b", "a/CON.txt", "a/trailing.", "C:/bad"])
def test_portable_paths_reject_traversal_and_windows_aliases(name):
    with pytest.raises(ValueError, match="Unsafe portable"):
        exporter().portable(name)


def test_source_mutation_after_validation_aborts_and_preserves_partial(tmp_path, monkeypatch):
    module, roots = exporter(), finished_fixture(tmp_path)
    collect = module._collect

    def mutate(*args, **kwargs):
        result = collect(*args, **kwargs)
        (roots[2] / "deps.bin").write_bytes(b"mutated after SHA inventory")
        return result

    monkeypatch.setattr(module, "_collect", mutate)
    with pytest.raises(ValueError, match="Source changed"):
        module.export_results(*roots[:4], tmp_path / "export", config=roots[4], _expected_counts=(1, 6), _part_bytes=4096)
    assert not (tmp_path / "export").exists()
    assert len(list(tmp_path.glob("export.partial.*/failure.json"))) == 1


@pytest.mark.parametrize("kind", ["selection", "predictions", "aux_output", "aux_marker", "source_roster"])
def test_initial_export_rejects_changed_frozen_evidence(tmp_path, kind):
    module, roots = exporter(), finished_fixture(tmp_path)
    core, pipeline, aux1, _, config = roots
    if kind == "selection":
        path = next((pipeline / "selections").glob("*.json"))
    elif kind == "predictions":
        path = next(pipeline.glob("experiments/*/fit/evaluation/test_predictions.parquet"))
    elif kind == "aux_output":
        path = aux1 / "runs/postprocess/transfer/transfer_metrics.json"
    elif kind == "aux_marker":
        path = aux1 / "postprocessing_console.log"
    else:
        path = core / "src/virtual_imu/training/reviewer_hpo.py"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        module.export_results(*roots[:4], tmp_path / "export", config=config, _expected_counts=(1, 6))
    assert not (tmp_path / "export").exists()


def test_finite_prediction_scope_ignores_unused_nan_but_rejects_required_nan(tmp_path):
    import numpy as np
    import pandas as pd

    module, roots = exporter(), finished_fixture(tmp_path)
    path = next(roots[1].glob("experiments/*/fit/evaluation/test_predictions.parquet"))
    frame = pd.read_parquet(path)
    frame["unused_optional_temperature"] = np.nan
    frame.to_parquet(path, index=False)
    audit = module.SourceAudit({"core_project": roots[0]})
    assert module._finite_predictions(path, "paired", audit)["rows"] == 320
    frame.loc[0, "model_gx"] = np.nan
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Nonfinite final prediction"):
        module._finite_predictions(path, "paired", module.SourceAudit({"core_project": roots[0]}))


def test_nonfinite_checkpoint_is_rejected_even_when_its_new_hash_is_declared(tmp_path):
    import torch

    module, roots = exporter(), finished_fixture(tmp_path)
    state = json.loads((roots[1] / "pipeline.json").read_text(encoding="utf-8"))
    job = next(job for job in state["jobs"].values() if job["phase"] == "search" and job["status"] == "complete")
    path = Path(job["run_dir"]) / "best.ckpt"
    checkpoint = torch.load(path, weights_only=True)
    next(iter(checkpoint["model_state_dict"].values())).view(-1)[0] = float("nan")
    torch.save(checkpoint, path)
    job["checkpoint_sha256"] = sha(path)
    job["artifact_sha256"]["best.ckpt"] = sha(path)
    with pytest.raises(ValueError, match="Nonfinite or missing completed model tensors"):
        module._validate_completed_core(roots[0], roots[1], state, module.SourceAudit({"core_project": roots[0]}))


def test_source_symlink_is_forbidden(tmp_path):
    module = exporter()
    source, link = tmp_path / "source.txt", tmp_path / "link.txt"
    source.write_text("regular", encoding="utf-8")
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("Host does not authorize creating symlinks")
    with pytest.raises(ValueError, match="Symlink/reparse"):
        module.SourceAudit({"source": tmp_path}).inspect(link)
