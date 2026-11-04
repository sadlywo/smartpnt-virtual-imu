"""Four-project transport and read-only source verification, synthetic only."""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def api():
    path = PROJECT / "scripts/server/export_reviewer_shared_results.py"
    assert path.is_file(), "Four-project exporter has not been implemented"
    return load(path, "shared_export_test")


def four_sources(tmp_path, module):
    roots = {name + "_project": tmp_path / name for name in ("core", "aux1", "aux2", "aux3")}
    files = {}
    for key, root in roots.items():
        path = root / "runs/exact/result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"origin": key}), encoding="utf-8")
        prefix = "core_project" if key == "core_project" else key.removesuffix("_project")
        member = f"server/{prefix}/runs/exact/result.json"
        files[member] = {
            "source_path": str(path),
            "sha256": module.base._sha(path),
            "size_bytes": path.stat().st_size,
            "role": "synthetic_four_root_result",
        }
    return roots, files


def test_four_project_stream_is_received_and_original_aux3_identity_survives(tmp_path, monkeypatch):
    module = api()
    roots, files = four_sources(tmp_path, module)
    audit = module.base.SourceAudit(roots)
    for entry in files.values():
        audit.inspect(Path(entry["source_path"]))
    run_roots = {
        "core_pipeline": roots["core_project"] / "runs/exact",
        **{name: roots[name + "_project"] / "runs/exact" for name in ("aux1", "aux2", "aux3")},
    }
    out = tmp_path / "export"
    result = module.write_package(
        roots,
        run_roots,
        out,
        files,
        {},
        audit,
        validation={"status": "passed", "analysis_profile": module.PROFILE},
        support={},
        settings={"compression_level": 1},
        synthetic=True,
        part_bytes=2048,
    )
    assert len(result["parts"]) > 4
    inner = json.loads((out / module.base.INNER_PATH).read_text(encoding="utf-8"))
    assert len(inner["path_mappings"]) == 4 and inner["analysis_profile"] == module.PROFILE
    assert {
        "metadata/processing/queue_reviewer_shared_export.py",
        "metadata/processing/queue_reviewer_export.py",
    } <= set(inner["files"])
    assert inner["source_roots"]["aux3"] == str(run_roots["aux3"])
    assert module.verify_export(out) == result
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    receiver = load(PROJECT / "scripts/receive_reviewer_results.py", "shared_export_receiver")
    mirror = tmp_path / "mirror"
    received = receiver.receive_results(
        out / "export_manifest.json", out, mirror, _allow_synthetic=True
    )
    assert received["status"] == "complete"
    helper = load(PROJECT / "scripts/reviewer_artifact_package.py", "shared_export_package")
    package = helper.load_package(mirror, allow_synthetic=True)
    target = package.resolve_path(str(run_roots["aux3"] / "result.json"))
    assert package.read_json(target) == {"origin": "aux3_project"}
    assert package.root_path("aux3") == target.parent
    source = roots["aux3_project"] / "runs/exact/result.json"
    source.write_text(source.read_text().replace("aux3", "aux9"))
    with pytest.raises(ValueError, match="source changed"):
        module.verify_export(out)


def test_fourth_root_must_be_disjoint_and_exports_cannot_live_in_any_source(tmp_path):
    module = api()
    roots, _ = four_sources(tmp_path, module)
    module.validate_roots(roots, tmp_path / "separate")
    with pytest.raises(ValueError, match="independent"):
        module.validate_roots(
            {**roots, "aux3_project": roots["aux2_project"] / "nested"}, tmp_path / "separate"
        )
    with pytest.raises(ValueError, match="output"):
        module.validate_roots(roots, roots["aux3_project"] / "export")


def test_old_complete_core_and_auxiliaries_cannot_export_without_shared_workflow(tmp_path):
    module = api()
    fixture = load(PROJECT / "tests/test_reviewer_export.py", "shared_export_original_fixture")
    core, pipeline, aux1, aux2, config = fixture.finished_fixture(tmp_path)
    aux3 = tmp_path / "aux3"
    aux3.mkdir()
    assert callable(getattr(module, "export_results", None)), (
        "Shared completion gates have not been implemented"
    )
    with pytest.raises(ValueError, match="exact auxiliary run identity"):
        module.export_results(
            core,
            pipeline,
            aux1,
            aux2,
            aux3,
            tmp_path / "export",
            config=config,
            _expected_counts=(1, 6),
        )
    assert not (tmp_path / "export").exists()


def test_aux3_header_uses_coherent_scientific_fixture_and_binds_actual_prior_states():
    module = api()
    fixture = load(
        PROJECT / "tests/test_reviewer_shared_controls_queue.py", "shared_header_runtime_fixture"
    )
    tool, out, core_sha = fixture.complete_synthetic_run(materialize_header=True)
    context = tool.read_json(out.parents[2] / "SYNTHETIC_fixture_context.json")
    project = Path(context["core_project"])
    pipeline = Path(context["pipeline_root"])
    aux3 = Path(context["aux3_project"])
    priors = {name: Path(context[name + "_state"]) for name in ("aux1", "aux2")}
    roots = {
        "core_project": project,
        "aux3_project": aux3,
        **{name + "_project": path.parents[2] for name, path in priors.items()},
    }
    # The stored fixture's source-contract intentionally names the actual local
    # producer files. This extra audit owner is only for source bytes in this unit
    # test; the public exporter always constructs exactly four source roots.
    owners = {**roots, "synthetic_source_checkout": PROJECT}
    state_path = out / "shared_controls_state.json"
    state = tool.read_json(state_path)
    state["contract"].update(
        out=str(out),
        analysis_profile=module.PROFILE,
        source_fingerprint=context["source_fingerprint"],
        dataset_manifest_sha256=context["dataset_manifest_sha256"],
        aux1_state=str(priors["aux1"]),
        preceding_state=str(priors["aux2"]),
        aux1_manifest_sha256=tool.sha(roots["aux1_project"] / "aux_manifest.json"),
        aux2_manifest_sha256=tool.sha(roots["aux2_project"] / "aux_manifest.json"),
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    settings = {
        "shared_aux_manifest_sha256": context["aux3_manifest_sha256"],
        "aux3_console_log": "shared_controls_console.log",
        "source_fingerprint": context["source_fingerprint"],
        "dataset_manifest_sha256": context["dataset_manifest_sha256"],
    }
    before = {str(path): path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()}
    run, saved, index = module._shared_header(
        aux3,
        project,
        pipeline,
        roots["aux1_project"],
        roots["aux2_project"],
        core_sha,
        settings,
        module.base.SourceAudit(owners),
        _validator=tool.validate_complete_artifacts,
    )
    assert run == out and saved["status"] == "complete"
    assert index["counts"]["shared_logical_controls"] == 54
    assert before == {
        str(path): path.stat().st_mtime_ns for path in out.rglob("*") if path.is_file()
    }
    changed = tool.read_json(priors["aux1"])
    changed["detached_source_change"] = True
    priors["aux1"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        module._shared_header(
            aux3,
            project,
            pipeline,
            roots["aux1_project"],
            roots["aux2_project"],
            core_sha,
            settings,
            module.base.SourceAudit(owners),
            _validator=tool.validate_complete_artifacts,
        )
