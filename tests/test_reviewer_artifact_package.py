"""Exact-source mapping and byte verification for portable result mirrors."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/reviewer_artifact_package.py"
    spec = importlib.util.spec_from_file_location("reviewer_package_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, *, synthetic=False):
    api = module()
    root = tmp_path / "mirror"
    path = root / "server/core_project/runs/exact/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"value": 0.25}', encoding="utf-8")
    member = path.relative_to(root).as_posix()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": synthetic,
        "roots": {"core_pipeline": "server/core_project/runs/exact"},
        "source_roots": {"core_pipeline": "/remote/project/runs/exact"},
        "path_mappings": [
            {"source_prefix": "/remote", "mirror_prefix": "server"},
            {"source_prefix": "/remote/project", "mirror_prefix": "server/core_project"},
        ],
        "files": {member: api.sha256_file(path)},
        "file_metadata": {
            member: {
                "size_bytes": path.stat().st_size,
                "source_path": "/remote/project/runs/exact/metrics.json",
            }
        },
    }
    saved = root / api.MANIFEST_MEMBER
    saved.parent.mkdir()
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    return api, root, path, saved, manifest


def test_exact_longest_prefix_mapping_preserves_original_identities(tmp_path):
    api, root, path, saved, _ = fixture(tmp_path)
    package = api.load_package(root, expected_manifest_sha256=api.sha256_file(saved))
    assert package.resolve_path("/remote/project/runs/exact/metrics.json") == path
    assert package.resolve_path(path) == path
    assert package.source_path(path) == "/remote/project/runs/exact/metrics.json"
    assert package.read_json(path) == {"value": 0.25}
    assert len(package.verified_files) == 1
    with pytest.raises(ValueError, match="No explicit package mapping"):
        package.resolve_path("/remotely/project/runs/exact/metrics.json")


def test_inventory_detects_same_size_mutation_missing_file_and_wrong_transport_digest(tmp_path):
    api, root, path, _, _ = fixture(tmp_path)
    package = api.load_package(root)
    path.write_text('{"value": 0.75}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        package.verify_all()
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        package.verify_all()
    with pytest.raises(ValueError, match="transport receipt"):
        api.load_package(root, expected_manifest_sha256="0" * 64)


def test_synthetic_and_traversal_are_never_accepted_as_formal_sources(tmp_path):
    api, root, _, _, _ = fixture(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="Synthetic"):
        api.load_package(root)
    package = api.load_package(root, allow_synthetic=True)
    for value in (
        "../outside",
        "/remote/project/../outside",
        "server\\..\\outside",
        "server/core_project/../../../../outside",
        "server//double",
        "C:relative",
    ):
        with pytest.raises(ValueError):
            package.resolve_path(value)


def test_source_file_metadata_cannot_relabel_a_different_run(tmp_path):
    api, root, _, saved, manifest = fixture(tmp_path)
    next(iter(manifest["file_metadata"].values()))["source_path"] = (
        "/remote/project/runs/another/metrics.json"
    )
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different file"):
        api.load_package(root)


def test_generated_audit_has_explicit_null_original_source(tmp_path):
    api, root, _, saved, manifest = fixture(tmp_path)
    generated = root / "metadata/completion_audit.json"
    generated.write_text('{"status":"passed"}', encoding="utf-8")
    member = generated.relative_to(root).as_posix()
    manifest["files"][member] = api.sha256_file(generated)
    manifest["file_metadata"][member] = {
        "size_bytes": generated.stat().st_size,
        "source_path": None,
    }
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    package = api.load_package(root)
    assert package.source_path(generated) is None
    assert package.read_json(generated) == {"status": "passed"}
