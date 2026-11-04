"""Real multipart gzip/tar extraction, interruption, and integrity behavior."""

import hashlib
import importlib.util
import io
import json
import random
import tarfile
from pathlib import Path

import pytest


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "reviewer_receive_test", scripts / "receive_reviewer_results.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def transport_fixture(tmp_path, api, *, extra=None, omit_model=False):
    source = {
        "server/core_project/runs/run/metrics.json": b'{"value":0.25}',
        "server/core_project/runs/run/best.ckpt": random.Random(15).randbytes(40960),
    }

    def sha(value):
        return hashlib.sha256(value).hexdigest()

    inner = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": True,
        "roots": {"core_pipeline": "server/core_project/runs/run"},
        "source_roots": {"core_pipeline": "/srv/frozen/runs/run"},
        "path_mappings": [{"source_prefix": "/srv/frozen", "mirror_prefix": "server/core_project"}],
        "files": {name: sha(value) for name, value in source.items()},
        "file_metadata": {
            name: {
                "size_bytes": len(value),
                "source_path": "/srv/frozen/" + name.removeprefix("server/core_project/"),
            }
            for name, value in source.items()
        },
    }
    inner_bytes = json.dumps(inner).encode()
    target = io.BytesIO()
    with tarfile.open(fileobj=target, mode="w:gz") as archive:
        for name, content in [(api.MANIFEST_MEMBER, inner_bytes), *source.items()]:
            if omit_model and name.endswith("best.ckpt"):
                continue
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if extra is not None:
            info = tarfile.TarInfo(extra)
            info.type = tarfile.SYMTYPE
            info.linkname = str(tmp_path / "outside")
            archive.addfile(info)
    compressed = target.getvalue()
    parts_root = tmp_path / "download"
    (parts_root / "parts").mkdir(parents=True)
    parts = []
    for index, start in enumerate(range(0, len(compressed), 1024)):
        data = compressed[start : start + 1024]
        name = f"parts/results.part{index:05d}"
        (parts_root / name).write_bytes(data)
        parts.append({"path": name, "sha256": sha(data), "size_bytes": len(data)})
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "archive": {"format": "tar.gz", "sha256": sha(compressed), "size_bytes": len(compressed)},
        "inner_manifest": {"path": api.MANIFEST_MEMBER, "sha256": sha(inner_bytes)},
        "parts": parts,
    }
    saved = parts_root / "export_manifest.json"
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    return saved, parts_root, source, manifest


def test_multipart_roundtrip_and_completed_resume_needs_no_temporary_archive(tmp_path, monkeypatch):
    api = module(monkeypatch)
    saved, parts_root, sources, manifest = transport_fixture(tmp_path, api)
    out = tmp_path / "mirror"
    result = api.receive_results(saved, parts_root, out, _allow_synthetic=True)
    assert result["status"] == "complete" and result["verified_files"] == 2
    assert len(manifest["parts"]) > 30
    for name, content in sources.items():
        assert (out / name).read_bytes() == content
    assert not list(out.rglob(".receive-partial-*"))
    # A verified completed mirror can be checked after transport caches move away.
    for part in manifest["parts"]:
        (parts_root / part["path"]).unlink()
    resumed = api.receive_results(saved, parts_root, out, resume=True, _allow_synthetic=True)
    assert resumed["resumed_without_extraction"]


def test_interrupted_commit_preserves_partial_then_skips_completed_file(tmp_path, monkeypatch):
    api = module(monkeypatch)
    saved, parts_root, sources, _ = transport_fixture(tmp_path, api)
    out = tmp_path / "mirror"
    original, calls = api.os.link, []

    def interrupt(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("synthetic interrupted commit")
        return original(source, destination)

    monkeypatch.setattr(api.os, "link", interrupt)
    with pytest.raises(OSError, match="interrupted commit"):
        api.receive_results(saved, parts_root, out, _allow_synthetic=True)
    retained = list(out.rglob(".receive-partial-*"))
    assert len(retained) == 1 and retained[0].stat().st_size == len(
        next(v for k, v in sources.items() if k.endswith("best.ckpt"))
    )
    monkeypatch.setattr(api.os, "link", original)
    result = api.receive_results(saved, parts_root, out, resume=True, _allow_synthetic=True)
    assert result["written"] == 1 and result["skipped"] == 1
    assert retained[0].exists()


def test_partial_volume_plan_and_same_size_corruption_reject_before_extract(tmp_path, monkeypatch):
    api = module(monkeypatch)
    saved, parts_root, _, manifest = transport_fixture(tmp_path, api)
    assert all(row["status"] == "valid" for row in api.download_plan(manifest, parts_root))
    part = parts_root / manifest["parts"][3]["path"]
    payload = part.read_bytes()
    part.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    assert api.download_plan(manifest, parts_root)[3]["status"] == "different"
    out = tmp_path / "mirror"
    with pytest.raises(ValueError, match="volume checksum"):
        api.receive_results(saved, parts_root, out, _allow_synthetic=True)
    assert not out.exists()


@pytest.mark.parametrize("extra", ["../outside", "server/core_project/escape"])
def test_unsafe_or_link_archive_entries_never_write_outside_mirror(tmp_path, monkeypatch, extra):
    api = module(monkeypatch)
    saved, parts_root, _, _ = transport_fixture(tmp_path, api, extra=extra)
    out = tmp_path / "mirror"
    with pytest.raises(ValueError):
        api.receive_results(saved, parts_root, out, _allow_synthetic=True)
    assert not (tmp_path / "outside").exists()
    assert json.loads((out / "metadata/receive_state.json").read_text())["status"] == "failed"


def test_missing_declared_model_and_synthetic_production_guard(tmp_path, monkeypatch):
    api = module(monkeypatch)
    saved, parts_root, _, _ = transport_fixture(tmp_path, api, omit_model=True)
    out = tmp_path / "formal"
    with pytest.raises(ValueError, match="Synthetic"):
        api.receive_results(saved, parts_root, out)
    assert not out.exists()
    with pytest.raises(ValueError, match="roster differs"):
        api.receive_results(saved, parts_root, tmp_path / "synthetic", _allow_synthetic=True)


def test_resume_preserves_user_changes_and_rejects_wrong_transport_identity(tmp_path, monkeypatch):
    api = module(monkeypatch)
    saved, parts_root, _, manifest = transport_fixture(tmp_path, api)
    out = tmp_path / "mirror"
    api.receive_results(saved, parts_root, out, _allow_synthetic=True)
    changed = out / "server/core_project/runs/run/metrics.json"
    changed.write_bytes(b'{"value":0.75}')
    with pytest.raises(ValueError, match="checksum changed"):
        api.receive_results(saved, parts_root, out, resume=True, _allow_synthetic=True)
    assert changed.read_bytes() == b'{"value":0.75}'
    manifest["archive"]["sha256"] = "0" * 64
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="different transport"):
        api.receive_results(saved, parts_root, out, resume=True, _allow_synthetic=True)
