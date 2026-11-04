from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from virtual_imu.io.pilot_download import (
    _gdrive_direct_download_url,
    download_artifact,
    ensure_dataset_layout,
    extract_artifact,
    load_pilot_catalog,
    resolve_gdrive_matches,
)


def test_catalog_creates_standard_dataset_layout(tmp_path: Path) -> None:
    config = tmp_path / "pilot.yaml"
    config.write_text(
        """
raw_root: data/raw
artifacts:
  - id: navinst_imu
    dataset: navinst_pilot
    source_type: http
    url: https://example.test/imus.bag
    destination: 00_archives/imus.bag
    extraction: none
""".strip(),
        encoding="utf-8",
    )

    catalog = load_pilot_catalog(config)
    roots = ensure_dataset_layout(tmp_path / "raw", catalog)

    root = roots["navinst_pilot"]
    assert (root / "00_archives").is_dir()
    assert (root / "01_extracted").is_dir()
    assert (root / "_manifests").is_dir()


def test_project_catalog_contains_only_the_three_pilot_datasets() -> None:
    catalog = load_pilot_catalog(Path("configs/multi_dataset_pilot_download.yaml"))

    assert {artifact["dataset"] for artifact in catalog["artifacts"]} == {
        "fusionportable_pilot",
        "navinst_pilot",
        "penncosyvio_pilot",
    }
    assert {artifact["id"] for artifact in catalog["artifacts"]} >= {
        "navinst_urban03_day_imus",
        "penncosyvio_tango_top",
        "penncosyvio_visensor",
        "fusionportable_static_imus",
        "fusionportable_mcr_slow",
    }


def test_catalog_rejects_destination_escape(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
artifacts:
  - id: bad
    dataset: navinst_pilot
    source_type: http
    url: https://example.test/file
    destination: ../escape.bin
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="destination"):
        load_pilot_catalog(config)


def test_existing_complete_artifact_is_verified_and_skipped(tmp_path: Path) -> None:
    dataset_root = tmp_path / "navinst_pilot"
    archive = dataset_root / "00_archives" / "imus.bag"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"complete")
    artifact = {
        "id": "navinst_imu",
        "dataset": "navinst_pilot",
        "source_type": "http",
        "url": "https://example.test/imus.bag",
        "destination": "00_archives/imus.bag",
        "expected_size": len(b"complete"),
        "extraction": "none",
    }

    def unexpected_command(*args, **kwargs):
        raise AssertionError("verified files must not invoke a downloader")

    result = download_artifact(artifact, dataset_root, command_runner=unexpected_command)

    assert result["status"] == "verified_skip"
    assert result["size"] == len(b"complete")
    assert len(result["sha256"]) == 64


def test_http_artifact_downloads_to_partial_then_promotes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "navinst_pilot"
    payload = b"downloaded"
    artifact = {
        "id": "navinst_imu",
        "dataset": "navinst_pilot",
        "source_type": "http",
        "url": "https://example.test/imus.bag",
        "destination": "00_archives/imus.bag",
        "expected_size": len(payload),
        "extraction": "none",
    }
    observed: dict[str, object] = {}

    def fake_command(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        partial = Path(args[args.index("--output") + 1])
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(args, 0, "", "")

    result = download_artifact(
        artifact,
        dataset_root,
        command_runner=fake_command,
        min_free_space_gb=0,
    )

    destination = dataset_root / "00_archives" / "imus.bag"
    assert result["status"] == "downloaded"
    assert destination.read_bytes() == payload
    assert not destination.with_name("imus.bag.part").exists()
    assert observed["args"][:4] == ["curl", "-L", "--fail", "--retry"]
    assert "--continue-at" in observed["args"]
    assert observed["args"][observed["args"].index("--continue-at") + 1] == "-"
    assert observed["args"][observed["args"].index("--speed-time") + 1] == "60"
    assert observed["args"][observed["args"].index("--speed-limit") + 1] == "1024"
    assert "--retry-all-errors" in observed["args"]
    assert "--silent" in observed["args"]
    assert "--show-error" in observed["args"]
    assert observed["kwargs"]["encoding"] == "utf-8"
    assert observed["kwargs"]["errors"] == "replace"


def test_failed_http_download_is_blocked_and_keeps_partial(tmp_path: Path) -> None:
    dataset_root = tmp_path / "navinst_pilot"
    artifact = {
        "id": "navinst_imu",
        "dataset": "navinst_pilot",
        "source_type": "http",
        "url": "https://example.test/imus.bag",
        "destination": "00_archives/imus.bag",
        "expected_size": 100,
    }

    def failed_command(args, **kwargs):
        partial = Path(args[args.index("--output") + 1])
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        return subprocess.CompletedProcess(args, 22, "", "server rejected request")

    result = download_artifact(
        artifact,
        dataset_root,
        command_runner=failed_command,
        min_free_space_gb=0,
    )

    assert result["status"] == "blocked"
    assert "server rejected" in result["reason"]
    assert Path(result["local_path"]).read_bytes() == b"partial"


def test_gdrive_folder_listing_is_filtered_by_configured_regex() -> None:
    artifact = {
        "id": "fusion_static",
        "dataset": "fusionportable_pilot",
        "source_type": "gdrive_folder_match",
        "folder_url": "https://drive.google.com/drive/folders/root",
        "match_regex": r"(?i).*Static_IMUs.*\.bag\.7z$",
        "destination": "00_archives/fusionportable_static",
    }
    listing = [
        {"url": "https://drive.google.com/file/d/static", "path": "root/Static_IMUs/run.bag.7z"},
        {"url": "https://drive.google.com/file/d/other", "path": "root/MCR/run.bag.7z"},
    ]

    def fake_command(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps(listing), "")

    matches = resolve_gdrive_matches(artifact, command_runner=fake_command)

    assert matches == [listing[0]]


def test_gdrive_listing_url_is_converted_to_resumable_direct_http() -> None:
    url = _gdrive_direct_download_url("https://drive.google.com/uc?id=file_123")

    assert url == (
        "https://drive.usercontent.google.com/download"
        "?id=file_123&export=download&confirm=t"
    )


def test_tar_extraction_is_isolated_and_inventoried(tmp_path: Path) -> None:
    archive = tmp_path / "sample.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"imu"
        info = tarfile.TarInfo("sequence/imu.txt")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    dataset_root = tmp_path / "penncosyvio_pilot"
    artifact = {"id": "tango", "extraction": "tar"}
    download = {"status": "downloaded", "local_path": str(archive)}

    result = extract_artifact(artifact, download, dataset_root)

    assert result["extraction_status"] == "extracted"
    assert result["inventory"] == [{"path": "sequence/imu.txt", "size": 3}]


def test_tar_extraction_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as handle:
        payload = b"escape"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    dataset_root = tmp_path / "penncosyvio_pilot"
    artifact = {"id": "unsafe", "extraction": "tar"}
    download = {"status": "downloaded", "local_path": str(archive)}

    with pytest.raises(ValueError, match="outside"):
        extract_artifact(artifact, download, dataset_root)

    assert not (dataset_root / "01_extracted" / "escape.txt").exists()


def test_pilot_download_cli_help() -> None:
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "scripts/download_pilot_datasets.py", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--dataset" in completed.stdout
    assert "--dry-run" in completed.stdout
