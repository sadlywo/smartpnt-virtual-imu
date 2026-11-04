"""Reproducible download and organization helpers for pilot IMU datasets."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from virtual_imu.io.manifest import sha256_file
from virtual_imu.utils.paths import utc_now_iso

SUPPORTED_SOURCE_TYPES = {"http", "gdrive_folder_match"}
STANDARD_DATASET_DIRS = ("00_archives", "01_extracted", "_manifests")


def _relative_destination(value: Any) -> Path:
    destination = Path(str(value))
    if destination.is_absolute() or ".." in destination.parts:
        raise ValueError(f"Artifact destination must stay inside its dataset root: {destination}")
    return destination


def load_pilot_catalog(path: Path) -> dict[str, Any]:
    """Load and validate a pilot dataset YAML catalog."""
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("Pilot catalog must be a YAML mapping")
    artifacts = catalog.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Pilot catalog requires a non-empty artifacts list")

    artifact_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("Every artifact must be a mapping")
        missing = [key for key in ("id", "dataset", "source_type", "destination") if key not in artifact]
        if missing:
            raise ValueError(f"Artifact is missing required keys: {missing}")
        artifact_id = str(artifact["id"])
        if artifact_id in artifact_ids:
            raise ValueError(f"Duplicate artifact id: {artifact_id}")
        artifact_ids.add(artifact_id)
        source_type = str(artifact["source_type"])
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {source_type}")
        _relative_destination(artifact["destination"])
    return catalog


def ensure_dataset_layout(raw_root: Path, catalog: dict[str, Any]) -> dict[str, Path]:
    """Create standard raw-data directories for every catalog dataset."""
    roots: dict[str, Path] = {}
    datasets = sorted({str(artifact["dataset"]) for artifact in catalog["artifacts"]})
    for dataset in datasets:
        root = raw_root / dataset
        for directory in STANDARD_DATASET_DIRS:
            (root / directory).mkdir(parents=True, exist_ok=True)
        roots[dataset] = root
    return roots


def download_artifact(
    artifact: dict[str, Any],
    dataset_root: Path,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
    min_free_space_gb: float = 50.0,
) -> dict[str, Any]:
    """Download one artifact, or verify and skip an existing complete file."""
    destination = dataset_root / _relative_destination(artifact["destination"])
    expected_size = artifact.get("expected_size")
    if destination.is_file() and (expected_size is None or destination.stat().st_size == int(expected_size)):
        return {
            "artifact_id": str(artifact["id"]),
            "dataset": str(artifact["dataset"]),
            "status": "verified_skip",
            "local_path": str(destination),
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    if destination.exists():
        return _blocked_result(
            artifact,
            destination,
            f"existing destination has unexpected size: {destination.stat().st_size}",
        )
    source_type = str(artifact["source_type"])
    if source_type not in {"http", "gdrive_file"}:
        raise NotImplementedError(f"Downloader is not implemented for artifact: {artifact['id']}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    remaining_bytes = max(0, int(expected_size or 0) - (partial.stat().st_size if partial.exists() else 0))
    free_bytes = shutil.disk_usage(destination.parent).free
    reserve_bytes = int(min_free_space_gb * 1024**3)
    if free_bytes - remaining_bytes < reserve_bytes:
        return _blocked_result(
            artifact,
            destination,
            f"insufficient free space: need {remaining_bytes} bytes plus {reserve_bytes} byte reserve",
        )

    if source_type == "http":
        args = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--speed-time",
            "60",
            "--speed-limit",
            "1024",
            "--silent",
            "--show-error",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            str(artifact["url"]),
        ]
    else:
        args = [
            "uvx",
            "--from",
            "gdown==6.1.0",
            "gdown",
            str(artifact["url"]),
            "--continue",
            "--output",
            str(partial),
        ]
    completed = command_runner(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "curl failed").strip()
        return _blocked_result(artifact, partial, detail)
    if expected_size is not None and partial.stat().st_size != int(expected_size):
        return _blocked_result(
            artifact,
            partial,
            f"downloaded size {partial.stat().st_size} does not match expected {expected_size}",
        )

    partial.replace(destination)
    return {
        "artifact_id": str(artifact["id"]),
        "dataset": str(artifact["dataset"]),
        "status": "downloaded",
        "local_path": str(destination),
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _blocked_result(artifact: dict[str, Any], path: Path, reason: str) -> dict[str, Any]:
    return {
        "artifact_id": str(artifact["id"]),
        "dataset": str(artifact["dataset"]),
        "status": "blocked",
        "local_path": str(path),
        "reason": reason,
    }


def resolve_gdrive_matches(
    artifact: dict[str, Any],
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> list[dict[str, str]]:
    """List one Google Drive folder and retain paths selected by the catalog regex."""
    args = [
        "uvx",
        "--from",
        "gdown==6.1.0",
        "gdown",
        str(artifact["folder_url"]),
        "--folder",
        "--json",
        "--quiet",
    ]
    completed = command_runner(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "gdown folder listing failed").strip()
        raise RuntimeError(detail)
    stdout = completed.stdout.strip()
    start, end = stdout.find("["), stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError("gdown folder listing did not return a JSON list")
    listing = json.loads(stdout[start : end + 1])
    if not isinstance(listing, list):
        raise ValueError("gdown folder listing JSON must be a list")
    pattern = re.compile(str(artifact["match_regex"]))
    matches: list[dict[str, str]] = []
    for item in listing:
        if not isinstance(item, dict) or "path" not in item or "url" not in item:
            continue
        if pattern.search(str(item["path"])):
            matches.append({"path": str(item["path"]), "url": str(item["url"])})
    return matches


def extract_artifact(
    artifact: dict[str, Any],
    download: dict[str, Any],
    dataset_root: Path,
    *,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Safely extract one downloaded artifact and return its inventory."""
    result = dict(download)
    if download["status"] == "blocked":
        result["extraction_status"] = "skipped_download_blocked"
        return result

    archive = Path(str(download["local_path"]))
    mode = str(artifact.get("extraction", "none"))
    if mode == "auto":
        lower_name = archive.name.lower()
        mode = "7z" if lower_name.endswith(".7z") else "tar" if lower_name.endswith(".tar") else "none"
    if mode == "none":
        result["extraction_status"] = "not_required"
        result["inventory"] = []
        return result

    target = dataset_root / "01_extracted" / str(artifact["id"])
    if target.is_dir() and any(target.rglob("*")):
        result["extraction_status"] = "verified_skip"
        result["extracted_root"] = str(target)
        result["inventory"] = _inventory(target)
        return result

    if mode == "tar":
        with tarfile.open(archive) as handle:
            target_resolved = target.resolve()
            members = handle.getmembers()
            for member in members:
                member_path = (target / member.name).resolve()
                if member_path != target_resolved and target_resolved not in member_path.parents:
                    raise ValueError(f"TAR member resolves outside extraction root: {member.name}")
                if member.issym() or member.islnk():
                    raise ValueError(f"TAR links are not allowed in pilot extraction: {member.name}")
            target.mkdir(parents=True, exist_ok=True)
            handle.extractall(target, members=members)
    elif mode == "7z":
        executable = _find_7z()
        if executable is None:
            result["extraction_status"] = "downloaded_not_extracted"
            result["extraction_reason"] = "7z executable not found"
            result["inventory"] = []
            return result
        target.mkdir(parents=True, exist_ok=True)
        args = [executable, "x", "-y", f"-o{target}", str(archive)]
        completed = command_runner(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            result["extraction_status"] = "extraction_blocked"
            result["extraction_reason"] = (completed.stderr or completed.stdout).strip()
            return result
    else:
        result["extraction_status"] = "unsupported"
        result["extraction_reason"] = f"unsupported extraction mode: {mode}"
        return result

    result["extraction_status"] = "extracted"
    result["extracted_root"] = str(target)
    result["inventory"] = _inventory(target)
    return result


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size}
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _find_7z() -> str | None:
    executable = shutil.which("7z") or shutil.which("7z.exe")
    if executable is not None:
        return executable
    for candidate in (
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files (x86)/7-Zip/7z.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def run_pilot_download(
    config_path: Path,
    *,
    datasets: list[str] | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Resolve, download, extract, and manifest the configured pilot artifacts."""
    catalog = load_pilot_catalog(config_path)
    raw_root = Path(str(catalog.get("raw_root", "data/raw")))
    uv_cache = (raw_root / "_tool_cache" / "uv").resolve()
    uv_cache.mkdir(parents=True, exist_ok=True)
    os.environ["UV_CACHE_DIR"] = str(uv_cache)
    logger.info("WRITES isolated uv tool cache under {}", uv_cache)
    selected = set(datasets or [])
    artifacts = [
        artifact
        for artifact in catalog["artifacts"]
        if not selected or str(artifact["dataset"]) in selected
    ]
    if selected - {str(item["dataset"]) for item in catalog["artifacts"]}:
        unknown = sorted(selected - {str(item["dataset"]) for item in catalog["artifacts"]})
        raise ValueError(f"Unknown dataset selection: {unknown}")

    selected_catalog = {**catalog, "artifacts": artifacts}
    roots = ensure_dataset_layout(raw_root, selected_catalog)
    records_by_dataset: dict[str, list[dict[str, Any]]] = {name: [] for name in roots}
    min_free_space_gb = float(catalog.get("min_free_space_gb", 50))

    for artifact in artifacts:
        dataset = str(artifact["dataset"])
        dataset_root = roots[dataset]
        resolved = dict(artifact)
        logger.info("READS source definition for {}", artifact["id"])
        try:
            if artifact["source_type"] == "gdrive_folder_match":
                logger.info("RESOLVES Google Drive folder for {}", artifact["id"])
                matches = resolve_gdrive_matches(artifact, command_runner=command_runner)
                if len(matches) != 1:
                    raise ValueError(f"expected exactly one Drive match, found {len(matches)}")
                match = matches[0]
                resolved["source_type"] = "http"
                resolved["url"] = _gdrive_direct_download_url(match["url"])
                resolved["resolved_listing_url"] = match["url"]
                resolved["resolved_source_path"] = match["path"]
                resolved["destination"] = str(Path(str(artifact["destination"])) / Path(match["path"]).name)

            if dry_run:
                logger.info("SKIPS download for {} because dry-run is active", artifact["id"])
                record = {
                    "artifact_id": str(artifact["id"]),
                    "dataset": dataset,
                    "status": "planned",
                    "local_path": str(dataset_root / _relative_destination(resolved["destination"])),
                    "extraction_status": "planned",
                }
            else:
                logger.info("WRITES artifact {} under {}", artifact["id"], dataset_root)
                download = download_artifact(
                    resolved,
                    dataset_root,
                    command_runner=command_runner,
                    min_free_space_gb=min_free_space_gb,
                )
                record = extract_artifact(
                    artifact,
                    download,
                    dataset_root,
                    command_runner=command_runner,
                )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("BLOCKS artifact {}: {}", artifact["id"], exc)
            record = _blocked_result(
                artifact,
                dataset_root / _relative_destination(artifact["destination"]),
                str(exc),
            )
            record["extraction_status"] = "skipped_download_blocked"

        record.update(_artifact_provenance(artifact, resolved, dataset_root, record))
        records_by_dataset[dataset].append(record)
        _write_dataset_manifest(dataset_root, config_path, records_by_dataset[dataset])
        _write_dataset_readme(dataset_root, records_by_dataset[dataset])
        logger.info("WRITES dataset manifest for {}", dataset)

    all_records = [record for records in records_by_dataset.values() for record in records]
    summary = {
        "created_utc": utc_now_iso(),
        "config": str(config_path),
        "dry_run": dry_run,
        "datasets": sorted(records_by_dataset),
        "artifact_count": len(all_records),
        "status_counts": _status_counts(all_records),
        "total_complete_bytes": sum(int(item.get("size", 0)) for item in all_records),
        "artifacts": all_records,
    }
    summary_path = raw_root / "_pilot_manifests" / "pilot_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _artifact_provenance(
    source: dict[str, Any],
    resolved: dict[str, Any],
    dataset_root: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    local_path = Path(str(record["local_path"]))
    try:
        relative_path = local_path.relative_to(dataset_root).as_posix()
    except ValueError:
        relative_path = str(local_path)
    return {
        "source_type": str(source["source_type"]),
        "source_page": source.get("source_page"),
        "source_url": resolved.get("url", resolved.get("folder_url")),
        "resolved_listing_url": resolved.get("resolved_listing_url"),
        "resolved_source_path": resolved.get("resolved_source_path"),
        "relative_path": relative_path,
        "recorded_utc": utc_now_iso(),
        "sequence": source.get("sequence"),
        "platform": source.get("platform"),
        "low_sensors": source.get("low_sensors", []),
        "reference_sensor": source.get("reference_sensor"),
        "nominal_rates_hz": source.get("nominal_rates_hz", {}),
        "synchronization": source.get("synchronization"),
        "license": source.get("license"),
    }


def _write_dataset_manifest(root: Path, config_path: Path, records: list[dict[str, Any]]) -> None:
    manifest = {
        "created_utc": utc_now_iso(),
        "config": str(config_path),
        "dataset_root": str(root),
        "artifact_count": len(records),
        "status_counts": _status_counts(records),
        "artifacts": records,
    }
    _write_json(root / "_manifests" / "pilot_manifest.json", manifest)


def _write_dataset_readme(root: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        f"# {root.name}",
        "",
        "This directory is generated by the multi-dataset pilot downloader.",
        "Original files remain under `00_archives`; derived extraction lives under `01_extracted`.",
        "The authoritative provenance record is `_manifests/pilot_manifest.json`.",
        "",
        "| Artifact | Status | Extraction |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {item['artifact_id']} | {item['status']} | {item.get('extraction_status', '')} |"
        for item in records
    )
    path = root / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _gdrive_direct_download_url(url: str) -> str:
    match = re.search(r"(?:[?&]id=|/d/)([-\w]+)", url)
    if match is None:
        raise ValueError(f"Cannot extract Google Drive file id from URL: {url}")
    file_id = match.group(1)
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
