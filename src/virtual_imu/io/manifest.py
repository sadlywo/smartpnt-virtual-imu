"""Data manifest generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from virtual_imu.utils.paths import utc_now_iso


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: str | Path, source_url: str, include_sha256: bool = False) -> dict[str, Any]:
    """Build a manifest for files under a root directory."""
    root_path = Path(root)
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        rel = path.relative_to(root_path).as_posix()
        item: dict[str, Any] = {"path": rel, "size_bytes": path.stat().st_size}
        if include_sha256:
            item["sha256"] = sha256_file(path)
        files.append(item)
    return {
        "source_url": source_url,
        "download_time_utc": utc_now_iso(),
        "local_path": str(root_path),
        "file_count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a manifest as pretty JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
