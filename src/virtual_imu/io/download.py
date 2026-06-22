"""Dataset download helpers with graceful fallback messages."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from loguru import logger

from virtual_imu.io.manifest import build_manifest, write_manifest


def _run_command(args: list[str], cwd: Path | None = None) -> bool:
    """Run a command and return whether it succeeded."""
    try:
        logger.info("Running: {}", " ".join(args))
        subprocess.run(args, cwd=cwd, check=True)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("Command failed: {} ({})", " ".join(args), exc)
        return False


def download_smartpnt(
    repo_url: str,
    out_dir: str | Path,
    manifest_config: str | Path | None = None,
    include_sha256: bool = False,
) -> Path:
    """Try to clone/download SmartPNT-MSF and always write a local manifest if files exist."""
    out_path = Path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and any(out_path.iterdir()):
        logger.info("Output directory already contains files; skipping fresh download: {}", out_path)
    else:
        success = False
        if shutil.which("git"):
            success = _run_command(["git", "clone", "--progress", repo_url, str(out_path)])
        if not success and shutil.which("aria2c"):
            success = _run_command(["aria2c", "-c", "-d", str(out_path.parent), repo_url])
        if not success and shutil.which("wget"):
            success = _run_command(["wget", "-c", "-P", str(out_path), repo_url])
        if not success and shutil.which("curl"):
            out_path.mkdir(parents=True, exist_ok=True)
            success = _run_command(["curl", "-L", "-C", "-", "-o", str(out_path / "download.bin"), repo_url])
        if not success:
            out_path.mkdir(parents=True, exist_ok=True)
            logger.error(
                "Automated download failed. Manually download SmartPNT-MSF from {} and place it under {}.",
                repo_url,
                out_path,
            )

    manifest = build_manifest(out_path, repo_url, include_sha256=include_sha256)
    manifest["manifest_config"] = str(manifest_config) if manifest_config else None
    write_manifest(manifest, out_path / "data_manifest.json")
    logger.info("Wrote manifest with {} files to {}", manifest["file_count"], out_path / "data_manifest.json")
    return out_path
