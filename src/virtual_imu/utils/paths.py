"""Path and provenance utilities."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path


def ensure_parent(path: str | Path) -> Path:
    """Create a file's parent directory and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def git_commit(root: str | Path = ".") -> str:
    """Return the current git commit, or 'unknown' outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def project_root_from_file(file: str | Path) -> Path:
    """Return the repository root assuming the file lives below it."""
    path = Path(file).resolve()
    for parent in [path, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()
