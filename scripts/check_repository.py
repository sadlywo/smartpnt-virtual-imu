#!/usr/bin/env python3
"""Validate the public repository boundary and reproducibility scaffolding."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "AGENTS.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "configs/README.md",
    "data/README.md",
    "docs/ARTIFACTS.md",
    "docs/PROJECT_FRAMEWORK.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS_SNAPSHOT.md",
    "runs/README.md",
)
REQUIRED_DIRS = (
    "src/virtual_imu",
    "scripts",
    "configs",
    "docs",
    "tests",
    ".agents/skills/imu-research-visualization",
)
ARTIFACT_PREFIXES = (
    "data/raw/",
    "data/processed/",
    "data/interim/",
    "data/external/",
    "runs/",
    "outputs/",
    "checkpoints/",
    "artifacts/",
    ".venv/",
    ".venv312/",
    ".superpowers/",
    ".codex/",
)
ARTIFACT_ALLOWLIST = {
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",
    "runs/README.md",
}
MAX_TRACKED_BYTES = 2 * 1024 * 1024
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\0") if item]


def validate_yaml(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = ROOT / relative
        try:
            with path.open(encoding="utf-8") as handle:
                yaml.safe_load(handle)
        except Exception as exc:
            errors.append(f"invalid YAML {relative}: {exc}")
    return errors


def scan_secrets(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if path.suffix.lower() not in {".py", ".toml", ".yaml", ".yml", ".md", ".cff", ".json"}:
            continue
        if path.stat().st_size > MAX_TRACKED_BYTES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"possible credential in tracked file: {relative}")
    return errors


def validate_links() -> list[str]:
    errors: list[str] = []
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            cleaned = target.split("#", 1)[0]
            if not cleaned or "://" in cleaned or cleaned.startswith(("mailto:", "codex:")):
                continue
            resolved = (document.parent / cleaned).resolve()
            if ROOT.resolve() not in (resolved, *resolved.parents) or not resolved.exists():
                errors.append(f"broken local link in {document.relative_to(ROOT)}: {target}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="Optional JSON receipt path")
    args = parser.parse_args()
    print("READS: Git index, repository structure, tracked YAML, docs, and secret patterns", flush=True)
    print("WRITES: stdout summary and optional JSON receipt; SKIPS: installing dependencies or executing experiments", flush=True)

    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (ROOT / name).is_file():
            errors.append(f"missing required file: {name}")
    for name in REQUIRED_DIRS:
        if not (ROOT / name).is_dir():
            errors.append(f"missing required directory: {name}")

    files = tracked_files()
    for relative in files:
        text = relative.as_posix()
        if text.startswith(ARTIFACT_PREFIXES) and text not in ARTIFACT_ALLOWLIST:
            errors.append(f"generated artifact is tracked: {text}")
        path = ROOT / relative
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 2 MiB: {text} ({path.stat().st_size} bytes)")

    yaml_files = [path for path in files if path.suffix.lower() in {".yaml", ".yml"}]
    errors.extend(validate_yaml(yaml_files))
    errors.extend(scan_secrets(files))
    errors.extend(validate_links())

    result = {
        "status": "PASS" if not errors else "FAIL",
        "tracked_files": len(files),
        "yaml_files": len(yaml_files),
        "source_python_files": sum(path.suffix == ".py" and path.parts[0] in {"src", "scripts", "tests"} for path in files),
        "errors": errors,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"WRITES: {args.json_out}", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
