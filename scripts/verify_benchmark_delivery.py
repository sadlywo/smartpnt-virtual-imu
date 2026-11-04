"""Safely extract and verify the completed benchmark's immutable result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"READS: immutable result archive {args.archive}, server verification inventory", flush=True
    )
    print(f"WRITES: verified extraction {args.destination}, {args.report}", flush=True)
    print(
        "SKIPS: server changes, existing mismatched files, raw datasets, training and inference",
        flush=True,
    )
    if digest(args.archive) != args.sha256:
        raise ValueError("Result archive SHA256 differs from server receipt")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                not member.isfile()
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.name
                or ":" in member.name
                or member.name.casefold() in names
            ):
                raise ValueError(f"Unsafe or duplicate result member: {member.name}")
            names.add(member.name.casefold())
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Member resolves outside destination: {member.name}")
            if any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError(f"Symlink in result destination: {member.name}")
        verification_name = "runs/benchmark_export_20261024/verification.json"
        verification = json.loads(archive.extractfile(verification_name).read())
        if (
            verification["status"] != "passed"
            or verification["pipeline_status"] != "complete"
            or verification["runs"] != 45
            or verification["new_runs"] != 39
            or verification["reused_runs"] != 6
        ):
            raise ValueError("Server verification is not a completed 45-run benchmark")
        inventory = {entry["path"]: entry for entry in verification["inventory"]}
        if set(inventory) | {verification_name} != {member.name for member in members}:
            raise ValueError("Archive and server inventory have different membership")
        written = matched = 0
        for member in members:
            data = archive.extractfile(member).read()
            expected = inventory.get(member.name)
            if expected and (
                len(data) != expected["bytes"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Server inventory mismatch: {member.name}")
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f"Refusing to overwrite mismatched result: {target}")
                matched += 1
            else:
                with target.open("xb") as stream:
                    stream.write(data)
                written += 1
    for relative, entry in inventory.items():
        if digest(destination / relative) != entry["sha256"]:
            raise ValueError(f"Post-extraction SHA256 mismatch: {relative}")
    report = {
        "status": "passed",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(args.archive.resolve()),
        "archive_sha256": args.sha256,
        "archive_bytes": args.archive.stat().st_size,
        "archive_members": len(members),
        "inventory_hashes_verified": len(inventory),
        "existing_identical_files_kept": matched,
        "new_files_written": written,
        "server_verification": verification_name,
        "source_fingerprint": verification["bundle_verification"].get("source_fingerprint"),
        "old_run_links": verification["old_run_links"],
        "notes": "Reused original TCN results retain their existing local archive and source identity. Every new archive member and every inventoried extracted file was verified.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"VERIFIED: {len(members)} safe members; {len(inventory)} extracted SHA256 checks; {matched} existing identical metadata files retained",
        flush=True,
    )


if __name__ == "__main__":
    main()
