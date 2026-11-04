"""Verify downloaded result volumes and safely materialize an immutable mirror.

No network operations or credentials are implemented here. Obtain the completed
transport manifest and its volumes through the authorized interactive SSH/SCP
session, then use --plan to skip already verified volumes on later attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from reviewer_artifact_package import (
    MANIFEST_MEMBER,
    load_package,
    member_name,
    sha256_file,
    valid_sha,
)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(root: Path, name: str) -> Path:
    name = member_name(name)
    target = (root / Path(*PurePosixPath(name).parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("A transport path resolves outside its declared root")
    return target


def transport_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("A completed schema-1 transport manifest is required")
    if data.get("archive", {}).get("format") != "tar.gz":
        raise ValueError("Only the declared streaming tar.gz format is supported")
    valid_sha(data["archive"]["sha256"])
    if type(data["archive"]["size_bytes"]) is not int or data["archive"]["size_bytes"] <= 0:
        raise ValueError("Invalid compressed archive size")
    inner = data["inner_manifest"]
    if inner["path"] != MANIFEST_MEMBER:
        raise ValueError("Unexpected package manifest location")
    valid_sha(inner["sha256"])
    parts = data.get("parts", [])
    if not parts or not isinstance(parts, list):
        raise ValueError("Transport volumes are missing")
    names = []
    for part in parts:
        names.append(member_name(part["path"]))
        valid_sha(part["sha256"])
        if type(part["size_bytes"]) is not int or part["size_bytes"] <= 0:
            raise ValueError("Invalid transport volume size")
    if (
        len(set(names)) != len(names)
        or sum(part["size_bytes"] for part in parts) != data["archive"]["size_bytes"]
    ):
        raise ValueError("Transport volume identity/size coverage is inconsistent")
    return data


def download_plan(manifest: dict, parts_root: Path) -> list[dict]:
    result = []
    for part in manifest["parts"]:
        path = safe_path(parts_root, part["path"])
        status = "missing"
        if path.exists():
            status = (
                "valid"
                if path.is_file()
                and path.stat().st_size == part["size_bytes"]
                and sha256_file(path) == part["sha256"].lower()
                else "different"
            )
        result.append({**part, "local_path": str(path), "status": status})
    return result


def verify_transport(manifest: dict, parts_root: Path) -> list[Path]:
    paths, whole = [], hashlib.sha256()
    for part in manifest["parts"]:
        path = safe_path(parts_root, part["path"])
        if not path.is_file() or path.stat().st_size != part["size_bytes"]:
            raise ValueError(f"Missing or incomplete downloaded volume: {part['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                whole.update(block)
        if digest.hexdigest() != part["sha256"].lower():
            raise ValueError(f"Downloaded volume checksum differs: {part['path']}")
        paths.append(path)
    if whole.hexdigest() != manifest["archive"]["sha256"].lower():
        raise ValueError("Combined compressed-stream checksum differs")
    return paths


class VolumeReader(io.RawIOBase):
    """A bounded-memory sequential reader; no concatenated temporary archive."""

    def __init__(self, paths: list[Path]):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            raise ValueError("Unbounded transport reads are prohibited")
        pieces, remaining = [], size
        while remaining:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            block = self.current.read(remaining)
            if block:
                pieces.append(block)
                remaining -= len(block)
            else:
                self.current.close()
                self.current = None
        return b"".join(pieces)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def read_inner_manifest(paths: list[Path], expected_sha: str) -> bytes:
    with VolumeReader(paths) as reader, tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if member.name != MANIFEST_MEMBER or not member.isfile():
                raise ValueError("The first regular archive member must be the package manifest")
            if member.size > 512 * 1024 * 1024:
                raise ValueError("Package manifest exceeds the bounded metadata size")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Package manifest is unreadable")
            content = stream.read(member.size + 1)
            if len(content) != member.size or hashlib.sha256(content).hexdigest() != valid_sha(
                expected_sha
            ):
                raise ValueError("Inner package manifest checksum/size differs")
            return content
    raise ValueError("Package manifest is absent from the result archive")


def receive_results(
    manifest_path: Path,
    parts_root: Path,
    out: Path,
    *,
    resume: bool = False,
    _allow_synthetic: bool = False,
) -> dict:
    manifest_path, parts_root, out = [Path(p).resolve() for p in (manifest_path, parts_root, out)]
    manifest = transport_manifest(manifest_path)
    contract = {
        "transport_manifest_sha256": sha256_file(manifest_path),
        "archive_sha256": manifest["archive"]["sha256"].lower(),
        "inner_manifest_sha256": manifest["inner_manifest"]["sha256"].lower(),
    }
    state_path = out / "metadata/receive_state.json"
    if out.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError(
                "Existing mirror requires --resume and its own matching receive state"
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Result mirror belongs to a different transport identity")
        if state.get("status") == "complete":
            try:
                package = load_package(
                    out,
                    expected_manifest_sha256=contract["inner_manifest_sha256"],
                    allow_synthetic=_allow_synthetic,
                )
            except Exception as error:
                state.update(status="failed", error_type=type(error).__name__, error=str(error))
                write_json(state_path, state)
                raise
            return {
                "status": "complete",
                "verified_files": len(package.files),
                "resumed_without_extraction": True,
                "synthetic": bool(
                    package.manifest.get("synthetic") or package.manifest.get("synthetic_fixture")
                ),
            }
    else:
        state = {"contract": contract, "status": "verifying_transport"}
    paths = verify_transport(manifest, parts_root)
    inner_bytes = read_inner_manifest(paths, contract["inner_manifest_sha256"])
    inner = json.loads(inner_bytes)
    if (inner.get("synthetic") or inner.get("synthetic_fixture")) and not _allow_synthetic:
        raise ValueError("Synthetic fixtures cannot be received as formal experiment results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    write_json(state_path, state)
    saved_manifest = out / MANIFEST_MEMBER
    if saved_manifest.exists() and saved_manifest.read_bytes() != inner_bytes:
        raise ValueError("Preserving a different existing package manifest")
    if not saved_manifest.exists():
        with saved_manifest.open("xb") as stream:
            stream.write(inner_bytes)
    package = load_package(
        out,
        verify=False,
        expected_manifest_sha256=contract["inner_manifest_sha256"],
        allow_synthetic=_allow_synthetic,
    )
    needed_bytes = sum(
        package.manifest["file_metadata"][name]["size_bytes"]
        for name in package.files
        if not safe_path(out, name).exists()
    )
    if needed_bytes > shutil.disk_usage(out).free:
        raise OSError(f"Insufficient space to extract declared files: need {needed_bytes} bytes")
    state.update(
        status="extracting", started_utc=state.get("started_utc", datetime.now(UTC).isoformat())
    )
    write_json(state_path, state)
    seen, written, skipped = set(), 0, 0
    allowed_parents = {
        parent.as_posix()
        for name in [MANIFEST_MEMBER, *package.files]
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    try:
        with VolumeReader(paths) as reader, tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                name = member_name(member.name.rstrip("/") if member.isdir() else member.name)
                if member.isdir():
                    if name not in allowed_parents:
                        raise ValueError(f"Unexpected archive directory: {name}")
                    safe_path(out, name).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or name in seen:
                    raise ValueError(f"Unsupported or duplicate archive entry: {name}")
                seen.add(name)
                if name == MANIFEST_MEMBER:
                    if member.size != len(inner_bytes):
                        raise ValueError("Repeated manifest size differs")
                    continue
                if name not in package.files:
                    raise ValueError(f"Archive contains an undeclared file: {name}")
                if member.size != package.manifest["file_metadata"][name]["size_bytes"]:
                    raise ValueError(f"Archive member size differs: {name}")
                target = safe_path(out, name)
                if target.exists():
                    package.verify_file(name)
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Archive file is unreadable: {name}")
                digest, count = hashlib.sha256(), 0
                with (
                    source,
                    tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=".receive-partial-", delete=False
                    ) as temporary,
                ):
                    temp_path = Path(temporary.name)
                    for block in iter(lambda stream=source: stream.read(1024 * 1024), b""):
                        temporary.write(block)
                        digest.update(block)
                        count += len(block)
                if count != member.size or digest.hexdigest() != package.files[name]:
                    raise ValueError(
                        f"Extracted file checksum differs; preserved partial {temp_path}"
                    )
                # link is an atomic no-overwrite commit on Windows and POSIX.
                os.link(temp_path, target)
                temp_path.unlink()
                written += 1
                if written % 100 == 0:
                    state.update(written_this_attempt=written, skipped_this_attempt=skipped)
                    write_json(state_path, state)
        if seen != {MANIFEST_MEMBER, *package.files}:
            raise ValueError("Archive file roster differs from its complete package inventory")
        package.verify_all()
        state.update(
            status="complete",
            completed_utc=datetime.now(UTC).isoformat(),
            verified_files=len(package.files),
            written_this_attempt=written,
            skipped_this_attempt=skipped,
            synthetic=bool(inner.get("synthetic") or inner.get("synthetic_fixture")),
        )
        write_json(state_path, state)
        return {
            "status": "complete",
            "verified_files": len(package.files),
            "written": written,
            "skipped": skipped,
            "synthetic": state["synthetic"],
        }
    except Exception as error:
        state.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            written_this_attempt=written,
            skipped_this_attempt=skipped,
        )
        write_json(state_path, state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-manifest", type=Path, required=True)
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        f"READS: completed transport manifest {args.transport_manifest} and downloaded volumes under {args.parts_root}"
    )
    print(f"WRITES: {'none (download plan)' if args.plan else args.out}")
    print(
        "SKIPS: network/authentication, server mutation, existing-source overwrite and concatenated temporary archives"
    )
    if args.plan:
        result = download_plan(transport_manifest(args.transport_manifest), args.parts_root)
    else:
        if args.out is None:
            parser.error("--out is required unless --plan is used")
        result = receive_results(
            args.transport_manifest, args.parts_root, args.out, resume=args.resume
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
