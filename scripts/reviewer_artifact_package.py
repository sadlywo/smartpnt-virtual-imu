"""Read immutable reviewer result mirrors without rewriting server identities.

Only standard-library modules are used so report generation can use its own
document runtime. Business checks such as 93 selections/558 evaluations belong
to the experiment consumers, in addition to this byte-level package validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

MANIFEST_MEMBER = "metadata/package_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_name(value: str | Path) -> str:
    """Require a portable relative archive name, without alternate separators."""
    text = str(value)
    if not text or "\\" in text or ":" in text or "\x00" in text:
        raise ValueError(f"Invalid portable package member: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in text.split("/") or path.as_posix() != text:
        raise ValueError(f"Noncanonical package member: {text!r}")
    if text == ".":
        raise ValueError("A package member cannot denote the package root")
    return text


def valid_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("Expected a SHA256 hexadecimal digest")
    return value.lower()


def _source_name(value: str) -> str:
    result = str(value).replace("\\", "/").rstrip("/")
    if not result or "\x00" in result or ".." in result.split("/"):
        raise ValueError("Invalid source identity")
    if not (result.startswith("/") or re.match(r"^[A-Za-z]:/", result)):
        raise ValueError("Original source identities must be absolute")
    return result


class ArtifactPackage:
    def __init__(
        self,
        root: str | Path,
        *,
        verify: bool = True,
        allow_synthetic: bool = False,
        expected_manifest_sha256: str | None = None,
    ):
        self.root = Path(root).resolve()
        self.manifest_path = self._local_member(MANIFEST_MEMBER)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if expected_manifest_sha256 is not None and self.manifest_sha256 != valid_sha(
            expected_manifest_sha256
        ):
            raise ValueError("Package manifest checksum differs from its transport receipt")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ValueError("A complete schema-1 package source snapshot is required")
        if (manifest.get("synthetic") or manifest.get("synthetic_fixture")) and not allow_synthetic:
            raise ValueError("Synthetic fixtures cannot be used as formal experiment results")
        if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
            raise ValueError("Package source inventory is empty or missing")
        self.files = {
            member_name(name): valid_sha(digest) for name, digest in manifest["files"].items()
        }
        if MANIFEST_MEMBER in self.files:
            raise ValueError("The package manifest cannot recursively hash itself")
        metadata = manifest.get("file_metadata", {})
        if not isinstance(metadata, dict) or set(metadata) != set(self.files):
            raise ValueError("File sizes/source metadata must cover the complete source inventory")
        for name, row in metadata.items():
            if type(row.get("size_bytes")) is not int or row["size_bytes"] < 0:
                raise ValueError(f"Invalid recorded file size: {name}")
        declared_roots = manifest.get("roots", {})
        if not isinstance(declared_roots, dict) or not declared_roots:
            raise ValueError("Named package roots are required")
        self.roots = {key: self._local_member(value) for key, value in declared_roots.items()}
        self.path_mappings = []
        seen = set()
        for mapping in manifest.get("path_mappings", []):
            source = _source_name(mapping["source_prefix"])
            mirror = member_name(mapping["mirror_prefix"])
            if source in seen:
                raise ValueError("Original source prefixes must be unique")
            seen.add(source)
            self.path_mappings.append((source, mirror))
        if not self.path_mappings:
            raise ValueError("Explicit original-to-mirror path mappings are required")
        self.path_mappings.sort(key=lambda pair: len(pair[0]), reverse=True)
        for name, row in metadata.items():
            source = row.get("source_path")
            if source is not None and self.resolve_path(source) != self._local_member(name):
                raise ValueError(f"Recorded source identity maps to a different file: {name}")
        self.verified_files: dict[str, str] = {}
        if verify:
            self.verify_all()

    def _local_member(self, value: str | Path) -> Path:
        member = member_name(value)
        candidate = (self.root / Path(*PurePosixPath(member).parts)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Package member resolves outside its mirror root")
        return candidate

    def resolve_path(self, value: str | Path) -> Path:
        native = Path(value)
        if native.is_absolute():
            resolved = native.resolve()
            if resolved.is_relative_to(self.root):
                return resolved
        text = str(value).replace("\\", "/")
        if text.startswith("/") or re.match(r"^[A-Za-z]:/", text):
            source = _source_name(text)
            for prefix, mirror in self.path_mappings:
                if source == prefix:
                    return self._local_member(mirror)
                if source.startswith(prefix + "/"):
                    return self._local_member(mirror + source[len(prefix) :])
            raise ValueError(f"No explicit package mapping for original path: {text}")
        return self._local_member(text)

    resolve = resolve_path

    def root_path(self, key: str) -> Path:
        return self.roots[key]

    def relative_path(self, value: str | Path) -> str:
        return self.resolve_path(value).relative_to(self.root).as_posix()

    def source_path(self, value: str | Path) -> str | None:
        member = self.relative_path(value)
        metadata = self.manifest["file_metadata"].get(member, {})
        if "source_path" in metadata:
            # Generated audit/processing files have a declared null origin;
            # their creation provenance is distinct from a copied server file.
            return metadata["source_path"]
        matches = sorted(self.path_mappings, key=lambda pair: len(pair[1]), reverse=True)
        for source, mirror in matches:
            if member == mirror or member.startswith(mirror + "/"):
                return source + member[len(mirror) :]
        raise ValueError(f"No original source identity for {member}")

    def verify_file(self, value: str | Path, expected_sha256: str | None = None) -> str:
        path = self.resolve_path(value)
        member = path.relative_to(self.root).as_posix()
        expected = self.manifest_sha256 if member == MANIFEST_MEMBER else self.files.get(member)
        if expected is None:
            raise ValueError(f"File is not declared in the immutable source inventory: {member}")
        if expected_sha256 is not None and expected != valid_sha(expected_sha256):
            raise ValueError(f"Expected artifact identity differs from package inventory: {member}")
        if not path.is_file():
            raise ValueError(f"Declared source file is missing: {member}")
        if (
            member != MANIFEST_MEMBER
            and path.stat().st_size != self.manifest["file_metadata"][member]["size_bytes"]
        ):
            raise ValueError(f"Declared source file size changed: {member}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Declared source file checksum changed: {member}")
        self.verified_files[member] = actual
        return actual

    def verify_all(self) -> dict[str, str]:
        for member in self.files:
            self.verify_file(member)
        return dict(self.verified_files)

    def read_json(self, value: str | Path):
        self.verify_file(value)
        return json.loads(self.resolve_path(value).read_text(encoding="utf-8-sig"))

    def read_csv(self, value: str | Path) -> list[dict[str, str]]:
        self.verify_file(value)
        with self.resolve_path(value).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))


def load_package(root: str | Path, **kwargs) -> ArtifactPackage:
    return ArtifactPackage(root, **kwargs)
