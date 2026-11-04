"""Portable snapshots of the current source tree and a curated IMU training corpus.

Only this module's standard-library imports are needed for checksum verification or
safe extraction. Parquet schema validation imports pyarrow only when requested.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_DATASET_ROOT = "data/processed/unified_imu_v2"
MANIFEST_NAME = "bundle_manifest.json"
CHECKSUM_NAME = "SHA256SUMS"
ENVIRONMENT_PROFILE = "configs/server_environment.yaml"
DATASET_METADATA = (
    "manifest.json",
    "strict_manifest.json",
    "label_schema.json",
    "config_snapshot.yaml",
    "normalization.json",
    "archive/duplicates.json",
    "splits/paired_default.json",
    "splits/navigation_default.json",
)
BUILD_RUN_METADATA = (
    "config.yaml",
    "data_manifest.json",
    "metrics.json",
    "report.md",
    "verification.json",
    "source_state.json",
)
SOURCE_SUFFIXES = {
    "src": {".py"},
    "scripts": {".py", ".sh", ".ps1"},
    "configs": {".yaml", ".yml", ".json"},
    "docs": {".md", ".txt"},
    "tests": {".py", ".json", ".yaml", ".yml", ".txt"},
}
TOP_LEVEL_SOURCE = ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md")
EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv312",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "cache",
    "caches",
    "node_modules",
    "raw",
    "processed",
    "runs",
    "outputs",
    "checkpoints",
    ".ssh",
    ".aws",
}
SECRET_NAME = re.compile(
    r"(^|[._-])(credentials?|secrets?|private[._-]?key|api[._-]?key|access[._-]?token|passwords?)([._-]|$)",
    re.IGNORECASE,
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
TASK_SPLITS = {"paired_signal": "paired_default.json", "navigation": "navigation_default.json"}
IDENTITY_FIELDS = ("dataset_id", "pair_id", "sequence_id")


@dataclass(frozen=True)
class TrainingBundleResult:
    archive_path: Path
    checksum_path: Path
    run_dir: Path
    archive_sha256: str
    inventory: dict[str, Any]


def portable_relative_path(value: str) -> PurePosixPath:
    """Reject paths that change meaning between Windows, Linux, or tar extraction."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Not a portable relative path: {value!r}")
    if value.startswith("/") or any(ord(char) < 32 for char in value):
        raise ValueError(f"Unsafe portable relative path: {value!r}")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Unsafe portable relative path: {value!r}")
    for part in value.split("/"):
        if part.endswith((".", " ")) or any(char in '<>"|?*' for char in part):
            raise ValueError(f"Not a portable relative path: {value!r}")
        if re.fullmatch(
            r"(con|prn|aux|nul|com[1-9]|lpt[1-9])", part.split(".", 1)[0], re.IGNORECASE
        ):
            raise ValueError(f"Not a portable relative path: {value!r}")
    return PurePosixPath(value)


def _file_at(root: Path, name: str) -> Path:
    relative = portable_relative_path(name)
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe path outside project: {name}")
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError(f"Nonportable symbolic link: {name}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _identity(record: dict[str, Any]) -> tuple[str, str, str]:
    if not isinstance(record, dict):
        raise ValueError("Dataset/split identity must be an object")
    values = tuple(str(record.get(field, "")).strip() for field in IDENTITY_FIELDS)
    if not all(values):
        raise ValueError(f"Dataset/split record requires identity fields {IDENTITY_FIELDS}")
    return values


def _dataset_records(root: Path, dataset_root: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    relative = portable_relative_path(dataset_root)
    if relative.parts[:2] != ("data", "processed") or len(relative.parts) < 3:
        raise ValueError("Curated dataset must be under data/processed/<dataset>")
    status_path = _file_at(root, f"{dataset_root}/dataset_status.json")
    if status_path.is_file():
        status = _read_json(status_path)
        if status.get("status") == "superseded_diagnostic":
            replacement = status.get("replacement", "a formal dataset")
            raise ValueError(
                f"Dataset is superseded_diagnostic: {dataset_root}; use {replacement}"
            )
    manifest = _read_json(_file_at(root, f"{dataset_root}/manifest.json"))
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Curated manifest requires a nonempty records list")
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Curated records must be objects")
        _identity(record)
        path = str(record.get("path", ""))
        portable_relative_path(path)
        if not path.startswith(f"{dataset_root}/sequences/") or not path.endswith(".parquet"):
            raise ValueError(f"Record path is outside curated sequences: {path}")
        if path.casefold() in seen_paths:
            raise ValueError(f"Duplicate curated Parquet path: {path}")
        seen_paths.add(path.casefold())
        if record.get("task_view") not in TASK_SPLITS:
            raise ValueError(f"Record has unsupported task_view: {record.get('task_view')}")
        if record.get("training_eligible") is not True or record.get("duplicate_of"):
            raise ValueError(f"Ineligible or duplicate record must not be exported: {path}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not HASH_PATTERN.fullmatch(digest):
            raise ValueError(f"Record requires a Parquet sha256 checksum: {path}")
    if manifest.get("build_run"):
        portable_relative_path(str(manifest["build_run"]))
    return manifest, records


def validate_curated_dataset(
    root: Path,
    dataset_root: str = DEFAULT_DATASET_ROOT,
    *,
    check_schemas: bool = True,
    check_hashes: bool = True,
) -> dict[str, Any]:
    """Validate export paths, data fingerprints, task schemas, and complete disjoint splits."""
    manifest, records = _dataset_records(root, dataset_root)
    for name in DATASET_METADATA:
        if not _file_at(root, f"{dataset_root}/{name}").is_file():
            raise FileNotFoundError(f"Required dataset metadata is missing: {name}")
    schema = _read_json(_file_at(root, f"{dataset_root}/label_schema.json"))
    tasks = schema.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("label_schema.json requires tasks with input_columns and target_columns")
    strict = _read_json(_file_at(root, f"{dataset_root}/strict_manifest.json"))
    strict_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in strict.get("records", []):
        identity = _identity(record)
        if identity in strict_index:
            raise ValueError(f"Duplicate strict identity: {identity}")
        strict_index[identity] = record
    count_by_task = {task: 0 for task in TASK_SPLITS}
    index_by_task: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {
        task: {} for task in TASK_SPLITS
    }
    data_bytes = 0
    for record in records:
        task = record["task_view"]
        identity = _identity(record)
        if identity in index_by_task[task]:
            raise ValueError(f"Duplicate dataset identity: {identity}")
        index_by_task[task][identity] = record
        count_by_task[task] += 1
        path = _file_at(root, record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Curated Parquet is missing: {path}")
        data_bytes += path.stat().st_size
        if check_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {record['path']}")
        if check_schemas:
            import pyarrow as pa
            import pyarrow.parquet as pq

            task_schema = tasks.get(task, {})
            inputs = task_schema.get("input_columns")
            targets = task_schema.get("target_columns")
            if (
                not isinstance(inputs, list)
                or not inputs
                or not isinstance(targets, list)
                or not targets
            ):
                raise ValueError(
                    f"Label schema requires input_columns and target_columns for {task}"
                )
            parquet = pq.ParquetFile(path)
            required = ["timestamp_s", "sequence_id", *inputs, *targets]
            if task == "paired_signal":
                required.append("training_eligible")
            missing = sorted(set(required) - set(parquet.schema_arrow.names))
            if missing or parquet.metadata.num_rows <= 0:
                raise ValueError(
                    f"Parquet schema lacks required fields or rows at {record['path']}: {missing}"
                )
            nonnumeric = [
                column
                for column in ["timestamp_s", *inputs, *targets]
                if not (
                    pa.types.is_floating(parquet.schema_arrow.field(column).type)
                    or pa.types.is_integer(parquet.schema_arrow.field(column).type)
                )
            ]
            if nonnumeric:
                raise ValueError(
                    f"Parquet schema requires numeric signal/label types at {record['path']}: {nonnumeric}"
                )
        if task == "paired_signal":
            match = strict_index.get(identity)
            if match is None or match.get("training_eligible") is not True:
                raise ValueError(f"Paired record requires an eligible strict record: {identity}")
            portable_relative_path(str(match.get("strict_path", "")))
            if (
                match["strict_path"] != record["path"]
                or not match.get("fingerprint")
                or match["fingerprint"] != record.get("fingerprint")
            ):
                raise ValueError(f"Paired strict path/fingerprint mismatch: {identity}")
    if set(strict_index) != set(index_by_task["paired_signal"]):
        raise ValueError("Strict manifest must contain exactly the exported paired records")
    for task, filename in TASK_SPLITS.items():
        split_manifest = _read_json(_file_at(root, f"{dataset_root}/splits/{filename}"))
        if split_manifest.get("task_view") != task:
            raise ValueError(f"Split task_view differs from {task}: {filename}")
        splits = split_manifest.get("splits")
        if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
            raise ValueError(f"Split manifest requires train/validation/test: {filename}")
        assigned: set[tuple[str, str, str]] = set()
        sequence_split: dict[tuple[str, str], str] = {}
        physical_group_split: dict[tuple[str, str, str], str] = {}
        for split_name, split_records in splits.items():
            if not isinstance(split_records, list) or not split_records:
                raise ValueError(f"Split {task}/{split_name} is empty or invalid")
            for split_record in split_records:
                identity = _identity(split_record)
                if identity in assigned:
                    raise ValueError(f"Cross-split leakage or repeated split record: {identity}")
                assigned.add(identity)
                match = index_by_task[task].get(identity)
                portable_relative_path(str(split_record.get("path", "")))
                if match is None or split_record["path"] != match["path"]:
                    raise ValueError(f"Split record differs from curated manifest: {identity}")
                if split_record.get("fingerprint") != match.get("fingerprint"):
                    raise ValueError(
                        f"Split record fingerprint differs from curated manifest: {identity}"
                    )
                sequence = (
                    str(match.get("original_sequence_id", match["sequence_id"]))
                    if task == "navigation"
                    else str(match["sequence_id"])
                )
                group = (str(match["dataset_id"]), sequence)
                if group in sequence_split and sequence_split[group] != split_name:
                    raise ValueError(f"Cross-split original sequence leakage: {group}")
                sequence_split[group] = split_name
                for field in ("physical_group_id", "reference_measurement_sha256"):
                    value = match.get(field)
                    if value is None:
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"Invalid physical/reference group metadata: {field}")
                    if field == "reference_measurement_sha256" and not HASH_PATTERN.fullmatch(
                        value
                    ):
                        raise ValueError("Invalid reference_measurement_sha256 group checksum")
                    physical_group = (field, str(match["dataset_id"]), value)
                    if (
                        physical_group in physical_group_split
                        and physical_group_split[physical_group] != split_name
                    ):
                        raise ValueError(f"Cross-split {field} group leakage: {physical_group}")
                    physical_group_split[physical_group] = split_name
        if assigned != set(index_by_task[task]):
            raise ValueError(f"Split {task} does not cover exactly its curated records")
    normalization = _read_json(_file_at(root, f"{dataset_root}/normalization.json"))
    if normalization.get("fit_split", "train") != "train":
        raise ValueError("Normalization must be fit on train records only")
    return {
        "dataset_records": len(records),
        "task_records": count_by_task,
        "data_bytes": data_bytes,
        "data_build_run": manifest.get("build_run"),
        "schema_checked": check_schemas,
    }


def _selected_build_provenance(
    root: Path, dataset_root: str, manifest: dict[str, Any]
) -> list[str]:
    build_run = manifest.get("build_run", "")
    relative = portable_relative_path(build_run)
    if len(relative.parts) != 2 or relative.parts[0] != "runs":
        raise ValueError("Dataset build_run must identify runs/<build_id>")
    names = [f"{build_run}/{name}" for name in BUILD_RUN_METADATA]
    for name in names:
        if not _file_at(root, name).is_file():
            raise FileNotFoundError(f"Required selected build provenance is missing: {name}")
    if _read_json(_file_at(root, f"{build_run}/data_manifest.json")) != manifest:
        raise ValueError(
            "Selected build run data_manifest differs from the curated dataset manifest"
        )
    verification = _read_json(_file_at(root, f"{build_run}/verification.json"))
    if verification.get("passed") is not True:
        raise ValueError("Selected dataset build verification did not pass")
    if verification.get("sha256") != sha256_file(_file_at(root, f"{dataset_root}/manifest.json")):
        raise ValueError(
            "Selected build verification checksum differs from the curated dataset manifest"
        )
    return names


def _source_paths(root: Path, log: Callable[[str], None]) -> list[str]:
    paths = []
    skipped = 0
    for name in TOP_LEVEL_SOURCE:
        if not _file_at(root, name).is_file():
            raise FileNotFoundError(f"Required source file is missing: {name}")
        paths.append(name)
    for directory, suffixes in SOURCE_SUFFIXES.items():
        for current, directories, filenames in os.walk(root / directory, followlinks=False):
            kept = []
            for name in sorted(directories):
                if (
                    name.casefold() in EXCLUDED_DIRECTORIES
                    or SECRET_NAME.search(name)
                    or (Path(current) / name).is_symlink()
                ):
                    skipped += 1
                else:
                    kept.append(name)
            directories[:] = kept
            for name in sorted(filenames):
                candidate = Path(current) / name
                if (
                    candidate.suffix.lower() not in suffixes
                    or SECRET_NAME.search(name)
                    or name.startswith(".env")
                ):
                    skipped += 1
                    continue
                relative = candidate.relative_to(root).as_posix()
                _file_at(root, relative)
                paths.append(relative)
    log(
        f"SKIPS: {skipped} disallowed source files/directories; .git, raw/old data, prior runs, outputs, environments, caches, keys"
    )
    return sorted(paths)


def _git_snapshot(root: Path) -> tuple[str | None, bool]:
    def read(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError:
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    return read("rev-parse", "HEAD") or None, bool(
        read("status", "--porcelain", "--untracked-files=normal")
    )


def _source_path_allowed(name: str) -> bool:
    parts = portable_relative_path(name).parts
    if len(parts) == 1:
        return name in TOP_LEVEL_SOURCE
    if any(
        part.casefold() in EXCLUDED_DIRECTORIES
        or SECRET_NAME.search(part)
        or part.startswith(".env")
        for part in parts
    ):
        return False
    return (
        parts[0] in SOURCE_SUFFIXES
        and PurePosixPath(name).suffix.lower() in SOURCE_SUFFIXES[parts[0]]
    )


def verify_training_bundle(root: Path, *, check_schemas: bool = True) -> dict[str, Any]:
    """Verify extracted bytes before reading their task schemas or using them in training."""
    root = root.resolve()
    inventory = _read_json(_file_at(root, MANIFEST_NAME))
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("entries"), list):
        raise ValueError("Unsupported bundle inventory schema")
    expected = {MANIFEST_NAME: sha256_file(_file_at(root, MANIFEST_NAME))}
    for entry in inventory["entries"]:
        name = entry.get("path", "")
        path = _file_at(root, name)
        if name in expected or any(name.casefold() == key.casefold() for key in expected):
            raise ValueError(f"Duplicate inventory path: {name}")
        digest = entry.get("sha256", "")
        if not isinstance(digest, str) or not HASH_PATTERN.fullmatch(digest):
            raise ValueError(f"Invalid checksum in inventory: {name}")
        if entry.get("kind") == "source" and not _source_path_allowed(name):
            raise ValueError(f"Bundle source entry is outside the allowlist: {name}")
        if entry.get("kind") not in {"source", "data", "build_provenance"}:
            raise ValueError(f"Unknown bundle entry kind: {name}")
        if (
            not path.is_file()
            or path.stat().st_size != entry.get("size_bytes")
            or sha256_file(path) != digest
        ):
            raise ValueError(f"Bundle checksum mismatch: {name}")
        expected[name] = digest
    checksum_map = {}
    for line in _file_at(root, CHECKSUM_NAME).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError("Malformed checksum inventory")
        digest, name = match.groups()
        portable_relative_path(name)
        if name in checksum_map:
            raise ValueError(f"Repeated checksum path: {name}")
        checksum_map[name] = digest
    if checksum_map != expected:
        raise ValueError("Bundle checksum inventory does not match bundle_manifest.json")
    dataset_root = inventory["dataset_root"]
    manifest, records = _dataset_records(root, dataset_root)
    if inventory.get("data_build_run") != manifest.get("build_run"):
        raise ValueError("Bundle data_build_run differs from the dataset build_run")
    required_data = {f"{dataset_root}/{name}" for name in DATASET_METADATA} | {
        record["path"] for record in records
    }
    recorded_data = {entry["path"] for entry in inventory["entries"] if entry["kind"] == "data"}
    if recorded_data != required_data:
        raise ValueError("Bundle checksum inventory must cover exactly the required curated files")
    for record in records:
        if expected[record["path"]] != record["sha256"]:
            raise ValueError(f"Dataset checksum differs from bundle inventory: {record['path']}")
    source_hashes = {
        entry["path"]: entry["sha256"]
        for entry in inventory["entries"]
        if entry["kind"] == "source"
    }
    if hashlib.sha256(_json_bytes(source_hashes)).hexdigest() != inventory.get(
        "source_fingerprint"
    ):
        raise ValueError("Bundle source fingerprint differs from its source checksum inventory")
    if expected[f"{dataset_root}/manifest.json"] != inventory.get("dataset_manifest_sha256"):
        raise ValueError("Dataset manifest checksum differs from bundle inventory")
    environment_profile = inventory.get("environment_profile")
    if ENVIRONMENT_PROFILE in expected:
        required_profile = {"path": ENVIRONMENT_PROFILE, "sha256": expected[ENVIRONMENT_PROFILE]}
        if environment_profile != required_profile:
            raise ValueError("Bundle environment profile differs from its checksum inventory")
    elif environment_profile is not None:
        raise ValueError("Bundle environment profile is missing from its checksum inventory")
    report = validate_curated_dataset(
        root, inventory["dataset_root"], check_schemas=check_schemas, check_hashes=False
    )
    required_build_provenance = set(_selected_build_provenance(root, dataset_root, manifest))
    recorded_build_provenance = {
        entry["path"] for entry in inventory["entries"] if entry["kind"] == "build_provenance"
    }
    if recorded_build_provenance != required_build_provenance:
        raise ValueError(
            "Bundle checksum inventory must cover exactly the selected dataset build provenance"
        )
    return {
        "bundle_id": inventory["bundle_id"],
        "file_count": len(inventory["entries"]),
        "source_fingerprint": inventory["source_fingerprint"],
        "git_commit": inventory["git_commit"],
        "git_dirty": inventory["git_dirty"],
        "checksums_verified": True,
        "build_provenance_files": len(required_build_provenance),
        "environment_profile": environment_profile,
        **report,
    }


def validate_server_environment(
    report: dict[str, Any], profile: dict[str, Any], *, require_gpu: bool = False
) -> None:
    """Reject runtime drift from the explicitly selected server image profile."""
    if not isinstance(profile, dict):
        raise ValueError("Environment profile must be a mapping")
    required = {"profile_id", "python", "torch", "numpy", "cuda", "platform", "machine"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"Missing environment profile fields: {sorted(missing)}")
    actual = {
        "python": ".".join(str(report.get("python_version", "")).split(".")[:2]),
        "torch": str(report.get("torch", "")).split("+")[0],
        "numpy": report.get("numpy"),
        "platform": report.get("platform"),
        "machine": str(report.get("machine", "")).lower(),
    }
    for key, value in actual.items():
        if value != str(profile[key]):
            raise RuntimeError(
                f"Environment profile {profile['profile_id']} expected {key}={profile[key]}, got {value}"
            )
    if require_gpu and not report.get("cuda_available"):
        raise RuntimeError("Environment profile requires a working CUDA GPU")
    has_cuda_build = report.get("cuda_runtime") is not None
    if (require_gpu or report.get("cuda_available") or has_cuda_build) and report.get("cuda_runtime") != profile["cuda"]:
        raise RuntimeError(
            f"Environment profile expected CUDA {profile['cuda']}, got {report.get('cuda_runtime')}"
        )


def check_training_environment(
    *, require_gpu: bool = False, profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Require importable dependencies and, when requested, a working CUDA operation."""
    try:
        import numpy
        import pandas
        import pyarrow
        import torch
        import yaml
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"Training dependencies could not load; run Linux bootstrap first: {error}"
        ) from error
    if sys.version_info < (3, 11):  # noqa: UP036 - scripts may run before bootstrap.
        raise RuntimeError("Training requires Python >= 3.11; Linux bootstrap selects Python 3.12")
    available = bool(torch.cuda.is_available())
    if require_gpu and not available:
        raise RuntimeError(
            "GPU required but torch.cuda.is_available() is false; inspect the PyTorch build and NVIDIA driver"
        )
    report: dict[str, Any] = {
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine().lower(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
        "pyyaml": yaml.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": available,
        "gpu_operation_verified": False,
        "numpy_bridge_verified": False,
        "conv1d_backward_verified": False,
        "cudnn_version": None,
        "devices": [],
    }
    if profile is not None:
        validate_server_environment(report, profile, require_gpu=require_gpu)
        report["profile_id"] = profile["profile_id"]
    try:
        sample = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        if not numpy.array_equal(torch.from_numpy(sample).numpy(), sample):
            raise RuntimeError("NumPy/PyTorch round trip changed the sample")
        report["numpy_bridge_verified"] = True
    except (RuntimeError, OSError) as error:
        raise RuntimeError(f"NumPy/PyTorch bridge failed: {error}") from error
    if available:
        try:
            tensor = torch.ones((2, 2), device="cuda")
            value = float((tensor @ tensor).sum().item())
            torch.cuda.synchronize()
            if value != 8.0:
                raise RuntimeError(f"CUDA matrix operation returned {value}, expected 8.0")
            seed = int((profile or {}).get("seed", 0))
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                layer = torch.nn.Conv1d(6, 8, kernel_size=5, padding=4).to("cuda")
                signal = torch.ones((2, 6, 16), device="cuda", requires_grad=True)
                loss = layer(signal).square().mean()
                loss.backward()
                gradients = [signal.grad, *(parameter.grad for parameter in layer.parameters())]
                if not torch.isfinite(loss).item() or not all(
                    gradient is not None and torch.isfinite(gradient).all().item()
                    for gradient in gradients
                ):
                    raise RuntimeError("CUDA Conv1d backward produced missing/nonfinite gradients")
                torch.cuda.synchronize()
            report["conv1d_backward_verified"] = True
            report["cudnn_version"] = torch.backends.cudnn.version()
            report["devices"] = [
                {
                    "name": torch.cuda.get_device_properties(index).name,
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            report["gpu_operation_verified"] = True
        except (RuntimeError, OSError) as error:
            raise RuntimeError(f"CUDA GPU operation failed: {error}") from error
    return report


def extract_training_bundle(archive_path: Path, destination: Path) -> Path:
    """Check every member first, then extract regular files into a new empty directory."""
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Extraction destination must be empty: {destination}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        seen = set()
        for member in members:
            portable_relative_path(member.name)
            if not member.isfile():
                raise ValueError(f"Archive may contain regular files only: {member.name}")
            if member.name.casefold() in seen:
                raise ValueError(f"Duplicate archive path: {member.name}")
            seen.add(member.name.casefold())
            _file_at(destination, member.name)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            path = _file_at(destination, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            reader = archive.extractfile(member)
            if reader is None:
                raise ValueError(f"Cannot read archive regular file: {member.name}")
            with reader, path.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            path.chmod(member.mode & 0o777)
    return destination


def build_training_bundle(
    project_root: Path,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    *,
    bundle_id: str | None = None,
    output_dir: Path | str = "outputs/server_bundles",
    log: Callable[[str], None] = print,
) -> TrainingBundleResult:
    """Hash live source/data, create a deterministic archive, and verify its round trip."""
    root = project_root.resolve()
    dataset_arg = Path(dataset_root)
    dataset_name = (
        dataset_arg.relative_to(root).as_posix()
        if dataset_arg.is_absolute()
        else dataset_arg.as_posix()
    )
    portable_relative_path(dataset_name)
    if bundle_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")  # noqa: UP017 - supports bootstrap interpreters.
        bundle_id = f"server_bundle_{stamp}"
    if not re.fullmatch(r"server_bundle_[A-Za-z0-9_-]+", bundle_id):
        raise ValueError(
            "bundle_id must start with server_bundle_ and contain letters, digits, underscores or hyphens"
        )
    output_arg = Path(output_dir)
    output_name = (
        output_arg.relative_to(root).as_posix()
        if output_arg.is_absolute()
        else output_arg.as_posix()
    )
    portable_relative_path(output_name)
    if portable_relative_path(output_name).parts[:2] != ("outputs", "server_bundles"):
        raise ValueError("Training bundle destination must be under outputs/server_bundles")
    archive_path = _file_at(root, f"{output_name}/{bundle_id}.tar.gz")
    checksum_path = archive_path.with_name(f"{archive_path.name}.sha256")
    run_dir = root / "runs" / bundle_id
    if archive_path.exists() or checksum_path.exists() or run_dir.exists():
        raise FileExistsError(f"Bundle/run already exists; choose a new bundle id: {bundle_id}")
    log(
        f"READS: current allowlisted source tree and {dataset_name}; hashes include untracked source"
    )
    data_report = validate_curated_dataset(root, dataset_name)
    source_paths = _source_paths(root, log)
    manifest, records = _dataset_records(root, dataset_name)
    build_provenance_paths = _selected_build_provenance(root, dataset_name, manifest)
    data_paths = [f"{dataset_name}/{name}" for name in DATASET_METADATA] + [
        record["path"] for record in records
    ]
    paths = sorted(source_paths + data_paths + build_provenance_paths)
    if len({name.casefold() for name in paths}) != len(paths):
        raise ValueError("Source/data paths are not unique on a case-insensitive filesystem")
    entries = []
    for name in paths:
        path = _file_at(root, name)
        entries.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "source"
                if name in source_paths
                else "build_provenance"
                if name in build_provenance_paths
                else "data",
            }
        )
    source_hashes = {
        entry["path"]: entry["sha256"] for entry in entries if entry["kind"] == "source"
    }
    source_fingerprint = hashlib.sha256(_json_bytes(source_hashes)).hexdigest()
    commit, dirty = _git_snapshot(root)
    inventory = {
        "schema_version": 1,
        "bundle_id": bundle_id,
        "dataset_root": dataset_name,
        "data_build_run": manifest.get("build_run"),
        "dataset_manifest_sha256": sha256_file(root / dataset_name / "manifest.json"),
        "git_commit": commit,
        "git_dirty": dirty,
        "source_fingerprint": source_fingerprint,
        "entries": entries,
    }
    if ENVIRONMENT_PROFILE in source_hashes:
        inventory["environment_profile"] = {
            "path": ENVIRONMENT_PROFILE,
            "sha256": source_hashes[ENVIRONMENT_PROFILE],
        }
    inventory_bytes = _json_bytes(inventory)
    checksums = {entry["path"]: entry["sha256"] for entry in entries}
    checksums[MANIFEST_NAME] = hashlib.sha256(inventory_bytes).hexdigest()
    checksums_bytes = "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums)).encode(
        "utf-8"
    )
    generated = {MANIFEST_NAME: inventory_bytes, CHECKSUM_NAME: checksums_bytes}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True)
    source_snapshot = {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_fingerprint": source_fingerprint,
        "sha256_by_path": source_hashes,
    }
    (run_dir / "source_snapshot.json").write_bytes(_json_bytes(source_snapshot))
    shutil.copyfile(root / dataset_name / "manifest.json", run_dir / "data_manifest.json")
    (run_dir / MANIFEST_NAME).write_bytes(inventory_bytes)
    (run_dir / "config_snapshot.yaml").write_text(
        f"bundle_id: {bundle_id}\ndataset_root: {dataset_name}\noutput_dir: {output_name}\nsource_policy: current_allowlisted_files\narchive_format: deterministic_tar_gzip\nroundtrip_verification: true\n",
        encoding="utf-8",
    )
    log(
        f"WRITES: {archive_path.relative_to(root).as_posix()}, archive SHA256 sidecar, and source-of-truth metadata in runs/{bundle_id}"
    )
    partial = archive_path.with_name(f"{archive_path.name}.partial")
    try:
        with (
            partial.open("xb") as raw,
            gzip.GzipFile(
                fileobj=raw, mode="wb", filename="", compresslevel=1, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            for name in sorted([*paths, *generated]):
                info = tarfile.TarInfo(name)
                info.mtime = 0
                info.uid = info.gid = 0
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                if name in generated:
                    content = generated[name]
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
                else:
                    path = _file_at(root, name)
                    info.size = path.stat().st_size
                    with path.open("rb") as reader:
                        archive.addfile(info, reader)
        with tempfile.TemporaryDirectory(
            prefix=f".{bundle_id}_verify_", dir=archive_path.parent
        ) as temporary:
            extracted = extract_training_bundle(partial, Path(temporary) / "project")
            verified = verify_training_bundle(extracted)
        with partial.open("rb") as reader, archive_path.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        archive_digest = sha256_file(archive_path)
        checksum_path.write_bytes(f"{archive_digest}  {archive_path.name}\n".encode())
        (run_dir / "archive_sha256.txt").write_text(
            f"{archive_digest}  {archive_path.relative_to(root).as_posix()}\n", encoding="utf-8"
        )
        (run_dir / "verification.json").write_bytes(
            _json_bytes({"roundtrip_verified": True, "archive_sha256": archive_digest, **verified})
        )
        metrics = {
            **data_report,
            "source_bytes": sum(
                entry["size_bytes"] for entry in entries if entry["kind"] == "source"
            ),
            "payload_bytes": sum(entry["size_bytes"] for entry in entries),
            "archive_bytes": archive_path.stat().st_size,
            "file_count": len(entries),
        }
        (run_dir / "metrics.json").write_bytes(_json_bytes(metrics))
        (run_dir / "report.md").write_text(
            f"# Server training bundle {bundle_id}\n\nArchive: `{archive_path.relative_to(root).as_posix()}`\n\nSHA256: `{archive_digest}`\n\nCurated dataset: `{dataset_name}`; build run: `{manifest.get('build_run')}` (six selected provenance files are exported).\n\nLive source fingerprint: `{source_fingerprint}`; git commit: `{commit}`; dirty: `{dirty}`.\n\nExported {len(records)} sequences, {data_report['data_bytes']} Parquet bytes, {len(entries)} files. Every SHA256, portable path, task schema, disjoint split and selected build provenance passed isolated extraction verification.\n\nSource-of-truth files in this run: `config_snapshot.yaml`, `data_manifest.json`, `source_snapshot.json`, `bundle_manifest.json`, `metrics.json`, `verification.json`, `archive_sha256.txt`, `report.md`. Raw data, other runs, outputs, environments, caches and keys are excluded. No remote operation or full training was performed.\n",
            encoding="utf-8",
        )
        log(
            f"VERIFIED: isolated archive round trip, {len(entries)} file checksums, {len(records)} curated records; archive_sha256={archive_digest}"
        )
        return TrainingBundleResult(archive_path, checksum_path, run_dir, archive_digest, inventory)
    except Exception as error:
        (run_dir / "verification.json").write_bytes(
            _json_bytes({"roundtrip_verified": False, "error": str(error)})
        )
        raise
    finally:
        partial.unlink(missing_ok=True)
