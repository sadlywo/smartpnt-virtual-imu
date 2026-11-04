"""Calibration parsing, transform construction, and asset acquisition."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from virtual_imu.io.pilot_download import (
    _gdrive_direct_download_url,
    download_artifact,
    resolve_gdrive_matches,
    sha256_file,
)
from virtual_imu.utils.paths import utc_now_iso

FUSION_CALIBRATION_FILES = (
    "body_imu.yaml",
    "event_cam00.yaml",
    "event_cam00_imu.yaml",
    "event_cam01.yaml",
    "event_cam01_imu.yaml",
    "ouster00.yaml",
    "ouster00_imu.yaml",
)


@dataclass(frozen=True)
class RigidTransform:
    """A sensor pose expressed in the common body frame."""

    rotation_body_from_sensor: np.ndarray
    position_body_to_sensor_m: np.ndarray
    calibration_id: str

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_body_from_sensor, dtype=float)
        translation = np.asarray(self.position_body_to_sensor_m, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {translation.shape}")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("RigidTransform values must be finite")
        rotation = rotation.copy()
        translation = translation.copy()
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation_body_from_sensor", rotation)
        object.__setattr__(self, "position_body_to_sensor_m", translation)


def load_opencv_yaml(path: Path) -> dict[str, Any]:
    """Load an OpenCV-flavored YAML mapping without changing global YAML state."""

    logger.info("READS OpenCV calibration YAML {}", path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("%YAML:"):
        text = "\n".join(lines[1:])

    class OpenCVSafeLoader(yaml.SafeLoader):
        pass

    def construct_opencv_matrix(
        loader: OpenCVSafeLoader,
        node: yaml.nodes.MappingNode,
    ) -> dict[str, Any]:
        return loader.construct_mapping(node, deep=True)

    OpenCVSafeLoader.add_constructor(
        "tag:yaml.org,2002:opencv-matrix",
        construct_opencv_matrix,
    )
    loaded = yaml.load(text, Loader=OpenCVSafeLoader)
    if not isinstance(loaded, dict):
        raise ValueError(f"OpenCV calibration YAML must contain a mapping: {path}")
    return loaded


def opencv_matrix(value: dict[str, Any]) -> np.ndarray:
    """Convert an OpenCV rows/cols/data mapping to a floating-point array."""

    if not isinstance(value, dict):
        raise ValueError("OpenCV matrix must be a mapping with rows, cols, and data")
    try:
        rows = int(value["rows"])
        cols = int(value["cols"])
        data = np.asarray(value["data"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OpenCV matrix requires valid rows, cols, and data") from exc
    if rows <= 0 or cols <= 0 or data.ndim != 1 or data.size != rows * cols:
        raise ValueError(
            "OpenCV matrix rows and cols do not match the one-dimensional data length"
        )
    if not np.isfinite(data).all():
        raise ValueError("OpenCV matrix data must be finite")
    return data.reshape(rows, cols)


def quaternion_wxyz_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a near-unit ``(w, x, y, z)`` quaternion to a rotation matrix."""

    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Quaternion must contain four finite values in (w, x, y, z) order")
    norm = float(np.linalg.norm(values))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"Quaternion norm differs from one by more than 1e-3: {norm}")
    values = values / norm
    w, x, y, z = values
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def compose_reference_to_body(
    sensor_to_body: np.ndarray,
    sensor_to_local_imu: np.ndarray,
) -> np.ndarray:
    """Compose reference-to-body and reference-to-local-IMU rotations."""

    body_from_reference = _rotation_matrix(sensor_to_body, "sensor_to_body")
    local_from_reference = _rotation_matrix(sensor_to_local_imu, "sensor_to_local_imu")
    return body_from_reference @ local_from_reference.T


def calibration_status(path: Path, required: bool) -> dict[str, Any]:
    """Report availability and strict-training eligibility for one calibration file."""

    if path.is_file():
        return {"available": True, "training_eligible": True, "reason": "available"}
    if required:
        return {
            "available": False,
            "training_eligible": False,
            "reason": "missing_calibration",
        }
    return {
        "available": False,
        "training_eligible": True,
        "reason": "optional_calibration_missing",
    }


def load_penn_body_transforms(path: Path) -> dict[str, RigidTransform]:
    """Load published Penn body-to-sensor matrices as body-from-sensor transforms."""

    logger.info("READS Penn body calibration {}", path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("transforms"), dict):
        raise ValueError("Penn calibration YAML requires a transforms mapping")

    result: dict[str, RigidTransform] = {}
    for sensor_name, definition in loaded["transforms"].items():
        if not isinstance(definition, dict):
            raise ValueError(f"Penn transform {sensor_name} must be a mapping")
        if definition.get("direction") != "body_to_sensor":
            raise ValueError(f"Penn transform {sensor_name} must declare body_to_sensor direction")
        matrix = np.asarray(definition.get("matrix_3x4"), dtype=float)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"Penn transform {sensor_name} matrix_3x4 must be finite 3x4")
        rotation_sensor_from_body = matrix[:, :3]
        translation_sensor_from_body = matrix[:, 3]
        rotation_body_from_sensor = rotation_sensor_from_body.T
        position_body_to_sensor = -rotation_body_from_sensor @ translation_sensor_from_body
        result[str(sensor_name)] = RigidTransform(
            rotation_body_from_sensor=rotation_body_from_sensor,
            position_body_to_sensor_m=position_body_to_sensor,
            calibration_id=str(definition["calibration_id"]),
        )
    return result


def load_fusion_body_transforms(calibration_root: Path) -> dict[str, RigidTransform]:
    """Build FusionPortable IMU transforms from the official 20220209 chain."""

    documents: dict[str, dict[str, Any]] = {}
    for filename in FUSION_CALIBRATION_FILES:
        path = calibration_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing FusionPortable calibration file: {path}")
        documents[filename] = load_opencv_yaml(path)

    result = {
        "stim300": RigidTransform(
            rotation_body_from_sensor=np.eye(3),
            position_body_to_sensor_m=np.zeros(3),
            calibration_id="fusionportable_20220209_stim300_body",
        )
    }
    result["davis_left_imu"] = _fusion_local_imu_transform(
        documents["event_cam00.yaml"],
        local_name="ecimu",
        calibration_id="fusionportable_20220209_davis_left_imu",
    )
    result["davis_right_imu"] = _fusion_local_imu_transform(
        documents["event_cam01.yaml"],
        local_name="ecimu",
        calibration_id="fusionportable_20220209_davis_right_imu",
    )
    result["ouster_imu"] = _fusion_local_imu_transform(
        documents["ouster00.yaml"],
        local_name="osimu00",
        calibration_id="fusionportable_20220209_ouster_imu",
    )
    return result


def acquire_calibration_assets(config: dict[str, Any]) -> dict[str, Any]:
    """Acquire declared calibration assets and atomically write per-dataset manifests."""

    raw_root = Path(str(config.get("raw_root", "data/raw")))
    min_free_space_gb = float(config.get("min_free_space_gb", 50.0))
    assets = config.get("calibration_assets", [])
    if not isinstance(assets, list):
        raise ValueError("calibration_assets must be a list")

    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    dataset_roots: dict[str, Path] = {}
    for source in assets:
        if not isinstance(source, dict):
            raise ValueError("Every calibration asset must be a mapping")
        artifact = dict(source)
        dataset = str(artifact["dataset"])
        dataset_root = _dataset_root(raw_root, dataset)
        dataset_roots[dataset] = dataset_root
        _ensure_calibration_layout(dataset_root)
        destination = _calibration_destination(artifact["destination"])
        required = bool(artifact.get("required_for_strict", False))
        logger.info("READS calibration source definition for {}", artifact["id"])

        resolved = dict(artifact)
        try:
            source_type = str(artifact["source_type"])
            if source_type == "gdrive_folder_match":
                logger.info("READS Google Drive listing for {}", artifact["id"])
                matches = resolve_gdrive_matches(artifact)
                if len(matches) != 1:
                    raise ValueError(f"expected exactly one Drive match, found {len(matches)}")
                match = matches[0]
                resolved["source_type"] = "http"
                resolved["url"] = _gdrive_direct_download_url(match["url"])
                resolved["resolved_listing_url"] = match["url"]
                resolved["resolved_source_path"] = match["path"]
                record = download_artifact(
                    resolved,
                    dataset_root,
                    min_free_space_gb=min_free_space_gb,
                )
            elif source_type == "local_file":
                record = _copy_local_asset(resolved, dataset_root)
            elif source_type == "http":
                record = download_artifact(
                    resolved,
                    dataset_root,
                    min_free_space_gb=min_free_space_gb,
                )
            else:
                raise ValueError(f"Unsupported calibration source_type: {source_type}")
        except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("BLOCKS calibration asset {}: {}", artifact.get("id"), exc)
            record = {
                "artifact_id": str(artifact.get("id", "unknown")),
                "dataset": dataset,
                "status": "blocked",
                "local_path": str(dataset_root / destination),
                "reason": str(exc),
            }

        record.update(_calibration_provenance(artifact, resolved, required))
        if record["status"] in {"downloaded", "verified_skip", "copied"}:
            status = calibration_status(Path(str(record["local_path"])), required)
            if status["available"]:
                logger.info("SKIPS or completes acquisition for {}: {}", artifact["id"], record["status"])
        else:
            status = {
                "available": False,
                "training_eligible": not required,
                "reason": "calibration_acquisition_failed",
            }
        record.update(status)
        records_by_dataset.setdefault(dataset, []).append(record)

    navinst = config.get("navinst")
    if isinstance(navinst, dict):
        dataset = "navinst_pilot"
        dataset_root = _dataset_root(raw_root, dataset)
        dataset_roots[dataset] = dataset_root
        _ensure_calibration_layout(dataset_root)
        path = _navinst_calibration_path(raw_root, navinst["expected_path"])
        required = bool(navinst.get("required_for_strict", False))
        logger.info("READS expected NavINST calibration path {}", path)
        status = calibration_status(path, required)
        if status["available"]:
            logger.info("SKIPS NavINST acquisition because calibration is already available")
        else:
            logger.warning("BLOCKS NavINST strict preparation: {}", status["reason"])
        nav_record = {
            "artifact_id": "navinst_setup_transformation",
            "dataset": dataset,
            "status": "available" if status["available"] else "missing",
            "local_path": str(path),
            "source_type": "unresolved_directory",
            "source_directory": navinst.get("source_directory"),
            "required_for_strict": required,
            "missing_behavior": navinst.get("missing_behavior"),
            "recorded_utc": utc_now_iso(),
            **status,
        }
        if path.is_file():
            nav_record.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
        records_by_dataset.setdefault(dataset, []).append(nav_record)

    all_records: list[dict[str, Any]] = []
    manifest_paths: dict[str, str] = {}
    for dataset, records in records_by_dataset.items():
        dataset_root = dataset_roots[dataset]
        manifest_path = dataset_root / "_manifests" / "calibration_manifest.json"
        manifest = {
            "created_utc": utc_now_iso(),
            "dataset_root": str(dataset_root),
            "artifact_count": len(records),
            "status_counts": _status_counts(records),
            "artifacts": records,
        }
        logger.info("WRITES calibration manifest {}", manifest_path)
        _write_json(manifest_path, manifest)
        manifest_paths[dataset] = str(manifest_path)
        all_records.extend(records)

    return {
        "created_utc": utc_now_iso(),
        "raw_root": str(raw_root),
        "artifact_count": len(all_records),
        "status_counts": _status_counts(all_records),
        "manifests": manifest_paths,
        "artifacts": all_records,
    }


def _rotation_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return result


def _opencv_vector(document: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = opencv_matrix(document[key]).reshape(-1)
    if value.size != size:
        raise ValueError(f"{key} must contain {size} values")
    return value


def _fusion_local_imu_transform(
    document: dict[str, Any],
    *,
    local_name: str,
    calibration_id: str,
) -> RigidTransform:
    body_from_reference = quaternion_wxyz_to_matrix(
        _opencv_vector(document, "quaternion_sensor_bodyimu", 4)
    )
    local_from_reference = quaternion_wxyz_to_matrix(
        _opencv_vector(document, f"quaternion_sensor_{local_name}", 4)
    )
    body_from_local = compose_reference_to_body(body_from_reference, local_from_reference)
    reference_origin_in_body = _opencv_vector(document, "translation_sensor_bodyimu", 3)
    reference_origin_in_local = _opencv_vector(
        document,
        f"translation_sensor_{local_name}",
        3,
    )
    local_origin_in_body = reference_origin_in_body - body_from_local @ reference_origin_in_local
    return RigidTransform(
        rotation_body_from_sensor=body_from_local,
        position_body_to_sensor_m=local_origin_in_body,
        calibration_id=calibration_id,
    )


def _calibration_destination(value: Any) -> Path:
    destination = Path(str(value))
    if (
        destination.is_absolute()
        or ".." in destination.parts
        or not destination.parts
        or destination.parts[0] != "02_calibration"
    ):
        raise ValueError(f"Calibration destination must stay under 02_calibration: {destination}")
    return destination


def _dataset_root(raw_root: Path, value: Any) -> Path:
    dataset = Path(str(value))
    if (
        dataset.is_absolute()
        or dataset.drive
        or len(dataset.parts) != 1
        or dataset.name in {"", ".", ".."}
    ):
        raise ValueError(f"Dataset must be a single safe path segment: {value}")

    resolved_raw_root = raw_root.resolve()
    resolved_dataset_root = (resolved_raw_root / dataset).resolve()
    try:
        resolved_dataset_root.relative_to(resolved_raw_root)
    except ValueError as exc:
        raise ValueError(f"Dataset root escapes raw_root: {resolved_dataset_root}") from exc
    return resolved_dataset_root


def _navinst_calibration_path(raw_root: Path, value: Any) -> Path:
    configured_path = Path(str(value))
    if configured_path.is_absolute() or configured_path.drive or ".." in configured_path.parts:
        raise ValueError(f"NavINST expected_path must be a safe relative path: {value}")

    resolved_path = configured_path.resolve()
    calibration_root = (_dataset_root(raw_root, "navinst_pilot") / "02_calibration").resolve()
    try:
        resolved_path.relative_to(calibration_root)
    except ValueError as exc:
        raise ValueError(
            "NavINST expected_path must stay under "
            f"{calibration_root}: {configured_path}"
        ) from exc
    return resolved_path


def _ensure_calibration_layout(dataset_root: Path) -> None:
    (dataset_root / "02_calibration").mkdir(parents=True, exist_ok=True)
    (dataset_root / "_manifests").mkdir(parents=True, exist_ok=True)


def _copy_local_asset(artifact: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    source = Path(str(artifact["source_path"]))
    destination = dataset_root / _calibration_destination(artifact["destination"])
    logger.info("READS local calibration asset {}", source)
    if not source.is_file():
        raise FileNotFoundError(f"Local calibration source is missing: {source}")
    source_sha256 = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != source_sha256:
            raise ValueError(f"Existing calibration destination differs from source: {destination}")
        logger.info("SKIPS verified local calibration asset {}", destination)
        status = "verified_skip"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        logger.info("WRITES local calibration asset {}", destination)
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        status = "copied"
    return {
        "artifact_id": str(artifact["id"]),
        "dataset": str(artifact["dataset"]),
        "status": status,
        "local_path": str(destination),
        "size": destination.stat().st_size,
        "sha256": source_sha256,
    }


def _calibration_provenance(
    source: dict[str, Any],
    resolved: dict[str, Any],
    required: bool,
) -> dict[str, Any]:
    return {
        "source_type": str(source["source_type"]),
        "source_page": source.get("source_page"),
        "source_url": resolved.get("url", resolved.get("folder_url")),
        "source_path": source.get("source_path"),
        "resolved_listing_url": resolved.get("resolved_listing_url"),
        "resolved_source_path": resolved.get("resolved_source_path"),
        "required_for_strict": required,
        "recorded_utc": utc_now_iso(),
    }


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
