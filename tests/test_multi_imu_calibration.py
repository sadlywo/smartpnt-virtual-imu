from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from virtual_imu.io.multi_imu_calibration import (
    acquire_calibration_assets,
    calibration_status,
    compose_reference_to_body,
    load_fusion_body_transforms,
    load_opencv_yaml,
    load_penn_body_transforms,
    opencv_matrix,
    quaternion_wxyz_to_matrix,
)


def test_quaternion_wxyz_to_matrix_rotates_about_z() -> None:
    half_angle = np.sqrt(0.5)

    rotation = quaternion_wxyz_to_matrix([half_angle, 0.0, 0.0, half_angle])

    np.testing.assert_allclose(
        rotation,
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_compose_reference_to_body_uses_inverse_local_imu_rotation() -> None:
    sensor_to_body = quaternion_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0])
    sensor_to_local_imu = quaternion_wxyz_to_matrix([0.0, 0.0, 0.0, 1.0])

    result = compose_reference_to_body(sensor_to_body, sensor_to_local_imu)

    np.testing.assert_allclose(result, sensor_to_body @ sensor_to_local_imu.T)


def test_missing_required_calibration_is_not_training_eligible(tmp_path: Path) -> None:
    status = calibration_status(tmp_path / "setup_transformation.txt", required=True)

    assert status == {
        "available": False,
        "training_eligible": False,
        "reason": "missing_calibration",
    }


def test_load_opencv_yaml_parses_matrix_without_registering_global_tag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calibration.yaml"
    path.write_text(
        """%YAML:1.0
rotation: !!opencv-matrix
  rows: 2
  cols: 3
  dt: f
  data: [1, 2, 3, 4, 5, 6]
""",
        encoding="utf-8",
    )

    loaded = load_opencv_yaml(path)

    np.testing.assert_array_equal(
        opencv_matrix(loaded["rotation"]),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.safe_load(
            """rotation: !!opencv-matrix
  rows: 1
  cols: 1
  data: [1]
"""
        )


def test_opencv_matrix_rejects_declared_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"rows.*cols.*data"):
        opencv_matrix({"rows": 2, "cols": 2, "data": [1.0, 2.0, 3.0]})


def test_quaternion_norm_error_above_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="norm"):
        quaternion_wxyz_to_matrix([1.002, 0.0, 0.0, 0.0])


def test_load_penn_body_transforms_inverts_published_body_to_sensor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "penn.yaml"
    path.write_text(
        """version: 1
source_url: https://example.test/extrinsics
transforms:
  tango_top_imu:
    calibration_id: penn_tango
    published_notation: T_B_TTI
    direction: body_to_sensor
    matrix_3x4:
      - [0, -1, 0, 1]
      - [1, 0, 0, 2]
      - [0, 0, 1, 3]
""",
        encoding="utf-8",
    )

    transform = load_penn_body_transforms(path)["tango_top_imu"]

    published_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    published_translation = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(transform.rotation_body_from_sensor, published_rotation.T)
    np.testing.assert_allclose(
        transform.position_body_to_sensor_m,
        -published_rotation.T @ published_translation,
    )
    assert transform.calibration_id == "penn_tango"


def test_load_fusion_body_transforms_composes_camera_and_local_imu(
    tmp_path: Path,
) -> None:
    _write_fusion_fixture(tmp_path)

    transforms = load_fusion_body_transforms(tmp_path)

    expected_event_rotation = quaternion_wxyz_to_matrix(
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    )
    np.testing.assert_allclose(
        transforms["davis_left_imu"].rotation_body_from_sensor,
        expected_event_rotation,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        transforms["davis_left_imu"].position_body_to_sensor_m,
        [1.0, -1.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_array_equal(transforms["stim300"].rotation_body_from_sensor, np.eye(3))
    assert set(transforms) == {"stim300", "davis_left_imu", "davis_right_imu", "ouster_imu"}


def test_acquire_calibration_assets_writes_separate_manifest_and_skips_existing(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    dataset_root = raw_root / "fusionportable_pilot"
    calibration = dataset_root / "02_calibration" / "body_imu.yaml"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("update_rate: 200\n", encoding="utf-8")
    pilot_manifest = dataset_root / "_manifests" / "pilot_manifest.json"
    pilot_manifest.parent.mkdir(parents=True)
    pilot_manifest.write_text('{"owner": "pilot"}', encoding="utf-8")
    config = {
        "raw_root": str(raw_root),
        "min_free_space_gb": 0,
        "calibration_assets": [
            {
                "id": "fusion_body_imu",
                "dataset": "fusionportable_pilot",
                "source_type": "http",
                "url": "https://example.test/body_imu.yaml",
                "destination": "02_calibration/body_imu.yaml",
                "required_for_strict": True,
            }
        ],
    }

    summary = acquire_calibration_assets(config)

    record = summary["artifacts"][0]
    assert record["status"] == "verified_skip"
    assert record["training_eligible"] is True
    assert len(record["sha256"]) == 64
    assert json.loads(pilot_manifest.read_text(encoding="utf-8")) == {"owner": "pilot"}
    calibration_manifest = dataset_root / "_manifests" / "calibration_manifest.json"
    assert json.loads(calibration_manifest.read_text(encoding="utf-8"))["artifact_count"] == 1


def test_acquire_calibration_assets_rejects_absolute_dataset(tmp_path: Path) -> None:
    config = _single_asset_config(tmp_path / "raw", str(tmp_path / "outside"))

    with pytest.raises(ValueError, match="single safe path segment"):
        acquire_calibration_assets(config)


def test_acquire_calibration_assets_rejects_dataset_parent_escape(tmp_path: Path) -> None:
    config = _single_asset_config(tmp_path / "raw", "../outside")

    with pytest.raises(ValueError, match="single safe path segment"):
        acquire_calibration_assets(config)


def test_acquire_calibration_assets_rejects_absolute_navinst_path(tmp_path: Path) -> None:
    config = {
        "raw_root": str(tmp_path / "raw"),
        "navinst": {
            "expected_path": str(tmp_path / "outside" / "setup_transformation.txt"),
            "required_for_strict": True,
        },
    }

    with pytest.raises(ValueError, match="NavINST expected_path"):
        acquire_calibration_assets(config)


def test_acquire_calibration_assets_rejects_navinst_parent_escape(tmp_path: Path) -> None:
    config = {
        "raw_root": str(tmp_path / "raw"),
        "navinst": {
            "expected_path": "../outside/setup_transformation.txt",
            "required_for_strict": True,
        },
    }

    with pytest.raises(ValueError, match="NavINST expected_path"):
        acquire_calibration_assets(config)


def test_project_navinst_path_is_not_prefixed_with_raw_root_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = Path("data/raw/navinst_pilot/02_calibration/setup_transformation.txt")
    config = {
        "raw_root": "data/raw",
        "navinst": {
            "expected_path": str(expected),
            "required_for_strict": True,
            "missing_behavior": "diagnostic_only",
        },
    }

    summary = acquire_calibration_assets(config)

    assert Path(summary["artifacts"][0]["local_path"]) == expected.resolve()
    assert Path(summary["manifests"]["navinst_pilot"]) == (
        tmp_path / "data/raw/navinst_pilot/_manifests/calibration_manifest.json"
    )


def _single_asset_config(raw_root: Path, dataset: str) -> dict[str, object]:
    return {
        "raw_root": str(raw_root),
        "min_free_space_gb": 0,
        "calibration_assets": [
            {
                "id": "unsafe",
                "dataset": dataset,
                "source_type": "unsupported",
                "destination": "02_calibration/calibration.yaml",
            }
        ],
    }


def _write_fusion_fixture(root: Path) -> None:
    (root / "body_imu.yaml").write_text("update_rate: 200\n", encoding="utf-8")
    (root / "event_cam00.yaml").write_text(
        _fusion_extrinsic_yaml(
            body_quaternion=[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            body_translation=[1.0, 0.0, 0.0],
            local_quaternion=[1.0, 0.0, 0.0, 0.0],
            local_translation=[1.0, 0.0, 0.0],
            local_name="ecimu",
        ),
        encoding="utf-8",
    )
    (root / "event_cam01.yaml").write_text(
        _fusion_extrinsic_yaml(
            body_quaternion=[1.0, 0.0, 0.0, 0.0],
            body_translation=[0.0, 0.0, 0.0],
            local_quaternion=[1.0, 0.0, 0.0, 0.0],
            local_translation=[0.0, 0.0, 0.0],
            local_name="ecimu",
        ),
        encoding="utf-8",
    )
    (root / "ouster00.yaml").write_text(
        _fusion_extrinsic_yaml(
            body_quaternion=[1.0, 0.0, 0.0, 0.0],
            body_translation=[0.0, 0.0, 0.0],
            local_quaternion=[1.0, 0.0, 0.0, 0.0],
            local_translation=[0.0, 0.0, 0.0],
            local_name="osimu00",
        ),
        encoding="utf-8",
    )
    for name in ("event_cam00_imu.yaml", "event_cam01_imu.yaml", "ouster00_imu.yaml"):
        (root / name).write_text("update_rate: 100\n", encoding="utf-8")


def _fusion_extrinsic_yaml(
    *,
    body_quaternion: list[float],
    body_translation: list[float],
    local_quaternion: list[float],
    local_translation: list[float],
    local_name: str,
) -> str:
    return f"""quaternion_sensor_bodyimu: !!opencv-matrix
  rows: 1
  cols: 4
  data: {body_quaternion}
translation_sensor_bodyimu: !!opencv-matrix
  rows: 1
  cols: 3
  data: {body_translation}
quaternion_sensor_{local_name}: !!opencv-matrix
  rows: 1
  cols: 4
  data: {local_quaternion}
translation_sensor_{local_name}: !!opencv-matrix
  rows: 1
  cols: 3
  data: {local_translation}
"""
