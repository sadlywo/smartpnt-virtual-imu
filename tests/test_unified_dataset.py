import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _paired(times: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"timestamp_s": times})
    for suffix in ("low", "high"):
        for axis in ("gx", "gy", "gz", "ax", "ay", "az"):
            frame[f"{axis}_{suffix}"] = np.sin(2 * np.pi * times)
    return frame


def test_resampling_rejects_gap_bridging_and_keeps_paired_phase():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    times = np.r_[np.arange(500) / 100, 10 + np.arange(500) / 100]
    segments, metrics = regularize_segments(
        _paired(times),
        [
            f"{axis}_{suffix}"
            for suffix in ("low", "high")
            for axis in ("gx", "gy", "gz", "ax", "ay", "az")
        ],
        rate_hz=40,
        native_rate_hz=100,
        min_rows=80,
    )
    assert len(segments) == 2
    assert metrics["anti_alias_applied"]
    for segment in segments:
        assert np.allclose(np.diff(segment.timestamp_s), 0.025)
        assert np.allclose(segment.gx_low, segment.gx_high)
        assert np.isfinite(segment.gx_low).all()


def test_downsampling_suppresses_above_nyquist_signal():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    times = np.arange(2000) / 100
    frame = pd.DataFrame({"timestamp_s": times, "gx_low": np.sin(2 * np.pi * 30 * times)})
    segments, _ = regularize_segments(
        frame, ["gx_low"], rate_hz=40, native_rate_hz=100, min_rows=80
    )
    assert np.std(segments[0].gx_low) < 0.01


def test_native_rate_is_not_upsampled():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    with pytest.raises(ValueError, match="upsampl"):
        regularize_segments(
            _paired(np.arange(200) / 40), ["gx_low"], rate_hz=100, native_rate_hz=40, min_rows=80
        )


def test_quality_mask_even_one_sample_breaks_a_segment():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    frame = _paired(np.arange(800) / 40)
    frame["_label_valid"] = True
    frame.loc[400, "_label_valid"] = False
    segments, _ = regularize_segments(frame, ["gx_low"], rate_hz=40, native_rate_hz=40, min_rows=80)
    assert len(segments) == 2
    assert segments[0].timestamp_s.max() < segments[1].timestamp_s.min()


def test_heading_interpolation_respects_wrap():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    times = np.arange(400) / 100
    frame = _paired(times)
    frame["gt_heading_deg"] = (359 + 2 * times) % 360
    segments, _ = regularize_segments(
        frame,
        ["gx_low"],
        rate_hz=40,
        native_rate_hz=100,
        min_rows=80,
        label_columns=["gt_heading_deg"],
        angle_columns=["gt_heading_deg"],
    )
    heading = np.unwrap(np.deg2rad(segments[0].gt_heading_deg))
    assert np.max(abs(np.diff(heading))) < 0.001


def test_absent_native_sample_breaks_context():
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    times = np.delete(np.arange(1000) / 100, 500)
    segments, _ = regularize_segments(
        _paired(times), ["gx_low"], rate_hz=40, native_rate_hz=100, min_rows=80, max_gap_periods=1.5
    )
    assert len(segments) == 2


def test_native_smartpnt_enu_and_no_prior_aliasing():
    from virtual_imu.preprocessing.smartpnt_native import native_smartpnt_frame
    from virtual_imu.preprocessing.unified_signals import regularize_segments

    times = np.arange(4000) / 200
    imu = pd.DataFrame({"timestamp_s": times})
    for axis in ("gx", "gy", "gz"):
        imu[f"{axis}_radps"] = np.sin(2 * np.pi * 90 * times)
    for axis in ("ax", "ay", "az"):
        imu[f"{axis}_mps2"] = 9.81
    gt_times = np.arange(2000) / 100
    gt = pd.DataFrame(
        {
            "timestamp_s": gt_times,
            "latitude_deg": 0.0,
            "longitude_deg": 0.0,
            "x_ecef_m": 6378137.0,
            "y_ecef_m": gt_times,
            "z_ecef_m": 0.0,
            "vx_ecef_mps": 0.0,
            "vy_ecef_mps": 1.0,
            "vz_ecef_mps": 0.0,
            "heading_deg": 90.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "ambiguity_status": "Fixed",
            "quality": 1.0,
            "ins_flag": "NONE",
        }
    )
    frame = native_smartpnt_frame(imu, gt)
    assert np.allclose(frame.gt_x_local_m, frame.timestamp_s)
    assert np.allclose(frame.gt_v_east_mps, 1)
    segments, _ = regularize_segments(
        frame, ["gx_radps"], rate_hz=40, native_rate_hz=200, min_rows=80
    )
    assert np.std(segments[0].gx_radps) < 0.01
    gt.loc[101, "ambiguity_status"] = "Float"
    gt.loc[200, "timestamp_s"] += 0.02
    masked = native_smartpnt_frame(imu, gt.drop(index=[201, 202]))
    between = masked.timestamp_s.between(1.005, 1.015)
    assert not masked.loc[between, "_label_valid"].any()
    gap = masked.timestamp_s.between(1.995, 2.015)
    assert not masked.loc[gap, "_label_valid"].any()
    gt.loc[400, "heading_deg"] = np.nan
    masked = native_smartpnt_frame(imu, gt.drop(index=[201, 202]))
    later = masked.timestamp_s > 4.1
    assert np.isfinite(masked.loc[later, "gt_heading_deg"]).all()
    assert masked.loc[later, "_label_valid"].all()


def test_content_dedup_ignores_identity_but_not_measurements():
    from virtual_imu.io.unified_dataset import measurement_digest

    a = _paired(np.arange(200) / 40)
    a["sequence_id"] = "Traj1"
    b = a.copy()
    b["sequence_id"] = "Traj2"
    assert measurement_digest(a) == measurement_digest(b)
    b.loc[50, "gx_low"] += 0.01
    assert measurement_digest(a) != measurement_digest(b)


def test_builder_excludes_duplicate_and_navigation_from_signal_labels(tmp_path: Path):
    from virtual_imu.io.unified_dataset import (
        file_sha256,
        prepare_unified_dataset,
        verify_unified_dataset,
    )

    raw = tmp_path / "data/processed/inputs"
    raw.mkdir(parents=True)
    times = np.arange(1000) / 100
    for name in ("Traj1", "Traj2", "Traj3", "Traj4"):
        frame = pd.DataFrame({"timestamp_s": times})
        for axis in ("gx", "gy", "gz"):
            frame[f"{axis}_radps"] = np.sin(times)
        for axis in ("ax", "ay", "az"):
            frame[f"{axis}_mps2"] = np.cos(times)
        for axis in ("x", "y", "z"):
            frame[f"gt_{axis}_local_m"] = times + (
                0 if name in {"Traj1", "Traj2"} else int(name[-1])
            )
        for axis in ("east", "north", "up"):
            frame[f"gt_v_{axis}_mps"] = 1.0
        for angle in ("heading", "pitch", "roll"):
            frame[f"gt_{angle}_deg"] = times
        frame["gt_ambiguity_status"] = "Fixed"
        frame["sequence_id"] = name
        frame.to_parquet(raw / f"{name}_imu_gt_aligned.parquet", index=False)
    config = {
        "dataset_id": "test_v1",
        "seed": 7,
        "output_root": "data/processed/unified_imu_v1",
        "rate_hz": 40,
        "window_size": 80,
        "navigation_ambiguity": "Fixed",
        "datasets": {
            "smartpnt_msf": {
                "root": "data/processed/inputs",
                "native_rate_hz": 100,
                "splits": {"train": ["Traj3"], "validation": ["Traj4"], "test": ["Traj1"]},
            }
        },
    }
    manifest = prepare_unified_dataset(config, project_root=tmp_path)
    assert len(manifest["duplicates"]) == 1
    assert manifest["duplicates"][0]["original_sequence_id"] == "Traj2"
    assert not any(r["task"] == "paired_signal" for r in manifest["records"])
    for record in manifest["records"]:
        frame = pd.read_parquet(tmp_path / record["path"])
        assert frame.has_navigation_label.all()
        assert not frame.has_signal_label.any()
        assert "gx_high" not in frame
        assert "gx_low" in frame
    assert (tmp_path / manifest["build_run"] / "report.md").is_file()
    snapshot = tmp_path / config["output_root"] / "config_snapshot.yaml"
    assert file_sha256(snapshot) == manifest["config_sha256"]
    snapshot.write_bytes(snapshot.read_bytes() + b"\n# corrupted snapshot\n")
    with pytest.raises(ValueError, match="config snapshot checksum"):
        verify_unified_dataset(tmp_path / config["output_root"] / "manifest.json", tmp_path)


def test_dataset_balancing_assigns_equal_total_mass():
    from virtual_imu.training.unified_data import dataset_balanced_sampler

    frame = pd.concat(
        [
            pd.DataFrame(
                {"sequence_id": "a", "dataset_id": "alimu", "timestamp_s": np.arange(100)}
            ),
            pd.DataFrame(
                {"sequence_id": "b", "dataset_id": "penncosyvio", "timestamp_s": np.arange(20)}
            ),
        ],
        ignore_index=True,
    )
    sampler = dataset_balanced_sampler(frame, window_size=10, stride=10, seed=7)
    assert float(sampler.weights[:10].sum()) == pytest.approx(float(sampler.weights[10:].sum()))


def test_navigation_split_preserves_segment_boundaries():
    from virtual_imu.training.gt_supervised_dataset import _sequence_subset

    frame = pd.DataFrame(
        {
            "sequence_id": ["smart__Traj3__seg0", "smart__Traj4__seg0"],
            "original_sequence_id": ["Traj3", "Traj4"],
        }
    )
    selected = _sequence_subset(frame, ["Traj3"], "train")
    assert selected.sequence_id.tolist() == ["smart__Traj3__seg0"]


def test_shared_reference_recording_stays_in_one_split(tmp_path: Path):
    from virtual_imu.io.unified_dataset import prepare_unified_dataset

    source = tmp_path / "data/processed/alimu/all.parquet"
    source.parent.mkdir(parents=True)
    frames = []
    for recording in (1, 2, 3):
        for sensor, offset in (("BMX160", 0.01), ("MPU9250", 0.02)):
            frame = _paired(np.arange(160) / 40)
            frame[[f"{axis}_high" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]] += recording
            frame[[f"{axis}_low" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]] += offset
            frame["sequence_id"] = f"real_sensor_data_dyn_{recording:02d}_{sensor}"
            frame["low_sensor"], frame["high_sensor"] = sensor, "VN100"
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(source, index=False)
    config = {
        "dataset_id": "group_test",
        "output_root": "data/processed/group_test",
        "rate_hz": 40,
        "datasets": {
            "alimu": {
                "source": "data/processed/alimu/all.parquet",
                "native_rate_hz": 40,
                "recording_splits": {
                    "train": ["dyn_01"],
                    "validation": ["dyn_02"],
                    "test": ["dyn_03"],
                },
            }
        },
    }
    manifest = prepare_unified_dataset(config, project_root=tmp_path)
    assignments = {}
    for record in manifest["records"]:
        group = record["physical_group_id"]
        assert assignments.setdefault(group, record["split"]) == record["split"]
    assert len(assignments) == 3
    split = json.loads(
        (tmp_path / config["output_root"] / "splits/paired_default.json").read_text()
    )
    assert split["group_key"] == "physical_recording"
    # Different low measurements cannot conceal a repeated reference recording.
    damaged = frames[-1].copy()
    high = [f"{axis}_high" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]
    damaged[high] = frames[0][high].to_numpy()
    frames[-1] = damaged
    pd.concat(frames, ignore_index=True).to_parquet(source, index=False)
    config["output_root"] = "data/processed/group_bad"
    with pytest.raises(ValueError, match=r"reference recording.*split"):
        prepare_unified_dataset(config, project_root=tmp_path)


def test_partial_input_duplicates_are_archived_with_reference_differences(tmp_path: Path):
    from virtual_imu.io.unified_dataset import prepare_unified_dataset, verify_unified_dataset

    source = tmp_path / "data/processed/alimu/all.parquet"
    source.parent.mkdir(parents=True)
    frames = []
    for recording in (1, 2, 3, 4):
        frame = _paired(np.arange(120 if recording == 1 else 160) / 40)
        high = [f"{axis}_high" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]
        frame[high] += recording * 0.01
        frame["low_sensor"], frame["high_sensor"] = "BMX160", "VN100"
        frame["sequence_id"] = f"real_sensor_data_dyn_{recording:02d}_BMX160"
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(source, index=False)
    config = {
        "dataset_id": "partial",
        "output_root": "data/processed/partial",
        "rate_hz": 40,
        "datasets": {
            "alimu": {
                "source": "data/processed/alimu/all.parquet",
                "native_rate_hz": 40,
                "exclude_recordings": {"dyn_01": "dyn_02"},
                "recording_splits": {
                    "train": ["dyn_02"],
                    "validation": ["dyn_03"],
                    "test": ["dyn_04"],
                },
            }
        },
    }
    manifest = prepare_unified_dataset(config, project_root=tmp_path)
    assert len(manifest["records"]) == 3
    assert len(manifest["duplicates"]) == 1
    duplicate = manifest["duplicates"][0]
    assert duplicate["input_measurement_sha256"] == duplicate["canonical_prefix_sha256"]
    assert not duplicate["reference_prefix_equal"]
    assert duplicate["reference_prefix_max_difference"][0] == pytest.approx(0.01)
    status = tmp_path / config["output_root"] / "dataset_status.json"
    status.write_text(json.dumps({"status": "superseded_diagnostic"}), encoding="utf-8")
    with pytest.raises(ValueError, match="superseded diagnostic"):
        verify_unified_dataset(status.parent / "manifest.json", tmp_path)
