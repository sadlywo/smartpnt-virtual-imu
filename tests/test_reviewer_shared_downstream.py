"""Synthetic wrapper support tests; science is covered by downstream_attitude."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_reviewer_shared_downstream as downstream
import reviewer_shared_classical as shared


def physical():
    data = pd.DataFrame(np.zeros((320, 6)), columns=shared.LOW)
    data[shared.HIGH] = 0.0
    data["dataset_id"] = "penncosyvio"
    data["low_sensor"], data["high_sensor"] = "Tango Top IMU", "ADIS16488"
    data["sequence_id"], data["original_sequence_id"] = "fixture_segment", "bs"
    data["timestamp_s"] = np.arange(len(data)) / 40
    data["source_timestamp_s"] = data.timestamp_s + 100
    data["available_timestamp_s"] = data.timestamp_s + 0.1
    data["calibration_id"] = "penncosyvio_published_body_extrinsics_v1"
    data["lever_arm_compensated"] = True
    data["sample_rate_hz"] = 40.0
    return data


def test_exact_aux1_clock_and_physical_mapping():
    data = physical()
    pd.testing.assert_frame_equal(downstream.exact_physical_frame(data, data, "bs"), data)
    changed = data.copy()
    changed["source_timestamp_s"] += 1
    with pytest.raises(ValueError, match="clock"):
        downstream.exact_physical_frame(changed, data, "bs")
    changed = data.copy()
    changed["calibration_id"] = "refitted"
    with pytest.raises(ValueError, match="physical support"):
        downstream.exact_physical_frame(changed, data, "bs")


def test_wrapper_passes_original_gt_origin_and_past_initialization(monkeypatch):
    data = physical()
    states = {
        method: shared._state("penn_only", method, kind="identity") for method in shared.METHODS
    }
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/downstream_penn.yaml").read_text(
            encoding="utf-8"
        )
    )
    seen = []
    support = {
        "input_rows": len(data),
        "common_valid_rows": len(data),
        "common_methods": ["fixture"],
    }
    propagation = {"algorithm": "fixture_original"}

    def evaluator(frame, truth, **kwargs):
        seen.append(kwargs)
        return {"support": support, "protocol": propagation}

    monkeypatch.setattr(downstream, "evaluate_attitude_stream", evaluator)
    monkeypatch.setattr(downstream, "evaluate_strapdown_stream", evaluator)
    metadata = {
        "physical_support": {"source_clock_origin_s": 100.0},
        "support": {task: support for task in ("attitude", "strapdown")},
        "propagation_protocol": {task: propagation for task in ("attitude", "strapdown")},
    }
    downstream.replay_view(data, pd.DataFrame(), states, config, metadata)
    assert seen[0]["gt_timestamp_origin_s"] == 100.0
    assert seen[1]["initial_velocity_span_s"] == 0.1
    assert seen[1]["gravity_world_mps2"] == [0.0, 0.0, -9.80665]
    assert seen[0]["window_samples"] == [80, 200, 400, 800]
    changed = copy.deepcopy(metadata)
    changed["physical_support"]["source_clock_origin_s"] = 101
    with pytest.raises(ValueError, match="clock origin"):
        downstream.replay_view(data, pd.DataFrame(), states, config, changed)


def test_window_key_mapping_rejects_shift_without_reselecting():
    row = {
        "sequence_id": "fixture",
        "segment": 0,
        "samples": 80,
        "start_s": 0.0,
        "end_s": 1.975,
        "endpoint_deg": 0.0,
    }
    result = {
        task: {
            "window_details": [{**row, "method": method} for method in shared.METHODS],
            "full_recordings": [],
        }
        for task in ("attitude", "strapdown")
    }
    reference = {
        "window_details": [{**row, "task": task, "method": "identity"} for task in result],
        "full_segments": [],
    }
    downstream.match_reference_keys(result, reference)
    result["attitude"]["window_details"][1]["start_s"] = 0.025
    with pytest.raises(ValueError, match="window/segment keys"):
        downstream.match_reference_keys(result, reference)
