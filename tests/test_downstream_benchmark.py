"""Historical replay must preserve identities, support and optimizer seed semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "downstream_benchmark", Path(__file__).parents[1] / "scripts/evaluate_downstream_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_baseline_state_is_selected_by_exact_penn_sensor_mapping() -> None:
    identity = {
        "dataset_id": "penncosyvio",
        "low_sensor": "Tango Top IMU",
        "high_sensor": "ADIS16488",
    }
    state = {"identity": identity, "fit_split": "train", "bias": [0] * 6}
    assert (
        MODULE.select_penn_state([{"identity": {**identity, "high_sensor": "VN100"}}, state])
        == state
    )
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.select_penn_state([state, state])
    with pytest.raises(ValueError, match="train"):
        MODULE.select_penn_state([{**state, "fit_split": "test"}])


def test_source_time_recovery_uses_exact_sequence_timestamp_keys(tmp_path) -> None:
    paired = pd.DataFrame(
        {
            "sequence_id": ["penncosyvio__bs__seg0000"] * 3,
            "timestamp_s": [1.525, 1.55, 1.575],
            "source_timestamp_s": [1.53, 1.555, 1.58],
        }
    )
    paired.to_parquet(tmp_path / "penncosyvio__bs__seg0000.parquet")
    recovered, provenance = MODULE.recover_source_clock(
        paired.drop(columns="source_timestamp_s").iloc[[2, 0]], tmp_path
    )
    np.testing.assert_allclose(recovered.source_timestamp_s, [1.58, 1.53])
    assert provenance[0]["sha256"]
    missing = paired.drop(columns="source_timestamp_s").iloc[[0]].copy()
    missing["timestamp_s"] = 8.0
    with pytest.raises(ValueError, match="missing"):
        MODULE.recover_source_clock(missing, tmp_path)


def test_method_aggregation_counts_classical_once_and_seeds_as_optimization_variation() -> None:
    trials = pd.DataFrame(
        [
            {
                "task": "attitude",
                "method": "identity",
                "seed": None,
                "samples": 80,
                "horizon_s": 1.975,
                "count": 2,
                "endpoint_rmse_deg": 4.0,
            },
            *[
                {
                    "task": "attitude",
                    "method": "tcn_original",
                    "seed": seed,
                    "samples": 80,
                    "horizon_s": 1.975,
                    "count": 2,
                    "endpoint_rmse_deg": value,
                }
                for seed, value in [(1, 1.0), (2, 2.0), (3, 3.0)]
            ],
        ]
    )
    result = MODULE.aggregate_trials(trials)
    classic = result[result.method == "identity"].iloc[0]
    neural = result[result.method == "tcn_original"].iloc[0]
    assert classic.optimizer_runs == 1
    assert classic.endpoint_rmse_deg_sd == 0
    assert neural.optimizer_runs == 3
    assert neural.endpoint_rmse_deg_mean == 2
    assert neural.endpoint_rmse_deg_sd == 1
    assert neural.endpoint_rmse_deg_improvement_over_low_pct == 50


def test_changed_historical_baseline_arithmetic_is_rejected(tmp_path) -> None:
    current, historical = tmp_path / "current.py", tmp_path / "historical.py"
    current.write_text("def predict(x):\n    return x + 1\n")
    historical.write_text("def predict(x):\n    return x + 2\n")
    with pytest.raises(ValueError, match="arithmetic changed"):
        MODULE.verify_replay_source(current, historical, ["predict"])
    historical.write_text("def predict(x):\n    return x + 1\n")
    assert MODULE.verify_replay_source(current, historical, ["predict"])["current_sha256"]
