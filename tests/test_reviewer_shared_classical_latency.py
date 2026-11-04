"""SYNTHETIC proofs for shared/bank CPU kernels and accounting."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reviewer_shared_classical import LOW, _state, predict_state  # noqa: E402


def module():
    path = ROOT / "scripts/benchmark_reviewer_shared_classical_latency.py"
    spec = importlib.util.spec_from_file_location("shared_cpu_test", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def frame(seed=5, pair="low", length=640):
    values = np.random.default_rng(seed).normal(size=(length, 6))
    result = pd.DataFrame(values, columns=LOW)
    for name, value in {
        "dataset_id": "SYNTHETIC",
        "low_sensor": pair,
        "high_sensor": "high",
        "sequence_id": "SYNTHETIC_" + pair,
        "original_sequence_id": "SYNTHETIC_" + pair,
        "split": "train",
    }.items():
        result[name] = value
    result["timestamp_s"] = np.arange(length) / 40
    return result


def state(method, taps=None, delta=False, arithmetic="weights_bias"):
    if method == "identity":
        return _state("pooled", method, kind="identity")
    if method == "bias_mean":
        return _state("pooled", method, kind="bias", intercept=np.arange(6) / 10)
    features = 6 * ((taps or 1) + int(delta))
    return _state(
        "pooled",
        method,
        weights=np.random.default_rng(7).normal(size=(features, 6)),
        intercept=np.arange(6) / 10,
        taps=taps,
        delta=delta,
        arithmetic=arithmetic,
    )


@pytest.mark.parametrize(
    "method,taps,delta,arithmetic",
    [
        ("identity", None, False, "weights_bias"),
        ("bias_mean", None, False, "weights_bias"),
        ("affine", None, False, "design_matmul"),
        ("ridge_tuned", None, False, "weights_bias"),
        ("fir_tuned", 5, False, "weights_bias"),
        ("arx_tuned", 3, True, "weights_bias"),
    ],
)
def test_full_normalized_kernel_matches_predictor_start_and_real_history(
    method, taps, delta, arithmetic
):
    tool, data = module(), frame()
    fitted = state(method, taps, delta, arithmetic)
    compiled = tool.compile_state(fitted)
    expected = predict_state(fitted, data)
    # Start mode uses actual stream-first padding/delta=0; interior uses actual prior LOW.
    first = data[LOW].to_numpy(float)[None, :80]
    np.testing.assert_array_equal(tool.predict_blocks(compiled, first, 0), expected[None, :80])
    history = max((taps or 1) - 1, int(delta))
    block = data[LOW].to_numpy(float)[None, 120 - history : 200]
    np.testing.assert_allclose(
        tool.predict_blocks(compiled, block, history),
        expected[None, 120:200],
        rtol=1e-12,
        atol=1e-12,
    )
    if delta:
        reset = tool.predict_blocks(compiled, data[LOW].to_numpy(float)[None, 120:200], 0)
        assert not np.allclose(reset[:, 0], expected[None, 120])


@pytest.mark.parametrize(
    "method,taps,delta,count",
    [
        ("identity", None, False, 0),
        ("bias_mean", None, False, 6),
        ("affine", None, False, 42),
        ("ridge_affine", None, False, 42),
        ("learned_fir_causal", 9, False, 330),
        ("arx_linear", 12, True, 474),
        ("fir_tuned", 17, False, 618),
        ("arx_tuned", 24, True, 906),
    ],
)
def test_actual_array_inventory(method, taps, delta, count):
    row = module().coefficient_inventory(state(method, taps, delta))
    assert row["coefficient_count"] == count
    assert row["coefficient_bytes"] == 8 * count
    assert row["normalizer_count"] == row["normalizer_bytes"] == 0
    assert row["coefficient_interpretation"] == "stored_learned_float64_scalars_not_statistical_DoF"


def test_bank_totals_deduplicate_references_but_count_distinct_pairs_with_identical_numbers():
    tool = module()
    fitted = state("affine")
    records = [
        {
            "protocol": "pooled",
            "canonical_method": "affine",
            "information_scope": "known_device_bank",
            "variant": "bank",
            "pair_identity": {"dataset_id": "SYNTHETIC", "low_sensor": pair, "high_sensor": "h"},
            "state": fitted,
        }
        for pair in ("a", "a", "b")
    ]
    inventory, totals = tool.parameter_tables(records)
    assert len(inventory) == 2
    assert totals[0]["pair_count"] == 2
    assert totals[0]["bank_coefficient_count"] == 84
    assert totals[0]["numeric_unique_coefficient_count"] == 42
    assert totals[0]["reference_count"] == 3


def test_seeded_pool_retains_real_same_stream_history_and_physical_sha():
    tool, data = module(), frame()
    one, rows = tool.make_window_pool(data, 80, 4, 128, 20261024)
    two, again = tool.make_window_pool(data, 80, 4, 128, 20261024)
    np.testing.assert_array_equal(one, two)
    assert rows == again and rows[0]["physical_row_sha256"]
    for block, row in zip(one, rows, strict=True):
        np.testing.assert_array_equal(
            block, data[LOW].to_numpy(float)[row["history_start_index"] : row["end_index"] + 1]
        )
    bad = data.copy()
    bad.loc[8, "timestamp_s"] += 0.001
    with pytest.raises(ValueError, match="40Hz"):
        tool.make_window_pool(bad, 80, 4, 1, 1)


def test_workload_times_actual_full_pair_loop_and_independent_throughput():
    tool = module()
    states = [tool.compile_state(state("affine")) for _ in range(3)]
    pools = [np.ones((2, 80, 6)) * i for i in range(3)]
    calls = []

    def predictor(fitted, values, history):
        calls.append((id(fitted), len(values)))
        return values[:, history:]

    result = tool.time_workload(
        states, pools, history=0, warmup=1, iterations=2, repeats=3, predictor=predictor
    )
    assert len(calls) == 3 * (1 + 2 * 2 * 3)
    assert result["pair_count"] == 3 and result["windows_per_call"] == 6
    assert len(result["raw_wall_ms"]) == 6
    assert len(result["raw_throughput_loop_seconds"]) == 3
    assert result["loop_throughput_windows_per_second"] == pytest.approx(
        6 * 2 * 3 / sum(result["raw_throughput_loop_seconds"])
    )
    assert "lag/delta feature construction" in tool.TIMING_SCOPE


def test_actual_blas_threadpool_is_verified():
    tool = module()
    with tool.single_thread_runtime() as runtime:
        assert runtime["threadpoolctl_version"] == "3.6.0"
        assert runtime["threadpools"]
        assert all(row["num_threads"] == 1 for row in runtime["threadpools"])


def test_normalized_fixed_bank_kernel_matches_frozen_predictor_on_synthetic_stream():
    from reviewer_shared_classical import normalize_bank_pair
    from virtual_imu.eval.formal_evaluation import HIGH, fit_pair_baselines, predict_pair_baselines

    tool, data = module(), frame()
    data[HIGH] = data[LOW].to_numpy(float) * 1.01 + 0.03
    fixed = fit_pair_baselines(data, fir_taps=3, alpha=0.01)
    states = normalize_bank_pair("pooled", {"fixed": fixed})
    original = predict_pair_baselines(fixed, data)
    for method, fitted in states.items():
        compiled = tool.compile_state(fitted)
        np.testing.assert_allclose(
            tool.predict_blocks(compiled, data[LOW].to_numpy(float)[None, :80], 0)[0],
            original[method][:80],
            rtol=1e-12,
            atol=1e-12,
        )
        history = max(
            (fitted["feature_spec"]["taps"] or 1) - 1, int(fitted["feature_spec"]["include_delta"])
        )
        np.testing.assert_allclose(
            tool.predict_blocks(
                compiled, data[LOW].to_numpy(float)[None, 120 - history : 200], history
            )[0],
            original[method][120:200],
            rtol=1e-12,
            atol=1e-12,
        )


def test_public_cli_has_no_synthetic_or_subset_bypass():
    args = module().parser()
    options = {option for action in args._actions for option in action.option_strings}
    assert not options & {"--synthetic", "--subset", "--allow-subset", "--smoke"}
