"""Classical timing preserves frozen fits, true history and immutable records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.formal_evaluation import HIGH, IDENTITY, LOW, fit_pair_baselines
from virtual_imu.eval.formal_navigation import (
    fit_navigation_baselines,
    predict_navigation_baselines,
)
from virtual_imu.eval.reviewer_evaluation import _predict_paired_classical

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_reviewer_classical_latency.py"


def module():
    spec = importlib.util.spec_from_file_location("reviewer_classical_latency", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def frame(seed=1, length=400):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(length, 6))
    result = pd.DataFrame(values, columns=LOW)
    result[HIGH] = values * 1.1 + 0.2
    result[IDENTITY] = ["toy", "low", "high"]
    result["sequence_id"] = "train_segment"
    result["timestamp_s"] = np.arange(length) / 40
    result["split"] = "train"
    return result


def paired_state():
    train = frame()
    fixed = fit_pair_baselines(train, fir_taps=3)
    plain = {"weights": np.eye(6).tolist(), "bias": [0.1] * 6}
    tapped = {
        "weights": np.random.default_rng(2).normal(size=(18, 6)).tolist(),
        "bias": [0.2] * 6,
        "taps": 3,
        "include_delta": False,
    }
    delta = {**tapped, "weights": np.ones((24, 6)).tolist(), "include_delta": True}
    return {
        "fit_split": "train",
        "selection_split": "validation",
        "pairs": [
            {
                "identity": fixed["identity"],
                "fixed": fixed,
                "ridge_tuned": plain,
                "fir_tuned": tapped,
                "arx_tuned": delta,
            }
        ],
    }


def test_inventory_counts_only_actual_fitted_scalars_and_separates_normalizers():
    tool = module()
    specs = tool.method_specs("paired", paired_state())
    assert len(specs) == 9
    by_name = {spec["method"]: spec for spec in specs}
    assert tool.coefficient_inventory(by_name["identity"])["fitted_scalar_count"] == 0
    bias = tool.coefficient_inventory(by_name["bias_mean"])
    assert (bias["coefficient_count"], bias["bias_count"], bias["fitted_scalar_bytes"]) == (
        0,
        6,
        48,
    )
    affine = tool.coefficient_inventory(by_name["affine"])
    assert (affine["coefficient_count"], affine["bias_count"]) == (36, 6)
    x = np.random.default_rng(4).normal(size=(30, 80, 6))
    nav = fit_navigation_baselines(x, x.mean(1)[:, :3])
    specs = tool.method_specs("navigation", {"fixed": nav, "ridge_tuned": {"state": nav}})
    ridge = next(spec for spec in specs if spec["method"] == "ridge")
    count = tool.coefficient_inventory(ridge)
    assert (count["coefficient_count"], count["bias_count"], count["normalizer_scalar_count"]) == (
        54,
        3,
        36,
    )


def test_paired_forward_includes_true_history_delta_and_matches_frozen_predictor():
    tool = module()
    state = paired_state()
    specs = tool.method_specs("paired", state)
    block = np.random.default_rng(6).normal(size=(2, 92, 6))
    pieces = []
    for index, values in enumerate(block):
        part = pd.DataFrame(values, columns=LOW)
        part["sequence_id"] = str(index)
        part["timestamp_s"] = np.arange(92) / 40
        pieces.append(part)
    expected = _predict_paired_classical(state["pairs"][0], pd.concat(pieces, ignore_index=True))
    for spec in specs:
        actual = tool.predict_block(spec, block, history=12)
        reference = expected[spec["method"]].reshape(2, 92, 6)[:, 12:]
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    fir = next(spec for spec in specs if spec["method"] == "fir_tuned")
    changed = block.copy()
    changed[:, 11] += 20
    assert not np.array_equal(
        tool.predict_block(fir, changed, 12), tool.predict_block(fir, block, 12)
    )


def test_navigation_summary_forward_matches_frozen_baselines():
    tool = module()
    x = np.random.default_rng(9).normal(size=(25, 80, 6))
    fitted = fit_navigation_baselines(x, x.mean(1)[:, :3])
    specs = tool.method_specs("navigation", {"fixed": fitted, "ridge_tuned": {"state": fitted}})
    expected = predict_navigation_baselines(fitted, x[:2])
    for spec in specs:
        actual = tool.predict_block(spec, x[:2], history=0)
        np.testing.assert_allclose(
            actual, expected["ridge" if spec["method"] == "ridge_tuned" else spec["method"]]
        )
    bad = dict(specs[-1], weights=[[float("nan")]])
    with pytest.raises(ValueError, match=r"finite|shape"):
        tool.coefficient_inventory(bad)


def test_fixed_seed_train_windows_have_same_segment_history_and_reject_gaps_or_test():
    tool = module()
    training = frame(length=500)
    blocks, metadata = tool.make_window_pool(training, LOW, 80, 12, 4, 20261024)
    again, same = tool.make_window_pool(training, LOW, 80, 12, 4, 20261024)
    np.testing.assert_array_equal(blocks, again)
    assert metadata == same
    for values, record in zip(blocks, metadata, strict=True):
        assert record["start_index"] >= 12
        np.testing.assert_array_equal(
            values, training[LOW].iloc[record["start_index"] - 12 : record["start_index"] + 80]
        )
    training.loc[60, "timestamp_s"] += 0.1
    with pytest.raises(ValueError, match="contiguous"):
        tool.make_window_pool(training, LOW, 80, 12, 4, 1)
    training = frame()
    training["split"] = "test"
    with pytest.raises(ValueError, match="train"):
        tool.make_window_pool(training, LOW, 80, 12, 4, 1)


def test_incomplete_gate_prevents_loading_training_payloads(tmp_path, monkeypatch):
    tool = module()
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "pipeline.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr(
        tool, "load_train", lambda *_: pytest.fail("Training payload read before gate")
    )
    with pytest.raises(ValueError, match=r"93|558"):
        tool.main(["--pipeline-root", str(pipeline), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_actual_cpu_timing_and_resume_keep_partial_and_reject_hash_mutation(tmp_path):
    tool = module()
    spec = {"method": "identity", "kind": "identity", "task": "paired", "pair_identity": {}}
    values = np.random.default_rng(1).normal(size=(1, 80, 6))
    contract = {"state_sha256": "frozen", "input_sha256": tool.array_digest(values)}
    partial = tmp_path / "partial" / "interrupted"
    partial.mkdir(parents=True)
    (partial / "kept.txt").write_text("preserve")
    row = tool.persist_record(
        tmp_path, "one", contract, spec, values, 0, warmup=1, iterations=2, repeats=2, resume=False
    )
    assert row["wall"]["count"] == 4
    assert row["output_shape"] == [1, 80, 6]
    assert row["loop_throughput_windows_per_second"] > 0
    assert row["fitted_scalar_count"] == 0
    record = tmp_path / "records/one/record.json"
    stamp = record.stat().st_mtime_ns
    tool.persist_record(
        tmp_path, "one", contract, spec, values, 0, warmup=1, iterations=2, repeats=2, resume=True
    )
    assert record.stat().st_mtime_ns == stamp
    assert (partial / "kept.txt").read_text() == "preserve"
    with pytest.raises(ValueError, match="contract"):
        tool.persist_record(
            tmp_path,
            "one",
            {**contract, "state_sha256": "changed"},
            spec,
            values,
            0,
            warmup=1,
            iterations=2,
            repeats=2,
            resume=True,
        )
    record.write_text(record.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        tool.persist_record(
            tmp_path,
            "one",
            contract,
            spec,
            values,
            0,
            warmup=1,
            iterations=2,
            repeats=2,
            resume=True,
        )


def synthetic_full_pipeline(tmp_path, tool):
    root = tmp_path / "pipeline"
    root.mkdir()
    protocols = [{"id": f"paired_{i}", "task": "paired"} for i in range(6)] + [
        {"id": f"nav_{i}", "task": "navigation"} for i in range(3)
    ]
    state = {
        "status": "complete",
        "smoke": False,
        "cells": {},
        "jobs": {},
        "contract": {"config": {"protocols": protocols}},
        "final_seeds": [1, 2, 3],
        "final_variants": ["default", "tuned"],
    }
    state["contract"]["source_sha256"] = {
        str(path): tool.sha(path) for path in (tool.PROJECT / "src/virtual_imu").rglob("*.py")
    }
    paired = paired_state()
    x = np.random.default_rng(3).normal(size=(25, 80, 6))
    navfit = fit_navigation_baselines(x, x.mean(1)[:, :3])
    nav = {
        "fit_split": "train",
        "selection_split": "validation",
        "fixed": navfit,
        "ridge_tuned": {"state": navfit},
    }
    for protocol in protocols:
        protocol_id, task = protocol["id"], protocol["task"]
        fitted = paired if task == "paired" else nav
        training_file = root / f"{protocol_id}_train.bin"
        training_file.write_bytes(b"synthetic train only")
        split_manifest = root / f"{protocol_id}_split.json"
        split_manifest.write_text(
            json.dumps(
                {
                    "splits": {
                        "train": [
                            {
                                "path": str(training_file),
                                "sha256": tool.sha(training_file),
                                "sequence_id": "train",
                                "original_sequence_id": "train_original",
                            }
                        ]
                    }
                }
            )
        )
        contract = {
            "protocol_id": protocol_id,
            "task": task,
            "implementation": {},
            "manifests": {
                "split_manifest": {"path": str(split_manifest), "sha256": tool.sha(split_manifest)}
            },
            "train_validation": {"rows_fingerprint": protocol_id},
        }
        cache = root / "classical" / protocol_id / "cache.json"
        cache.parent.mkdir(parents=True)
        cache.write_text(
            json.dumps(
                {"contract": contract, "states": fitted, "state_sha256": tool._digest(fitted)}
            )
        )
        for family in range(10 if task == "paired" else 11):
            cell = f"{protocol_id}/model_{family}"
            selected = {"trial": {"parameters": {"width": 4}}}
            selection = root / "selections" / f"{cell.replace('/', '__')}.json"
            selection.parent.mkdir(exist_ok=True)
            selection.write_text(json.dumps(selected))
            state["cells"][cell] = {"selected": selected, "selection_sha256": tool.sha(selection)}
            for variant in ("default", "tuned"):
                for seed in (1, 2, 3):
                    identifier = f"{cell.replace('/', '__')}__{variant}__{seed}"
                    run = root / "experiments" / identifier
                    (run / "evaluation").mkdir(parents=True)
                    config = {
                        "seed": seed,
                        "reviewer_protocol": {
                            "protocol_id": protocol_id,
                            "phase": f"final_{variant}",
                            "variant": variant,
                            "model_family": f"model_{family}",
                        },
                        "model": {"in_channels": 6, "out_channels": 6 if task == "paired" else 3},
                        "data": {"window_size": 80, "split_manifest": str(split_manifest)},
                    }
                    config_file = run / "config.yaml"
                    config_file.write_text(json.dumps(config))
                    state_file = run / "evaluation/baseline_train_state.json"
                    state_file.write_text(json.dumps(fitted))
                    metrics = run / "evaluation/metrics.json"
                    metrics.write_text(
                        json.dumps(
                            {
                                "classical_cache": {
                                    "path": str(cache),
                                    "sha256": tool.sha(cache),
                                    "contract_sha256": tool._digest(contract),
                                }
                            }
                        )
                    )
                    summary = root / "evaluations" / f"{identifier}.json"
                    summary.parent.mkdir(exist_ok=True)
                    summary.write_text("{}")
                    state["jobs"][identifier] = {
                        "id": identifier,
                        "cell": cell,
                        "phase": f"final_{variant}",
                        "seed": seed,
                        "run_dir": str(run),
                        "status": "complete",
                        "evaluation_status": "complete",
                        "run_config_sha256": tool.sha(config_file),
                        "evaluation_sha256": tool.sha(summary),
                        "evaluation_artifact_sha256": {
                            str(path.resolve()): tool.sha(path)
                            for path in (metrics, state_file, cache)
                        },
                    }
    return root, state


def test_full_93_558_gate_deduplicates_only_after_every_final_state_hash_check(
    tmp_path, monkeypatch
):
    tool = module()
    root, state = synthetic_full_pipeline(tmp_path, tool)
    checked = []
    monkeypatch.setitem(
        tool.verify_pipeline.__globals__, "_verify_selection", lambda cell, *_: checked.append(cell)
    )
    sources, mapping = tool.collect_sources(state, root)
    assert len(checked) == 93
    assert len(sources) == 9
    assert len(mapping) == 558
    assert len({row["job_id"] for row in mapping}) == 558
    assert all(
        sum(row["source_id"] == source["id"] for row in mapping)
        == (60 if source["task"] == "paired" else 66)
        for source in sources
    )
    training_file = Path(sources[0]["training_inputs"][0]["path"])
    original = training_file.read_bytes()
    training_file.write_bytes(b"changed train input")
    with pytest.raises(ValueError, match="TRAIN file checksum"):
        tool.collect_sources(state, root)
    training_file.write_bytes(original)
    # Even a duplicate slot must verify its own saved state, not blindly skip it.
    final = list(state["jobs"].values())[-1]
    path = Path(final["run_dir"]) / "evaluation/baseline_train_state.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="checksum"):
        tool.collect_sources(state, root)
    final["evaluation_status"] = "running"
    with pytest.raises(ValueError, match=r"93|558"):
        tool.collect_sources(state, root)


def test_csv_partial_resume_rejects_same_size_corruption(tmp_path):
    tool = module()
    spec = {"method": "identity", "kind": "identity", "task": "paired", "pair_identity": {}}
    values = np.ones((1, 80, 6))
    contract = {
        "protocol_id": "toy",
        "source_id": "toy",
        "state_sha256": "state",
        "input_sha256": "input",
    }
    row = tool.persist_record(
        tmp_path, "one", contract, spec, values, 0, warmup=1, iterations=2, repeats=1, resume=False
    )
    tool._csvs(tmp_path, [row], [{"job_id": "x"}])
    tool._csvs(tmp_path, [row], [{"job_id": "x"}])
    path = tmp_path / "classical_latency.csv"
    path.write_text(path.read_text().replace("identity", "corrupt!"))
    with pytest.raises(ValueError, match="content"):
        tool._csvs(tmp_path, [row], [{"job_id": "x"}])


def test_first_run_source_freeze_and_complete_roster_are_portable_and_fail_closed(
    tmp_path, monkeypatch
):
    tool = module()
    project = tmp_path / "frozen"
    package = project / "src/virtual_imu"
    package.mkdir(parents=True)
    source = package / "model.py"
    source.write_text("original source")
    frozen = {
        "contract": {"source_sha256": {r"C:\\server\\src\\virtual_imu\\model.py": tool.sha(source)}}
    }
    # Canonical source matching also accepts a Windows source receipt on this host.
    # Use ordinary separators rather than doubled literal backslashes in metadata.
    key = next(iter(frozen["contract"]["source_sha256"]))
    frozen["contract"]["source_sha256"] = {key.replace("\\\\", "\\"): tool.sha(source)}
    result = tool._processing_sources(project, frozen)
    assert result["matched_trained_source_sha256"] == {"src/virtual_imu/model.py": tool.sha(source)}
    extra = package / "extra.py"
    extra.write_text("unfrozen")
    with pytest.raises(ValueError, match="roster"):
        tool._processing_sources(project, frozen)
    extra.unlink()
    source.write_text("changed source")
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "pipeline.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(tool, "PROJECT", project)
    monkeypatch.setattr(tool, "verify_pipeline", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        tool, "load_train", lambda *_: pytest.fail("Payload read despite changed source")
    )
    with pytest.raises(ValueError, match="Executed source differs"):
        tool.main(["--pipeline-root", str(pipeline), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_main_synthetic_navigation_single_thread_and_interrupted_summary_resume(
    tmp_path, monkeypatch
):
    tool = module()
    rng = np.random.default_rng(20261024)
    values = rng.normal(size=(800, 6))
    training = pd.DataFrame(values, columns=tool.SMARTPNT_GT_IMU_COLUMNS)
    training["sequence_id"] = "train_segment"
    training["original_sequence_id"] = "train_recording"
    training["timestamp_s"] = np.arange(len(training)) / 40
    training["split"] = "train"
    windows = np.stack([values[start : start + 80] for start in range(0, len(values) - 79, 40)])
    fitted = fit_navigation_baselines(windows, windows.mean(1)[:, :3])
    source = {
        "id": "synthetic_nav",
        "protocol_id": "nav",
        "task": "navigation",
        "states": {"fixed": fitted, "ridge_tuned": {"state": fitted}},
        "cache_sha256": "toy",
        "training_inputs": [{"path": "toy_train", "sha256": "frozen"}],
    }
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    state = {
        "contract": {
            "source_sha256": {
                str(path): tool.sha(path)
                for path in (tool.PROJECT / "src/virtual_imu").rglob("*.py")
            }
        }
    }
    (pipeline / "pipeline.json").write_text(json.dumps(state))
    monkeypatch.setattr(tool, "collect_sources", lambda *_: ([source], [{"job_id": "synthetic"}]))
    monkeypatch.setattr(tool, "load_train", lambda *_: (training, tool.SMARTPNT_GT_IMU_COLUMNS))
    out = tmp_path / "result"
    argv = ["--pipeline-root", str(pipeline), "--out", str(out)]
    assert tool.main(argv) == 0
    report = json.loads((out / "latency.json").read_text(encoding="utf-8"))
    assert len(report["records"]) == 8
    assert report["environment"]["threadpoolctl_version"] == "3.6.0"
    assert all(pool["num_threads"] == 1 for pool in report["environment"]["native_thread_pools"])
    record = next((out / "records").glob("*/record.json"))
    stamp = record.stat().st_mtime_ns
    (out / "artifact_manifest.json").unlink()  # interrupted after valid summary writes
    interrupted = out / "partial/unrelated_interrupted_state.txt"
    interrupted.parent.mkdir(exist_ok=True)
    interrupted.write_text("retained")
    assert tool.main([*argv, "--resume"]) == 0
    assert record.stat().st_mtime_ns == stamp
    assert interrupted.read_text() == "retained"
    assert tool.main([*argv, "--resume"]) == 0
    assert record.stat().st_mtime_ns == stamp
    pool = next((out / "input_pools").glob("*/physical_low_history_blocks.npy"))
    pool.write_bytes(pool.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        tool.main([*argv, "--resume"])


def test_aux_script_bundle_uses_actual_imported_frozen_source_project(tmp_path):
    auxiliary = tmp_path / "aux2/scripts/benchmark_reviewer_classical_latency.py"
    auxiliary.parent.mkdir(parents=True)
    auxiliary.write_bytes(SCRIPT.read_bytes())
    spec = importlib.util.spec_from_file_location("isolated_classical_aux", auxiliary)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    assert value.SCRIPT_ROOT == tmp_path / "aux2"
    assert value.PROJECT == SCRIPT.parents[1]
    assert value.PROJECT != value.SCRIPT_ROOT
