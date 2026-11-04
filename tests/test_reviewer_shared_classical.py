"""Synthetic-only shared-control contracts; never open the active corpus."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import evaluate_reviewer_shared_classical as cli  # noqa: E402
import reviewer_shared_classical as shared  # noqa: E402


def frame(role="train", *, n=320, seed=3, dataset="fixture", sequence=None):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n, 6))
    result = pd.DataFrame(values, columns=shared.LOW)
    result[shared.HIGH] = values @ (np.eye(6) * 1.1) + np.arange(6) / 20
    result["dataset_id"] = dataset
    result["low_sensor"], result["high_sensor"] = "low", "high"
    result["sequence_id"] = sequence or dataset + "_" + role
    result["original_sequence_id"] = role
    result["timestamp_s"] = np.arange(n) / 40
    result["split"] = role
    return result


@pytest.fixture(scope="module")
def fitted():
    return shared.fit_protocol(
        "pooled",
        frame(),
        frame("validation", seed=4),
        search={"ridge_alpha": [1e-6, 0.1], "fir_taps": [1, 3], "arx_taps": [3]},
        synthetic=True,
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_sequence",
        "empty_sequence",
        "collision",
        "gap",
        "duplicate",
        "nonfinite",
        "recording_collision",
    ],
)
def test_physical_boundary_rejections(corruption):
    data = frame()
    if corruption == "missing_sequence":
        data = data.drop(columns="sequence_id")
    elif corruption == "empty_sequence":
        data.loc[0, "sequence_id"] = " "
    elif corruption == "collision":
        data.loc[0, "low_sensor"] = "other"
    elif corruption == "recording_collision":
        data.loc[0, "original_sequence_id"] = "other"
    elif corruption == "gap":
        data.loc[160:, "timestamp_s"] += 0.025
    elif corruption == "duplicate":
        data.loc[1, "timestamp_s"] = 0
    else:
        data.loc[1, shared.LOW[0]] = np.nan
    with pytest.raises(ValueError):
        shared.validate_streams(data)


def test_authoritative_manifest_identity_rejects():
    data = frame()
    record = {
        name: data[name].iloc[0]
        for name in [*shared.IDENTITY, "sequence_id", "original_sequence_id"]
    }
    record["original_sequence_id"] = "wrong_recording"
    with pytest.raises(ValueError, match="Manifest physical identity"):
        shared.validate_streams(data, [record])


@pytest.mark.parametrize(("n", "covered"), [(320, 320), (359, 320), (360, 360), (399, 360)])
def test_train_n_and_evaluation_c_preserve_original_order(n, covered):
    data = frame("validation", n=n).sample(frac=1, random_state=10).reset_index(drop=True)
    selected, registry = shared.physical_support(data, "validation")
    expected = data[data.timestamp_s < covered / 40].reset_index(drop=True)
    pd.testing.assert_frame_equal(selected, expected)
    assert registry[0]["eligible_rows"] == covered
    data["split"] = "train"
    train, _ = shared.physical_support(data, "train")
    pd.testing.assert_frame_equal(train, data)


def test_short_stream_explicit_skip_and_no_concatenation():
    data = pd.concat([frame(n=319), frame(n=321, sequence="long")], ignore_index=True)
    result, registry = shared.physical_support(data, "train")
    assert len(result) == 321
    assert registry[0]["skip_reason"] == "segment_shorter_than_320"


def test_stream_local_lags_and_delta_are_independent():
    left, right = frame(sequence="a"), frame(sequence="b")
    left[shared.LOW], right[shared.LOW] = 1.0, 100.0
    data = pd.concat([left, right], ignore_index=True)
    features = shared._sequence_lag_features(data, 3, include_delta=True)
    np.testing.assert_array_equal(features[0, :18], np.ones(18))
    np.testing.assert_array_equal(features[320, :18], np.ones(18) * 100)
    np.testing.assert_array_equal(features[[0, 320], 18:], np.zeros((2, 6)))


def test_predictor_cannot_route_device_or_read_high(fitted):
    target = frame("test", seed=20)
    changed = target.copy()
    changed[shared.IDENTITY] = ["unknown_dataset", "unknown_LOW", "unknown_HIGH"]
    changed[shared.HIGH] = np.nan
    for state in fitted["states"].values():
        np.testing.assert_array_equal(
            shared.predict_state(state, target), shared.predict_state(state, changed)
        )
    assert fitted["fit_role"] == "train" and fitted["selection_role"] == "validation"
    assert "test" not in __import__("inspect").signature(shared.fit_protocol).parameters


def test_macro_objective_does_not_reward_large_dataset():
    a, b = frame("validation", n=3200, dataset="large"), frame("validation", n=320, dataset="small")
    data = pd.concat([a, b], ignore_index=True)
    data[shared.HIGH] = 0.0
    data[shared.LOW] = 1.0
    candidate_a = np.zeros((len(data), 6))
    candidate_a[len(a) :] = 2
    candidate_b = np.ones((len(data), 6)) * 0.9
    assert np.mean(candidate_a**2) < np.mean(candidate_b**2)
    assert (
        shared.macro_breakdown(candidate_b, data)["score"]
        < shared.macro_breakdown(candidate_a, data)["score"]
    )
    source = SCRIPTS / "evaluate_reviewer_transfer.py"
    spec = importlib.util.spec_from_file_location("shared_score_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert shared.macro_breakdown(candidate_a, data)["score"] == reference._macro_score(
        candidate_a, data
    )


def test_candidates_and_counts_are_actual_and_identity_n1(fitted):
    assert len(fitted["states"]) == 9
    assert len(fitted["candidate_history"]) == 8
    assert fitted["compute_counts"]["total_linear_solver_calls"] == 12
    assert fitted["states"]["identity"]["coefficient_count"] == 0
    assert fitted["states"]["bias_mean"]["coefficient_count"] == 6
    assert fitted["states"]["arx_linear"]["coefficient_count"] == 474
    assert all(
        row["finite"] and row["error"] is None and row["elapsed_s"] >= 0
        for row in fitted["candidate_history"]
    )
    assert [row["order"] for row in fitted["candidate_history"]] == list(range(8))
    for method, selected in fitted["selections"].items():
        candidates = [
            row for row in fitted["candidate_history"] if row["canonical_method"] == method
        ]
        best = min(candidates, key=lambda row: row["validation_score"])
        assert selected["alpha"] == best["alpha"] and selected["taps"] == best["taps"]


def test_per_protocol_state_and_source_contract_isolation(fitted):
    changed = copy.deepcopy(fitted["states"]["affine"])
    changed["protocol"] = "penn_only"
    assert shared.digest(changed) != shared.digest(fitted["states"]["affine"])
    assert changed["numeric_sha256"] == fitted["states"]["affine"]["numeric_sha256"]
    assert shared.frame_sha(frame()) != shared.frame_sha(frame(seed=21))
    changed["weights"][0][0] += 1
    with pytest.raises(ValueError, match="fields differ"):
        shared.verify_state(changed)


def test_single_pair_alias_is_honest_and_requires_support(fitted):
    state = fitted["states"]["ridge_tuned"]
    proof = {
        key: True
        for key in (
            "same_protocol",
            "train_rows_exact",
            "validation_rows_exact",
            "grid_order_exact",
            "tie_rule_exact",
            "bank_own_selection_valid",
        )
    }
    exact = shared.alias_audit(
        state, copy.deepcopy(state), frame("validation", seed=4), contract_proof=proof
    )
    assert exact["alias_status"] == "exact_alias" and exact["n"] == 1 and exact["sd"] is None
    other = shared._state(
        "pooled", "ridge_tuned", weights=state["weights"], intercept=state["intercept"], alpha=10
    )
    assert (
        shared.alias_audit(state, other, frame("validation", seed=4), contract_proof=proof)[
            "alias_status"
        ]
        == "not_alias"
    )
    with pytest.raises(ValueError, match="proof"):
        shared.alias_audit(state, state, frame("validation"), contract_proof={})


def test_failed_candidate_is_retained_and_aborts(monkeypatch):
    observed = []
    calls = 0
    original = np.linalg.solve

    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 5:  # Three fixed ridge solves + first grid solve; fail next.
            raise np.linalg.LinAlgError("fixture solver failure")
        return original(*args)

    monkeypatch.setattr(np.linalg, "solve", failing)
    with pytest.raises(np.linalg.LinAlgError, match="fixture"):
        shared.fit_protocol(
            "pooled",
            frame(),
            frame("validation"),
            search={"ridge_alpha": [1e-6, 0.1], "fir_taps": [1], "arx_taps": [3]},
            synthetic=True,
            observer=observed.append,
        )
    assert observed[-1]["finite"] is False and "fixture solver failure" in observed[-1]["error"]


def test_immutable_manifest_detects_same_length_mutation(tmp_path):
    staging = tmp_path / "attempt"
    staging.mkdir()
    summary = shared.finish_stage(
        staging,
        tmp_path / "fit_select",
        {"stage": "fit_select", "contract_sha256": "fixture"},
        {"control_registry": pd.DataFrame({"x": [1]})},
    )
    assert summary["tables"]["control_registry"]["path"] == "fit_select/control_registry.csv"
    manifest = shared.verify_artifact_manifest(tmp_path / "fit_select", contract_sha="fixture")
    assert manifest["root_kind"] == "aux3_run"
    (tmp_path / "fit_select/control_registry.csv").write_text("x\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        shared.verify_artifact_manifest(tmp_path / "fit_select", contract_sha="fixture")


def test_no_test_audit_before_global_selection_freeze(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "full_original_audit", lambda *args: called.append("TEST audit"))
    with pytest.raises(FileNotFoundError):
        cli.signal_evaluation(tmp_path, {}, {}, {}, [], {}, tmp_path, tmp_path)
    assert called == []


def test_formal_cli_rejects_incomplete_package_without_output(tmp_path):
    config = yaml.safe_load(
        (SCRIPTS.parent / "configs/reviewer_shared_classical.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    root = tmp_path / "core"
    root.mkdir()
    (root / "pipeline.json").write_text(
        json.dumps({"status": "complete", "smoke": False, "cells": {}, "jobs": {}}),
        encoding="utf-8",
    )
    output = tmp_path / "aux3"
    args = SimpleNamespace(
        project_root=tmp_path,
        pipeline_root=root,
        aux1_state=tmp_path / "aux1.json",
        preceding_state=tmp_path / "aux2.json",
        out=output,
        config=config_path,
        stage="fit_select",
        resume=False,
    )
    with pytest.raises(ValueError, match="93/558"):
        cli.run(args)
    assert not output.exists()


def test_normalized_predictions_equal_frozen_shared_source(fitted):
    source = SCRIPTS / "evaluate_reviewer_transfer.py"
    spec = importlib.util.spec_from_file_location("shared_predict_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    train, val, target = frame(), frame("validation", seed=4), frame("test", seed=8)
    original = reference.fit_source_classical(
        train, val, {"ridge_alpha": [1e-6, 0.1], "fir_taps": [1, 3], "arx_taps": [3]}
    )
    predictions = reference.predict_source_classical(original, target)
    for method, state in fitted["states"].items():
        key = (
            "identity"
            if method == "identity"
            else "source_" + method + ("" if method.endswith("tuned") else "_fixed")
        )
        np.testing.assert_allclose(
            shared.predict_state(state, target), predictions[key], rtol=1e-12, atol=1e-12
        )


def test_frozen_bundle_gate_hashes_source_only_and_rejects_mutation(tmp_path):
    source = tmp_path / "fixture.py"
    source.write_bytes(b"x=1\n")
    source_hashes = {"fixture.py": shared.sha(source)}
    fingerprint = (
        __import__("hashlib")
        .sha256(
            (
                json.dumps(source_hashes, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode()
        )
        .hexdigest()
    )
    manifest = {
        "entries": [
            {
                "path": "fixture.py",
                "kind": "source",
                "size_bytes": 4,
                "sha256": source_hashes["fixture.py"],
            },
            {"path": "forbidden_TEST.parquet", "kind": "data", "size_bytes": 4, "sha256": "a" * 64},
        ],
        "source_fingerprint": fingerprint,
        "dataset_manifest_sha256": "fixture_dataset",
    }
    shared.write_once(tmp_path / "bundle_manifest.json", manifest)
    config = {
        "expected_source_fingerprint": fingerprint,
        "expected_dataset_manifest_sha256": "fixture_dataset",
    }
    assert cli.verify_frozen_bundle_sources(tmp_path, config)["source_file_count"] == 1
    assert not (tmp_path / "forbidden_TEST.parquet").exists()
    source.write_bytes(b"x=2\n")
    with pytest.raises(ValueError, match="source checksum"):
        cli.verify_frozen_bundle_sources(tmp_path, config)


def test_completed_cache_rejects_changed_source_or_split_contract(tmp_path):
    staging = tmp_path / "attempt"
    staging.mkdir()
    original = {"protocol": "pooled", "source_sha256": "source_a", "split_sha256": "split_a"}
    shared.finish_stage(
        staging,
        tmp_path / "fit_select",
        {"stage": "fit_select", "contract_sha256": shared.digest(original)},
        {"control_registry": pd.DataFrame({"method": ["identity"]})},
    )
    for changed in (
        {**original, "source_sha256": "source_b"},
        {**original, "split_sha256": "split_b"},
        {**original, "protocol": "penn_only"},
    ):
        with pytest.raises(ValueError, match="contract"):
            shared.verify_artifact_manifest(
                tmp_path / "fit_select", contract_sha=shared.digest(changed)
            )


def test_all_six_states_freeze_and_formal_rejects_synthetic(tmp_path, fitted):
    staging = tmp_path / "attempt"
    staging.mkdir()
    descriptors = {}
    for protocol in shared.PROTOCOLS:
        selected = copy.deepcopy(fitted)
        selected["protocol"] = protocol
        for state in selected["states"].values():
            state["protocol"] = protocol
        selected["state_sha256"] = {
            name: shared.digest(state) for name, state in selected["states"].items()
        }
        for name, state in selected["states"].items():
            shared.write_once(staging / protocol / f"{name}.state.json", state)
        path = staging / protocol / "selection.json"
        shared.write_once(path, selected)
        descriptors[protocol] = {
            "path": f"fit_select/{protocol}/selection.json",
            "sha256": shared.sha(path),
        }
    shared.write_once(
        staging / "selection_set.json", {"protocols": descriptors, "contract_sha256": "fixture"}
    )
    shared.finish_stage(
        staging,
        tmp_path / "fit_select",
        {
            "stage": "fit_select",
            "contract_sha256": "fixture",
            "selection_set_sha256": shared.sha(staging / "selection_set.json"),
        },
        {},
    )
    _, states = cli.verify_selection_set(tmp_path, formal=False)
    assert len(states) == 6 and sum(len(value["states"]) for value in states.values()) == 54
    with pytest.raises(ValueError, match="formal selection"):
        cli.verify_selection_set(tmp_path)


def test_partial_fit_failure_retains_candidate_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_roles",
        lambda *args: (
            {"train": frame(), "validation": frame("validation")},
            {"train": [], "validation": []},
        ),
    )

    def fail(*args, observer, **kwargs):
        observer({"finite": False, "error": "fixture failure", "order": 0})
        raise ValueError("fixture failure")

    monkeypatch.setattr(cli, "fit_protocol", fail)
    with pytest.raises(ValueError, match="fixture failure"):
        cli.fit_select(
            tmp_path,
            {"source_sha256": {}},
            {protocol: {} for protocol in shared.PROTOCOLS},
            tmp_path,
        )
    evidence = list((tmp_path / ".partial").rglob("candidate_attempts.jsonl"))
    assert len(evidence) == 1 and "fixture failure" in evidence[0].read_text()
    assert not (tmp_path / "fit_select").exists()


def test_six_protocol_fixture_signal(tmp_path, monkeypatch):
    """Exercise real grids/solvers and all720 mappings with explicit toy data."""
    from virtual_imu.eval.reviewer_evaluation import (
        _fit_paired_classical,
        _frame_digest,
        _predict_paired_classical,
    )

    frames = {
        role: frame(role, seed=20 + index)
        for index, role in enumerate(("train", "validation", "test"))
    }
    for data in frames.values():
        data["source_timestamp_s"] = data.timestamp_s + 100
        data["available_timestamp_s"] = data.timestamp_s + 0.1
        data["calibration_id"], data["lever_arm_compensated"] = "fixture", True
        data["sample_rate_hz"] = 40.0
    calls = []

    def loader(manifests, roles, project):
        calls.append(tuple(roles))
        return {role: frames[role].copy() for role in roles}, {role: [] for role in roles}

    monkeypatch.setattr(cli, "load_roles", loader)
    original_fit = shared.fit_protocol
    monkeypatch.setattr(
        cli, "fit_protocol", lambda *args, **kwargs: original_fit(*args, **kwargs, synthetic=True)
    )
    contract = {"source_sha256": {}, "synthetic_fixture": True}
    manifests = {
        protocol: {
            name: {"path": f"fixture/{name}", "sha256": name + "_fixture_hash"}
            for name in (
                "dataset_manifest",
                "split_manifest",
                "strict_manifest",
                "assignment_manifest",
            )
        }
        for protocol in shared.PROTOCOLS
    }
    out = tmp_path / "o"
    summary = cli.fit_select(out, contract, manifests, tmp_path)
    assert calls == [("train", "validation")] * 6
    assert summary["synthetic_fixture"] is True
    assert summary["counts"]["candidate_solver_calls"] == 546
    assert summary["counts"]["total_solver_calls"] == 570
    registry = pd.read_csv(out / "fit_select/control_registry.csv")
    assert len(registry) == 54 and registry.n.eq(1).all() and registry.sd.isna().all()
    assert registry.learned_state.sum() == 48
    original_verifier = cli.verify_selection_set
    monkeypatch.setattr(
        cli, "verify_selection_set", lambda output: original_verifier(output, formal=False)
    )

    def audit(*args):
        assert len(calls) == 6
        assert (out / "fit_select/selection_set.json").is_file()
        return {"synthetic_fixture": True}

    monkeypatch.setattr(cli, "full_original_audit", audit)
    monkeypatch.setattr(
        cli,
        "original_helpers",
        lambda: SimpleNamespace(recorded_path=lambda value, *args: Path(value)),
    )
    jobs = []
    columns = [*shared.IDENTITY, "sequence_id", "timestamp_s", *shared.LOW, *shared.HIGH]
    # One legitimate original deterministic bank per protocol; the60 original
    # neural mappings refer to it, without turning classical n into60.
    for protocol in shared.PROTOCOLS:
        bank = _fit_paired_classical(frames["train"], frames["validation"], shared.SEARCH)
        directory = tmp_path / "c" / protocol / "evaluation"
        directory.mkdir(parents=True)
        saved = {
            "states": bank,
            "state_sha256": shared.digest(bank),
            "contract": {
                "protocol_id": protocol,
                "task": "paired",
                "manifests": manifests[protocol],
                "search": shared.SEARCH,
                "train_validation": {
                    "train_rows": 320,
                    "validation_rows": 320,
                    "train_sha256": _frame_digest(frames["train"], columns),
                    "validation_sha256": _frame_digest(frames["validation"], columns),
                },
            },
        }
        cache_path = directory.parent / "cache.json"
        shared.write_once(cache_path, saved)
        shared.write_once(directory / "baseline_train_state.json", bank)
        shared.write_once(
            directory / "provenance.json",
            {"classical_cache": {"path": str(cache_path), "sha256": shared.sha(cache_path)}},
        )
        for role in ("validation", "test"):
            table = frames[role].copy()
            for method, values in _predict_paired_classical(bank["pairs"][0], table).items():
                for axis, values_axis in zip(shared.AXES, values.T, strict=True):
                    table[f"{method}_{axis}"] = values_axis
            table.to_parquet(directory / f"{role}_predictions.parquet", index=False)
        for index in range(60):
            jobs.append(
                {
                    "id": f"{protocol}_{index}",
                    "cell": protocol + "/fixture",
                    "phase": "final_default" if index < 30 else "final_tuned",
                    "seed": 20261010 + index % 3,
                    "run_dir": str(directory.parent),
                }
            )
    result = cli.signal_evaluation(out, contract, manifests, {}, jobs, {}, tmp_path, tmp_path)
    assert result["synthetic_fixture"] is True
    assert result["counts"]["logical_signal_slots"] == 108
    assert result["counts"]["paired_job_mappings"] == 360
    assert result["counts"]["neural_split_support_references"] == 720
    assert calls[6:] == [("train", "validation", "test")] * 6
    shared.verify_artifact_manifest(out / "signal_evaluation")
    aliases = pd.read_csv(out / "signal_evaluation/alias_audit.csv")
    assert len(aliases) == 54 and aliases.n.eq(1).all() and aliases.sd.isna().all()


def test_auxiliary_numeric_audit_retains_blank_unsupported_and_rejects_literal_nonfinite(tmp_path):
    (tmp_path / "values.csv").write_text(
        "metric,value,sd\nunsupported,,\nsupported,1,\n", encoding="utf-8"
    )
    shared.write_once(tmp_path / "summary.json", {"value": 1.0, "unsupported": None})
    result = cli.audit_auxiliary_payloads(tmp_path)
    assert result["status"] == "passed" and result["null_numeric_cells"] > 0
    (tmp_path / "values.csv").write_text("metric,value\ncorrupt,NaN\n", encoding="utf-8")
    with pytest.raises(ValueError, match="literal nonfinite"):
        cli.audit_auxiliary_payloads(tmp_path)


def test_original_timing_profile_recomputes_samples_and_loops():
    record = {
        "warmup": 30,
        "iterations": 100,
        "repeats": 3,
        "raw_wall_ms": [[1.0] * 100] * 3,
        "raw_throughput_loop_seconds": [1.0, 1.0, 1.0],
        "batch_size": 1,
        "wall": {"count": 300, "mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0},
        "loop_throughput_windows_per_second": 100.0,
    }
    cli.audit_timing_measurements([record])
    record["wall"]["p95_ms"] = 2.0
    with pytest.raises(ValueError, match="statistics"):
        cli.audit_timing_measurements([record])


def test_prediction_null_cannot_be_hidden_as_optional_metadata(tmp_path):
    data = frame("test")
    data["optional_quality"] = np.nan
    path = tmp_path / "prediction.parquet"
    data.to_parquet(path, index=False)
    assert cli.audit_auxiliary_payloads(tmp_path)["status"] == "passed"
    data.loc[0, shared.LOW[0]] = np.nan
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="required prediction"):
        cli.audit_auxiliary_payloads(tmp_path)


def test_role_loader_never_opens_test_and_rejects_missing_raw_id(tmp_path, monkeypatch):
    entries, strict_records = {}, []
    for role in ("train", "validation", "test"):
        path = tmp_path / f"{role}.parquet"
        data = frame(role, n=359)
        data["training_eligible"] = True
        data["pair_id"] = "low_high"
        if role != "test":
            data.to_parquet(path, index=False)
        record = {
            name: str(data[name].iloc[0])
            for name in [*shared.IDENTITY, "sequence_id", "original_sequence_id", "pair_id"]
        }
        record.update(
            path=str(path),
            strict_path=str(path),
            sha256=shared.sha(path) if role != "test" else "never_open",
            fingerprint=role,
            training_eligible=True,
            group_value=role,
        )
        entries[role] = [record]
        strict_records.append(record)
    split_path, strict_path = tmp_path / "split.json", tmp_path / "strict.json"
    shared.write_once(split_path, {"split_kind": "leave_one_sequence_out", "splits": entries})
    shared.write_once(strict_path, {"records": strict_records})
    manifests = {
        "split_manifest": {"path": str(split_path), "sha256": shared.sha(split_path)},
        "strict_manifest": {"path": str(strict_path), "sha256": shared.sha(strict_path)},
    }
    original_read = pd.read_parquet
    reads = []

    def read(path, *args, **kwargs):
        reads.append(Path(path).name)
        assert Path(path).name != "test.parquet"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    loaded, _ = cli.load_roles(manifests, ("train", "validation"), tmp_path)
    assert reads == ["train.parquet", "validation.parquet"]
    assert len(loaded["train"]) == 359 and len(loaded["validation"]) == 320
    invalid = original_read(tmp_path / "train.parquet").drop(columns="sequence_id")
    invalid.to_parquet(tmp_path / "train.parquet", index=False)
    entries["train"][0]["sha256"] = shared.sha(tmp_path / "train.parquet")
    split_path.write_text(
        json.dumps({"split_kind": "leave_one_sequence_out", "splits": entries}), encoding="utf-8"
    )
    strict_path.write_text(json.dumps({"records": strict_records}), encoding="utf-8")
    for descriptor in manifests.values():
        descriptor["sha256"] = shared.sha(Path(descriptor["path"]))
    with pytest.raises(ValueError, match="requires columns"):
        cli.load_roles(manifests, ("train", "validation"), tmp_path)


def test_exact_alias_requires_equal_numeric_content_hash():
    left = shared._state("pooled", "affine", weights=np.eye(6), intercept=np.zeros(6))
    signed_zero = np.zeros(6)
    signed_zero[0] = -0.0
    right = shared._state("pooled", "affine", weights=np.eye(6), intercept=signed_zero)
    assert left["numeric_sha256"] != right["numeric_sha256"]
    proof = {
        key: True
        for key in (
            "same_protocol",
            "train_rows_exact",
            "validation_rows_exact",
            "grid_order_exact",
            "tie_rule_exact",
            "bank_own_selection_valid",
        )
    }
    assert (
        shared.alias_audit(left, right, frame("validation"), contract_proof=proof)["alias_status"]
        == "numerically_equivalent"
    )
