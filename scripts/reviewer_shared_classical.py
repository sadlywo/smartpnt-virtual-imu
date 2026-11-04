"""Protocol-shared raw-SI controls, strict physical support, immutable artifacts.

This module has no TEST loader. Fit accepts only TRAIN and VAL frames. The
selection observer repeats the frozen transfer solver verbatim in order to retain
candidate timing and dataset/pair evidence without modifying the frozen helper.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from virtual_imu.eval.baselines import _sequence_lag_features
from virtual_imu.eval.formal_evaluation import (
    HIGH,
    IDENTITY,
    LOW,
    fit_pair_baselines,
    physical_error_metrics,
    predict_pair_baselines,
)

PROFILE = "reviewer_shared_classical_v1"
SCOPE = "protocol_shared_classical"
PROTOCOLS = ("pooled", "alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf")
METHODS = (
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "ridge_tuned",
    "fir_tuned",
    "arx_tuned",
)
SEARCH = {
    "ridge_alpha": [1e-6, 1e-4, 0.01, 0.1, 1, 10, 100],
    "fir_taps": [1, 3, 5, 9, 17, 33, 65],
    "arx_taps": [3, 6, 12, 24, 48],
}
AXES = ("gx", "gy", "gz", "ax", "ay", "az")
PHYSICAL = [
    *IDENTITY,
    "sequence_id",
    "original_sequence_id",
    "timestamp_s",
    "source_timestamp_s",
    "available_timestamp_s",
    "calibration_id",
    "lever_arm_compensated",
    "sample_rate_hz",
    *LOW,
    *HIGH,
]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path):
    print(f"READS: {path}", flush=True)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"WRITES: {path}", flush=True)


def frame_sha(frame: pd.DataFrame, columns=None) -> str:
    """Bind column names, row order and values; never sort away a support change."""
    selected = frame if columns is None else frame[list(columns)]
    value = hashlib.sha256(json.dumps(list(selected.columns)).encode())
    value.update(pd.util.hash_pandas_object(selected, index=False).to_numpy().tobytes())
    return value.hexdigest()


def validate_streams(frame: pd.DataFrame, records=None, *, require_target=True) -> list[dict]:
    required = [*IDENTITY, "sequence_id", "original_sequence_id", "timestamp_s", *LOW]
    if require_target:
        required += HIGH
    missing = set(required) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Nonempty physical support requires columns: {sorted(missing)}")
    for name in [*IDENTITY, "sequence_id", "original_sequence_id"]:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing/nonempty physical identity required: {name}")
    numerical = ["timestamp_s", *LOW, *(HIGH if require_target else [])]
    if not np.isfinite(frame[numerical].to_numpy(float)).all():
        raise ValueError("Nonfinite timestamp or six-axis LOW/HIGH")
    manifest = {}
    for record in records or []:
        key = str(record.get("sequence_id", "")).strip()
        if not key or key in manifest:
            raise ValueError("Authoritative manifest has missing/conflicting sequence_id")
        manifest[key] = record
    registry = []
    for sequence, positions in frame.groupby("sequence_id", sort=False).indices.items():
        group = frame.iloc[positions]
        fields = [*IDENTITY, "original_sequence_id"]
        fields += [
            name
            for name in ("segment_id", "recording_id", "reference_recording_id", "pair_id")
            if name in group
        ]
        if any(group[name].nunique(dropna=False) != 1 for name in fields):
            raise ValueError(f"sequence_id collision across physical streams: {sequence}")
        if records is not None:
            record = manifest.get(str(sequence))
            if record is None:
                raise ValueError(f"Sequence absent from authoritative manifest: {sequence}")
            for name in fields:
                if name in record and str(group[name].iloc[0]) != str(record[name]):
                    raise ValueError(f"Manifest physical identity mismatch: {sequence}/{name}")
            if any(name not in record for name in [*IDENTITY, "original_sequence_id"]):
                raise ValueError("Manifest must bind dataset, original sensors and recording")
        times = np.sort(group.timestamp_s.to_numpy(float), kind="stable")
        if len(times) > 1 and not np.allclose(np.diff(times), 0.025, atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"Physical stream has duplicate timestamps, gap or non-40Hz spacing: {sequence}"
            )
        registry.append(
            {
                "sequence_id": str(sequence),
                **{name: str(group[name].iloc[0]) for name in fields},
                "rows": len(group),
                "time_start_s": float(times[0]),
                "time_end_s": float(times[-1]),
                "row_sha256": frame_sha(group),
                "boundary_status": "continuous_40hz",
                "source_path": manifest.get(str(sequence), {}).get("path"),
                "source_sha256": manifest.get(str(sequence), {}).get("sha256"),
            }
        )
    return registry


def physical_support(
    frame: pd.DataFrame, role: str, records=None
) -> tuple[pd.DataFrame, list[dict]]:
    if role not in {"train", "validation", "test"}:
        raise ValueError("Unknown split role")
    registry = validate_streams(frame, records)
    if "split" in frame and set(frame.split.astype(str)) != {role}:
        raise ValueError("Loaded frame split role differs from requested role")
    selected = np.zeros(len(frame), dtype=bool)
    by_sequence = {row["sequence_id"]: row for row in registry}
    for sequence, positions in frame.groupby("sequence_id", sort=False).indices.items():
        ordered = positions[np.argsort(frame.iloc[positions].timestamp_s.to_numpy(), kind="stable")]
        n = len(ordered)
        count = n if role == "train" else n // 40 * 40
        count = count if n >= 320 else 0
        selected[ordered[:count]] = True
        row = by_sequence[str(sequence)]
        row.update(
            role=role,
            eligible_rows=count,
            skipped_rows=n - count,
            skip_reason="segment_shorter_than_320"
            if n < 320
            else ("complete_stride40_prefix" if count < n else None),
        )
    result = frame.iloc[np.flatnonzero(selected)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"No eligible >=320-point physical streams for {role}")
    return result, registry


def macro_breakdown(prediction: np.ndarray, validation: pd.DataFrame) -> dict:
    pairs, datasets = [], []
    for identity, positions in validation.groupby(IDENTITY, sort=True).indices.items():
        group = validation.iloc[positions]
        measured = physical_error_metrics(prediction[positions], group[HIGH].to_numpy(float))
        baseline = physical_error_metrics(group[LOW].to_numpy(float), group[HIGH].to_numpy(float))
        ratios = [
            measured[key] / max(baseline[key], 1e-12)
            for key in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
        ]
        pairs.append(
            {
                **dict(zip(IDENTITY, map(str, identity), strict=True)),
                "rows": len(group),
                "score": float(np.mean(ratios)),
                "gyro_ratio": ratios[0],
                "accel_ratio": ratios[1],
                "near_zero_gyro_identity": baseline["gyro_vector_rmse_radps"] <= 1e-12,
                "near_zero_accel_identity": baseline["accel_vector_rmse_mps2"] <= 1e-12,
            }
        )
    for dataset in sorted({row["dataset_id"] for row in pairs}):
        values = [row["score"] for row in pairs if row["dataset_id"] == dataset]
        datasets.append(
            {"dataset_id": dataset, "pairs": len(values), "score": float(np.mean(values))}
        )
    return {
        "score": float(np.mean([row["score"] for row in datasets])),
        "pairs": pairs,
        "datasets": datasets,
    }


def _state(
    protocol,
    method,
    *,
    kind="linear",
    weights=None,
    intercept=None,
    taps=None,
    delta=False,
    alpha=None,
    arithmetic="weights_bias",
) -> dict:
    weights = [] if weights is None else np.asarray(weights, dtype=np.float64).tolist()
    intercept = [] if intercept is None else np.asarray(intercept, dtype=np.float64).tolist()
    count = np.asarray(weights).size + np.asarray(intercept).size
    if kind == "identity":
        expected = 0
    elif kind == "bias":
        expected = 6
    else:
        expected = 36 * ((taps or 1) + int(delta)) + 6
    feature_count = 6 * ((taps or 1) + int(delta))
    shape_valid = (
        (kind == "identity" and not weights and not intercept)
        or (kind == "bias" and not weights and np.shape(intercept) == (6,))
        or (
            kind == "linear"
            and np.shape(weights) == (feature_count, 6)
            and np.shape(intercept) == (6,)
        )
    )
    if (
        not shape_valid
        or count != expected
        or not np.isfinite(np.asarray([*np.asarray(weights).ravel(), *intercept], float)).all()
    ):
        raise ValueError("Nonfinite or wrongly shaped learned coefficient state")
    numeric = {
        "state_kind": kind,
        "weights": weights,
        "intercept": intercept,
        "hyperparameters": {"alpha": alpha, "taps": taps},
        "feature_spec": {
            "taps": taps,
            "include_delta": delta,
            "edge_padding": "first_sample",
            "initial_delta": 0,
            "split_local": True,
            "stream_key": "sequence_id",
        },
        "residual_add_low": kind == "bias",
        "normalization": {"mode": "none_raw_si", "coefficient_count": 0},
        "arithmetic": arithmetic,
    }
    state = {
        "schema_version": 1,
        "protocol": protocol,
        "information_scope": SCOPE,
        "canonical_method": method,
        "variant": "shared",
        "input_channels": 6,
        "output_channels": 6,
        **numeric,
        "coefficient_count": int(count),
        "coefficient_bytes": int(count * 8),
        "normalizer_count": 0,
        "normalizer_bytes": 0,
        "numeric_sha256": digest(numeric),
    }
    return state


def verify_state(state: dict) -> None:
    if state.get("information_scope") != SCOPE:
        raise ValueError("Normalized shared state checksum/scope differs")
    recreated = _state(
        state["protocol"],
        state["canonical_method"],
        kind=state["state_kind"],
        weights=state["weights"],
        intercept=state["intercept"],
        taps=state["feature_spec"]["taps"],
        delta=state["feature_spec"]["include_delta"],
        alpha=state["hyperparameters"]["alpha"],
        arithmetic=state["arithmetic"],
    )
    if recreated != state:
        raise ValueError("Normalized predictor state fields differ from coefficient specification")


def normalize_bank_pair(protocol: str, bank: dict) -> dict[str, dict]:
    fixed = bank["fixed"]
    result = {
        "identity": _state(protocol, "identity", kind="identity"),
        "bias_mean": _state(protocol, "bias_mean", kind="bias", intercept=fixed["bias"]),
    }
    for name in ("affine", "ridge_affine"):
        coefficients = np.asarray(fixed[name], float)
        result[name] = _state(
            protocol,
            name,
            weights=coefficients[:-1],
            intercept=coefficients[-1],
            alpha=0.01 if name == "ridge_affine" else None,
            arithmetic="design_matmul",
        )
    system = fixed["system_id"]
    for name, prefix, delta in (("learned_fir_causal", "fir", False), ("arx_linear", "arx", True)):
        result[name] = _state(
            protocol,
            name,
            weights=system[prefix + "_weights"],
            intercept=system[prefix + "_bias"],
            taps=system[prefix + "_taps"],
            delta=delta,
            alpha=system["alpha"],
        )
    for name in ("ridge_tuned", "fir_tuned", "arx_tuned"):
        if name in bank:
            fitted = bank[name]
            result[name] = _state(
                protocol,
                name,
                weights=fitted["weights"],
                intercept=fitted["bias"],
                taps=fitted.get("taps"),
                delta=fitted.get("include_delta", False),
                alpha=fitted["alpha"],
            )
    return result


def predict_state(state: dict, frame: pd.DataFrame) -> np.ndarray:
    """Metadata controls stream boundaries only. HIGH is never inspected."""
    verify_state(state)
    validate_streams(frame, require_target=False)
    low = frame[LOW].to_numpy(float)
    if state["state_kind"] == "identity":
        return low.copy()
    bias = np.asarray(state["intercept"], float)
    if state["state_kind"] == "bias":
        return low + bias
    spec = state["feature_spec"]
    features = (
        _sequence_lag_features(frame, spec["taps"], include_delta=spec["include_delta"])
        if spec["taps"]
        else low
    )
    weights = np.asarray(state["weights"], float)
    result = (
        np.column_stack([features, np.ones(len(features))]) @ np.vstack([weights, bias])
        if state["arithmetic"] == "design_matmul"
        else features @ weights + bias
    )
    if result.shape != (len(frame), 6) or not np.isfinite(result).all():
        raise ValueError("Invalid shared predictions")
    return result


def fit_protocol(
    protocol: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    search=None,
    observer=None,
    synthetic=False,
) -> dict:
    """Fit one protocol. No TEST argument/path/metrics enter this API."""
    if protocol not in PROTOCOLS:
        raise ValueError("Unknown paired protocol")
    search = SEARCH if search is None else search
    if not synthetic and search != SEARCH:
        raise ValueError("Formal selection requires the exact frozen grid/order")
    validate_streams(train)
    validate_streams(validation)
    if not synthetic and ("split" not in train or "split" not in validation):
        raise ValueError("Formal fit API requires explicit TRAIN/VAL roles")
    if "split" in train and set(train.split.astype(str)) != {"train"}:
        raise ValueError("Fit frame role must be train")
    if "split" in validation and set(validation.split.astype(str)) != {"validation"}:
        raise ValueError("Selection frame role must be validation")
    if set(train.sequence_id.astype(str)) & set(validation.sequence_id.astype(str)):
        raise ValueError("TRAIN/VAL physical sequence identities overlap")
    train_ids = set(map(tuple, train[IDENTITY].astype(str).to_numpy()))
    if set(map(tuple, validation[IDENTITY].astype(str).to_numpy())) != train_ids:
        raise ValueError("TRAIN/VAL device identities differ")
    helper = train.copy()
    helper[IDENTITY] = "protocol_shared"
    started = time.perf_counter()
    fixed = fit_pair_baselines(helper, fir_taps=9, alpha=0.01)
    fixed_elapsed = time.perf_counter() - started
    states = normalize_bank_pair(protocol, {"fixed": fixed})
    for method, values in predict_pair_baselines(fixed, validation).items():
        np.testing.assert_allclose(
            predict_state(states[method], validation), values, rtol=1e-12, atol=1e-12
        )
    history, selections = [], {}
    for method, taps_grid, delta in (
        ("ridge_tuned", [None], False),
        ("fir_tuned", search["fir_taps"], False),
        ("arx_tuned", search["arx_taps"], True),
    ):
        best = None
        for taps in taps_grid:
            tx = (
                _sequence_lag_features(train, taps, include_delta=delta)
                if taps
                else train[LOW].to_numpy(float)
            )
            vx = (
                _sequence_lag_features(validation, taps, include_delta=delta)
                if taps
                else validation[LOW].to_numpy(float)
            )
            design = np.column_stack([tx, np.ones(len(tx))])
            validation_design = np.column_stack([vx, np.ones(len(vx))])
            gram, rhs = design.T @ design, design.T @ train[HIGH].to_numpy(float)
            penalty = np.eye(design.shape[1])
            penalty[-1, -1] = 0
            for alpha in search["ridge_alpha"]:
                start = time.perf_counter()
                row = {
                    "protocol": protocol,
                    "information_scope": SCOPE,
                    "canonical_method": method,
                    "variant": "shared",
                    "order": len(history),
                    "alpha": alpha,
                    "taps": taps,
                    "include_delta": delta,
                }
                try:
                    coefficients = np.linalg.solve(gram + alpha * penalty, rhs)
                    breakdown = macro_breakdown(validation_design @ coefficients, validation)
                    score = breakdown["score"]
                    if not np.isfinite(coefficients).all() or not np.isfinite(score):
                        raise ValueError("Nonfinite shared calibration candidate")
                    row.update(validation_score=score, finite=True, error=None, breakdown=breakdown)
                except Exception as error:
                    row.update(
                        validation_score=None,
                        finite=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                    row["elapsed_s"] = time.perf_counter() - start
                    history.append(row)
                    if observer:
                        observer(row)
                    raise
                row["elapsed_s"] = time.perf_counter() - start
                history.append(row)
                if observer:
                    observer(row)
                if best is None or score < best["validation_score"]:
                    best = {**row, "weights": coefficients[:-1], "bias": coefficients[-1]}
        selections[method] = {
            key: value for key, value in best.items() if key not in {"weights", "bias"}
        }
        states[method] = _state(
            protocol,
            method,
            weights=best["weights"],
            intercept=best["bias"],
            taps=best["taps"],
            delta=delta,
            alpha=best["alpha"],
        )
    return {
        "schema_version": 1,
        "protocol": protocol,
        "information_scope": SCOPE,
        "synthetic_fixture": synthetic,
        "fit_role": "train",
        "selection_role": "validation",
        "fit_rows": len(train),
        "selection_rows": len(validation),
        "train_row_sha256": frame_sha(train),
        "validation_row_sha256": frame_sha(validation),
        "search": search,
        "tie_rule": "strict_smaller_first_in_grid_order",
        "states": states,
        "selections": selections,
        "candidate_history": history,
        "compute_counts": {
            "fixed_learned_computations": 5,
            "fixed_linear_solver_calls": 4,
            "tuned_solver_calls": len(history),
            "total_linear_solver_calls": len(history) + 4,
            "bias_mean_computations": 1,
            "fixed_elapsed_s": fixed_elapsed,
            "learned_state_slots": 8,
            "identity_references": 1,
            "cache_hits": 0,
        },
        "selection_metric": "equal_dataset_equal_pair_mean_gyro_accel_vector_rmse_relative_identity",
    }


def alias_audit(
    shared: dict, bank: dict, validation: pd.DataFrame, *, contract_proof: dict
) -> dict:
    """Numerical aliases require separate provenance/support proof supplied by caller."""
    required = (
        "same_protocol",
        "train_rows_exact",
        "validation_rows_exact",
        "grid_order_exact",
        "tie_rule_exact",
        "bank_own_selection_valid",
    )
    if not all(contract_proof.get(key) is True for key in required):
        raise ValueError("Alias support/source/selection proof is incomplete")
    verify_state(shared)
    verify_state(bank)
    hp_same = shared["hyperparameters"] == bank["hyperparameters"]
    same_structure = all(
        shared[key] == bank[key]
        for key in ("state_kind", "feature_spec", "normalization", "residual_add_low")
    )
    left, right = predict_state(shared, validation), predict_state(bank, validation)
    exact = (
        hp_same
        and same_structure
        and shared["numeric_sha256"] == bank["numeric_sha256"]
        and shared["weights"] == bank["weights"]
        and shared["intercept"] == bank["intercept"]
        and left.tobytes() == right.tobytes()
    )
    equivalent = (
        hp_same
        and same_structure
        and np.shape(shared["weights"]) == np.shape(bank["weights"])
        and np.allclose(shared["weights"], bank["weights"], rtol=1e-10, atol=1e-12)
        and np.allclose(shared["intercept"], bank["intercept"], rtol=1e-10, atol=1e-12)
        and np.allclose(left, right, rtol=1e-10, atol=1e-12)
    )
    scores = [macro_breakdown(values, validation)["score"] for values in (left, right)]
    exact = exact and scores[0] == scores[1]
    equivalent = equivalent and bool(np.isclose(*scores, rtol=1e-10, atol=1e-12))
    return {
        "protocol": shared["protocol"],
        "canonical_method": shared["canonical_method"],
        "information_scope": SCOPE,
        "variant": "shared",
        "alias_status": "exact_alias"
        if exact
        else "numerically_equivalent"
        if equivalent
        else "not_alias",
        "shared_state_sha256": digest(shared),
        "bank_state_sha256": digest(bank),
        "shared_validation_score": scores[0],
        "bank_validation_score": scores[1],
        "same_hyperparameters": hp_same,
        "max_prediction_absolute_difference": float(np.max(np.abs(left - right))),
        "rtol": 1e-10,
        "atol": 1e-12,
        "proof": contract_proof,
        "n": 1,
        "sd": None,
    }


def verify_artifact_manifest(directory: Path, *, contract_sha=None) -> dict:
    manifest = read_json(directory / "artifact_manifest.json")
    if manifest.get("status") != "complete" or (
        contract_sha is not None and manifest.get("contract_sha256") != contract_sha
    ):
        raise ValueError("Completed stage contract/status differs")
    root = directory.parent
    actual = {
        path.relative_to(root).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Completed stage output file inventory changed")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Completed output checksum changed: {name}")
    return manifest


def finish_stage(
    staging: Path, destination: Path, summary: dict, tables: dict[str, pd.DataFrame]
) -> dict:
    if destination.exists():
        raise FileExistsError("Never replace a completed stage")
    descriptors = {}
    for role, table in tables.items():
        path = staging / f"{role}.csv"
        table.to_csv(path, index=False)
        descriptors[role] = {
            "path": f"{destination.name}/{role}.csv",
            "sha256": sha(path),
            "rows": len(table),
            "columns": list(table.columns),
        }
        print(f"WRITES: {path}", flush=True)
    summary = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "status": "complete",
        **summary,
        "tables": descriptors,
    }
    write_once(staging / "stage_summary.json", summary)
    files = {
        f"{destination.name}/{path.relative_to(staging).as_posix()}": sha(path)
        for path in staging.rglob("*")
        if path.is_file()
    }
    write_once(
        staging / "artifact_manifest.json",
        {
            "schema_version": 1,
            "analysis_profile": PROFILE,
            "root_kind": "aux3_run",
            "status": "complete",
            "contract_sha256": summary["contract_sha256"],
            "files": files,
        },
    )
    staging.rename(destination)
    return summary
