"""Independent-GT replay of frozen shared controls on the original five views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate_reviewer_shared_classical import (
    _staging,
    metadata_gate,
    original_helpers,
    verify_selection_set,
)
from reviewer_shared_classical import (
    AXES,
    METHODS,
    PHYSICAL,
    PROFILE,
    SCOPE,
    digest,
    finish_stage,
    predict_state,
    read_json,
    sha,
    verify_artifact_manifest,
    write_once,
)
from virtual_imu.eval.downstream_attitude import (
    evaluate_attitude_stream,
    evaluate_strapdown_stream,
    source_clock_origin_s,
)

VIEWS = {
    "penn_loso_af": "af",
    "penn_loso_as": "as",
    "penn_loso_bf": "bf",
    "penn_only": "bs",
    "pooled": "bs",
}


def exact_physical_frame(
    shared: pd.DataFrame, original: pd.DataFrame, recording: str
) -> pd.DataFrame:
    shared = (
        shared.loc[shared.dataset_id.eq("penncosyvio")]
        .sort_values(["sequence_id", "timestamp_s"], kind="stable")
        .reset_index(drop=True)
    )
    original = original.sort_values(["sequence_id", "timestamp_s"], kind="stable").reset_index(
        drop=True
    )
    if shared.empty or set(shared.original_sequence_id.astype(str)) != {recording}:
        raise ValueError("Downstream recording/physical support differs")
    if any(column not in shared or column not in original for column in PHYSICAL):
        raise ValueError("Downstream physical support/clock fields are incomplete")
    try:
        pd.testing.assert_frame_equal(
            shared[PHYSICAL], original[PHYSICAL], check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError("Shared/aux1 physical support or clock differs") from error
    if (
        not shared.calibration_id.eq("penncosyvio_published_body_extrinsics_v1").all()
        or not shared.lever_arm_compensated.eq(True).all()
        or not shared.sample_rate_hz.eq(40.0).all()
    ):
        raise ValueError("Original rig frame/extrinsics/40Hz contract differs")
    source_clock_origin_s(shared)
    return shared


def replay_view(
    frame: pd.DataFrame, truth: pd.DataFrame, states: dict, config: dict, original_metadata: dict
) -> dict:
    """Thin wrapper: original midpoint SO3/strapdown integrators own all science."""
    helper = original_helpers()
    helper.validate_protocol(config)
    clock = source_clock_origin_s(frame)
    if clock != original_metadata["physical_support"]["source_clock_origin_s"]:
        raise ValueError("Original source clock origin must remain fixed")
    combined = frame.copy()
    for method in METHODS:
        values = predict_state(states[method], frame)
        for axis, vector in zip(AXES, values.T, strict=True):
            combined[f"{axis}_{method}"] = vector
    args = {
        "prefixes": list(METHODS),
        "window_samples": config["protocol"]["window_samples"],
        "rate_hz": 40.0,
        "max_gt_gap_s": config["max_gt_gap_s"],
        "gt_timestamp_origin_s": clock,
    }
    result = {
        "attitude": evaluate_attitude_stream(combined, truth, **args),
        "strapdown": evaluate_strapdown_stream(
            combined,
            truth,
            gravity_world_mps2=config["protocol"]["gravity_world_mps2"],
            initial_velocity_span_s=0.1,
            **args,
        ),
    }
    for task, measured in result.items():
        previous = original_metadata["support"][task]
        if {
            key: value for key, value in measured["support"].items() if key != "common_methods"
        } != {key: value for key, value in previous.items() if key != "common_methods"}:
            raise ValueError(
                "Shared downstream common support or initialization skips differ from aux1"
            )
        if measured["protocol"] != original_metadata["propagation_protocol"][task]:
            raise ValueError("Shared downstream propagation/GT initialization contract differs")
    return result


def support_keys(rows, *, section):
    names = ["task", "sequence_id", "segment", "start_s", "end_s"]
    names += ["samples"] if section == "window_details" else []
    return [tuple(row[name] for name in names) for row in rows]


def match_reference_keys(result: dict, reference: dict) -> None:
    for section, key in (
        ("window_details", "window_details"),
        ("full_segments", "full_recordings"),
    ):
        original = [row for row in reference[section] if row["method"] == "identity"]
        expected = support_keys(original, section=section)
        for method in METHODS:
            measured = [
                {**row, "task": task}
                for task, value in result.items()
                for row in value[key]
                if row["method"] == method
            ]
            if support_keys(measured, section=section) != expected:
                raise ValueError(
                    "Shared downstream window/segment keys differ from the original aux1"
                )
    # Identity is a common alias. Its scientific errors must also match before
    # it can reference the original deterministic stream.
    for section, key in (
        ("window_details", "window_details"),
        ("full_segments", "full_recordings"),
    ):
        original = [row for row in reference[section] if row["method"] == "identity"]
        measured = [
            {**row, "task": task}
            for task, value in result.items()
            for row in value[key]
            if row["method"] == "identity"
        ]
        for left, right in zip(original, measured, strict=True):
            for name, value in right.items():
                if name == "method":
                    continue
                if isinstance(value, (int, float, list)):
                    if not np.allclose(value, left[name], rtol=1e-10, atol=1e-12):
                        raise ValueError(
                            "Original/shared identity downstream numerical alias differs"
                        )
                elif value != left[name]:
                    raise ValueError("Original/shared identity downstream metadata differs")


def run(args):
    print(
        f"READS: frozen shared selections/signals and original aux1 GT/support; WRITES: {args.out}/downstream; SKIPS: neural inference, bank rerun, GT fitting and selection",
        flush=True,
    )
    out, project, root = (
        args.out.resolve(),
        args.project_root.resolve(),
        args.pipeline_root.resolve(),
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    state, _, gate = metadata_gate(root, project, args.aux1_state, args.preceding_state, config)
    del state
    _, selections = verify_selection_set(out)
    signal_manifest = verify_artifact_manifest(out / "signal_evaluation")
    aux1 = gate["prior"]["aux1"]
    previous = Path(aux1["path"]).parent / "downstream"
    original_inventory = {
        path.relative_to(previous).as_posix(): sha(path)
        for path in sorted(previous.rglob("*"))
        if path.is_file()
    }
    if original_inventory != aux1["state"]["stages"]["downstream"]["artifact_sha256"]:
        raise ValueError("Original downstream source artifacts changed after signal audit")
    old_config = previous / "config_snapshot.yaml"
    propagation = yaml.safe_load(old_config.read_text(encoding="utf-8"))
    helper = original_helpers()
    helper.validate_protocol(propagation)
    contract = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "fit_selection_set_sha256": sha(out / "fit_select/selection_set.json"),
        "signal_manifest_sha256": sha(out / "signal_evaluation/artifact_manifest.json"),
        "signal_contract_sha256": signal_manifest["contract_sha256"],
        "aux1_state_sha256": sha(args.aux1_state),
        "aux2_state_sha256": sha(args.preceding_state),
        "propagation_config_sha256": sha(old_config),
        "script_sha256": sha(Path(__file__)),
        "integrator_sha256": sha(project / "src/virtual_imu/eval/downstream_attitude.py"),
    }
    contract_sha = digest(contract)
    destination = out / "downstream"
    if destination.exists():
        if not args.resume:
            raise FileExistsError("Completed downstream requires resume")
        verify_artifact_manifest(destination, contract_sha=contract_sha)
        return read_json(destination / "stage_summary.json")
    staging = _staging(out, "downstream")
    write_once(staging / "contract.json", contract)
    summaries, windows, segments, proofs = [], [], [], []
    for protocol, recording in VIEWS.items():
        old = previous / "units" / protocol
        metadata = read_json(old / "metrics.json")
        old_completion = read_json(previous / "complete.json")
        if sha(old / "artifact_manifest.json") != old_completion["unit_manifests"][protocol]:
            raise ValueError("Original downstream unit manifest differs")
        helper.verify_completed(old, old_completion["contract_sha256"])
        reference = {
            section: read_json(old / f"{section}.json")
            for section in ("window_details", "full_segments")
        }
        print(
            f"READS: manifest-bound shared signal and original downstream physical rows {protocol}",
            flush=True,
        )
        frame = exact_physical_frame(
            pd.read_parquet(out / "signal_evaluation" / protocol / "test_predictions.parquet"),
            pd.read_parquet(old / "evaluated_signals.parquet"),
            recording,
        )
        gt_path = Path(metadata["independent_gt_path"])
        if sha(gt_path) != metadata["independent_gt_sha256"]:
            raise ValueError("Original independent optical GT SHA differs")
        print(f"READS: exact original independent GT {gt_path}", flush=True)
        result = replay_view(
            frame, pd.read_parquet(gt_path), selections[protocol]["states"], propagation, metadata
        )
        match_reference_keys(result, reference)
        for task, values in result.items():
            for section, key, accumulator in (
                ("summary", "windows", summaries),
                ("windows", "window_details", windows),
                ("segments", "full_recordings", segments),
            ):
                for row in values[key]:
                    method = row["method"]
                    base = {
                        "protocol": protocol,
                        "information_scope": SCOPE,
                        "canonical_method": method,
                        "variant": "shared",
                        "recording": recording,
                        "view": "separate_same_recording_pooled_reference"
                        if protocol == "pooled"
                        else "outer_recording_holdout",
                        "task": task,
                        "n": 1,
                        "sd": None,
                        "identity_alias": method == "identity",
                        "state_content_sha256": digest(selections[protocol]["states"][method]),
                        **{name: value for name, value in row.items() if name != "method"},
                    }
                    if section == "summary":
                        base["optimizer_runs"] = 1
                        base["status"] = (
                            "supported" if row["count"] else "unsupported_no_fixed_windows"
                        )
                    accumulator.append(base)
            proofs.append(
                {
                    "protocol": protocol,
                    "recording": recording,
                    "task": task,
                    "original_unit_manifest_sha256": sha(old / "artifact_manifest.json"),
                    "independent_gt_sha256": sha(gt_path),
                    "support": values["support"],
                    "propagation_protocol": values["protocol"],
                    "identity_alias_verified": True,
                }
            )
    slots = {
        (row["protocol"], row["task"], row["samples"], row["canonical_method"]) for row in summaries
    }
    if len(slots) != 360 or len(summaries) != 360:
        raise ValueError("Expected 5 x 2 x 4 x 9 shared downstream logical summary slots")
    write_once(staging / "support_proofs.json", proofs)
    write_once(staging / "downstream_windows.json", windows)
    write_once(staging / "downstream_segments.json", segments)
    return finish_stage(
        staging,
        destination,
        {
            "stage": "downstream",
            "contract_sha256": contract_sha,
            "selection_set_sha256": contract["fit_selection_set_sha256"],
            "counts": {
                "downstream_views": 5,
                "logical_downstream_summary_slots": 360,
                "independent_recordings": 4,
                "neural_inference_calls": 0,
                "comparison_neural_variants_per_view": 20,
                "comparison_shared_controls_per_view": 9,
                "comparison_high_references_per_view": 1,
                "comparison_logical_methods_per_view": 30,
                "comparison_logical_streams_per_view": 70,
            },
            "pooled_bs": "same recording as penn_only/bs; separate training-scope reference",
        },
        {
            "downstream_summary": pd.DataFrame(summaries),
            "downstream_windows": pd.DataFrame(windows),
            "downstream_segments": pd.DataFrame(segments),
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "pipeline-root", "aux1-state", "preceding-state", "out", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args(argv))
    print(json.dumps({"status": result["status"], "stage": result["stage"]}), flush=True)


if __name__ == "__main__":
    main()
