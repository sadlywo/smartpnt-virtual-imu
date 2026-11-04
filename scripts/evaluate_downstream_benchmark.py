"""Replay every historical paired method against independent Penn optical GT.

The contract/config are frozen and printed before prediction arrays are read.
No estimator, initialization, lag, calibration or model is fitted or selected.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from virtual_imu.eval.downstream_attitude import (
    evaluate_attitude_stream,
    evaluate_strapdown_stream,
    source_clock_origin_s,
)
from virtual_imu.eval.formal_evaluation import (
    HIGH,
    IDENTITY,
    LOW,
    VIRTUAL,
    json_safe,
    predict_pair_baselines,
)

NEURAL_METHODS = (
    "tcn_original",
    "plain_cnn",
    "gru",
    "transformer",
    "cnn_bilstm",
    "deepgyro_adapter",
    "tlio_resnet18_adapter",
)
CLASSIC_METHODS = ("identity", "bias_mean", "affine", "ridge_affine", "fir", "arx")
PENN_IDENTITY = {
    "dataset_id": "penncosyvio",
    "low_sensor": "Tango Top IMU",
    "high_sensor": "ADIS16488",
}
KEYS = ["sequence_id", "timestamp_s"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("WRITES {}", path)


def select_penn_state(states: list[dict]) -> dict:
    selected = [state for state in states if state.get("identity") == PENN_IDENTITY]
    if len(selected) != 1:
        raise ValueError("Need exactly one baseline state for the exact Penn sensor mapping")
    if selected[0].get("fit_split") != "train":
        raise ValueError("Penn baseline state must declare train-only fitting")
    return selected[0]


def verify_replay_source(current: Path, historical: Path, names: list[str]) -> dict:
    """Bind source snapshots and require identical actual prediction arithmetic."""
    logger.info("READS current and historical baseline replay source {} / {}", current, historical)
    functions = []
    for path in (current, historical):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        if set(selected) != set(names):
            raise ValueError("Baseline replay source is missing required functions")
        functions.append(selected)
    if functions[0] != functions[1]:
        raise ValueError("Baseline replay arithmetic changed from the saved historical source")
    return {
        "current_path": current.as_posix(),
        "current_sha256": sha(current),
        "historical_path": historical.as_posix(),
        "historical_sha256": sha(historical),
        "identical_function_AST": names,
    }


def recover_source_clock(frame: pd.DataFrame, paired_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    if "source_timestamp_s" in frame:
        return frame, []
    sources, provenance = [], []
    for sequence in frame.sequence_id.unique():
        if Path(str(sequence)).name != sequence:
            raise ValueError("Unsafe sequence identifier")
        path = paired_root / f"{sequence}.parquet"
        logger.info("READS immutable source clock recovery {}", path)
        source = pd.read_parquet(path, columns=[*KEYS, "source_timestamp_s"])
        sources.append(source)
        provenance.append({"path": path.as_posix(), "sha256": sha(path)})
    merged = frame.merge(
        pd.concat(sources, ignore_index=True),
        on=KEYS,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(frame) or merged.source_timestamp_s.isna().any():
        raise ValueError("Source clock recovery has missing sequence/timestamp keys")
    return merged, provenance


def aggregate_trials(trials: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "method", "samples", "horizon_s"]
    excluded = set([*keys, "seed", "count"])
    metrics = [
        column for column in trials.select_dtypes(include="number") if column not in excluded
    ]
    rows = []
    for values, group in trials.groupby(keys, sort=False, dropna=False):
        if len(group) > 1 and group.seed.isna().any():
            raise ValueError(
                "Repeated deterministic classical results cannot be counted as optimizer runs"
            )
        if group.seed.dropna().duplicated().any():
            raise ValueError("Repeated optimization seed")
        row = dict(zip(keys, values, strict=True))
        row.update({"optimizer_runs": len(group), "windows_per_run": int(group["count"].iloc[0])})
        if group["count"].nunique() != 1:
            raise ValueError("Method seeds do not share window support")
        for metric in metrics:
            values_array = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values_array)) if len(values_array) else None
            row[f"{metric}_sd"] = (
                float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    for metric in metrics:
        mean_column = f"{metric}_mean"
        improvements = []
        for _, row in result.iterrows():
            low = result[
                (result.task == row.task)
                & (result.method == "identity")
                & (result.samples == row.samples)
            ]
            denominator = low[mean_column].iloc[0] if len(low) else np.nan
            improvements.append(
                100 * (1 - row[mean_column] / denominator)
                if pd.notna(denominator) and denominator > 1e-12 and pd.notna(row[mean_column])
                else None
            )
        result[f"{metric}_improvement_over_low_pct"] = improvements
    return result


def _registry(pipeline_root: Path, previous_root: Path) -> tuple[list[dict], Path]:
    path = pipeline_root / "pipeline.json"
    logger.info("READS pipeline identities {}", path)
    pipeline = json.loads(path.read_text(encoding="utf-8"))
    entries = []
    for entry in pipeline["runs"]:
        if entry["task"] != "paired":
            continue
        if entry["status"] != "evaluated":
            raise ValueError("Historical paired run is not evaluated")
        basename = str(entry["run_dir"]).replace("\\", "/").rstrip("/").split("/")[-1]
        candidates = [root / "experiments" / basename for root in (pipeline_root, previous_root)]
        matched = [candidate for candidate in candidates if (candidate / "config.yaml").is_file()]
        if len(matched) != 1:
            raise ValueError(f"Cannot uniquely map historical run: {basename}")
        entries.append({**entry, "local_run": matched[0].as_posix()})
    expected = {
        (method, seed) for method in NEURAL_METHODS for seed in (20261010, 20261011, 20261012)
    }
    if {(entry["candidate_id"], entry["seed"]) for entry in entries} != expected or len(
        entries
    ) != 21:
        raise ValueError(
            "Historical registry must contain all seven paired methods and all three fixed seeds"
        )
    return entries, path


def run(pipeline_root: Path, previous_root: Path, config_path: Path, out: Path) -> dict:
    logger.info("READS frozen downstream YAML {}", config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["window_samples"] != [80, 200, 400, 800] or protocol["gravity_world_mps2"] != [
        0.0,
        0.0,
        -9.80665,
    ]:
        raise ValueError(
            "Historical comparison requires the frozen common horizon/gravity contract"
        )
    entries, pipeline_path = _registry(pipeline_root, previous_root)
    historical_eval = pipeline_root.parent.parent / "src/virtual_imu/eval"
    replay_sources = [
        verify_replay_source(
            Path("src/virtual_imu/eval/formal_evaluation.py"),
            historical_eval / "formal_evaluation.py",
            ["predict_pair_baselines"],
        ),
        verify_replay_source(
            Path("src/virtual_imu/eval/baselines.py"),
            historical_eval / "baselines.py",
            [
                "predict_system_id_from_state",
                "_sequence_lag_features",
                "_lag_features",
                "_ridge_predict",
            ],
        ),
    ]
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output run already contains artifacts; choose a new source-of-truth run")
    out.mkdir(parents=True, exist_ok=True)
    logger.add(out / "evaluation.log", encoding="utf-8")
    snapshot = out / "config_snapshot.yaml"
    snapshot.write_bytes(config_path.read_bytes())
    gt_path = Path(config["output_root"]) / "bs.parquet"
    gt_manifest = Path(config["output_root"]) / "manifest.json"
    source_files = [
        config_path,
        pipeline_path,
        gt_path,
        gt_manifest,
        Path(__file__),
        Path("src/virtual_imu/eval/downstream_attitude.py"),
        Path("src/virtual_imu/eval/formal_evaluation.py"),
        Path("src/virtual_imu/eval/baselines.py"),
        historical_eval / "formal_evaluation.py",
        historical_eval / "baselines.py",
    ]
    contract = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "scope": "all historical paired methods; only held-out Penn bs; no downstream selection or refitting",
        "protocol": protocol,
        "source_files_sha256": {p.as_posix(): sha(p) for p in source_files},
        "registry": [
            {
                k: e.get(k)
                for k in [
                    "candidate_id",
                    "seed",
                    "local_run",
                    "run_dir",
                    "temporal_context",
                    "fidelity",
                    "reused",
                ]
            }
            for e in entries
        ],
        "classic_methods": list(CLASSIC_METHODS),
        "baseline_replay_source": replay_sources,
        "reference": "HIGH is a measured inertial reference, never the independent pose ground truth",
        "aggregation": "mean and sample SD over three optimizer seeds only; classical predictions evaluated once; no window confidence intervals",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract["contract_sha256"] = contract_sha
    write_json(out / "contract.json", contract)
    print(f"FROZEN_DOWNSTREAM_CONTRACT_SHA256={contract_sha}", flush=True)
    logger.info(
        "WRITES frozen config snapshot {}; SKIPS all model/clock/calibration fitting", snapshot
    )
    common = None
    common_fingerprint = None
    states_fingerprint = None
    state = None
    predictions, provenance = {}, []
    mapping = {"high": {"method": "HIGH", "seed": None, "category": "measured_reference"}}
    for index, entry in enumerate(entries):
        run_dir = Path(entry["local_run"])
        pred_path = run_dir / "evaluation/test_predictions.parquet"
        baseline_path = run_dir / "evaluation/baseline_train_state.json"
        run_config = run_dir / "config.yaml"
        logger.info("READS historical prediction {}", pred_path)
        frame = pd.read_parquet(pred_path)
        if len(frame) != 96320:
            raise ValueError("Historical paired test sample coverage differs from 96,320")
        group = (
            frame[frame.dataset_id.eq("penncosyvio")]
            .sort_values(KEYS, kind="stable")
            .reset_index(drop=True)
        )
        if (
            len(group) != 6560
            or group.duplicated(KEYS).any()
            or set(group.original_sequence_id) != {"bs"}
        ):
            raise ValueError("Historical Penn test support must be 6,560 unique bs samples")
        for column, value in PENN_IDENTITY.items():
            if not group[column].eq(value).all():
                raise ValueError("Unexpected Penn sensor mapping")
        group, recovered = recover_source_clock(group, Path(config["paired_root"]))
        physical_columns = [*KEYS, *IDENTITY, "source_timestamp_s", *LOW, *HIGH]
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(group[physical_columns], index=False).values.tobytes()
        ).hexdigest()
        if common_fingerprint is not None and fingerprint != common_fingerprint:
            raise ValueError(
                "Historical methods do not share exactly the same physical samples and raw six-axis values"
            )
        common_fingerprint = fingerprint
        source_clock_origin_s(group)
        if not np.isfinite(group[LOW + HIGH + VIRTUAL].to_numpy(dtype=float)).all():
            raise ValueError(
                "Nonfinite six-axis prediction; common support must not change per method"
            )
        run_cfg = yaml.safe_load(run_config.read_text(encoding="utf-8"))
        if run_cfg["seed"] != entry["seed"]:
            raise ValueError("Run config and registry seeds disagree")
        logger.info("READS already train-fitted baseline state {}", baseline_path)
        selected = select_penn_state(json.loads(baseline_path.read_text(encoding="utf-8")))
        state_digest = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        if states_fingerprint is not None and state_digest != states_fingerprint:
            raise ValueError("Historical deterministic Penn baseline states differ across runs")
        states_fingerprint, state = state_digest, selected
        if common is None:
            common = group.drop(columns=VIRTUAL).copy()
        prefix = f"neural_{index:02d}"
        predictions[prefix] = group[VIRTUAL].to_numpy(dtype=float)
        mapping[prefix] = {
            "method": entry["candidate_id"],
            "seed": entry["seed"],
            "category": "neural",
        }
        provenance.append(
            {
                "method": entry["candidate_id"],
                "seed": entry["seed"],
                "temporal_context": entry["temporal_context"],
                "fidelity": entry["fidelity"],
                "prediction_path": pred_path.as_posix(),
                "prediction_sha256": sha(pred_path),
                "config_path": run_config.as_posix(),
                "config_sha256": sha(run_config),
                "baseline_path": baseline_path.as_posix(),
                "baseline_sha256": sha(baseline_path),
                "clock_recovery": recovered,
                "physical_sample_sha256": fingerprint,
            }
        )
    if common is None or state is None:
        raise ValueError("No paired predictions")
    original_classic = predict_pair_baselines(state, common)
    classic = {
        ({"learned_fir_causal": "fir", "arx_linear": "arx"}).get(method, method): values
        for method, values in original_classic.items()
    }
    if set(classic) != set(CLASSIC_METHODS):
        raise ValueError(f"Unexpected replay baseline names: {sorted(classic)}")
    for method in CLASSIC_METHODS:
        predictions[method] = classic[method]
        mapping[method] = {"method": method, "seed": None, "category": "classical"}
    axes = ["gx", "gy", "gz", "ax", "ay", "az"]
    extra_columns = {}
    for prefix, values in predictions.items():
        if values.shape != (len(common), 6) or not np.isfinite(values).all():
            raise ValueError("Nonfinite or malformed replay signal")
        extra_columns.update(
            {f"{axis}_{prefix}": values[:, index] for index, axis in enumerate(axes)}
        )
    common = pd.concat([common, pd.DataFrame(extra_columns, index=common.index)], axis=1)
    # Keep all streams in one call: every method shares exactly the same support.
    prefixes = list(mapping)
    origin = source_clock_origin_s(common)
    logger.info("READS independent optical test GT {}; source clock origin {}s", gt_path, origin)
    truth = pd.read_parquet(gt_path)
    args = {
        "prefixes": prefixes,
        "window_samples": protocol["window_samples"],
        "rate_hz": protocol["rate_hz"],
        "max_gt_gap_s": config["max_gt_gap_s"],
        "gt_timestamp_origin_s": origin,
    }
    results = {
        "attitude": evaluate_attitude_stream(common, truth, **args),
        "strapdown": evaluate_strapdown_stream(
            common, truth, gravity_world_mps2=protocol["gravity_world_mps2"], **args
        ),
    }
    trial_rows, detail_rows, full_rows = [], [], []
    for task, result in results.items():
        for row in result["windows"]:
            translated = {**row, **mapping[row["method"]], "task": task}
            details = [
                d
                for d in result["window_details"]
                if d["method"] == row["method"] and d["samples"] == row["samples"]
            ]
            if task == "attitude":
                translated["endpoint_mean_deg"] = (
                    float(np.mean([d["endpoint_deg"] for d in details])) if details else None
                )
            trial_rows.append(translated)
        detail_rows.extend(
            {**row, **mapping[row["method"]], "task": task} for row in result["window_details"]
        )
        full_rows.extend(
            {**row, **mapping[row["method"]], "task": task} for row in result["full_recordings"]
        )
    trials = pd.DataFrame(trial_rows)
    summary = aggregate_trials(trials)
    for name, table in [
        ("trials", trials),
        ("summary", summary),
        ("window_details", pd.DataFrame(detail_rows)),
        ("full_segments", pd.DataFrame(full_rows)),
    ]:
        path = out / f"{name}.csv"
        table.to_csv(path, index=False)
        logger.info("WRITES {}", path)
        write_json(out / f"{name}.json", table.to_dict(orient="records"))
    signal_path = out / "evaluated_signals.parquet"
    common.to_parquet(signal_path, index=False)
    logger.info("WRITES auditable physical-time six-axis streams {}", signal_path)
    metrics = {
        "contract_sha256": contract_sha,
        "scope": "one held-out Penn bs recording only; optimization SD is not recording-level generalization or significance",
        "input_samples": len(common),
        "source_clock_origin_s": origin,
        "physical_sample_sha256": common_fingerprint,
        "penn_baseline_state_sha256": states_fingerprint,
        "methods": sorted({v["method"] for v in mapping.values()}),
        "neural_optimizer_runs": len(entries),
        "classical_evaluations": len(CLASSIC_METHODS),
        "groundtruth_provenance": gt_manifest.as_posix(),
        "input_provenance": provenance,
        "protocols": {task: result["protocol"] for task, result in results.items()},
        "support": {task: result["support"] for task, result in results.items()},
        "signal_columns": mapping,
        "warning": "All methods retained. Offline/full-window contexts differ in latency. No downstream model selection or test-fitted lag/rotation/bias. Optical GT about10cm position accuracy; rotation accuracy not numerically assumed. Noisier bf retained in source audit, but historical held-out evaluation contains bs only.",
    }
    for path_string, expected in contract["source_files_sha256"].items():
        if sha(Path(path_string)) != expected:
            raise ValueError("Frozen source/config changed during downstream evaluation")
    write_json(out / "metrics.json", metrics)
    lines = [
        "# Historical independent-GT downstream diagnostics",
        "",
        f"Frozen contract: `{contract_sha}`. Penn bs only, 6,560 common physical samples. Each fixed window initializes once from independent optical GT; no GT updates during propagation. Position includes oracle initial velocity from a past 0.1s difference.",
        "",
        "Mean ± sample SD describes optimization seeds only. Classical methods and HIGH are evaluated once. All methods are retained; no downstream winner is selected.",
        "",
    ]
    for task, metric, unit in [
        ("attitude", "endpoint_rmse_deg", "deg"),
        ("strapdown", "position_endpoint_rmse_m", "m"),
    ]:
        lines += [
            f"{task}: fixed-window endpoint RMSE ({unit})",
            "",
            "| Method | Optimizer runs | 1.975 s | 4.975 s | 9.975 s | 19.975 s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method in [*NEURAL_METHODS, *CLASSIC_METHODS, "HIGH"]:
            group = summary[(summary.task == task) & (summary.method == method)].sort_values(
                "samples"
            )
            values = [
                f"{row[f'{metric}_mean']:.5g} ± {row[f'{metric}_sd']:.3g}"
                for _, row in group.iterrows()
            ]
            lines.append(
                f"| {method} | {int(group.optimizer_runs.iloc[0])} | " + " | ".join(values) + " |"
            )
        lines.append("")
    lines += [
        "This is an oracle-initialized propagation diagnostic, not a complete deployable navigation system. One recording cannot establish all-scene generalization. Different temporal contexts carry different inference latency. Negative improvements and all seeds are retained."
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("WRITES {}; SKIPS ranking, model selection and all fitting", report)
    artifact_paths = [
        path
        for path in out.iterdir()
        if path.is_file() and path.name not in {"artifact_manifest.json", "evaluation.log"}
    ]
    write_json(
        out / "artifact_manifest.json",
        {
            "contract_sha256": contract_sha,
            "files": {path.name: sha(path) for path in artifact_paths},
        },
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pipeline_root, args.previous_root, args.config, args.out)
    print(
        json.dumps(
            {
                "methods": len(result["methods"]),
                "samples": result["input_samples"],
                "contract_sha256": result["contract_sha256"],
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
