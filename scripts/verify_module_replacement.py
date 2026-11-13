"""Independent supports, full132-file CUDA inference, and exact table audits.

Replay uses the original AMP setting and batch64. Physical outputs must match
with rtol=1e-7, atol=1e-10. No production window/evaluate/predict/report helper
is called. This CLI writes verification.json and never changes selections.
"""

# The experiment's pinned Python 3.9 runtime does not support zip(strict=...).
# ruff: noqa: B905
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from verify_cnn_bilstm_ablation import (  # noqa: E402
    AXES,
    HIGH,
    KEYS,
    LOW,
    check_close,
    read_json,
    replay_row,
    sha,
    support_hash,
)

REPLAY_RTOL = 1e-7
REPLAY_ATOL = 1e-10
CONDITION = ["protocol", "split", "variant", "window", "seed"]
SUPPORT = ["aggregation", "dataset_id", "pair_id", "original_sequence_id"]
IDENTITY = CONDITION + SUPPORT
GROUP = [key for key in IDENTITY if key != "seed"]
PHYSICAL = [
    f"{g}_{m}" for g in ("gyro", "accel") for m in ("rmse", "vector_rmse", "mae", "bias_norm")
]
PHYSICAL += [axis + "_rmse" for axis in AXES]
ORIGINS = ["reused", "source_run", "source_variant"]
FIT_COST = ["best_epoch", "actual_epochs", "elapsed_seconds", "amp_skipped_steps"]
INFERENCE_COST = ["batch1_ms_mean", "batch1_ms_median"]
CONTROL_MAP = {
    "cnn_lstm64": "cnn_lstm_wide",
    "cnn_bilstm32": "cnn_bilstm",
    "cnn_identity": "cnn_only",
}
ARCHIVE_IDENTITY = [
    "job_id",
    "split",
    "checkpoint_sha256",
    "initialization_sha256",
    "config_sha256",
    "support_sha256",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        digest.update(header.encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_csv(path):
    frame = pd.read_csv(path)
    for column in frame:
        if column.endswith(("job_id", "source_run", "source_variant")):
            frame[column] = frame[column].fillna("")
    return frame


def exact_table(actual, expected, keys, label):
    """Require a bijection of identities, then compare expected value columns."""
    require(set(expected.columns).issubset(actual.columns), label + " missing columns")
    require(not actual[keys].isna().any().any(), label + " empty identity")
    require(not actual.duplicated(keys).any(), label + " duplicate identities")
    require(not expected.duplicated(keys).any(), label + " duplicate expected identities")
    merged = expected.merge(
        actual[list(expected.columns)],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_expected", "_actual"),
    )
    require(merged._merge.eq("both").all(), label + " missing or extra identities")
    for column in expected.columns.difference(keys):
        actual_values, expected_values = merged[column + "_actual"], merged[column + "_expected"]
        if pd.api.types.is_numeric_dtype(expected[column]) and not pd.api.types.is_bool_dtype(
            expected[column]
        ):
            require(pd.api.types.is_numeric_dtype(actual_values), label + " nonnumeric " + column)
            check_close(actual_values.to_numpy(), expected_values.to_numpy(), label + " " + column)
        else:
            require(
                actual_values.fillna("").eq(expected_values.fillna("")).all(),
                label + " " + column + " differs",
            )


def support_layout(frame):
    rows = [
        dict(
            aggregation="overall",
            dataset_id="__all__",
            pair_id="__all__",
            original_sequence_id="__all__",
            samples=len(frame),
        )
    ]
    for aggregation, columns in [
        ("dataset", ["dataset_id"]),
        ("pair", ["dataset_id", "pair_id"]),
        ("recording", ["dataset_id", "pair_id", "original_sequence_id"]),
    ]:
        for identity, group in frame.groupby(columns, sort=False):
            values = identity if isinstance(identity, tuple) else (identity,)
            row = dict(
                aggregation=aggregation,
                dataset_id="__all__",
                pair_id="__all__",
                original_sequence_id="__all__",
                samples=len(group),
            )
            row.update(dict(zip(columns, values)))
            rows.append(row)
    return rows


def _expected_metrics(frame, jobs, supports, config):
    if jobs is None:
        jobs = [
            dict(
                id=f"{p}__{v}__w{w}__s{s}",
                protocol=p,
                variant=v,
                window=w,
                seed=s,
                reused=v in CONTROL_MAP,
            )
            for p, v, w, s in product(
                frame.protocol.unique(),
                frame.loc[frame.variant.ne("raw"), "variant"].unique(),
                frame.loc[frame.variant.ne("raw"), "window"].unique(),
                frame.loc[frame.variant.ne("raw"), "seed"].unique(),
            )
        ]
    if config is not None:
        expected = set(
            product(config["protocols"], config["variants"], config["windows"], config["seeds"])
        )
        actual = {(j["protocol"], j["variant"], j["window"], j["seed"]) for j in jobs}
        require(actual == expected and len(jobs) == len(expected), "Planned metric grid differs")
    require(len({j["id"] for j in jobs}) == len(jobs), "Duplicate planned jobs")
    layouts = {}
    for protocol, split in product(
        dict.fromkeys(j["protocol"] for j in jobs), ["validation", "test"]
    ):
        if supports is not None:
            require((protocol, split) in supports, "Missing independent support")
            layouts[(protocol, split)] = support_layout(supports[(protocol, split)])
        else:
            raw = frame[
                frame.protocol.eq(protocol) & frame.split.eq(split) & frame.variant.eq("raw")
            ]
            require(not raw.empty, "Missing Raw support")
            layouts[(protocol, split)] = raw[[*SUPPORT, "samples"]].to_dict("records")
    rows = []
    for (protocol, split), layout in layouts.items():
        treatments = [dict(id="", variant="raw", window=0, seed=-1, reused=False)]
        treatments += [j for j in jobs if j["protocol"] == protocol]
        for job in treatments:
            for support in layout:
                row = dict(
                    protocol=protocol,
                    split=split,
                    variant=job["variant"],
                    window=job["window"],
                    seed=job["seed"],
                    job_id=job["id"],
                    **support,
                )
                if "reused" in frame:
                    row["reused"] = job.get("reused", job["variant"] in CONTROL_MAP)
                rows.append(row)
    return pd.DataFrame(rows), jobs


def _origins(row, group, fields=ORIGINS):
    if set(fields).issubset(group.columns):
        for field in fields:
            require(group[field].nunique(dropna=False) == 1, "Mixed " + field + " within summary")
            row[field] = group[field].iloc[0]


def registry_provenance(registry, jobs):
    """Derive omitted source variants and authenticate cost/report origins."""
    registry = registry.copy()
    if "source_variant" not in registry:
        registry["source_variant"] = registry.variant.map(CONTROL_MAP).fillna("")
    required = ["job_id", "source_run", "source_job_id", "source_variant"]
    require(set(required).issubset(registry), "Missing native registry provenance")
    for column in required[1:]:
        registry[column] = registry[column].fillna("")
    expected = []
    for job in jobs:
        legacy = CONTROL_MAP.get(job["variant"], "")
        row = dict(
            job_id=job["id"],
            **{k: job[k] for k in ["protocol", "variant", "window", "seed", "reused"]},
            source_variant=legacy,
            source_job_id=(
                f"{job['protocol']}__{legacy}__w{job['window']}__s{job['seed']}"
                if job["reused"]
                else ""
            ),
            fit_kind="reused_control" if job["reused"] else "new_fit",
            training_cost_origin="prior_run" if job["reused"] else "current_run",
            job_config_sha256=json_digest(job),
        )
        expected.append(row)
    exact_table(registry, pd.DataFrame(expected), ["job_id"], "registry provenance")
    indexed = registry.set_index("job_id")
    for job in jobs:
        source = indexed.loc[job["id"], "source_run"]
        require(bool(source) == bool(job["reused"]), "Registry source-run reuse differs")
        if job["reused"]:
            require(
                "config" in job and "reuse" in job["config"], "Missing planned source-run identity"
            )
            base = Path(job.get("root", ROOT))
            planned = base / job["config"]["reuse"]["run_dir"]
            require(
                (base / source).resolve() == planned.resolve(),
                "Registry source run differs from plan",
            )
    return registry


def verify_tables(run_dir, metrics, *, jobs=None, supports=None, config=None):
    """Check metric and derived-table completeness, provenance and arithmetic."""
    run_dir = Path(run_dir)
    if jobs is None and (run_dir / "planned_jobs.json").is_file():
        jobs = read_json(run_dir / "planned_jobs.json")
    frame = metrics.copy()
    require(
        set([*IDENTITY, "job_id", "samples", *PHYSICAL]).issubset(frame),
        "Incomplete metrics schema",
    )
    frame["job_id"] = frame.job_id.fillna("")
    require(
        np.isfinite(frame[PHYSICAL]).all().all() and frame[PHYSICAL].ge(0).all().all(),
        "Invalid physical metrics",
    )
    require(frame.samples.gt(0).all() and frame.samples.mod(1).eq(0).all(), "Invalid sample counts")
    expected_metrics, jobs = _expected_metrics(frame, jobs, supports, config)
    exact_table(frame, expected_metrics, IDENTITY, "metrics")
    if (run_dir / "fit_registry.csv").is_file():
        registry = registry_provenance(read_csv(run_dir / "fit_registry.csv"), jobs)
        if set(ORIGINS).issubset(registry):
            frame = frame.drop(columns=[c for c in [*ORIGINS, "source_job_id"] if c in frame])
            fields = ["job_id", *ORIGINS] + (
                ["source_job_id"] if "source_job_id" in registry else []
            )
            frame = frame.merge(registry[fields], on="job_id", how="left", validate="many_to_one")
            frame.loc[frame.variant.eq("raw"), "reused"] = False
            for field in fields[1:]:
                if field != "reused":
                    frame[field] = frame[field].fillna("")
    original_score = frame.get("normalized_score")
    frame["normalized_score"] = np.nan
    raw = frame[frame.variant.eq("raw") & frame.aggregation.eq("pair")]
    baseline = {
        (r.protocol, r.split, r.dataset_id, r.pair_id): (r.gyro_rmse, r.accel_rmse)
        for r in raw.itertuples()
    }
    for _, indexes in frame.groupby(CONDITION, sort=False).groups.items():
        group, datasets = frame.loc[indexes], {}
        for index, row in group[group.aggregation.eq("pair")].iterrows():
            g, a = baseline[(row.protocol, row.split, row.dataset_id, row.pair_id)]
            value = 0.5 * (row.gyro_rmse / max(g, 1e-12) + row.accel_rmse / max(a, 1e-12))
            frame.loc[index, "normalized_score"] = value
            datasets.setdefault(row.dataset_id, []).append(value)
        require(bool(datasets), "Missing pair metrics")
        for index, row in group[group.aggregation.eq("dataset")].iterrows():
            frame.loc[index, "normalized_score"] = np.mean(datasets[row.dataset_id])
        frame.loc[group[group.aggregation.eq("overall")].index, "normalized_score"] = np.mean(
            [np.mean(v) for v in datasets.values()]
        )
    if original_score is not None:
        check_close(original_score, frame.normalized_score, "stored normalized score")
    extra = [c for c in [*ORIGINS, "source_job_id"] if c in frame]
    long = frame.melt(
        id_vars=[*IDENTITY, "samples", "job_id", *extra],
        value_vars=[*PHYSICAL, "normalized_score"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    keys, summary_rows = [*GROUP, "metric"], []
    for identity, group in long.groupby(keys, sort=False):
        row = dict(zip(keys, identity))
        require(group.samples.nunique() == 1, "Unequal seed supports")
        row.update(
            mean=group.value.mean(),
            sd=group.value.std(ddof=1),
            n=group.seed.nunique(),
            samples=group.samples.iloc[0],
        )
        _origins(row, group)
        summary_rows.append(row)
    expected_summary = pd.DataFrame(summary_rows)
    exact_table(read_csv(run_dir / "summary.csv"), expected_summary, keys, "summary")
    lookup = {tuple(row[k] for k in [*IDENTITY, "metric"]): row for row in long.to_dict("records")}
    paired_rows = []
    for row in long[~long.variant.isin(["raw", "cnn_lstm64"])].to_dict("records"):
        reference = lookup.get(
            tuple(({**row, "variant": "cnn_lstm64"})[k] for k in [*IDENTITY, "metric"])
        )
        require(
            reference is not None and row["samples"] == reference["samples"],
            "Missing paired reference",
        )
        item = dict(row)
        for field in ["variant", "value", "samples", "job_id", *extra]:
            item["reference_" + field] = reference[field]
        item.update(
            reference_seed=row["seed"],
            reference_window=row["window"],
            delta=row["value"] - reference["value"],
        )
        item["relative_delta_pct"] = (
            100 * item["delta"] / reference["value"] if reference["value"] else np.nan
        )
        paired_rows.append(item)
    paired = pd.DataFrame(paired_rows)
    require(not paired.empty, "Missing paired comparisons")
    exact_table(
        read_csv(run_dir / "paired_deltas.csv"), paired, [*IDENTITY, "metric"], "paired per-seed"
    )
    rows = []
    for identity, group in paired.groupby(keys, sort=False):
        delta, relative = group.delta, group.relative_delta_pct.dropna()
        row = dict(zip(keys, identity))
        row.update(
            delta_mean=delta.mean(),
            delta_sd=delta.std(ddof=1),
            n=group.seed.nunique(),
            relative_delta_pct_mean=relative.mean(),
            relative_delta_pct_sd=relative.std(ddof=1),
            relative_n=len(relative),
            worse_seeds=int(delta.gt(0).sum()),
            better_seeds=int(delta.lt(0).sum()),
            tied_seeds=int(delta.eq(0).sum()),
            samples=group.samples.iloc[0],
            reference_samples=group.reference_samples.iloc[0],
            reference_variant="cnn_lstm64",
            reference_window=group.reference_window.iloc[0],
        )
        _origins(row, group)
        _origins(row, group, ["reference_reused", "reference_source_run"])
        rows.append(row)
    expected_pairs = pd.DataFrame(rows)
    exact_table(read_csv(run_dir / "paired_summary.csv"), expected_pairs, keys, "paired summary")
    ranking_rows = []
    val = frame[
        frame.split.eq("validation") & frame.aggregation.eq("overall") & frame.variant.ne("raw")
    ]
    for identity, group in val.groupby(["protocol", "variant", "window"], sort=False):
        row = dict(zip(["protocol", "variant", "window"], identity))
        row.update(
            mean=group.normalized_score.mean(),
            sd=group.normalized_score.std(ddof=1),
            n=group.seed.nunique(),
            selection_split="validation",
            reused=identity[1] in CONTROL_MAP,
        )
        _origins(row, group)
        ranking_rows.append(row)
    ranking = pd.DataFrame(ranking_rows)
    ranking["rank"] = ranking.groupby("protocol")["mean"].rank(method="min").astype(int)
    exact_table(
        read_csv(run_dir / "validation_ranking.csv"),
        ranking,
        ["protocol", "variant", "window"],
        "VALID ranking",
    )
    return dict(
        summary_rows=len(expected_summary),
        paired_rows=len(paired),
        paired_summary_rows=len(expected_pairs),
        validation_rank_rows=len(ranking),
        metric_rows=len(frame),
    ), frame


def verify_compute(run_dir, *, jobs=None):
    """Check every planned per-seed cost and all seven cost summaries."""
    run_dir = Path(run_dir)
    jobs = read_json(run_dir / "planned_jobs.json") if jobs is None else jobs
    registry, cost, table = [
        read_csv(run_dir / name)
        for name in ["fit_registry.csv", "compute_cost.csv", "compute_summary.csv"]
    ]
    registry = registry_provenance(registry, jobs)
    expected = pd.DataFrame(
        [
            dict(
                job_id=j["id"],
                **{k: j[k] for k in ["protocol", "variant", "window", "seed", "reused"]},
            )
            for j in jobs
        ]
    )
    exact_table(registry, expected, ["job_id"], "fit registry")
    exact_table(cost, expected, ["job_id"], "compute costs")
    require(
        set(["parameters", *FIT_COST, "training_cost_origin", "source_run"]).issubset(registry),
        "Missing fit costs",
    )
    require(
        set(["parameters", *INFERENCE_COST, "latency_origin"]).issubset(cost),
        "Missing inference costs",
    )
    for field in ["parameters", *FIT_COST]:
        require(
            np.isfinite(registry[field]).all() and registry[field].ge(0).all(),
            "Invalid fit cost " + field,
        )
    for field in ["parameters", "best_epoch", "actual_epochs", "amp_skipped_steps"]:
        require(registry[field].mod(1).eq(0).all(), "Noninteger fit cost " + field)
    require(
        registry.parameters.gt(0).all()
        and registry.best_epoch.ge(1).all()
        and registry.best_epoch.le(registry.actual_epochs).all()
        and registry.actual_epochs.le(80).all(),
        "Invalid fit epochs/parameters",
    )
    for field in ["parameters", *INFERENCE_COST]:
        require(
            np.isfinite(cost[field]).all() and cost[field].gt(0).all(),
            "Invalid inference cost " + field,
        )
    require(cost.latency_origin.eq("current_run_uniform_measurement").all(), "Wrong latency origin")
    fits, rows = registry.set_index("job_id"), []
    for item in cost.to_dict("records"):
        fit = fits.loc[item["job_id"]]
        require(
            item["parameters"] == fit["parameters"], "Parameter count differs between fit and cost"
        )
        require(
            fit["training_cost_origin"] == ("prior_run" if item["reused"] else "current_run"),
            "Training origin differs",
        )
        require(bool(fit["source_run"]) == bool(item["reused"]), "Source-run identity differs")
        for metric in ["parameters", *INFERENCE_COST, *FIT_COST]:
            row = {k: item[k] for k in ["protocol", "variant", "window", "seed", "reused"]}
            row.update(
                source_run=fit["source_run"],
                training_cost_origin=fit["training_cost_origin"],
                metric=metric,
                value=fit[metric] if metric in FIT_COST else item[metric],
                cost_scope="architecture"
                if metric == "parameters"
                else (
                    "uniform_new_inference"
                    if metric in INFERENCE_COST
                    else ("source_training" if item["reused"] else "new_training")
                ),
            )
            rows.append(row)
    long, summaries = pd.DataFrame(rows), []
    keys = ["protocol", "variant", "window", "metric"]
    for identity, group in long.groupby(keys, sort=False):
        row = dict(zip(keys, identity))
        row.update(mean=group.value.mean(), sd=group.value.std(ddof=1), n=group.seed.nunique())
        _origins(row, group, ["reused", "source_run", "training_cost_origin", "cost_scope"])
        if identity[-1] == "parameters":
            require(group.value.nunique() == 1, "Architecture capacity varies across seeds")
        summaries.append(row)
    exact_table(table, pd.DataFrame(summaries), keys, "compute summary")
    return len(summaries)


def reconstruct_windows(frames, *, role, stats=None, include_inputs=True):
    """Independent per-sequence320/stride40/center40 reconstruction."""
    require(role in {"train", "validation", "test"}, "Unknown reconstruction role")
    pieces = []
    for original in frames:
        if len(original) < 320:
            continue
        keep = len(original) if role == "train" else len(original) // 40 * 40
        pieces.append(original.iloc[:keep].copy())
    require(bool(pieces), "No eligible source recordings")
    frame = pd.concat(pieces, ignore_index=True)
    low = frame[LOW].to_numpy(np.float64)
    residual = frame[HIGH].to_numpy(np.float64) - low
    require(np.isfinite(low).all() and np.isfinite(residual).all(), "Nonfinite source signals")
    if stats is None:
        require(role == "train", "Only TRAIN may fit statistics")
        stats = {}
        for name, values in [("feature", low), ("residual", residual)]:
            std = values.std(0)
            stats[name + "_mean"] = values.mean(0).tolist()
            stats[name + "_std"] = np.where(std < 1e-8, 1.0, std).tolist()
    for name in ["feature_mean", "feature_std", "residual_mean", "residual_std"]:
        values = np.asarray(stats[name])
        require(values.shape == (6,) and np.isfinite(values).all(), "Invalid TRAIN statistics")
        if name.endswith("std"):
            require((values > 0).all(), "Nonpositive TRAIN standard deviations")
    starts, targets, labels = [], [], []
    for _, group in frame.groupby("sequence_id", sort=False):
        indices = group.index.to_numpy()
        require(
            np.all(np.diff(indices) == 1) and group.dataset_id.nunique() == 1,
            "Noncontiguous source sequence",
        )
        require(np.all(np.diff(group.timestamp_s.to_numpy()) > 0), "Non-increasing source time")
        for start in range(int(indices[0]), int(indices[-1]) + 2 - 320, 40):
            starts.append(start)
            targets.extend(range(start + 140, start + 180))
            labels.append(group.dataset_id.iloc[0])
    require(bool(starts), "No complete source windows")
    target = frame.iloc[targets].reset_index(drop=True)
    require(not target.duplicated(KEYS).any(), "Overlapping source targets")
    inputs = None
    if include_inputs:
        normalized = ((low - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
        inputs = np.ascontiguousarray(np.stack([normalized[s : s + 320].T for s in starts]))
    return dict(inputs=inputs, target_frame=target, stats=stats, labels=np.asarray(labels))


def independent_forward(model, inputs, target_frame, stats, *, device, amp, batch_size=64):
    """Direct forward and independent residual-to-physical conversion."""
    device = torch.device(device)
    require(batch_size == 64, "Replay must preserve original batch64")
    require(inputs.ndim == 3 and inputs.shape[1:] == (6, 320), "Wrong replay input shape")
    require(
        len(target_frame) == len(inputs) * 40 and np.isfinite(inputs).all(),
        "Incomplete replay support",
    )
    require(not amp or device.type == "cuda", "AMP replay requires CUDA")
    model = model.to(device).eval()
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(inputs[start : start + batch_size])).to(
                device
            )
            with torch.autocast(device_type=device.type, enabled=amp):
                output = model(batch)
            require(
                output.shape == batch.shape and torch.isfinite(output).all(),
                "Invalid checkpoint inference",
            )
            pieces.append(output[..., 140:180].float().cpu().numpy())
    normalized = np.concatenate(pieces).transpose(0, 2, 1).reshape(-1, 6).astype(np.float64)
    return (
        normalized * stats["residual_std"]
        + stats["residual_mean"]
        + target_frame[LOW].to_numpy(np.float64)
    )


def compare_predictions(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(
        actual.shape == expected.shape
        and actual.ndim == 2
        and actual.shape[1] == 6
        and len(actual) > 0,
        "Replayed prediction shape differs",
    )
    require(
        np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite replay comparison"
    )
    difference = np.abs(actual - expected)
    require(
        np.allclose(actual, expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
        f"Checkpoint inference differs from exported prediction; max_abs={difference.max():.9g}",
    )
    return dict(
        samples=len(actual),
        elements=int(actual.size),
        max_abs_error=float(difference.max()),
        max_relative_error=float((difference / np.maximum(np.abs(expected), REPLAY_ATOL)).max()),
        rtol=REPLAY_RTOL,
        atol=REPLAY_ATOL,
    )


def replay_prediction_file(path, model, rebuilt, stats, identity, *, device, amp, batch_size=64):
    require(set(identity) == set(ARCHIVE_IDENTITY), "Incomplete expected NPZ identity")
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(["prediction", *ARCHIVE_IDENTITY]).issubset(archive.files),
            "Missing NPZ identity fields",
        )
        for key in ARCHIVE_IDENTITY:
            value = archive[key]
            require(
                value.ndim == 0 and str(value.item()) == identity[key],
                "NPZ " + key + " differs from lock",
            )
        prediction = archive["prediction"]
    direct = independent_forward(
        model,
        rebuilt["inputs"],
        rebuilt["target_frame"],
        stats,
        device=device,
        amp=amp,
        batch_size=batch_size,
    )
    result = compare_predictions(prediction, direct)
    return {
        **result,
        **identity,
        "prediction_file_sha256": sha(path),
        "batch_size": batch_size,
        "device": str(device),
        "amp": bool(amp),
        "coverage": "all_samples",
    }


def verify_lock_identity(run_dir, jobs, lock, access):
    run_dir = Path(run_dir)
    ids = {j["id"] for j in jobs}
    require(
        len(ids) == len(jobs) and lock.get("test_loaded") is False,
        "Invalid immutable selection snapshot",
    )
    require(lock.get("config_sha256") == sha(run_dir / "config.yaml"), "Lock config hash differs")
    require(lock.get("jobs_sha256") == json_digest(jobs), "Lock planned jobs differ")
    require(
        lock.get("analysis_plan_sha256") == sha(run_dir / "analysis_plan.json"),
        "Lock analysis hash differs",
    )
    for field in ["checkpoints_sha256", "initialization_sha256"]:
        require(set(lock.get(field, {})) == ids, "Incomplete lock " + field)
    require(
        lock.get("new_fits") == sum(not j["reused"] for j in jobs)
        and lock.get("reused_controls") == sum(j["reused"] for j in jobs),
        "Lock reuse counts differ",
    )
    for job in jobs:
        directory = Path(job["directory"])
        require(
            sha(directory / "best.pt") == lock["checkpoints_sha256"][job["id"]],
            "Locked checkpoint changed",
        )
        require(
            sha(directory / "initialization.json") == lock["initialization_sha256"][job["id"]],
            "Locked initialization changed",
        )
    require(
        isinstance(access, dict) and access.get("numerical_TEST_access") is True,
        "Missing durable TEST access",
    )
    require(
        access.get("selection_lock_sha256") == sha(run_dir / "selections.lock.json"),
        "TEST access lock differs",
    )
    try:
        timestamp = datetime.fromisoformat(access["first_access_utc"].replace("Z", "+00:00"))
        require(timestamp.tzinfo is not None, "TEST access must have a timezone")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise AssertionError("Invalid first TEST access timestamp") from error


def _verify_jobs(cfg, jobs, run_dir):
    expected = set(product(cfg["protocols"], cfg["variants"], [320], cfg["seeds"]))
    require(len(jobs) == len(expected) == 66, "Incomplete planned matrix")
    require(
        {(j["protocol"], j["variant"], j["window"], j["seed"]) for j in jobs} == expected,
        "Planned matrix differs",
    )
    for job in jobs:
        name = f"{job['protocol']}__{job['variant']}__w320__s{job['seed']}"
        require(job["id"] == name and job["config"] == cfg, "Planned job configuration differs")
        require(
            Path(job["directory"]).resolve() == (run_dir / "fits" / name).resolve(),
            "Planned fit directory differs",
        )
        require(job["reused"] == (job["variant"] in CONTROL_MAP), "Planned reuse identity differs")
    expected_files = {
        j["id"] + "__" + role + ".npz" for j in jobs for role in ["validation", "test"]
    }
    require(
        {p.name for p in (run_dir / "predictions").glob("*.npz")} == expected_files,
        "Incomplete or extra prediction files",
    )


def verify(run_dir):
    """Formal read-only CUDA replay after all fits, exports and reporting."""
    run_dir = Path(run_dir).resolve()
    torch.set_num_threads(1)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    require(
        cfg["training"]["device"] == "cuda" and cfg["evaluation"]["batch_size"] == 64,
        "Formal replay requires the original CUDA/batch64 recipe",
    )
    require(torch.cuda.is_available(), "CUDA unavailable for original-precision replay")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(cfg["training"].get("deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = False
    device, amp = torch.device("cuda"), bool(cfg["training"]["amp"])
    prov, plan, jobs, done, lock, access = [
        read_json(run_dir / name)
        for name in [
            "provenance.json",
            "analysis_plan.json",
            "planned_jobs.json",
            "training.complete.json",
            "selections.lock.json",
            "test_access.json",
        ]
    ]
    _verify_jobs(cfg, jobs, run_dir)
    # Run this gate before any original TEST payload is opened.
    verify_lock_identity(run_dir, jobs, lock, access)
    gate_sha256 = {
        name: sha(run_dir / name) for name in ["selections.lock.json", "test_access.json"]
    }
    require(
        done["planned"] == done["completed"] == 66
        and done["new_fits"] == 48
        and done["reused_controls"] == 18,
        "Incomplete training receipt",
    )
    require(plan["test_unlocked"] is False, "Analysis was not frozen before TEST")
    require(
        sha(run_dir / "config.yaml") == prov["config_sha256"]
        and sha(ROOT / cfg["data"]["manifest"]) == prov["manifest_sha256"],
        "Frozen configuration/data differs",
    )
    for receipt, folder in [(prov, "source_snapshot"), (plan, "analysis_source")]:
        for relative, expected in receipt["source_sha256"].items():
            require(
                sha(ROOT / relative) == sha(run_dir / folder / relative) == expected,
                "Frozen source differs: " + relative,
            )
    old = ROOT / cfg["reuse"]["run_dir"]
    require(
        sha(old / "complete.json") == prov["reused_run_complete_sha256"],
        "Old completion seal differs",
    )
    manifest = read_json(ROOT / cfg["data"]["manifest"])
    supports, rebuilt, stats_by_protocol, labels, input_hashes = {}, {}, {}, {}, {}
    for protocol in cfg["protocols"]:
        directory = run_dir / "supports" / protocol
        records = read_json(directory / "records.json")
        eligible = [
            r
            for r in manifest["records"]
            if r.get("task") == "paired_signal"
            and r.get("training_eligible") is True
            and r["dataset_id"] in ({"alimu", "penncosyvio"} if protocol == "pooled" else {"alimu"})
        ]
        expected_records = {
            role: [r for r in eligible if r["split"] == role]
            for role in ["train", "validation", "test"]
        }
        require(records == expected_records, "Support records differ from the pinned manifest")
        physical = {role: {r["physical_group_id"] for r in rows} for role, rows in records.items()}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            require(not physical[a] & physical[b], "Physical recording leakage")
        for role in ["train", "validation", "test"]:
            frames = []
            for record in records[role]:
                path = ROOT / record["path"]
                digest = sha(path)
                require(digest == record["sha256"], "Original data hash differs")
                input_hashes[record["path"]] = digest
                if record["rows"] < 320:
                    continue
                raw = (
                    pd.read_parquet(path)
                    .sort_values("timestamp_s", kind="stable")
                    .reset_index(drop=True)
                )
                require(len(raw) == record["rows"], "Original row count differs")
                require(
                    np.allclose(np.diff(raw.timestamp_s), 1 / 40, atol=1e-6, rtol=1e-5),
                    "Original sampling rate differs",
                )
                frames.append(raw)
            result = reconstruct_windows(
                frames,
                role=role,
                stats=stats_by_protocol.get(protocol),
                include_inputs=role != "train",
            )
            if role == "train":
                stats = read_json(directory / "statistics.json")
                for name, value in result["stats"].items():
                    check_close(stats[name], value, "Independent TRAIN " + name)
                stats_by_protocol[protocol], labels[protocol] = stats, result["labels"]
            saved = pd.read_parquet(directory / (role + ".parquet"))
            pd.testing.assert_frame_equal(
                saved[KEYS], result["target_frame"][KEYS], check_dtype=False
            )
            check_close(
                saved[LOW + HIGH],
                result["target_frame"][LOW + HIGH],
                "Original support values",
                atol=0,
                rtol=0,
            )
            supports[(protocol, role)], rebuilt[(protocol, role)] = result["target_frame"], result
        print("REPLAY source/windows/TRAIN-statistics PASS " + protocol, flush=True)

    from virtual_imu.ablations.models import build_ablation_model
    from virtual_imu.replacements.models import build_replacement_model

    metrics = read_csv(run_dir / "metrics_per_seed.csv")
    counts, scored = verify_tables(run_dir, metrics, jobs=jobs, supports=supports, config=cfg)
    counts["compute_summary_rows"] = verify_compute(run_dir, jobs=jobs)
    registry = read_csv(run_dir / "fit_registry.csv").set_index("job_id")
    exact_table(
        pd.DataFrame(done["fits"]), registry.reset_index(), ["job_id"], "training complete registry"
    )
    replayed, replays, init_by_seed, order_cache = 0, [], {}, {}
    for protocol, role in product(cfg["protocols"], ["validation", "test"]):
        frame = supports[(protocol, role)]
        for _, row in metrics[
            metrics.protocol.eq(protocol) & metrics.split.eq(role) & metrics.variant.eq("raw")
        ].iterrows():
            replay_row(frame, frame[LOW].to_numpy(float), row)
            replayed += 1
    for index, job in enumerate(jobs, 1):
        directory, identity = Path(job["directory"]), job["id"]
        fit, init = read_json(directory / "fit.json"), read_json(directory / "initialization.json")
        require(
            read_json(directory / "job.json") == job
            and fit["job_config_sha256"] == json_digest(job),
            "Fit job identity differs",
        )
        require(
            sha(directory / "best.pt")
            == fit["checkpoint_sha256"]
            == lock["checkpoints_sha256"][identity],
            "Checkpoint checksum differs",
        )
        saved = torch.load(directory / "best.pt", map_location="cpu")
        require(
            all(torch.isfinite(t).all() for t in saved["model"].values()),
            "Nonfinite checkpoint state",
        )
        history = read_csv(directory / "history.csv")
        require(
            1 <= len(history) == fit["actual_epochs"] <= 80
            and history.epoch.tolist() == list(range(1, len(history) + 1)),
            "Epoch history differs",
        )
        require(
            np.isfinite(history[["train_loss", "val_score", "lr"]]).all().all(),
            "Nonfinite fitting history",
        )
        require(
            fit["best_epoch"]
            == saved["epoch"]
            == int(history.loc[history.val_score.idxmin(), "epoch"]),
            "Checkpoint epoch is not VALID minimum",
        )
        check_close(history.val_score.min(), fit["best_val_score"], "Fit VALID minimum")
        check_close(saved["val_score"], fit["best_val_score"], "Native checkpoint VALID minimum")
        require(
            saved["stats"]
            == stats_by_protocol[job["protocol"]]
            == read_json(directory / "stats.json"),
            "Checkpoint TRAIN statistics differ",
        )
        for role in ["train", "validation"]:
            require(
                saved[role + "_support_sha256"]
                == fit[role + "_support_sha256"]
                == support_hash(supports[(job["protocol"], role)]),
                "Checkpoint support differs",
            )
        require(
            fit["amp_skipped_steps"] == history.amp_skipped_steps.sum(),
            "AMP skipped-step count differs",
        )
        model, meta = build_replacement_model(
            job["variant"], job["seed"], cfg["model"], cfg["replacement_models"]
        )
        require(
            meta == init and state_digest(model.state_dict()) == init["init_state_sha256"],
            "Initialization replay differs",
        )
        require(
            fit["parameters"]
            == registry.loc[identity, "parameters"]
            == sum(p.numel() for p in model.parameters()),
            "Fit capacity differs",
        )
        reference, _ = build_ablation_model("cnn_lstm_wide", job["seed"], cfg["model"])
        for key in init["retained_keys"]:
            require(
                torch.equal(model.state_dict()[key], reference.state_dict()[key]),
                "Retained initialization differs",
            )
        del reference
        model.load_state_dict(saved["model"], strict=True)
        if job["reused"]:
            receipt = read_json(directory / "import.json")
            require(
                receipt["training_performed"] is False and saved["job"] == receipt["source_job"],
                "Native import identity differs",
            )
            native_init = read_json(directory / "source_initialization.json")
            require(
                saved["metadata"] == native_init
                and init["init_state_sha256"] == native_init["init_state_sha256"],
                "Native import initialization differs",
            )
            source_id = f"{job['protocol']}__{CONTROL_MAP[job['variant']]}__w320__s{job['seed']}"
            require(
                saved["job"]["id"] == source_id
                and saved["job"]["variant"] == CONTROL_MAP[job["variant"]]
                and saved["job"]["window"] == 320,
                "Native source variant/window differs",
            )
            for path, expected in receipt["source_sha256"].items():
                require(sha(path) == expected, "Native source artifact changed")
            require(
                sha(directory / "best.pt")
                == receipt["source_fit"]["checkpoint_sha256"]
                == sha(old / "fits" / source_id / "best.pt"),
                "Imported bytes differ from source",
            )
            require(fit["training_cost_origin"] == "prior_run", "Imported cost origin differs")
        else:
            require(
                saved["job"] == job
                and init == saved["metadata"]
                and fit["training_cost_origin"] == "current_run",
                "New-fit identity differs",
            )
        key = (job["variant"], job["seed"])
        require(
            key not in init_by_seed or init_by_seed[key] == init,
            "Cross-protocol initialization differs",
        )
        init_by_seed[key] = init
        key = (job["protocol"], job["seed"])
        if key not in order_cache:
            lab = labels[job["protocol"]]
            dataset_counts = {d: np.count_nonzero(lab == d) for d in set(lab)}
            weights = torch.tensor([1 / dataset_counts[d] for d in lab], dtype=torch.float64)
            cumulative, values = hashlib.sha256(), []
            for epoch in range(1, 81):
                order = torch.multinomial(
                    weights,
                    len(lab),
                    replacement=True,
                    generator=torch.Generator().manual_seed(job["seed"] + epoch * 1000003),
                ).numpy()
                cumulative.update(order.tobytes())
                values.append((hashlib.sha256(order.tobytes()).hexdigest(), cumulative.hexdigest()))
            order_cache[key] = values
        orders = order_cache[key][: len(history)]
        require(
            history.order_sha256.tolist() == [item[0] for item in orders]
            and fit["order_sha256"] == orders[-1][1],
            "Minibatch order replay differs",
        )
        for role in ["validation", "test"]:
            frame, reconstruction = (
                supports[(job["protocol"], role)],
                rebuilt[(job["protocol"], role)],
            )
            expected_identity = dict(
                job_id=identity,
                split=role,
                checkpoint_sha256=lock["checkpoints_sha256"][identity],
                initialization_sha256=lock["initialization_sha256"][identity],
                config_sha256=lock["config_sha256"],
                support_sha256=support_hash(frame),
            )
            path = run_dir / "predictions" / (identity + "__" + role + ".npz")
            replays.append(
                replay_prediction_file(
                    path,
                    model,
                    reconstruction,
                    stats_by_protocol[job["protocol"]],
                    expected_identity,
                    device=device,
                    amp=amp,
                    batch_size=64,
                )
            )
            with np.load(path, allow_pickle=False) as archive:
                prediction = archive["prediction"]
            for _, row in metrics[metrics.job_id.eq(identity) & metrics.split.eq(role)].iterrows():
                replay_row(frame, prediction, row)
                replayed += 1
        score = scored[
            scored.job_id.eq(identity)
            & scored.split.eq("validation")
            & scored.aggregation.eq("overall")
        ].normalized_score.iloc[0]
        check_close(score, fit["best_val_score"], "VALID physical replay", rtol=1e-7, atol=1e-8)
        del model, saved
        print(f"REPLAY checkpoint/inference PASS {index}/66 ({2 * index}/132 files)", flush=True)
    require(
        len(replays) == 132 and len({(r["job_id"], r["split"]) for r in replays}) == 132,
        "Incomplete independent inference coverage",
    )
    require(
        read_json(run_dir / "selections.lock.json") == lock
        and read_json(run_dir / "test_access.json") == access
        and all(sha(run_dir / name) == expected for name, expected in gate_sha256.items()),
        "Selection lock or TEST access receipt changed during replay",
    )
    verify_lock_identity(run_dir, jobs, lock, access)
    for relative, expected in plan["source_sha256"].items():
        require(sha(ROOT / relative) == expected, "Analysis changed during replay")
    outputs = {
        p.relative_to(run_dir).as_posix(): sha(p)
        for p in run_dir.rglob("*")
        if p.is_file()
        and p.suffix != ".log"
        and p.name
        not in [
            "verification.json",
            "independent_review.json",
            "complete.json",
            "artifact_manifest.json",
        ]
    }
    receipt = dict(
        status="PASS",
        conditions=66,
        new_fits=48,
        reused_controls=18,
        prediction_files=132,
        independent_inference_files=len(replays),
        independent_inference_all_samples=True,
        inference_device=str(device),
        inference_amp=amp,
        inference_batch_size=64,
        prediction_comparison=dict(rtol=REPLAY_RTOL, atol=REPLAY_ATOL, units="physical"),
        prediction_replays=replays,
        metric_rows_replayed=replayed,
        data_files=len(input_hashes),
        data_sha256=input_hashes,
        source_files=len(prov["source_sha256"]),
        supports_recomputed_from_sources=True,
        TRAIN_only_statistics=True,
        checkpoint_VALID_minima=True,
        common_seed_minibatch_orders=True,
        exact_retained_initialization=True,
        native_import_identity_preserved=True,
        exact_table_identity_coverage=True,
        immutable_TEST_access_verified=True,
        gate_sha256=gate_sha256,
        **counts,
        output_sha256=outputs,
        verifier_sha256=sha(__file__),
    )
    (run_dir / "verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "WRITES verification.json; PASS66 conditions /132 full checkpoint inference replays",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        "READS locked checkpoints, original paired sources and exports; WRITES numerical verification; SKIPS training, selection changes and sampled inference",
        flush=True,
    )
    verify(args.run_dir.resolve())
