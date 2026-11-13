"""Report the frozen CNN-LSTM module replacement matrix, without data/model IO.

API: report(run_dir: Path) -> JSON-compatible reporting receipt.
CLI: python scripts/report_module_replacement.py --run-dir runs/<experiment_id>

Numerics implement the prepare_metrics/summarize_metrics contract of the sealed
report_cnn_bilstm_ablation.py independently. Keeping the small numerical
equivalent here makes every executed analysis dependency part of the pipeline's
analysis_plan.json freeze (no import of an unfrozen legacy reporting script).
Physical pooled errors are inputs, never reconstructed by averaging RMSEs.
Normalized score = equal datasets / equal pairs / mean gyro+accel Raw ratios.
SD is sample SD of optimization seeds, not a recording confidence interval.

CSV contract: paired_deltas has the complete metric identity, value,
reference_value, delta, relative_delta_pct, samples/reference_samples and exact
reference_seed/window/variant/job_id. Positive delta means worse. A zero
reference produces NaN percentage. paired_summary groups by identity minus seed
plus metric, and exports delta_mean/delta_sd/n,
relative_delta_pct_mean/relative_delta_pct_sd/relative_n,
worse_seeds/better_seeds/tied_seeds. validation_ranking exports
protocol/variant/window/mean/sd/n/rank/selection_split. Every table keeps reused.
"""

# Experiment runtime is Python 3.9; Chinese prose intentionally uses Chinese punctuation.
# ruff: noqa: B905, RUF001, UP017
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = logging.getLogger(__name__)
ALL = "__all__"
FIT_KEYS = ["protocol", "variant", "window", "seed"]
CONDITION = ["protocol", "split", "variant", "window", "seed"]
SUPPORT = ["aggregation", "dataset_id", "pair_id", "original_sequence_id"]
IDENTITY = CONDITION + SUPPORT
GROUP = [key for key in IDENTITY if key != "seed"]
PHYSICAL = [
    f"{g}_{m}" for g in ("gyro", "accel") for m in ("rmse", "vector_rmse", "mae", "bias_norm")
]
PHYSICAL += [f"{a}_rmse" for a in ("gx", "gy", "gz", "ax", "ay", "az")]
METRICS = [*PHYSICAL, "normalized_score"]
ORIGIN = ["reused", "source_run", "source_variant"]
FIT_COST = ["best_epoch", "actual_epochs", "elapsed_seconds", "amp_skipped_steps"]
INFERENCE_COST = ["batch1_ms_mean", "batch1_ms_median"]
ANALYSIS_SOURCES = (
    "scripts/report_module_replacement.py",
    "src/virtual_imu/viz/plots_module_replacement.py",
    "src/virtual_imu/viz/style.py",
)


def _require(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"{label}: missing fields/rows {sorted(missing)}")


def _integers(frame, fields):
    for field in fields:
        values = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError("Nonintegral/nonfinite " + field)
        frame[field] = values.astype("int64")


def _booleans(series, label):
    # bool('False') is True; never use truthiness to validate CSV provenance.
    values = series.astype(str).str.lower()
    if not values.isin(("true", "false")).all():
        raise ValueError("Invalid boolean " + label)
    return values.eq("true")


def prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Equivalent to the old physical-schema preparation; no legacy variants.

    All aggregation supports must exist for every condition. Raw is deterministic
    window0/seed-1. Recording normalized scores remain undefined. The Raw-zero
    denominator floor (1e-12) matches the historical grouped_score definition;
    paired percentages separately remain undefined at a zero model reference.
    """
    _require(metrics, [*IDENTITY, "samples", "job_id", *PHYSICAL], "metrics")
    frame = metrics.copy().reset_index(drop=True)
    if frame[IDENTITY].isna().any().any():
        raise ValueError("Missing metric identity")
    _integers(frame, ["window", "seed", "samples"])
    frame[PHYSICAL] = frame[PHYSICAL].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[PHYSICAL]).all().all() or (frame[PHYSICAL] < 0).any().any():
        raise ValueError("Invalid physical metrics")
    if (frame.samples <= 0).any() or frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate metric identities or invalid support samples")
    raw = frame.variant.eq("raw")
    if not raw.any() or frame.loc[raw, "window"].ne(0).any() or frame.loc[raw, "seed"].ne(-1).any():
        raise ValueError("Raw must have window0/seed-1")
    if frame.loc[~raw, "window"].le(0).any() or frame.loc[~raw, "seed"].lt(0).any():
        raise ValueError("Invalid learned seed/window")
    if not frame.aggregation.isin(("overall", "dataset", "pair", "recording")).all():
        raise ValueError("Unknown aggregation")
    for field, active in (
        ("dataset_id", ("dataset", "pair", "recording")),
        ("pair_id", ("pair", "recording")),
        ("original_sequence_id", ("recording",)),
    ):
        if not frame[field].ne(ALL).eq(frame.aggregation.isin(active)).all():
            raise ValueError("Inconsistent aggregation identity: " + field)
    if frame.groupby(["protocol", "split", *SUPPORT]).samples.nunique().gt(1).any():
        raise ValueError("Sample supports differ across conditions")
    for _, group in frame.groupby(["protocol", "split"], sort=False):
        baseline = group[group.variant.eq("raw")]
        if set(baseline.aggregation) != {"overall", "dataset", "pair", "recording"}:
            raise ValueError("Missing required aggregation support")
        datasets = baseline[baseline.aggregation.eq("dataset")]
        pairs = baseline[baseline.aggregation.eq("pair")]
        recordings = baseline[baseline.aggregation.eq("recording")]
        if baseline[baseline.aggregation.eq("overall")].samples.iloc[0] != datasets.samples.sum():
            raise ValueError("Overall/dataset sample counts disagree")
        for parent, child, keys in (
            (datasets, pairs, ["dataset_id"]),
            (pairs, recordings, ["dataset_id", "pair_id"]),
        ):
            if (
                not parent.set_index(keys)
                .samples.sort_index()
                .equals(child.groupby(keys).samples.sum().sort_index())
            ):
                raise ValueError("Parent/child support sample counts disagree")
        expected = set(map(tuple, baseline[SUPPORT].to_numpy()))
        for _, condition in group.groupby(["variant", "window", "seed"], sort=False):
            if set(map(tuple, condition[SUPPORT].to_numpy())) != expected:
                raise ValueError("Missing or mismatched aggregation supports")
    frame["job_id"] = frame.job_id.fillna("")
    supplied = frame.get("normalized_score")
    frame["normalized_score"] = np.nan
    pairs = frame[frame.aggregation.eq("pair")].reset_index(names="row_index")
    keys = ["protocol", "split", "dataset_id", "pair_id"]
    baseline = pairs[pairs.variant.eq("raw")][[*keys, "gyro_rmse", "accel_rmse"]]
    scored = pairs.merge(baseline, on=keys, suffixes=("", "_raw"), validate="many_to_one")
    scored["score"] = 0.5 * sum(
        scored[g + "_rmse"] / np.maximum(scored[g + "_rmse_raw"], 1e-12) for g in ("gyro", "accel")
    )
    frame.loc[scored.row_index, "normalized_score"] = scored.score.to_numpy()
    dataset_scores = scored.groupby([*CONDITION, "dataset_id"], sort=False).score.mean()
    overall_scores = dataset_scores.groupby(level=CONDITION, sort=False).mean()
    for index, row in frame[frame.aggregation.isin(("dataset", "overall"))].iterrows():
        key = tuple(row[field] for field in CONDITION)
        source = overall_scores
        if row.aggregation == "dataset":
            key += (row.dataset_id,)
            source = dataset_scores
        frame.at[index, "normalized_score"] = source.loc[key]
    if supplied is not None:
        known = supplied.notna()
        if not np.allclose(
            pd.to_numeric(supplied[known], errors="raise"),
            frame.loc[known, "normalized_score"],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("Supplied normalized scores disagree with balanced Raw ratios")
    return frame


def _long(frame):
    extra = [c for c in [*ORIGIN, "source_job_id"] if c in frame]
    return frame.melt(
        id_vars=[*IDENTITY, "samples", "job_id", *extra],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])


def _origins(frame, keys):
    if not set(ORIGIN).issubset(frame):
        return None
    origin = frame[[*keys, *ORIGIN]].drop_duplicates()
    if origin.duplicated(keys).any():
        raise ValueError("Mixed source/reuse identities in a seed summary")
    return origin


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """All physical metrics, normalized scores, sample SD, n and support size."""
    result = (
        _long(frame)
        .groupby([*GROUP, "metric"], as_index=False, sort=True)
        .agg(
            mean=("value", "mean"),
            sd=("value", "std"),
            n=("seed", "nunique"),
            samples=("samples", "first"),
        )
    )
    origin = _origins(frame, GROUP)
    return result if origin is None else result.merge(origin, on=GROUP, validate="many_to_one")


def paired_deltas(frame: pd.DataFrame, reference_variant="cnn_lstm64") -> pd.DataFrame:
    """Exact protocol/split/seed/window/support/metric pairs; positive is worse."""
    learned = frame[~frame.variant.eq("raw")]
    if learned.duplicated(IDENTITY).any():
        raise ValueError("Duplicate paired metric identities")
    long = _long(learned)
    keys = [key for key in IDENTITY if key != "variant"] + ["metric"]
    ref = long[long.variant.eq(reference_variant)].copy()
    other = long[~long.variant.eq(reference_variant)].copy()
    if ref.empty or other.empty:
        raise ValueError("Missing reference or treatment for pairing")
    fields = ["variant", "value", "samples", "job_id"]
    fields += [c for c in [*ORIGIN, "source_job_id"] if c in ref]
    ref = ref[[*keys, *fields]].rename(columns={field: "reference_" + field for field in fields})
    result = other.merge(ref, on=keys, how="left", validate="many_to_one")
    if result.reference_value.isna().any() or not result.samples.eq(result.reference_samples).all():
        raise ValueError("Missing reference or mismatched paired support/seed/window")
    if not np.isfinite(result[["value", "reference_value"]]).all().all():
        raise ValueError("Invalid paired values")
    result["reference_seed"] = result.seed
    result["reference_window"] = result.window
    result["delta"] = result.value - result.reference_value
    result["relative_delta_pct"] = 100.0 * result.delta / result.reference_value.replace(0, np.nan)
    return result.sort_values([*IDENTITY, "metric"]).reset_index(drop=True)


def summarize_pairs(deltas: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired arithmetic before averaging; separate percentage n."""
    frame = deltas.assign(
        worse=deltas.delta.gt(0).astype(int),
        better=deltas.delta.lt(0).astype(int),
        tied=deltas.delta.eq(0).astype(int),
    )
    result = frame.groupby([*GROUP, "metric"], as_index=False, sort=True).agg(
        delta_mean=("delta", "mean"),
        delta_sd=("delta", "std"),
        n=("seed", "nunique"),
        relative_delta_pct_mean=("relative_delta_pct", "mean"),
        relative_delta_pct_sd=("relative_delta_pct", "std"),
        relative_n=("relative_delta_pct", "count"),
        worse_seeds=("worse", "sum"),
        better_seeds=("better", "sum"),
        tied_seeds=("tied", "sum"),
        samples=("samples", "first"),
        reference_samples=("reference_samples", "first"),
        reference_variant=("reference_variant", "first"),
        reference_window=("reference_window", "first"),
    )
    origin = _origins(frame, GROUP)
    if origin is not None:
        result = result.merge(origin, on=GROUP, validate="many_to_one")
        refs = frame[[*GROUP, "reference_reused", "reference_source_run"]].drop_duplicates()
        result = result.merge(refs, on=GROUP, validate="many_to_one")
    return result


def validation_ranking(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Rank mean balanced VALID across seeds per training scope; ties share rank."""
    val = frame[frame.split.eq("validation") & frame.aggregation.eq("overall")]
    rows = []
    expected = set(product(config["windows"], config["seeds"]))
    for protocol, variant in product(config["protocols"], config["variants"]):
        group = val[val.protocol.eq(protocol) & val.variant.eq(variant)]
        if (
            len(group) != len(expected)
            or set(zip(group.window, group.seed)) != expected
            or not np.isfinite(group.normalized_score).all()
        ):
            raise ValueError("Incomplete validation window/seed grid")
        for window, values in group.groupby("window", sort=True):
            rows.append(
                dict(
                    protocol=protocol,
                    variant=variant,
                    window=window,
                    mean=values.normalized_score.mean(),
                    sd=values.normalized_score.std(ddof=1),
                    n=values.seed.nunique(),
                    selection_split="validation",
                    reused=variant in config["reuse"]["variants"],
                )
            )
    result = pd.DataFrame(rows)
    result["rank"] = result.groupby("protocol")["mean"].rank(method="min").astype(int)
    origin = _origins(val, ["protocol", "variant", "window"])
    if origin is not None:
        result = result.drop(columns="reused").merge(
            origin, on=["protocol", "variant", "window"], validate="one_to_one"
        )
    return result.sort_values(["protocol", "rank", "variant", "window"]).reset_index(drop=True)


def summarize_compute(registry, cost):
    """Keep source training cost separate from newly measured inference cost."""
    fields = ["job_id", *FIT_COST, "training_cost_origin", "source_run"]
    merged = cost.merge(registry[fields], on="job_id", validate="one_to_one")
    rows = []
    for metric in ["parameters", *INFERENCE_COST, *FIT_COST]:
        part = merged[[*FIT_KEYS, "reused", "source_run", "training_cost_origin", metric]].rename(
            columns={metric: "value"}
        )
        part["metric"] = metric
        if metric in INFERENCE_COST:
            part["cost_scope"] = "uniform_new_inference"
        elif metric == "parameters":
            part["cost_scope"] = "architecture"
        else:
            part["cost_scope"] = np.where(part.reused, "source_training", "new_training")
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    return long.groupby(
        [
            "protocol",
            "variant",
            "window",
            "reused",
            "source_run",
            "training_cost_origin",
            "cost_scope",
            "metric",
        ],
        as_index=False,
    ).agg(mean=("value", "mean"), sd=("value", "std"), n=("seed", "nunique"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_source_hashes():
    """Required executed analysis dependencies; freeze these before reading TEST.

    The coordinator additionally freezes the independent verifier. All entries
    actually listed in analysis_plan.json (including that verifier) are checked
    and traced, without importing or running it here.
    """
    return {relative: _sha(ROOT / relative) for relative in ANALYSIS_SOURCES}


def _read(run, name, kind, hashes):
    path = run / name
    LOG.info("READS %s", path)
    before = _sha(path)
    if kind == "csv":
        value = pd.read_csv(path)
    elif kind == "yaml":
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if _sha(path) != before:
        raise ValueError("Input changed while reading: " + name)
    hashes[name] = before
    return value


def _check_analysis(run, plan, config, hashes):
    sources = plan.get("source_sha256", {})
    if (
        plan.get("test_unlocked") is not False
        or not set(ANALYSIS_SOURCES).issubset(sources)
        or plan.get("reference") != config["reference_variant"]
        or plan.get("window") != 320
    ):
        raise ValueError("Missing/mismatched frozen pre-TEST analysis plan")
    for relative, expected in sources.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe analysis source path")
        current, snapshot = ROOT / path, run / "analysis_source" / path
        LOG.info("READS analysis source and frozen snapshot %s", relative)
        if _sha(current) != expected or _sha(snapshot) != expected:
            raise ValueError("Frozen analysis source changed: " + relative)
        hashes[(Path("analysis_source") / path).as_posix()] = expected
    return sources


def _validate_config(config):
    for key in ("protocols", "variants", "seeds", "windows"):
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError("Empty/duplicate config " + key)
    if (
        config["windows"] != [320]
        or config["reference_variant"] not in config["variants"]
        or not set(config["reuse"]["variants"]).issubset(config["variants"])
    ):
        raise ValueError("Invalid fixed-window/reference/reuse configuration")
    if any(type(seed) is not int or seed < 0 for seed in config["seeds"]):
        raise ValueError("Invalid config seeds")
    if any(
        config["data"][key] != value
        for key, value in (("max_window", 320), ("block_size", 40), ("stride", 40), ("rate_hz", 40))
    ):
        raise ValueError("Require fixed320 center40 common support at40Hz")


def _validate_tables(run, cfg, planned, prov, complete, evaluation, lock, registry, cost, metrics):
    expected = set(product(cfg["protocols"], cfg["variants"], cfg["windows"], cfg["seeds"]))
    jobs = pd.DataFrame(planned).rename(columns={"id": "job_id"})
    _require(jobs, [*FIT_KEYS, "job_id", "reused"], "planned jobs")
    if (
        len(jobs) != len(expected)
        or jobs.job_id.duplicated().any()
        or set(map(tuple, jobs[FIT_KEYS].to_numpy())) != expected
    ):
        raise ValueError("Planned job grid differs from config")
    jobs["reused"] = _booleans(jobs.reused, "planned reused")
    expected_reuse = jobs.variant.isin(cfg["reuse"]["variants"])
    if not jobs.reused.eq(expected_reuse).all():
        raise ValueError("Plan reuse differs from config")
    if (
        "source_variant" in jobs
        and not jobs.source_variant.fillna("")
        .eq(jobs.variant.map(cfg["reuse"]["variants"]).fillna(""))
        .all()
    ):
        raise ValueError("Planned source variant differs from config reuse mapping")
    for job in planned:
        if "config" in job and job["config"] != cfg:
            raise ValueError("Planned embedded config differs from snapshot")
    count, reused = len(jobs), int(jobs.reused.sum())
    counts = dict(conditions=count, new=count - reused, reused=reused)
    if (
        prov.get("planned_conditions") != count
        or prov.get("planned_new_fits") != count - reused
        or prov.get("planned_reused_controls") != reused
        or complete.get("status") != "complete"
        or complete.get("planned") != count
        or complete.get("completed") != count
        or complete.get("new_fits") != count - reused
        or complete.get("reused_controls") != reused
    ):
        raise ValueError("Training/provenance conditions are incomplete")
    if (
        evaluation.get("status") != "complete"
        or evaluation.get("conditions") != count
        or evaluation.get("prediction_files") != 2 * count
    ):
        raise ValueError("Evaluation completion/count declaration is incomplete")
    if (
        lock.get("test_loaded") is not False
        or prov.get("test_used_for_selection") is not False
        or lock.get("new_fits") != count - reused
        or lock.get("reused_controls") != reused
    ):
        raise ValueError("Missing validation-only pre-TEST checkpoint lock")
    _require(
        registry,
        [
            *FIT_KEYS,
            "job_id",
            "checkpoint_sha256",
            "job_config_sha256",
            "parameters",
            *FIT_COST,
            "reused",
            "fit_kind",
            "source_job_id",
            "source_run",
            "training_cost_origin",
        ],
        "fit registry",
    )
    _require(
        cost,
        [*FIT_KEYS, "job_id", "reused", "parameters", *INFERENCE_COST, "latency_origin"],
        "compute cost",
    )
    digests = {
        job["id"]: hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        for job in planned
    }
    for label, table in (("registry", registry), ("compute", cost)):
        _integers(table, ["window", "seed", "parameters"])
        table["reused"] = _booleans(table.reused, label + " reused")
        if table.job_id.duplicated().any() or set(table.job_id) != set(jobs.job_id):
            raise ValueError("Incomplete/duplicate job identities: " + label)
        joined = table.merge(
            jobs[[*FIT_KEYS, "job_id", "reused"]],
            on="job_id",
            suffixes=("", "_plan"),
            validate="one_to_one",
        )
        if any(not joined[c].eq(joined[c + "_plan"]).all() for c in [*FIT_KEYS, "reused"]):
            raise ValueError("Job metadata differs from plan: " + label)
        for field in ("directory", "job_id"):
            if (
                field in table
                and table[field].fillna("").str.contains("_guard_diagnostic", regex=False).any()
            ):
                raise ValueError("Excluded guard diagnostic identity")
    if any(digests[row.job_id] != row.job_config_sha256 for row in registry.itertuples()):
        raise ValueError("Fit config hash differs from planned job identity")
    if dict(zip(registry.job_id, registry.checkpoint_sha256)) != lock.get("checkpoints_sha256"):
        raise ValueError("Registry checkpoints differ from pre-TEST lock")
    completed = pd.DataFrame(complete.get("fits", []))
    _require(
        completed,
        [*FIT_KEYS, "job_id", "reused", "checkpoint_sha256", "job_config_sha256"],
        "completed fits",
    )
    columns = [*FIT_KEYS, "job_id", "reused", "checkpoint_sha256", "job_config_sha256"]
    if (
        len(completed) != count
        or completed.job_id.duplicated().any()
        or set(map(tuple, completed[columns].to_numpy()))
        != set(map(tuple, registry[columns].to_numpy()))
    ):
        raise ValueError("Training completion fit identities differ from registry")
    for table, fields in (
        (registry, ["parameters", *FIT_COST]),
        (cost, ["parameters", *INFERENCE_COST]),
    ):
        if (
            not np.isfinite(table[fields].to_numpy(float)).all()
            or (table[fields].astype(float) < 0).any().any()
        ):
            raise ValueError("Invalid compute/fit measurements")
    _integers(registry, ["best_epoch", "actual_epochs", "amp_skipped_steps"])
    if (
        registry.parameters.le(0).any()
        or registry.best_epoch.lt(1).any()
        or registry.best_epoch.gt(registry.actual_epochs).any()
        or registry.actual_epochs.gt(cfg["training"]["epochs"]).any()
    ):
        raise ValueError("Invalid parameter/epoch bounds")
    if (
        not registry.set_index("job_id")
        .parameters.sort_index()
        .equals(cost.set_index("job_id").parameters.sort_index())
    ):
        raise ValueError("Compute and registry parameters disagree")
    if not cost.latency_origin.eq("current_run_uniform_measurement").all():
        raise ValueError("Inference costs must be measured uniformly in current run")
    registry["source_run"] = registry.source_run.fillna("")
    registry["source_job_id"] = registry.source_job_id.fillna("")
    expected_source_variant = registry.variant.map(cfg["reuse"]["variants"]).fillna("")
    if (
        "source_variant" in registry
        and not registry.source_variant.fillna("").eq(expected_source_variant).all()
    ):
        raise ValueError("Registry source variant differs from config reuse mapping")
    registry["source_variant"] = expected_source_variant
    source_run = (ROOT / cfg["reuse"]["run_dir"]).resolve()
    for row in registry.itertuples():
        if row.fit_kind != (
            "reused_control" if row.reused else "new_fit"
        ) or row.training_cost_origin != ("prior_run" if row.reused else "current_run"):
            raise ValueError("Incorrect fit_kind/training cost origin")
        if row.reused:
            expected_id = f"{row.protocol}__{row.source_variant}__w{row.window}__s{row.seed}"
            if (
                not row.source_run
                or (ROOT / row.source_run).resolve() != source_run
                or row.source_job_id != expected_id
            ):
                raise ValueError("Reused control source run/job identity mismatch")
            if (
                "source_checkpoint_sha256" in registry
                and row.source_checkpoint_sha256 != row.checkpoint_sha256
            ):
                raise ValueError("Reused source checkpoint hash mismatch")
        elif row.source_run or row.source_job_id:
            raise ValueError("New fit must not claim source training")
    if "reused_run" in prov and (ROOT / prov["reused_run"]).resolve() != source_run:
        raise ValueError("Reused provenance run differs from config")
    learned = metrics[~metrics.variant.eq("raw")]
    observed = learned[[*CONDITION, "job_id"]].drop_duplicates()
    expected_eval = jobs[[*FIT_KEYS, "job_id"]].merge(
        pd.DataFrame({"split": ["validation", "test"]}), how="cross"
    )
    fields = [*CONDITION, "job_id"]
    if set(map(tuple, observed[fields].to_numpy())) != set(
        map(tuple, expected_eval[fields].to_numpy())
    ):
        raise ValueError("Evaluation seed/job coverage differs from config/plan")
    if set(map(tuple, metrics[["protocol", "split"]].drop_duplicates().to_numpy())) != set(
        product(cfg["protocols"], ("validation", "test"))
    ):
        raise ValueError("Unexpected protocol/split coverage")
    _require(metrics, ["reused"], "metrics reuse provenance")
    metric_reuse = _booleans(metrics.reused, "metrics reused")
    if not metric_reuse.eq(metrics.variant.isin(cfg["reuse"]["variants"])).all():
        raise ValueError("Metric reuse provenance differs from config")
    expected_domains = {"alimu_only": {"alimu"}, "pooled": {"alimu", "penncosyvio"}}
    for (protocol, _), part in metrics[metrics.aggregation.eq("dataset")].groupby(
        ["protocol", "split"]
    ):
        if protocol in expected_domains and set(part.dataset_id) != expected_domains[protocol]:
            raise ValueError("Missing/unexpected reporting domain")
    return counts


def _annotate(metrics, registry):
    frame = metrics.drop(columns=[c for c in [*ORIGIN, "source_job_id"] if c in metrics])
    fields = ["job_id", *ORIGIN, "source_job_id"]
    frame = frame.merge(registry[fields], on="job_id", how="left", validate="many_to_one")
    raw = frame.variant.eq("raw")
    frame.loc[raw, "reused"] = False
    for field in ("source_run", "source_variant", "source_job_id"):
        frame[field] = frame[field].fillna("")
    frame["reused"] = _booleans(frame.reused, "annotated reused")
    return frame


def _number(value):
    return "—" if pd.isna(value) else f"{value:.6g}"


def _estimate(mean, sd, n):
    if n == 0:
        return "—（n=0）"
    if n == 1:
        return _number(mean) + "（n=1；SD未定义）"
    return f"{_number(mean)} ± {_number(sd)}"


def _table(headers, rows):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    return (
        "\n".join(
            [
                "| " + " | ".join(map(escape, headers)) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + ["| " + " | ".join(map(escape, row)) + " |" for row in rows]
        )
        + "\n"
    )


def _metric_table(frame, identities, metrics):
    rows = []
    for key, part in frame.groupby(identities, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        lookup = part.set_index("metric")
        estimates = [
            _estimate(lookup.loc[m, "mean"], lookup.loc[m, "sd"], lookup.loc[m, "n"])
            if m in lookup.index
            else "—（未定义）"
            for m in metrics
        ]
        rows.append(
            [
                *key,
                bool(part.reused.iloc[0]),
                int(part.n.iloc[0]),
                int(part.samples.iloc[0]),
                *estimates,
            ]
        )
    return _table([*identities, "reused", "n种子", "每种子点数", *metrics], rows)


def _report_text(run, cfg, prov, counts, tables, metrics, registry, hashes):
    summary, paired, ranking, compute = (
        tables[name]
        for name in (
            "summary.csv",
            "paired_summary.csv",
            "validation_ranking.csv",
            "compute_summary.csv",
        )
    )
    lines = ["# CNN-LSTM 前后模块替换报告", ""]
    if prov.get("synthetic_fixture"):
        lines += ["**合成测试产物：所有数值仅用于验证报告代码，不能作为真实实验结论。**", ""]
    lines += [
        f"权威运行：`{run}`；实验：`{cfg['experiment_id']}`；训练源码 git：`{prov.get('git_commit', '未登记')}`。",
        f"已核对配置、计划、登记和完成声明：{len(cfg['variants'])}结构 × {len(cfg['protocols'])}训练范围 × {len(cfg['seeds'])}优化种子 = {counts['conditions']}条件；其中{counts['new']}项新拟合、{counts['reused']}项复用控制。VALID/TEST共{2 * counts['conditions']}份预测的完成声明已核对；实际预测字节及回放由独立核验器负责。",
        "## 固定方案与解释范围",
        "",
        f"主参照 `{cfg['reference_variant']}`。固定320点输入、40Hz（8 s），相同公共anchor内中心40点（1 s）监督与评价；TRAIN-only标准化，六轴LOW→六轴HIGH。配置优化种子为 {cfg['seeds']}。",
        "均值±样本标准差（SD，ddof=1）刻画三个优化种子的变化，不是独立录制的置信区间。Raw只有一次确定性读数，n=1、SD未定义。设备对或录制行数不能增加独立物理录制样本量。",
        "物理指标保留逐次先池化样本误差的结果，不以设备或录制RMSE平均重建总体。gyro及gx/gy/gz单位rad/s，accel及ax/ay/az单位m/s²；vector_rmse为三轴误差向量范数的RMSE；mae为分量绝对误差均值；bias_norm为三轴均值误差向量范数。",
        "normalized_score无量纲：每设备对先计算(gyro_RMSE/Raw_gyro_RMSE + accel_RMSE/Raw_accel_RMSE)/2，再等权设备对、等权数据集；Raw分母按历史定义以1e-12为下限。录制级normalized_score未定义。",
        f"固定历史配方 `{cfg['training']['optimizer']}`、最多{cfg['training']['epochs']}轮、patience={cfg['training']['early_stopping_patience']}；学习率{cfg['training']['lr']:.8g}，weight_decay={cfg['training']['weight_decay']:.8g}。本轮不做每模型HPO，结果表示固定配方下的替换表现。",
        "本轮在主TEST前冻结结构与分析源码，按每个训练范围的平均VALID归一化分数排名；TEST仅作描述。联合模型的AL/Penn TEST域不分别选结构。旧TEST已被查看，后续研究不能称为全新独立确认。",
        "有效上下文与未来信息范围随结构变化：逐点Linear/MLP前端不混合时间；CNN局部混合，单向LSTM/GRU聚合历史；双向循环块与Transformer利用窗口内更广信息。相同320点输入不意味着相同有效感受野或相同未来信息。",
        f"保留的CNN为{cfg['model']['conv_depth']}层k{cfg['model']['kernel_size']}；TCN另加膨胀{cfg['replacement_models']['tcn_dilations']}。后端GRU隐藏{cfg['replacement_models']['gru_hidden']}，BiGRU每方向{cfg['replacement_models']['bigru_hidden']}，不是相同宽度对照。",
        "七个新训练的非删除替换在训练前按参考参数量±5%预算选择宽度；历史BiLSTM32控制不属于这七个，参数量为73,222、比参考81,414少10.06%。线性前端和无后端属于低容量删除控制。参数量、规范化、有效感受野及方向性共同变化，因此这些替换不构成抽象算子的单因素因果实验。性能损失支持模块在当前方案中有用，不能证明某层专门消除了偏置或随机噪声。",
        "## 来源与支持",
        "",
        f"复用来源 `{cfg['reuse']['run_dir']}`，仅从当前登记保留原生source_job_id；SOURCE运行保持原样。旧训练成本属于prior_run，不计成本轮新训练；全部推理成本来自current_run_uniform_measurement。报告不读取SOURCE目录或加载其检查点。",
        _table(["当前variant", "来源variant"], sorted(cfg["reuse"]["variants"].items())),
    ]
    scope = []
    raw = metrics[metrics.variant.eq("raw")]
    for (protocol, split, dataset), part in raw[raw.aggregation.eq("dataset")].groupby(
        ["protocol", "split", "dataset_id"]
    ):
        records = raw[
            raw.protocol.eq(protocol)
            & raw.split.eq(split)
            & raw.dataset_id.eq(dataset)
            & raw.aggregation.eq("recording")
        ]
        scope.append(
            [
                protocol,
                split,
                dataset,
                int(part.samples.iloc[0]),
                records.pair_id.nunique(),
                records.original_sequence_id.nunique(),
            ]
        )
    lines += [
        _table(
            [
                "训练范围",
                "split",
                "评估域",
                "每seed点数",
                "设备对数",
                "录制标识数（不等同独立录制数）",
            ],
            scope,
        ),
        "## 所有域的VALID/TEST物理数值与SD",
        "",
        "下表覆盖全部结构和Raw；总体、设备对、录制及全部指标的逐行结果见summary.csv。",
        "",
    ]
    domain = summary[summary.aggregation.eq("dataset")]
    for split in ("validation", "test"):
        lines += [
            f"### {split}",
            "",
            _metric_table(
                domain[domain.split.eq(split)],
                ["protocol", "dataset_id", "variant", "window"],
                METRICS,
            ),
        ]
    lines += [
        "## 同seed配对变化",
        "",
        "在完全相同protocol/split/支持身份/seed/window下，处理减去参照；正值表示处理更差，负值表示处理更好。百分比先逐seed计算100×delta/reference，再求均值与SD，不是组均值之比。参照为零时百分比为空，relative_n另计；绝对差及方向仍保留。",
    ]
    cells = paired[
        paired.aggregation.eq("dataset")
        & paired.metric.isin(("gyro_rmse", "accel_rmse", "normalized_score"))
    ]
    lines += [
        _table(
            [
                "protocol",
                "split",
                "域",
                "variant",
                "metric",
                "reused",
                "n配对",
                "差均值±SD",
                "百分比均值±SD",
                "百分比n",
                "变差/改善/相等seed数",
            ],
            [
                [
                    r.protocol,
                    r.split,
                    r.dataset_id,
                    r.variant,
                    r.metric,
                    r.reused,
                    r.n,
                    _estimate(r.delta_mean, r.delta_sd, r.n),
                    _estimate(r.relative_delta_pct_mean, r.relative_delta_pct_sd, r.relative_n),
                    r.relative_n,
                    f"{r.worse_seeds}/{r.better_seeds}/{r.tied_seeds}",
                ]
                for r in cells.itertuples()
            ],
        ),
        "同号仅描述三个优化重复的一致性，不代表统计显著；域或指标出现取舍时不能宣称普遍优胜。",
        "## VALID排序（不按TEST选择）",
        "",
        "每个训练范围各自按整体等数据集/等设备对VALID分数的seed均值升序排序。完全相同的均值共享最小名次；展示次序按variant字典序，不使用TEST打破平局。",
        _table(
            ["protocol", "variant", "window", "reused", "VALID均值±SD", "n", "rank"],
            [
                [
                    r.protocol,
                    r.variant,
                    r.window,
                    r.reused,
                    _estimate(r.mean, r.sd, r.n),
                    r.n,
                    r.rank,
                ]
                for r in ranking.itertuples()
            ],
        ),
        "## 每设备对和录制的TEST读数",
        "",
    ]
    for aggregation in ("pair", "recording"):
        identities = ["protocol", "dataset_id", "pair_id"]
        if aggregation == "recording":
            identities += ["original_sequence_id"]
        identities += ["variant", "window"]
        lines += [
            f"### {aggregation}",
            "",
            _metric_table(
                summary[summary.split.eq("test") & summary.aggregation.eq(aggregation)],
                identities,
                ["gyro_rmse", "accel_rmse", "gyro_vector_rmse", "accel_vector_rmse"],
            ),
        ]
    lines += [
        "## 参数与计算成本",
        "",
        f"batch1统一新计时：warmup={cfg['evaluation']['latency_warmup']}、repeats={cfg['evaluation']['latency_repeats']}；均值/中位数先在每个作业重复前向计时，再跨seed求均值±SD。CUDA由runner同步计时，机器负载影响延迟。",
        "source_training是复用的旧训练，new_training是本轮新拟合，uniform_new_inference是所有结构本轮统一测量；architecture表示参数量。每作业训练墙钟包含VALID，并发作业墙钟之和不等于实验历时。",
        _table(
            [
                "protocol",
                "variant",
                "window",
                "reused",
                "cost_scope",
                "metric",
                "均值±SD",
                "n",
                "source_run",
            ],
            [
                [
                    r.protocol,
                    r.variant,
                    r.window,
                    r.reused,
                    r.cost_scope,
                    r.metric,
                    _estimate(r.mean, r.sd, r.n),
                    r.n,
                    r.source_run,
                ]
                for r in compute.itertuples()
            ],
        ),
        "### 复用原生作业身份",
        "",
        _table(
            ["job_id", "reused", "fit_kind", "source_run", "source_job_id", "training_cost_origin"],
            [
                [
                    r.job_id,
                    r.reused,
                    r.fit_kind,
                    r.source_run,
                    r.source_job_id,
                    r.training_cost_origin,
                ]
                for r in registry.itertuples()
            ],
        ),
        "## 追溯与核验范围",
        "",
        "本报告只读取显式运行目录中的表格、声明、数据清单和分析源码快照；不训练、不加载模型或原始数据、不重新预测。核对了配置/计划/登记/完成身份、源分析哈希和配对算术；SOURCE复用凭证、实际检查点及132份预测的数值回放归独立核验器负责。",
        "[图及图注](figures/figure_index.md)；[运行来源](provenance.json)；[配置](config.yaml)；[数据清单](data_manifest.json)；[TEST前分析冻结](analysis_plan.json)；[检查点锁](selections.lock.json)；[报告SHA回执](reporting_provenance.json)。",
        "[所有指标](summary.csv)；[逐seed配对](paired_deltas.csv)；[配对汇总](paired_summary.csv)；[VALID排序](validation_ranking.csv)；[成本汇总](compute_summary.csv)。所有派生表保留reused；SHA回执覆盖全部读取输入、执行分析源码和生成输出，回执本身不作循环自哈希。",
        _table(["报告输入", "SHA256"], sorted(hashes.items())),
    ]
    return "\n\n".join(lines) + "\n"


def report(run_dir: Path):
    """Validate the fixed run, write derived tables/figures/report, return receipt.

    Raises before any derived writes on missing, partial, nonfrozen or mismatched
    inputs. analysis_plan.json and its snapshots are checked before metric IO.
    The reporter never creates a freeze retroactively or infers a SOURCE run.
    """
    run = Path(run_dir).resolve()
    LOG.info(
        "READS explicit run %s; SKIPS training, checkpoints, raw data, SOURCE run and outputs/latest",
        run,
    )
    if any("_guard_diagnostic" in part for part in run.parts):
        raise ValueError("Excluded _guard_diagnostic run")
    hashes = {}
    cfg = _read(run, "config.yaml", "yaml", hashes)
    _validate_config(cfg)
    prov = _read(run, "provenance.json", "json", hashes)
    plan = _read(run, "analysis_plan.json", "json", hashes)
    sources = _check_analysis(run, plan, cfg, hashes)
    _read(run, "data_manifest.json", "json", hashes)
    if (
        hashes["config.yaml"] != prov.get("config_sha256")
        or hashes["data_manifest.json"] != prov.get("manifest_sha256")
        or hashes["data_manifest.json"] != cfg["data"].get("manifest_sha256")
    ):
        raise ValueError("Frozen config/manifest hashes disagree with provenance")
    planned = _read(run, "planned_jobs.json", "json", hashes)
    complete = _read(run, "training.complete.json", "json", hashes)
    evaluation = _read(run, "evaluation.complete.json", "json", hashes)
    lock = _read(run, "selections.lock.json", "json", hashes)
    registry = _read(run, "fit_registry.csv", "csv", hashes)
    cost = _read(run, "compute_cost.csv", "csv", hashes)
    metrics = prepare_metrics(_read(run, "metrics_per_seed.csv", "csv", hashes))
    counts = _validate_tables(
        run, cfg, planned, prov, complete, evaluation, lock, registry, cost, metrics
    )
    if plan.get("new_fit_count") != counts["new"] or plan.get("reuse_count") != counts["reused"]:
        raise ValueError("Frozen analysis count differs from actual grid")
    metrics = _annotate(metrics, registry)
    deltas = paired_deltas(metrics, cfg["reference_variant"])
    tables = {
        "summary.csv": summarize_metrics(metrics),
        "paired_deltas.csv": deltas,
        "paired_summary.csv": summarize_pairs(deltas),
        "validation_ranking.csv": validation_ranking(metrics, cfg),
        "compute_summary.csv": summarize_compute(registry, cost),
    }
    for name, digest in hashes.items():
        if _sha(run / name) != digest:
            raise ValueError("Input changed during reporting: " + name)
    for relative, digest in sources.items():
        if _sha(ROOT / relative) != digest:
            raise ValueError("Analysis source changed during reporting: " + relative)
    for name, table in tables.items():
        LOG.info("WRITES %s rows=%d", run / name, len(table))
        table.to_csv(run / name, index=False, float_format="%.17g")
    from virtual_imu.viz.plots_module_replacement import render_module_replacement_figures

    figures = render_module_replacement_figures(
        tables["summary.csv"], deltas, tables["compute_summary.csv"], run / "figures", cfg, metrics
    )
    text = _report_text(run, cfg, prov, counts, tables, metrics, registry, hashes)
    LOG.info("WRITES %s", run / "report.md")
    (run / "report.md").write_text(text, encoding="utf-8")
    paths = [run / name for name in tables] + [run / "report.md", run / "figures/figure_index.md"]
    paths += [Path(path) for entry in figures for path in entry["paths"]]
    receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run),
        git_commit=prov.get("git_commit"),
        inputs=hashes,
        source_sha256=sources,
        outputs={path.relative_to(run).as_posix(): _sha(path) for path in paths},
        counts=counts,
        training_performed=False,
        test_used_for_selection=False,
        frozen_analysis_verified=True,
        source_run_read=False,
        self_hash_excluded=True,
        schema_version=1,
    )
    LOG.info("WRITES %s; SKIPS self-referential receipt hash", run / "reporting_provenance.json")
    (run / "reporting_provenance.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        receipt = report(args.run_dir)
    except (ValueError, OSError, KeyError, TypeError, pd.errors.ParserError) as error:
        LOG.error("SKIPS formal reporting; WRITES no successful report: %s", error)
        return 2
    LOG.info(
        "WRITES complete reporting receipt conditions=%d; SKIPS training",
        receipt["counts"]["conditions"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
