"""Reporting regressions use synthetic physical errors, never a dataset or model.

Run with --basetemp runs/pytest_tmp/module_replacement_reporting_<unique>.
The integration test exports real paper figures under SYNTHETIC_REPORT, without monkeypatching
Matplotlib, and checks input/source immutability and the SHA receipt.
"""

# Python 3.9 is the experiment runtime.
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_module_replacement.py"
CONFIG = ROOT / "configs/ablations/cnn_lstm_module_replacement.yaml"
PHYSICAL = [
    f"{group}_{metric}"
    for group in ("gyro", "accel")
    for metric in ("rmse", "vector_rmse", "mae", "bias_norm")
] + [f"{axis}_rmse" for axis in ("gx", "gy", "gz", "ax", "ay", "az")]


def reporter():
    assert SCRIPT.is_file(), "Module replacement reporting entry point is missing"
    spec = importlib.util.spec_from_file_location("module_replacement_reporting", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["experiment_id"] = "SYNTHETIC_REPORT"
    cfg["reuse"]["run_dir"] = "SYNTHETIC_SOURCE_DO_NOT_OPEN"
    return cfg


def jobs(cfg):
    result = []
    for protocol in cfg["protocols"]:
        for variant in cfg["variants"]:
            for seed in cfg["seeds"]:
                result.append(
                    dict(
                        id=f"{protocol}__{variant}__w320__s{seed}",
                        protocol=protocol,
                        variant=variant,
                        window=320,
                        seed=seed,
                        reused=variant in cfg["reuse"]["variants"],
                        config=cfg,
                    )
                )
    return result


def physical_rows(cfg=None):
    """Build all pooled metrics from explicit, unequal synthetic error arrays."""
    cfg = cfg or config()
    rows = []
    for protocol in cfg["protocols"]:
        pairs = [("alimu", "A", 40, 1.0), ("alimu", "B", 400, 2.0)]
        if protocol == "pooled":
            pairs.append(("penncosyvio", "P", 80, 4.0))
        treatments = [dict(variant="raw", window=0, seed=-1)]
        treatments += [job for job in jobs(cfg) if job["protocol"] == protocol]
        for split in ("validation", "test"):
            for treatment in treatments:
                variant, seed = treatment["variant"], treatment["seed"]
                seed_index = cfg["seeds"].index(seed) if seed != -1 else 0
                pieces = []
                for index, (dataset, pair, count, base) in enumerate(pairs):
                    ratio = 1.0
                    if variant != "raw":
                        ratio = [0.1, 0.8, 0.2][index] * [1, 2, 4][seed_index]
                        if variant == "cnn_gru":
                            ratio *= [1.2, 0.9, 1.1][seed_index]
                        elif variant != cfg["reference_variant"]:
                            ratio *= 1.4 + cfg["variants"].index(variant) * 0.05
                        # VALID and TEST intentionally prefer different structures.
                        if split == "test" and variant == "linear_lstm64":
                            ratio *= 0.01
                    errors = np.tile(np.arange(1.0, 7.0) * base * ratio, (count, 1))
                    pieces.append((dataset, pair, errors))
                layouts = [("overall", "__all__", "__all__", "__all__", pieces)]
                for dataset in dict.fromkeys(piece[0] for piece in pieces):
                    layouts.append(
                        (
                            "dataset",
                            dataset,
                            "__all__",
                            "__all__",
                            [p for p in pieces if p[0] == dataset],
                        )
                    )
                for piece in pieces:
                    layouts.append(("pair", piece[0], piece[1], "__all__", [piece]))
                    layouts.append(("recording", piece[0], piece[1], "shared_recording", [piece]))
                for aggregation, dataset, pair, recording, group in layouts:
                    err = np.concatenate([p[2] for p in group])
                    row = dict(
                        protocol=protocol,
                        split=split,
                        variant=variant,
                        seed=seed,
                        window=treatment["window"],
                        aggregation=aggregation,
                        dataset_id=dataset,
                        pair_id=pair,
                        original_sequence_id=recording,
                        samples=len(err),
                        job_id=treatment.get("id", ""),
                    )
                    for name, section in (("gyro", slice(0, 3)), ("accel", slice(3, 6))):
                        e = err[:, section]
                        row.update(
                            {
                                name + "_rmse": np.sqrt(np.mean(e**2)),
                                name + "_vector_rmse": np.sqrt(np.mean(np.sum(e**2, axis=1))),
                                name + "_mae": np.mean(np.abs(e)),
                                name + "_bias_norm": np.linalg.norm(e.mean(axis=0)),
                            }
                        )
                    row.update(
                        {
                            a + "_rmse": np.sqrt(np.mean(err[:, i] ** 2))
                            for i, a in enumerate(("gx", "gy", "gz", "ax", "ay", "az"))
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def metrics():
    return physical_rows()


def test_physical_pooling_and_balanced_score_are_distinct(metrics):
    api = reporter()
    prepared = api.prepare_metrics(metrics)
    summary = api.summarize_metrics(prepared)
    domain = summary.query(
        "protocol == 'pooled' and split == 'validation' and "
        "variant == 'cnn_lstm64' and aggregation == 'dataset' and "
        "dataset_id == 'alimu' and metric == 'gyro_rmse'"
    ).iloc[0]
    one_seed = np.sqrt((40 * 0.1**2 + 400 * 1.6**2) * (14 / 3) / 440)
    assert domain["mean"] == pytest.approx(one_seed * 7 / 3)
    assert domain.sd == pytest.approx(one_seed * np.std([1, 2, 4], ddof=1))
    assert domain.n == 3 and domain.samples == 440
    balanced = summary.query(
        "protocol == 'pooled' and split == 'validation' and "
        "variant == 'cnn_lstm64' and aggregation == 'overall' and "
        "metric == 'normalized_score'"
    ).iloc[0]
    assert balanced["mean"] == pytest.approx(((0.1 + 0.8) / 2 + 0.2) / 2 * 7 / 3)
    assert balanced.n == 3 and balanced.samples == 520
    assert set(summary.metric) == set([*PHYSICAL, "normalized_score"])
    assert summary.query("variant == 'raw'").n.eq(1).all()
    assert summary.query("variant == 'raw'").sd.isna().all()
    assert not ((summary.aggregation == "recording") & (summary.metric == "normalized_score")).any()


def test_paired_arithmetic_is_per_seed_then_sample_sd(metrics):
    api = reporter()
    deltas = api.paired_deltas(api.prepare_metrics(metrics), "cnn_lstm64")
    cell = deltas.query(
        "protocol == 'pooled' and split == 'validation' and "
        "variant == 'cnn_gru' and aggregation == 'pair' and "
        "pair_id == 'A' and metric == 'gyro_rmse'"
    ).sort_values("seed")
    np.testing.assert_allclose(cell.relative_delta_pct, [20, -10, 10])
    np.testing.assert_allclose(cell.delta, np.sqrt(14 / 3) * np.array([0.02, -0.02, 0.04]))
    assert cell.reference_seed.equals(cell.seed)
    assert cell.reference_window.eq(320).all()
    assert cell.reference_variant.eq("cnn_lstm64").all()
    assert (cell.reference_job_id != cell.job_id).all()
    paired = api.summarize_pairs(deltas)
    row = paired.query(
        "protocol == 'pooled' and split == 'validation' and "
        "variant == 'cnn_gru' and aggregation == 'pair' and "
        "pair_id == 'A' and metric == 'gyro_rmse'"
    ).iloc[0]
    assert row.relative_delta_pct_mean == pytest.approx(20 / 3)
    assert row.relative_delta_pct_sd == pytest.approx(np.std([20, -10, 10], ddof=1))
    assert (row.n, row.relative_n, row.worse_seeds, row.better_seeds, row.tied_seeds) == (
        3,
        3,
        2,
        1,
        0,
    )
    ratio_of_means = 100 * cell.delta.mean() / cell.reference_value.mean()
    assert row.relative_delta_pct_mean != pytest.approx(ratio_of_means)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_seed",
        "wrong_seed",
        "wrong_split",
        "wrong_protocol",
        "wrong_window",
        "wrong_support",
        "wrong_samples",
        "duplicate",
    ],
)
def test_pairing_rejects_missing_or_misaligned_reference(metrics, damage):
    api = reporter()
    frame = api.prepare_metrics(metrics)
    ref = frame.variant.eq("cnn_lstm64") & frame.seed.eq(config()["seeds"][0])
    if damage == "missing_seed":
        frame = frame[~ref]
    elif damage == "duplicate":
        frame = pd.concat([frame, frame.loc[ref]])
    else:
        column, value = dict(
            wrong_seed=("seed", 999),
            wrong_split=("split", "train"),
            wrong_protocol=("protocol", "elsewhere"),
            wrong_window=("window", 40),
            wrong_support=("pair_id", "not_A"),
            wrong_samples=("samples", 999),
        )[damage]
        frame.loc[ref, column] = value
    with pytest.raises(ValueError, match=r"[Rr]eference|[Ss]upport|[Dd]uplicate|[Pp]air"):
        api.paired_deltas(frame, "cnn_lstm64")


def test_zero_reference_percentage_is_nan_not_zero_or_infinity(metrics):
    api = reporter()
    frame = metrics.copy()
    ref = frame.variant.eq("cnn_lstm64") & frame.seed.eq(config()["seeds"][0])
    frame.loc[ref, PHYSICAL] = 0.0
    delta = api.paired_deltas(api.prepare_metrics(frame), "cnn_lstm64")
    row = (
        api.summarize_pairs(delta)
        .query(
            "variant == 'cnn_gru' and protocol == 'pooled' and "
            "split == 'validation' and aggregation == 'pair' and "
            "pair_id == 'A' and metric == 'gyro_rmse'"
        )
        .iloc[0]
    )
    assert row.n == 3 and row.relative_n == 2
    assert row.relative_delta_pct_mean == pytest.approx(0.0)
    assert row.relative_delta_pct_sd == pytest.approx(np.std([-10.0, 10.0], ddof=1))
    assert not np.isinf(delta.relative_delta_pct).any()
    assert delta.loc[delta.seed.eq(config()["seeds"][0]), "relative_delta_pct"].isna().all()


def test_all_zero_references_keep_absolute_directions_and_zero_percentage_n(metrics):
    api = reporter()
    frame = metrics.copy()
    frame.loc[frame.variant.eq("cnn_lstm64"), PHYSICAL] = 0.0
    delta = api.paired_deltas(api.prepare_metrics(frame), "cnn_lstm64")
    paired = api.summarize_pairs(delta)
    assert paired.relative_n.eq(0).all()
    assert paired.relative_delta_pct_mean.isna().all()
    assert paired.relative_delta_pct_sd.isna().all()
    assert paired.n.eq(3).all() and paired.worse_seeds.eq(3).all()
    assert np.isfinite(paired.delta_mean).all()


@pytest.mark.parametrize("damage", ["nan", "infinite", "negative", "fractional_seed", "bad_score"])
def test_invalid_metric_values_are_not_hidden(metrics, damage):
    api = reporter()
    frame = metrics.copy()
    column, value = dict(
        nan=("gyro_rmse", np.nan),
        infinite=("accel_rmse", np.inf),
        negative=("gx_rmse", -1.0),
        fractional_seed=("seed", 2.5),
        bad_score=("normalized_score", 123.0),
    )[damage]
    frame[column] = frame.get(column, pd.Series(np.nan, index=frame.index)).astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        api.prepare_metrics(frame)


def test_validation_ranking_uses_balanced_valid_only(metrics):
    api = reporter()
    frame = api.prepare_metrics(metrics)
    ranking = api.validation_ranking(frame, config())
    assert len(ranking) == 22
    assert ranking.query("rank == 1").variant.eq("cnn_lstm64").all()
    assert ranking.selection_split.eq("validation").all()
    frame.loc[frame.split.eq("test"), "normalized_score"] = np.nan
    pd.testing.assert_frame_equal(ranking, api.validation_ranking(frame, config()))
    without_seed = frame[
        ~(
            frame.split.eq("validation")
            & frame.variant.eq("cnn_gru")
            & frame.seed.eq(config()["seeds"][0])
        )
    ]
    with pytest.raises(ValueError, match=r"[Vv]alidation|[Ss]eed"):
        api.validation_ranking(without_seed, config())


def test_validation_exact_ties_share_rank_without_test_tiebreak(metrics):
    api = reporter()
    frame = api.prepare_metrics(metrics)
    for protocol in config()["protocols"]:
        for seed in config()["seeds"]:
            ref = (
                frame.protocol.eq(protocol)
                & frame.seed.eq(seed)
                & frame.split.eq("validation")
                & frame.aggregation.eq("overall")
            )
            value = frame.loc[ref & frame.variant.eq("cnn_lstm64"), "normalized_score"].iloc[0]
            frame.loc[ref & frame.variant.eq("cnn_gru"), "normalized_score"] = value
    ranking = api.validation_ranking(frame, config())
    assert set(ranking.query("rank == 1").variant) == {"cnn_gru", "cnn_lstm64"}
    frame.loc[frame.split.eq("test"), "normalized_score"] = -1e100
    pd.testing.assert_frame_equal(ranking, api.validation_ranking(frame, config()))


def write_fixture(run):
    api = reporter()
    run.mkdir(parents=True)
    cfg = config()
    (run / "data_manifest.json").write_text('{"synthetic_fixture": true}', encoding="utf-8")
    cfg["data"]["manifest_sha256"] = sha(run / "data_manifest.json")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    planned = jobs(cfg)
    registry, costs = [], []
    for j in planned:
        common = {key: j[key] for key in ("protocol", "variant", "window", "seed", "reused")}
        old_variant = cfg["reuse"]["variants"].get(j["variant"], "")
        source_job = f"{j['protocol']}__{old_variant}__w320__s{j['seed']}" if j["reused"] else ""
        registry.append(
            dict(
                job_id=j["id"],
                **common,
                parameters=81414,
                best_epoch=2,
                actual_epochs=3,
                elapsed_seconds=10.0 + j["seed"] % 3,
                amp_skipped_steps=0,
                checkpoint_sha256=hashlib.sha256(j["id"].encode()).hexdigest(),
                source_run=cfg["reuse"]["run_dir"] if j["reused"] else "",
                source_job_id=source_job,
                source_variant=old_variant,
                fit_kind="reused_control" if j["reused"] else "new_fit",
                training_cost_origin="prior_run" if j["reused"] else "current_run",
                job_config_sha256=hashlib.sha256(
                    json.dumps(j, sort_keys=True).encode()
                ).hexdigest(),
                source_checkpoint_sha256=hashlib.sha256(j["id"].encode()).hexdigest()
                if j["reused"]
                else "",
            )
        )
        costs.append(
            dict(
                job_id=j["id"],
                **common,
                parameters=81414,
                batch1_ms_mean=0.10 + j["seed"] % 3 * 0.01,
                batch1_ms_median=0.09 + j["seed"] % 3 * 0.01,
                latency_origin="current_run_uniform_measurement",
            )
        )
    (run / "planned_jobs.json").write_text(json.dumps(planned), encoding="utf-8")
    pd.DataFrame(registry).to_csv(run / "fit_registry.csv", index=False)
    pd.DataFrame(costs).to_csv(run / "compute_cost.csv", index=False)
    frame = physical_rows(cfg)
    frame["reused"] = frame.variant.isin(cfg["reuse"]["variants"])
    frame.to_csv(run / "metrics_per_seed.csv", index=False)
    sources = api.analysis_source_hashes()
    for relative in sources:
        snapshot = run / "analysis_source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
    (run / "analysis_plan.json").write_text(
        json.dumps(
            dict(
                test_unlocked=False,
                source_sha256=sources,
                reference=cfg["reference_variant"],
                window=320,
                new_fit_count=48,
                reuse_count=18,
                selection="rank mean normalized VALID within protocol; TEST descriptive",
                paired_delta="exact protocol/split/support/window/seed; positive means worse",
                seed_SD="optimization variation, not recording CI",
            )
        ),
        encoding="utf-8",
    )
    provenance = dict(
        git_commit="SYNTHETIC",
        synthetic_fixture=True,
        planned_conditions=66,
        planned_new_fits=48,
        planned_reused_controls=18,
        config_sha256=sha(run / "config.yaml"),
        manifest_sha256=sha(run / "data_manifest.json"),
        test_used_for_selection=False,
        source_sha256={},
        reused_run=cfg["reuse"]["run_dir"],
        previous_TEST_was_seen=True,
    )
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run / "training.complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                planned=66,
                completed=66,
                new_fits=48,
                reused_controls=18,
                fits=registry,
            )
        ),
        encoding="utf-8",
    )
    (run / "evaluation.complete.json").write_text(
        json.dumps(dict(status="complete", conditions=66, prediction_files=132)), encoding="utf-8"
    )
    (run / "selections.lock.json").write_text(
        json.dumps(
            dict(
                test_loaded=False,
                new_fits=48,
                reused_controls=18,
                checkpoints_sha256={r["job_id"]: r["checkpoint_sha256"] for r in registry},
            )
        ),
        encoding="utf-8",
    )
    return run


def corrupt_json(run, name, **updates):
    path = run / name
    content = json.loads(path.read_text(encoding="utf-8"))
    content.update(updates)
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        "config_hash",
        "plan_grid",
        "plan_job",
        "fit_identity",
        "cost_identity",
        "missing_fit",
        "missing_eval",
        "not_complete",
        "checkpoint_lock",
        "test_selection",
        "not_frozen",
        "source_changed",
        "reused_false",
        "source_run",
        "source_job",
        "source_checkpoint",
        "latency_origin",
        "cost_origin",
        "fit_config_hash",
        "embedded_config",
        "fit_kind",
        "missing_evaluation",
        "evaluation_count",
        "metric_reuse",
        "plan_source_variant",
        "registry_source_variant",
    ],
)
def test_report_fails_closed_on_identity_or_provenance_damage(tmp_path, damage):
    api = reporter()
    # Keep any accidental successful RED-phase plot exports in the authorized area too.
    parent = tmp_path
    if damage in ("plan_source_variant", "registry_source_variant"):
        parent = (
            ROOT
            / "runs/module_replacement_preparation_20261107/SYNTHETIC_REPORT"
            / tmp_path.parent.name
        )
    run = write_fixture(parent / damage)
    if damage == "config_hash":
        corrupt_json(run, "provenance.json", config_sha256="broken")
    elif damage in ("plan_grid", "plan_job", "embedded_config", "plan_source_variant"):
        planned = json.loads((run / "planned_jobs.json").read_text(encoding="utf-8"))
        if damage == "plan_source_variant":
            planned[0]["source_variant"] = "wrong_original_model"
            # Rehash the fit binding so only the incorrect source mapping rejects it.
            registry = pd.read_csv(run / "fit_registry.csv")
            registry.loc[0, "job_config_sha256"] = hashlib.sha256(
                json.dumps(planned[0], sort_keys=True).encode()
            ).hexdigest()
            registry.to_csv(run / "fit_registry.csv", index=False)
            complete = json.loads((run / "training.complete.json").read_text(encoding="utf-8"))
            complete["fits"][0]["job_config_sha256"] = registry.loc[0, "job_config_sha256"]
            (run / "training.complete.json").write_text(json.dumps(complete), encoding="utf-8")
        elif damage == "embedded_config":
            planned[0]["config"]["training"]["lr"] = 100.0
        else:
            planned[0]["seed" if damage == "plan_grid" else "id"] = (
                999 if damage == "plan_grid" else "mislinked"
            )
        (run / "planned_jobs.json").write_text(json.dumps(planned), encoding="utf-8")
    elif damage in (
        "fit_identity",
        "cost_identity",
        "missing_fit",
        "reused_false",
        "source_run",
        "source_job",
        "source_checkpoint",
        "latency_origin",
        "cost_origin",
        "fit_config_hash",
        "fit_kind",
        "registry_source_variant",
    ):
        name = (
            "compute_cost.csv"
            if damage in ("cost_identity", "latency_origin")
            else "fit_registry.csv"
        )
        table = pd.read_csv(run / name)
        if damage == "missing_fit":
            table = table.iloc[1:]
        else:
            field, value = dict(
                fit_identity=("seed", 999),
                cost_identity=("variant", "unknown"),
                reused_false=("reused", False),
                source_run=("source_run", "unrelated"),
                source_job=("source_job_id", "pooled__wrong__w40__s999"),
                latency_origin=("latency_origin", "prior_run"),
                cost_origin=("training_cost_origin", "current_run"),
                fit_config_hash=("job_config_sha256", "0" * 64),
                fit_kind=("fit_kind", "new_fit"),
                registry_source_variant=("source_variant", "wrong_original_model"),
                source_checkpoint=("source_checkpoint_sha256", "0" * 64),
            )[damage]
            table.loc[0, field] = value
        table.to_csv(run / name, index=False)
    elif damage == "missing_eval":
        table = pd.read_csv(run / "metrics_per_seed.csv")
        table = table[~(table.variant.eq("cnn_gru") & table.seed.eq(config()["seeds"][0]))]
        table.to_csv(run / "metrics_per_seed.csv", index=False)
    elif damage == "not_complete":
        corrupt_json(run, "training.complete.json", completed=65)
    elif damage == "checkpoint_lock":
        corrupt_json(run, "selections.lock.json", checkpoints_sha256={})
    elif damage == "test_selection":
        corrupt_json(run, "provenance.json", test_used_for_selection=True)
    elif damage == "not_frozen":
        corrupt_json(run, "analysis_plan.json", source_sha256={})
    elif damage == "source_changed":
        corrupt_json(
            run,
            "analysis_plan.json",
            source_sha256={"scripts/report_module_replacement.py": "0" * 64},
        )
    elif damage == "missing_evaluation":
        (run / "evaluation.complete.json").unlink()
    elif damage == "evaluation_count":
        corrupt_json(run, "evaluation.complete.json", prediction_files=131)
    elif damage == "metric_reuse":
        table = pd.read_csv(run / "metrics_per_seed.csv")
        table.loc[table.variant.eq("cnn_lstm64"), "reused"] = False
        table.to_csv(run / "metrics_per_seed.csv", index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        api.report(run)
    assert not (run / "report.md").exists()
    assert not (run / "summary.csv").exists()


def test_unfrozen_analysis_is_rejected_before_metrics_read(tmp_path, caplog):
    api = reporter()
    run = write_fixture(tmp_path / "unfrozen")
    corrupt_json(run, "analysis_plan.json", source_sha256={})
    (run / "metrics_per_seed.csv").write_text("must not read this", encoding="utf-8")
    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match=r"[Aa]nalysis|[Ff]rozen"):
        api.report(run)
    assert not any(
        "READS" in r.message and "metrics_per_seed.csv" in r.message for r in caplog.records
    )


def test_complete_synthetic_report_exports_real_figures_and_all_hashes(tmp_path, caplog):
    api = reporter()
    run = write_fixture(
        ROOT
        / "runs/module_replacement_preparation_20261107/SYNTHETIC_REPORT"
        / f"{tmp_path.parent.name}__{tmp_path.name}"
    )
    original = {
        path.relative_to(run).as_posix(): sha(path) for path in run.rglob("*") if path.is_file()
    }
    sources = api.analysis_source_hashes()
    with caplog.at_level(logging.INFO):
        receipt = api.report(run)
    assert receipt["training_performed"] is False
    assert receipt["counts"] == dict(conditions=66, new=48, reused=18)
    assert receipt["source_sha256"] == sources
    assert api.analysis_source_hashes() == sources
    assert receipt["inputs"] == original
    assert all(sha(run / name) == digest for name, digest in original.items())
    for name in (
        "summary.csv",
        "paired_deltas.csv",
        "paired_summary.csv",
        "validation_ranking.csv",
        "compute_summary.csv",
    ):
        table = pd.read_csv(run / name)
        assert not table.empty and "reused" in table
        assert sha(run / name) == receipt["outputs"][name]
    cost = pd.read_csv(run / "compute_summary.csv")
    assert set(cost.query("metric == 'elapsed_seconds'").cost_scope) == {
        "source_training",
        "new_training",
    }
    assert set(cost.query("metric == 'batch1_ms_mean'").cost_scope) == {"uniform_new_inference"}
    assert set(cost.query("metric == 'batch1_ms_median'").cost_scope) == {"uniform_new_inference"}
    text = (run / "report.md").read_text(encoding="utf-8")
    for phrase in (
        "合成",
        "66",
        "48",
        "18",
        "320",
        "40",
        "样本标准差",
        "独立录制",
        "有效上下文",
        "HPO",
        "VALID",
        "TEST",
        "旧TEST",
        "SYNTHETIC_SOURCE_DO_NOT_OPEN",
        "rad/s",
        "m/s",
        "penncosyvio",
        "alimu",
        "±",
    ):
        assert phrase in text
    assert "置信区间" in text
    summary = pd.read_csv(run / "summary.csv")
    domains = summary.query("aggregation == 'dataset' and variant != 'raw'")
    for row in domains.itertuples():
        assert f"{row.mean:.6g} ± {row.sd:.6g}" in text
    index = (run / "figures/figure_index.md").read_text(encoding="utf-8")
    assert "sample SD" in index and "Positive" in index
    assert str(run) not in index
    for stem in (
        "back_physical",
        "front_physical",
        "back_paired_percent",
        "front_paired_percent",
        "normalized_validation_test",
        "compute_alimu_only",
        "compute_pooled",
    ):
        for ext in ("png", "svg", "pdf"):
            path = run / "figures" / f"{stem}.{ext}"
            assert path.stat().st_size > 1000
            assert sha(path) == receipt["outputs"][path.relative_to(run).as_posix()]
        svg = (run / "figures" / f"{stem}.svg").read_text(encoding="utf-8")
        assert "cnn_lstm64" in svg
    from PIL import Image

    with Image.open(run / "figures/back_physical.png") as picture:
        assert picture.info["dpi"] == pytest.approx((300, 300), abs=0.02)
    assert json.loads((run / "reporting_provenance.json").read_text(encoding="utf-8")) == receipt
    assert all(word in caplog.text for word in ("READS", "WRITES", "SKIPS"))


def test_plot_seed_points_retain_signed_percentages_and_missing_values(
    tmp_path, metrics, monkeypatch
):
    api = reporter()
    from virtual_imu.viz import plots_module_replacement as plots

    deltas = api.paired_deltas(api.prepare_metrics(metrics), "cnn_lstm64")
    observed = []

    def capture(fig, directory, stem):
        for ax in fig.axes:
            for collection in ax.collections:
                if hasattr(collection, "get_offsets"):
                    observed.extend(np.asarray(collection.get_offsets())[:, 0])
        plots.plt.close(fig)
        return []

    monkeypatch.setattr(plots, "save_figure", capture)
    plots.plot_paired_percent(deltas, tmp_path, ["cnn_gru"], "synthetic", config())
    assert min(observed) == pytest.approx(-10.0)
    assert max(observed) == pytest.approx(20.0)
    deltas["relative_delta_pct"] = np.nan
    plots.plot_paired_percent(deltas, tmp_path, ["cnn_gru"], "undefined", config())


def test_main_fails_without_writing_on_missing_run(tmp_path, caplog):
    api = reporter()
    run = tmp_path / "absent"
    with caplog.at_level(logging.INFO):
        assert api.main(["--run-dir", str(run)]) == 2
    assert not run.exists()
    assert "READS" in caplog.text and "SKIPS" in caplog.text
