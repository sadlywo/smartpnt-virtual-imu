"""Exercise independent arithmetic replay with meaningful artifact corruption."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_module_replacement import verify_tables  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def verifier():
    import verify_module_replacement

    return verify_module_replacement


def reporting_fixture():
    spec = importlib.util.spec_from_file_location(
        "verification_reporting_fixture",
        ROOT / "tests/test_module_replacement_reporting.py",
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


@pytest.fixture
def artifacts(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "synthetic_replacement_fixture", ROOT / "tests/test_module_replacement_reporting.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    cfg = helper.config()
    cfg["variants"] = ["cnn_lstm64", "cnn_gru"]
    metrics = helper.physical_rows(cfg)
    api = helper.reporter()
    scored = api.prepare_metrics(metrics)
    deltas = api.paired_deltas(scored)
    api.summarize_metrics(scored).to_csv(tmp_path / "summary.csv", index=False)
    deltas.to_csv(tmp_path / "paired_deltas.csv", index=False)
    api.summarize_pairs(deltas).to_csv(tmp_path / "paired_summary.csv", index=False)
    api.validation_ranking(scored, cfg).to_csv(tmp_path / "validation_ranking.csv", index=False)
    return tmp_path, metrics


def test_independent_table_replay_accepts_known_balanced_synthetic_arrays(artifacts):
    path, metrics = artifacts
    counts, scored = verify_tables(path, metrics)
    assert (
        counts["summary_rows"] > 0
        and counts["paired_rows"] > 0
        and counts["validation_rank_rows"] == 4
    )
    assert scored.normalized_score.notna().any()


@pytest.mark.parametrize(
    "name,column",
    [
        ("summary.csv", "mean"),
        ("summary.csv", "sd"),
        ("paired_deltas.csv", "relative_delta_pct"),
        ("paired_deltas.csv", "reference_seed"),
        ("paired_summary.csv", "worse_seeds"),
        ("paired_summary.csv", "delta_sd"),
        ("validation_ranking.csv", "mean"),
    ],
)
def test_independent_table_replay_rejects_corruption(artifacts, name, column):
    path, metrics = artifacts
    frame = pd.read_csv(path / name)
    index = frame[column].first_valid_index()
    frame.loc[index, column] = frame.loc[index, column] + 7
    frame.to_csv(path / name, index=False)
    with pytest.raises(AssertionError):
        verify_tables(path, metrics)


@pytest.mark.parametrize(
    "name",
    [
        "summary.csv",
        "paired_deltas.csv",
        "paired_summary.csv",
        "validation_ranking.csv",
    ],
)
@pytest.mark.parametrize("damage", ["duplicate_drop", "missing", "extra"])
def test_every_derived_table_requires_an_exact_identity_bijection(artifacts, name, damage):
    path, metrics = artifacts
    frame = pd.read_csv(path / name)
    if damage == "duplicate_drop":
        frame.iloc[-1] = frame.iloc[0]
    elif damage == "missing":
        frame = frame.iloc[:-1]
    else:
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        frame.loc[len(frame) - 1, "protocol"] = "unplanned"
    frame.to_csv(path / name, index=False)
    with pytest.raises(AssertionError):
        verify_tables(path, metrics)


@pytest.mark.parametrize("damage", ["condition", "recording", "pair", "wrong_job", "nan_metric"])
def test_metrics_completeness_is_checked_against_jobs_and_source_supports(artifacts, damage):
    path, metrics = artifacts
    helper = reporting_fixture()
    cfg = helper.config()
    cfg["variants"] = ["cnn_lstm64", "cnn_gru"]
    jobs = helper.jobs(cfg)
    supports = {}
    for (protocol, split), group in metrics[metrics.variant.eq("raw")].groupby(
        ["protocol", "split"]
    ):
        pieces = []
        for row in group[group.aggregation.eq("pair")].itertuples():
            pieces.append(
                pd.DataFrame(
                    {
                        "dataset_id": [row.dataset_id] * row.samples,
                        "pair_id": [row.pair_id] * row.samples,
                        "original_sequence_id": ["shared_recording"] * row.samples,
                    }
                )
            )
        supports[(protocol, split)] = pd.concat(pieces, ignore_index=True)
    verify_tables(path, metrics, jobs=jobs, supports=supports, config=cfg)
    damaged = metrics.copy()
    selected = damaged.variant.eq("cnn_gru") & damaged.seed.eq(cfg["seeds"][0])
    if damage == "condition":
        damaged = damaged[~selected]
    elif damage in ("recording", "pair"):
        damaged = damaged[~(selected & damaged.aggregation.eq(damage))]
    elif damage == "wrong_job":
        damaged.loc[selected, "job_id"] = "different-checkpoint"
    else:
        damaged.loc[selected, "gyro_rmse"] = np.nan
    with pytest.raises(AssertionError):
        verify_tables(path, damaged, jobs=jobs, supports=supports, config=cfg)


def test_validation_exact_ties_have_shared_rank_and_no_test_tiebreak(artifacts):
    path, metrics = artifacts
    helper = reporting_fixture()
    api = helper.reporter()
    frame = metrics.copy()
    identity = [c for c in api.IDENTITY if c != "variant"]
    ref = frame[frame.variant.eq("cnn_lstm64")].set_index(identity)
    mask = frame.variant.eq("cnn_gru")
    for index, row in frame[mask].iterrows():
        if row["split"] == "validation":
            frame.loc[index, api.PHYSICAL] = ref.loc[
                tuple(row[c] for c in identity), api.PHYSICAL
            ].to_numpy()
    cfg = helper.config()
    cfg["variants"] = ["cnn_lstm64", "cnn_gru"]
    scored = api.prepare_metrics(frame)
    deltas = api.paired_deltas(scored)
    for name, table in {
        "summary.csv": api.summarize_metrics(scored),
        "paired_deltas.csv": deltas,
        "paired_summary.csv": api.summarize_pairs(deltas),
        "validation_ranking.csv": api.validation_ranking(scored, cfg),
    }.items():
        table.to_csv(path / name, index=False)
    assert pd.read_csv(path / "validation_ranking.csv")["rank"].eq(1).all()
    verify_tables(path, frame)


@pytest.fixture
def compute_artifacts(tmp_path):
    helper = reporting_fixture()
    run = tmp_path / "synthetic_compute"
    helper.write_fixture(run)
    registry = pd.read_csv(run / "fit_registry.csv")
    registry["source_run"] = registry.source_run.fillna("")
    cost = pd.read_csv(run / "compute_cost.csv")
    helper.reporter().summarize_compute(registry, cost).to_csv(
        run / "compute_summary.csv", index=False
    )
    jobs = json.loads((run / "planned_jobs.json").read_text(encoding="utf-8"))
    return run, jobs


def test_compute_replay_preserves_source_training_and_uniform_inference_costs(compute_artifacts):
    path, jobs = compute_artifacts
    assert verifier().verify_compute(path, jobs=jobs) == 22 * 7


@pytest.mark.parametrize(
    "name,damage",
    [
        ("fit_registry.csv", "missing"),
        ("fit_registry.csv", "duplicate_drop"),
        ("compute_cost.csv", "missing"),
        ("compute_cost.csv", "duplicate_drop"),
        ("compute_cost.csv", "wrong_seed"),
        ("compute_cost.csv", "wrong_reused"),
        ("compute_cost.csv", "wrong_parameters"),
        ("compute_cost.csv", "negative_time"),
        ("compute_cost.csv", "latency_origin"),
        ("compute_summary.csv", "duplicate_drop"),
        ("compute_summary.csv", "missing"),
        ("compute_summary.csv", "extra"),
        ("compute_summary.csv", "wrong_scope"),
        ("compute_summary.csv", "wrong_mean"),
        ("compute_summary.csv", "wrong_sd"),
        ("compute_summary.csv", "wrong_source"),
    ],
)
def test_compute_replay_rejects_identity_capacity_origin_and_arithmetic_damage(
    compute_artifacts, name, damage
):
    path, jobs = compute_artifacts
    frame = pd.read_csv(path / name)
    if damage == "missing":
        frame = frame.iloc[:-1]
    elif damage == "duplicate_drop":
        frame.iloc[-1] = frame.iloc[0]
    elif damage == "extra":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        frame.loc[len(frame) - 1, "protocol"] = "unplanned"
    else:
        column, value = {
            "wrong_seed": ("seed", 123),
            "wrong_reused": ("reused", False),
            "wrong_parameters": ("parameters", 999),
            "negative_time": ("batch1_ms_mean", -1),
            "latency_origin": ("latency_origin", "source_run"),
            "wrong_scope": ("cost_scope", "new_training"),
            "wrong_mean": ("mean", 12345),
            "wrong_sd": ("sd", 12345),
            "wrong_source": ("source_run", "another_run"),
        }[damage]
        frame.loc[0, column] = value
    frame.to_csv(path / name, index=False)
    with pytest.raises(AssertionError):
        verifier().verify_compute(path, jobs=jobs)


@pytest.mark.parametrize("damage", ["source_run", "native_job", "fit_kind", "job_digest"])
def test_compute_provenance_cannot_be_rewritten_coherently_against_the_plan(
    compute_artifacts, damage
):
    path, jobs = compute_artifacts
    registry = pd.read_csv(path / "fit_registry.csv")
    table = pd.read_csv(path / "compute_summary.csv")
    selected = registry.protocol.eq("alimu_only") & registry.variant.eq("cnn_lstm64")
    if damage == "source_run":
        registry.loc[selected, "source_run"] = "substituted_source"
        table.loc[
            table.protocol.eq("alimu_only") & table.variant.eq("cnn_lstm64"), "source_run"
        ] = "substituted_source"
        table.to_csv(path / "compute_summary.csv", index=False)
    else:
        column, value = {
            "native_job": ("source_job_id", "another_job"),
            "fit_kind": ("fit_kind", "new_fit"),
            "job_digest": ("job_config_sha256", "bad"),
        }[damage]
        registry.loc[selected, column] = value
    registry.to_csv(path / "fit_registry.csv", index=False)
    with pytest.raises(AssertionError):
        verifier().verify_compute(path, jobs=jobs)


@pytest.mark.parametrize(
    "name,column",
    [
        ("summary.csv", "source_variant"),
        ("summary.csv", "source_run"),
        ("paired_deltas.csv", "source_job_id"),
        ("paired_deltas.csv", "reference_source_run"),
        ("paired_summary.csv", "reference_reused"),
        ("validation_ranking.csv", "source_variant"),
    ],
)
def test_registry_origin_is_verified_when_source_variant_is_derived(
    compute_artifacts, name, column
):
    path, jobs = compute_artifacts
    helper = reporting_fixture()
    reporter = helper.reporter()
    registry = pd.read_csv(path / "fit_registry.csv").fillna(
        {"source_run": "", "source_variant": "", "source_job_id": ""}
    )
    metrics = pd.read_csv(path / "metrics_per_seed.csv").fillna({"job_id": ""})
    annotated = reporter._annotate(reporter.prepare_metrics(metrics), registry)
    paired = reporter.paired_deltas(annotated)
    for filename, table in {
        "summary.csv": reporter.summarize_metrics(annotated),
        "paired_deltas.csv": paired,
        "paired_summary.csv": reporter.summarize_pairs(paired),
        "validation_ranking.csv": reporter.validation_ranking(annotated, helper.config()),
    }.items():
        table.to_csv(path / filename, index=False)
    registry.drop(columns="source_variant").to_csv(path / "fit_registry.csv", index=False)
    verify_tables(path, metrics, jobs=jobs)
    damaged = pd.read_csv(path / name)
    index = damaged[column].first_valid_index()
    damaged.loc[index, column] = False if column == "reference_reused" else "substituted"
    damaged.to_csv(path / name, index=False)
    with pytest.raises(AssertionError):
        verify_tables(path, metrics, jobs=jobs)


def source_frame(length, sequence="one", shift=0):
    time = np.arange(length, dtype=np.float64)
    low = time[:, None] * np.arange(1, 7)[None, :] / 100 + shift
    frame = pd.DataFrame(low, columns=verifier().LOW)
    frame[verifier().HIGH] = low + np.arange(1, 7)
    frame["dataset_id"] = "alimu"
    frame["pair_id"] = "A"
    frame["sequence_id"] = sequence
    frame["original_sequence_id"] = sequence
    frame["timestamp_s"] = time / 40
    return frame


def test_independent_windows_use_train_stats_center40_and_never_cross_recordings():
    api = verifier()
    frames = [source_frame(367), source_frame(401, "two", 100), source_frame(100, "short", 999)]
    train = api.reconstruct_windows(frames, role="train")
    raw = pd.concat(frames[:2], ignore_index=True)
    np.testing.assert_allclose(train["stats"]["feature_mean"], raw[api.LOW].mean())
    np.testing.assert_allclose(train["stats"]["feature_std"], raw[api.LOW].std(ddof=0))
    np.testing.assert_allclose(train["stats"]["residual_mean"], np.arange(1, 7))
    np.testing.assert_array_equal(train["stats"]["residual_std"], np.ones(6))
    result = api.reconstruct_windows(frames, role="validation", stats=train["stats"])
    assert result["inputs"].shape == (5, 6, 320)
    target = result["target_frame"]
    assert len(target) == 200
    np.testing.assert_allclose(target.iloc[:40].timestamp_s, np.arange(140, 180) / 40)
    assert target.iloc[:80].sequence_id.eq("one").all()
    assert target.iloc[80:].sequence_id.eq("two").all()
    expected = (
        (
            (frames[1].iloc[:320][api.LOW].to_numpy() - train["stats"]["feature_mean"])
            / train["stats"]["feature_std"]
        )
        .astype(np.float32)
        .T
    )
    np.testing.assert_array_equal(result["inputs"][2], expected)


class AffineReplay(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batches = []

    def forward(self, x):
        self.batches.append((len(x), x.shape[-1], self.training, torch.is_grad_enabled()))
        return x * 0.25 + 0.5


def replay_fixture():
    api = verifier()
    stats = {
        "feature_mean": [1.0] * 6,
        "feature_std": [2.0] * 6,
        "residual_mean": [3.0] * 6,
        "residual_std": [0.1] * 6,
    }
    source = source_frame(320 + 65 * 40)
    rebuilt = api.reconstruct_windows([source], role="test", stats=stats)
    normalized = rebuilt["inputs"][..., 140:180] * np.float32(0.25) + np.float32(0.5)
    expected = (
        normalized.transpose(0, 2, 1).reshape(-1, 6).astype(np.float64) * 0.1
        + 3
        + rebuilt["target_frame"][api.LOW].to_numpy()
    )
    identity = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in [
            "checkpoint_sha256",
            "initialization_sha256",
            "config_sha256",
            "support_sha256",
        ]
    }
    identity.update(job_id="synthetic__cnn_gru__w320__s1", split="test")
    return rebuilt, stats, expected, identity


def test_direct_reinference_batches64_preserves_order_eval_and_physical_units():
    api = verifier()
    rebuilt, stats, expected, _ = replay_fixture()
    model = AffineReplay()
    actual = api.independent_forward(
        model,
        rebuilt["inputs"],
        rebuilt["target_frame"],
        stats,
        device="cpu",
        amp=False,
        batch_size=64,
    )
    np.testing.assert_array_equal(actual, expected)
    assert model.batches == [(64, 320, False, False), (2, 320, False, False)]


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "test_array",
        "shape",
        "nan",
        "missing_identity",
        "job_id",
        "split",
        "checkpoint_sha256",
        "initialization_sha256",
        "config_sha256",
        "support_sha256",
    ],
)
def test_prediction_replay_rejects_substitution_even_with_coherent_metrics_and_metadata(
    tmp_path, damage
):
    api = verifier()
    rebuilt, stats, expected, identity = replay_fixture()
    contents = {"prediction": expected.copy(), **identity}
    if damage == "test_array":
        contents["prediction"][0, 0] += 0.25
        # All reported metric arithmetic can agree with the substituted array.
        from virtual_imu.ablations.data import physical_metrics

        rows = physical_metrics(rebuilt["target_frame"], contents["prediction"])
        for row in rows:
            api.replay_row(rebuilt["target_frame"], contents["prediction"], row)
        assert all(contents[key] == value for key, value in identity.items())
    elif damage == "shape":
        contents["prediction"] = contents["prediction"][:1]
    elif damage == "nan":
        contents["prediction"][0, 0] = np.nan
    elif damage == "missing_identity":
        del contents["checkpoint_sha256"]
    elif damage != "none":
        contents[damage] = "different"
    path = tmp_path / "synthetic_test.npz"
    np.savez_compressed(path, **contents)
    if damage == "none":
        receipt = api.replay_prediction_file(
            path, AffineReplay(), rebuilt, stats, identity, device="cpu", amp=False, batch_size=64
        )
        assert receipt["samples"] == len(expected) and receipt["max_abs_error"] == 0
    else:
        with pytest.raises(AssertionError):
            api.replay_prediction_file(
                path,
                AffineReplay(),
                rebuilt,
                stats,
                identity,
                device="cpu",
                amp=False,
                batch_size=64,
            )


def test_comparison_declares_tolerance_and_refuses_broadcasting_or_nonfinite_values():
    api = verifier()
    expected = np.ones((40, 6))
    receipt = api.compare_predictions(expected + api.REPLAY_ATOL / 2, expected)
    assert receipt["rtol"] == api.REPLAY_RTOL and receipt["atol"] == api.REPLAY_ATOL
    for damaged in [np.ones((1, 6)), np.full_like(expected, np.nan), expected + 0.01]:
        with pytest.raises(AssertionError):
            api.compare_predictions(damaged, expected)


@pytest.fixture
def lock_artifacts(tmp_path):
    api = verifier()
    (tmp_path / "config.yaml").write_text("synthetic: true\n", encoding="utf-8")
    (tmp_path / "analysis_plan.json").write_text("{}", encoding="utf-8")
    directory = tmp_path / "fits/job"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"synthetic checkpoint")
    (directory / "initialization.json").write_text("{}", encoding="utf-8")
    job = {"id": "job", "directory": str(directory), "reused": False}
    jobs = [job]
    lock = {
        "test_loaded": False,
        "checkpoints_sha256": {"job": api.sha(directory / "best.pt")},
        "initialization_sha256": {"job": api.sha(directory / "initialization.json")},
        "config_sha256": api.sha(tmp_path / "config.yaml"),
        "jobs_sha256": hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
        "analysis_plan_sha256": api.sha(tmp_path / "analysis_plan.json"),
        "new_fits": 1,
        "reused_controls": 0,
    }
    (tmp_path / "selections.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    access = {
        "first_access_utc": "2026-11-07T14:00:00+00:00",
        "numerical_TEST_access": True,
        "selection_lock_sha256": api.sha(tmp_path / "selections.lock.json"),
    }
    return tmp_path, jobs, lock, access


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "checkpoint",
        "initialization",
        "config",
        "analysis",
        "jobs",
        "access_lock",
        "missing_access",
        "no_test_access",
        "timestamp",
    ],
)
def test_lock_and_durable_test_access_bind_every_replay_identity(lock_artifacts, damage):
    path, jobs, lock, access = lock_artifacts
    if damage in ("checkpoint", "initialization"):
        filename = "best.pt" if damage == "checkpoint" else "initialization.json"
        (path / "fits/job" / filename).write_bytes(b"changed")
    elif damage in ("config", "analysis"):
        filename = "config.yaml" if damage == "config" else "analysis_plan.json"
        (path / filename).write_bytes(b"changed")
    elif damage == "jobs":
        jobs = copy.deepcopy(jobs)
        jobs[0]["extra"] = "changed"
    elif damage == "access_lock":
        access["selection_lock_sha256"] = "wrong"
    elif damage == "missing_access":
        access = None
    elif damage == "no_test_access":
        access["numerical_TEST_access"] = False
    elif damage == "timestamp":
        access["first_access_utc"] = "not-a-time"
    if damage == "none":
        verifier().verify_lock_identity(path, jobs, lock, access)
    else:
        with pytest.raises(AssertionError):
            verifier().verify_lock_identity(path, jobs, lock, access)


@pytest.mark.parametrize("replace_lock_during_replay", [False, True])
def test_formal_driver_replays_all132_files_with_only_synthetic_sources_and_cpu_math(
    tmp_path, monkeypatch, replace_lock_during_replay
):
    """Exercise orchestration, with only the CUDA hardware boundary substituted."""
    import yaml

    from virtual_imu.ablations import models as old_models
    from virtual_imu.ablations.data import physical_metrics
    from virtual_imu.replacements import models as new_models

    api = verifier()
    helper = reporting_fixture()
    reporter = helper.reporter()
    root = tmp_path / "SYNTHETIC_PROJECT"
    run = root / "run"
    run.mkdir(parents=True)
    cfg = helper.config()
    cfg["reuse"]["run_dir"] = "old"
    cfg["data"]["manifest"] = "manifest.json"

    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    def toy_model(seed):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = torch.nn.Conv1d(6, 6, 1)
        with torch.no_grad():
            model.weight.copy_(torch.eye(6).unsqueeze(-1) * 0.25)
            model.bias.fill_(0.5)
        return model

    def builder(variant, seed, *_args):
        model = toy_model(seed)
        metadata = dict(
            variant=variant,
            seed=seed,
            init_state_sha256=api.state_digest(model.state_dict()),
            retained_keys=sorted(model.state_dict()),
        )
        return model, metadata

    records, raw_frames = [], {}
    for dataset, role in __import__("itertools").product(
        ["alimu", "penncosyvio"], ["train", "validation", "test"]
    ):
        frame = source_frame(360, dataset + "_" + role, 10 if dataset != "alimu" else 0)
        frame["dataset_id"] = dataset
        relative = "source/" + dataset + "_" + role + ".parquet"
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / relative, index=False)
        record = dict(
            dataset_id=dataset,
            split=role,
            path=relative,
            sha256=api.sha(root / relative),
            rows=len(frame),
            physical_group_id=dataset + "_" + role,
            task="paired_signal",
            training_eligible=True,
        )
        records.append(record)
        raw_frames[(dataset, role)] = frame
    write_json(root / "manifest.json", dict(records=records, rate_hz=40))
    cfg["data"]["manifest_sha256"] = api.sha(root / "manifest.json")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    reconstructions, statistics = {}, {}
    for protocol in cfg["protocols"]:
        domains = ["alimu", "penncosyvio"] if protocol == "pooled" else ["alimu"]
        roles = {
            role: [r for r in records if r["dataset_id"] in domains and r["split"] == role]
            for role in ["train", "validation", "test"]
        }
        directory = run / "supports" / protocol
        write_json(directory / "records.json", roles)
        for role in ["train", "validation", "test"]:
            rebuilt = api.reconstruct_windows(
                [raw_frames[(d, role)] for d in domains], role=role, stats=statistics.get(protocol)
            )
            if role == "train":
                statistics[protocol] = rebuilt["stats"]
                write_json(directory / "statistics.json", rebuilt["stats"])
            rebuilt["target_frame"].to_parquet(directory / (role + ".parquet"), index=False)
            reconstructions[(protocol, role)] = rebuilt
    planned = []
    for job in helper.jobs(cfg):
        planned.append(dict(job, root=str(root), directory=str(run / "fits" / job["id"])))
    write_json(run / "planned_jobs.json", planned)
    metrics, registry, costs = [], [], []
    checkpoint_hashes, init_hashes = {}, {}
    for protocol in cfg["protocols"]:
        for role in ["validation", "test"]:
            target = reconstructions[(protocol, role)]["target_frame"]
            metrics.extend(
                dict(
                    protocol=protocol,
                    split=role,
                    variant="raw",
                    window=0,
                    seed=-1,
                    job_id="",
                    reused=False,
                    **row,
                )
                for row in physical_metrics(target, target[api.LOW].to_numpy())
            )
    for job in planned:
        directory = Path(job["directory"])
        directory.mkdir(parents=True)
        model, meta = builder(job["variant"], job["seed"])
        write_json(directory / "initialization.json", meta)
        init_hashes[job["id"]] = api.sha(directory / "initialization.json")
        stats = statistics[job["protocol"]]
        predictions = {}
        for role in ["validation", "test"]:
            reconstruction = reconstructions[(job["protocol"], role)]
            normalized = reconstruction["inputs"][..., 140:180] * np.float32(0.25) + np.float32(0.5)
            pred = normalized.transpose(0, 2, 1).reshape(-1, 6).astype(np.float64)
            pred = (
                pred * stats["residual_std"]
                + stats["residual_mean"]
                + reconstruction["target_frame"][api.LOW].to_numpy()
            )
            predictions[role] = pred
            metrics.extend(
                dict(
                    protocol=job["protocol"],
                    split=role,
                    variant=job["variant"],
                    window=320,
                    seed=job["seed"],
                    job_id=job["id"],
                    reused=job["reused"],
                    **row,
                )
                for row in physical_metrics(reconstruction["target_frame"], pred)
            )
        # Compute VALID selection independently of the verifier driver.
        val_rows = pd.DataFrame([r for r in metrics if r["job_id"] == job["id"]])
        raw_rows = pd.DataFrame(
            [r for r in metrics if r["variant"] == "raw" and r["protocol"] == job["protocol"]]
        )
        scored = reporter.prepare_metrics(pd.concat([val_rows, raw_rows], ignore_index=True))
        score = scored[
            scored.variant.ne("raw")
            & scored.split.eq("validation")
            & scored.aggregation.eq("overall")
        ].normalized_score.iloc[0]
        train = reconstructions[(job["protocol"], "train")]
        generator = torch.Generator().manual_seed(job["seed"] + 1000003)
        labels = train["labels"]
        weights = torch.tensor(
            [1 / np.count_nonzero(labels == d) for d in labels], dtype=torch.float64
        )
        order = torch.multinomial(weights, len(labels), True, generator=generator).numpy()
        order_hash = hashlib.sha256(order.tobytes()).hexdigest()
        native = job
        native_meta = meta
        source_id = ""
        if job["reused"]:
            legacy = cfg["reuse"]["variants"][job["variant"]]
            source_id = f"{job['protocol']}__{legacy}__w320__s{job['seed']}"
            native = dict(job, id=source_id, variant=legacy)
            native_meta = builder(legacy, job["seed"])[1]
        saved = dict(
            model=model.state_dict(),
            job=native,
            metadata=native_meta,
            epoch=1,
            val_score=score,
            stats=stats,
            train_support_sha256=api.support_hash(train["target_frame"]),
            validation_support_sha256=api.support_hash(
                reconstructions[(job["protocol"], "validation")]["target_frame"]
            ),
        )
        torch.save(saved, directory / "best.pt")
        digest = api.sha(directory / "best.pt")
        checkpoint_hashes[job["id"]] = digest
        record = dict(
            job_id=job["id"],
            **{k: job[k] for k in ["protocol", "variant", "window", "seed", "reused"]},
            checkpoint_sha256=digest,
            job_config_sha256=api.json_digest(job),
            best_epoch=1,
            best_val_score=score,
            actual_epochs=1,
            elapsed_seconds=1.0,
            amp_skipped_steps=0,
            order_sha256=order_hash,
            train_support_sha256=saved["train_support_sha256"],
            validation_support_sha256=saved["validation_support_sha256"],
            parameters=42,
            training_cost_origin="prior_run" if job["reused"] else "current_run",
            fit_kind="reused_control" if job["reused"] else "new_fit",
            source_run=str(root / "old") if job["reused"] else "",
            source_job_id=source_id,
            source_variant=cfg["reuse"]["variants"].get(job["variant"], ""),
        )
        pd.DataFrame(
            [
                dict(
                    epoch=1,
                    train_loss=0.1,
                    val_score=score,
                    lr=0.001,
                    amp_skipped_steps=0,
                    order_sha256=order_hash,
                )
            ]
        ).to_csv(directory / "history.csv", index=False)
        write_json(directory / "job.json", job)
        write_json(directory / "fit.json", record)
        write_json(directory / "stats.json", stats)
        if job["reused"]:
            origin = root / "old/fits" / source_id
            origin.mkdir(parents=True)
            (origin / "best.pt").write_bytes((directory / "best.pt").read_bytes())
            write_json(directory / "source_initialization.json", native_meta)
            write_json(
                directory / "import.json",
                dict(
                    training_performed=False,
                    source_job=native,
                    source_fit={"checkpoint_sha256": digest},
                    source_sha256={str(origin / "best.pt"): digest},
                ),
            )
        for role, pred in predictions.items():
            target = reconstructions[(job["protocol"], role)]["target_frame"]
            dest = run / "predictions"
            dest.mkdir(exist_ok=True)
            np.savez_compressed(
                dest / (job["id"] + "__" + role + ".npz"),
                prediction=pred,
                job_id=job["id"],
                split=role,
                checkpoint_sha256=digest,
                initialization_sha256=init_hashes[job["id"]],
                config_sha256=api.sha(run / "config.yaml"),
                support_sha256=api.support_hash(target),
            )
        registry.append(record)
        costs.append(
            dict(
                job_id=job["id"],
                **{k: job[k] for k in ["protocol", "variant", "window", "seed", "reused"]},
                parameters=42,
                batch1_ms_mean=0.1,
                batch1_ms_median=0.1,
                latency_origin="current_run_uniform_measurement",
            )
        )
    registry, costs, metrics = pd.DataFrame(registry), pd.DataFrame(costs), pd.DataFrame(metrics)
    registry.to_csv(run / "fit_registry.csv", index=False)
    costs.to_csv(run / "compute_cost.csv", index=False)
    metrics.to_csv(run / "metrics_per_seed.csv", index=False)
    annotated = reporter._annotate(reporter.prepare_metrics(metrics), registry)
    paired = reporter.paired_deltas(annotated)
    for name, frame in {
        "summary.csv": reporter.summarize_metrics(annotated),
        "paired_deltas.csv": paired,
        "paired_summary.csv": reporter.summarize_pairs(paired),
        "validation_ranking.csv": reporter.validation_ranking(annotated, cfg),
        "compute_summary.csv": reporter.summarize_compute(registry, costs),
    }.items():
        frame.to_csv(run / name, index=False)
    write_json(root / "old/complete.json", {"synthetic": True})
    write_json(
        run / "provenance.json",
        dict(
            config_sha256=api.sha(run / "config.yaml"),
            manifest_sha256=cfg["data"]["manifest_sha256"],
            source_sha256={},
            reused_run_complete_sha256=api.sha(root / "old/complete.json"),
        ),
    )
    write_json(run / "analysis_plan.json", dict(test_unlocked=False, source_sha256={}))
    write_json(
        run / "training.complete.json",
        dict(
            planned=66,
            completed=66,
            new_fits=48,
            reused_controls=18,
            fits=registry.to_dict("records"),
        ),
    )
    lock = dict(
        test_loaded=False,
        checkpoints_sha256=checkpoint_hashes,
        initialization_sha256=init_hashes,
        config_sha256=api.sha(run / "config.yaml"),
        jobs_sha256=api.json_digest(planned),
        analysis_plan_sha256=api.sha(run / "analysis_plan.json"),
        new_fits=48,
        reused_controls=18,
    )
    write_json(run / "selections.lock.json", lock)
    write_json(
        run / "test_access.json",
        dict(
            first_access_utc="2026-11-07T14:00:00+00:00",
            numerical_TEST_access=True,
            selection_lock_sha256=api.sha(run / "selections.lock.json"),
        ),
    )
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(old_models, "build_ablation_model", builder)
    monkeypatch.setattr(new_models, "build_replacement_model", builder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    direct = api.independent_forward
    calls = []

    def offline_cuda_boundary(model, inputs, target, stats, *, device, amp, batch_size):
        calls.append((str(device), amp, batch_size))
        if replace_lock_during_replay and len(calls) == 132:
            checkpoint = Path(planned[0]["directory"]) / "best.pt"
            changed = torch.load(checkpoint, map_location="cpu")
            changed["model"]["weight"].add_(0.1)
            torch.save(changed, checkpoint)
            changed_lock = copy.deepcopy(lock)
            changed_lock["checkpoints_sha256"][planned[0]["id"]] = api.sha(checkpoint)
            write_json(run / "selections.lock.json", changed_lock)
            changed_access = json.loads((run / "test_access.json").read_text(encoding="utf-8"))
            changed_access["selection_lock_sha256"] = api.sha(run / "selections.lock.json")
            write_json(run / "test_access.json", changed_access)
        return direct(model, inputs, target, stats, device="cpu", amp=False, batch_size=batch_size)

    monkeypatch.setattr(api, "independent_forward", offline_cuda_boundary)
    if replace_lock_during_replay:
        with pytest.raises(AssertionError, match=r"lock|Lock|changed|differ"):
            api.verify(run)
        assert not (run / "verification.json").exists()
        return
    api.verify(run)
    receipt = json.loads((run / "verification.json").read_text(encoding="utf-8"))
    assert len(calls) == 132 and set(calls) == {("cuda", True, 64)}
    assert receipt["independent_inference_files"] == 132
    assert receipt["independent_inference_all_samples"] is True
    assert {(r["job_id"], r["split"]) for r in receipt["prediction_replays"]} == {
        (j["id"], role) for j in planned for role in ["validation", "test"]
    }
