"""Numerical/reporting checks using explicitly synthetic run-local artifacts."""

# Python3.9 compatibility.
# ruff: noqa: B905
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_cnn_bilstm_ablation.py"
PRIMARY = ["cnn_bilstm", "bilstm_only", "cnn_only", "mlp_only", "cnn_lstm"]
EXTRA = ["cnn_lstm_wide", "pointwise_bilstm", "cnn_bilstm_dense"]
SEEDS = [11, 12, 13]


def reporter():
    assert SCRIPT.is_file(), "Reporting entry point has not been implemented"
    spec = importlib.util.spec_from_file_location("cnn_ablation_reporting", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return dict(
        experiment_id="SYNTHETIC_reporting_fixture",
        protocols=["alimu_only", "pooled"],
        variants=PRIMARY,
        supplemental=EXTRA,
        seeds=SEEDS,
        windows=[40, 320],
        data=dict(rate_hz=40, max_window=320, block_size=40, stride=40),
        model=dict(
            hidden_channels=64,
            conv_depth=3,
            kernel_size=5,
            lstm_hidden=32,
            dropout=0,
            in_channels=6,
            out_channels=6,
        ),
        training=dict(epochs=80, early_stopping_patience=15, optimizer="AdamW"),
        evaluation=dict(latency_warmup=5, latency_repeats=20),
    )


def jobs():
    return [
        dict(id=f"{p}__{v}__w{w}__s{s}", protocol=p, variant=v, window=w, seed=s)
        for p in ("alimu_only", "pooled")
        for v in PRIMARY + EXTRA
        for w in ([40, 320] if v in PRIMARY else [320])
        for s in SEEDS
    ]


def fixture_metrics():
    rows = []
    offsets = dict(zip(PRIMARY + EXTRA, [0, 0.08, 0.10, 0.15, 0.04, 0.03, 0.02, -0.01]))
    for protocol in ("alimu_only", "pooled"):
        # Unequal pair and dataset masses distinguish the balanced selection
        # score from a pooled RMSE ratio. rec_shared is deliberately reused.
        pairs = [("alimu", "A", 4, 1.0), ("alimu", "B", 40, 2.0)]
        if protocol == "pooled":
            pairs += [("penncosyvio", "P", 4, 4.0)]
        treatments = [dict(variant="raw", window=0, seed=-1)]
        treatments += [j for j in jobs() if j["protocol"] == protocol]
        for split in ("validation", "test"):
            for treatment in treatments:
                variant, window, seed = (treatment[k] for k in ("variant", "window", "seed"))
                pieces = []
                for index, (dataset, pair, count, base) in enumerate(pairs):
                    if variant == "raw":
                        ratio = 1.0
                    else:
                        ratio = ([0.1, 0.8, 0.2] if window == 40 else [0.6, 0.2, 0.5])[index]
                        if split == "test":
                            ratio = 0.9 if window == 40 else 0.1
                        ratio = (ratio + offsets[variant]) * (1 + 0.01 * (seed - 12))
                    error = np.tile(
                        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * base * ratio, (count, 1)
                    )
                    pieces.append((dataset, pair, error))
                layouts = [("overall", "__all__", "__all__", "__all__", pieces)]
                for dataset in dict.fromkeys(p[0] for p in pieces):
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
                    layouts.append(("recording", piece[0], piece[1], "rec_shared", [piece]))
                for aggregation, dataset, pair, recording, group in layouts:
                    err = np.concatenate([p[2] for p in group])
                    row = dict(
                        protocol=protocol,
                        split=split,
                        variant=variant,
                        window=window,
                        seed=seed,
                        aggregation=aggregation,
                        dataset_id=dataset,
                        pair_id=pair,
                        original_sequence_id=recording,
                        samples=len(err),
                        job_id=treatment.get("id", ""),
                    )
                    for name, sl in (("gyro", slice(0, 3)), ("accel", slice(3, 6))):
                        e = err[:, sl]
                        row.update(
                            {
                                name + "_rmse": np.sqrt(np.mean(e**2)),
                                name + "_vector_rmse": np.sqrt(np.mean(np.sum(e**2, axis=1))),
                                name + "_mae": np.mean(np.abs(e)),
                                name + "_bias_norm": np.linalg.norm(e.mean(0)),
                            }
                        )
                    row.update(
                        {
                            a + "_rmse": np.sqrt(np.mean(err[:, i] ** 2))
                            for i, a in enumerate(["gx", "gy", "gz", "ax", "ay", "az"])
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def test_summary_seed_sd_raw_and_physical_pooling():
    api = reporter()
    frame = api.prepare_metrics(fixture_metrics())
    result = api.summarize_metrics(frame)
    row = result.query(
        "protocol == 'pooled' and split == 'test' and variant == 'cnn_bilstm' and window == 320 and aggregation == 'dataset' and dataset_id == 'alimu' and metric == 'gyro_rmse'"
    ).iloc[0]
    physical = np.sqrt((4 * 14 / 3 + 40 * 4 * 14 / 3) / 44) * 0.1
    assert row["mean"] == pytest.approx(physical)
    assert row.sd == pytest.approx(physical * 0.01)
    assert row.n == 3 and row.samples == 44
    raw = result.query("variant == 'raw'")
    assert raw.n.eq(1).all() and raw.sd.isna().all()
    # Identical duplicate inputs must not inflate n; reject them explicitly.
    with pytest.raises(ValueError, match=r"[Dd]uplicate"):
        api.prepare_metrics(pd.concat([fixture_metrics(), fixture_metrics().iloc[:1]]))


def test_exact_seed_window_pairs_and_factorial_sign():
    api = reporter()
    frame = api.prepare_metrics(fixture_metrics())
    module = api.paired_deltas(frame, comparison="module")
    row = module.query(
        "protocol == 'pooled' and split == 'test' and variant == 'bilstm_only' and window == 40 and seed == 11 and aggregation == 'pair' and pair_id == 'A' and metric == 'gyro_rmse'"
    ).iloc[0]
    assert row.reference_window == 40 and row.reference_seed == 11
    assert row.relative_delta_pct == pytest.approx(100 * 0.08 / 0.9)
    windows = api.paired_deltas(frame, comparison="window")
    assert windows.reference_window.eq(320).all()
    assert windows.reference_variant.eq(windows.variant).all()
    interaction = api.factorial_interactions(frame)
    row = interaction.query(
        "protocol == 'pooled' and split == 'test' and window == 320 and seed == 12 and aggregation == 'pair' and pair_id == 'A' and metric == 'gyro_rmse'"
    ).iloc[0]
    assert row.interaction == pytest.approx(-0.03 * np.sqrt(14 / 3))


def test_selection_is_pair_dataset_balanced_and_ignores_test():
    api = reporter()
    frame = api.prepare_metrics(fixture_metrics())
    selected = api.select_validation_windows(frame, config())
    row = selected.query("protocol == 'pooled' and variant == 'cnn_bilstm'").iloc[0]
    assert row.selected_window == 40
    assert row.score_mean == pytest.approx((0.45 + 0.2) / 2)
    frame.loc[frame.split.eq("test"), "normalized_score"] = 1000
    pd.testing.assert_frame_equal(selected, api.select_validation_windows(frame, config()))
    # No overall-score shortcut: pooled physical weighting prefers window320.
    val = frame.query(
        "protocol == 'pooled' and split == 'validation' and variant == 'cnn_bilstm' and aggregation == 'overall'"
    )
    assert val.groupby("window").gyro_rmse.mean().idxmin() == 320


@pytest.mark.parametrize("field,value", [("gyro_rmse", np.nan), ("samples", 99), ("seed", 11.5)])
def test_invalid_metrics_fail_instead_of_generating_results(field, value):
    api = reporter()
    frame = fixture_metrics()
    frame[field] = frame[field].astype(float)
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        api.prepare_metrics(frame)


def write_fixture(run):
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config()), encoding="utf-8")
    (run / "data_manifest.json").write_text('{"synthetic": true}', encoding="utf-8")

    def digest(name):
        return hashlib.sha256((run / name).read_bytes()).hexdigest()

    provenance = dict(
        git_commit="SYNTHETIC",
        planned_fits=len(jobs()),
        synthetic_fixture=True,
        config_sha256=digest("config.yaml"),
        manifest_sha256=digest("data_manifest.json"),
        test_used_for_selection=False,
        source_sha256={},
    )
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run / "planned_jobs.json").write_text(json.dumps(jobs()), encoding="utf-8")
    registry, costs = [], []
    for j in jobs():
        common = {k: j[k] for k in ("protocol", "variant", "window", "seed")}
        registry.append(
            dict(
                job_id=j["id"],
                **common,
                parameters=100,
                best_epoch=2,
                actual_epochs=3,
                best_val_score=1.0,
                elapsed_seconds=2.0,
                amp_skipped_steps=0,
                checkpoint_sha256="synthetic_" + j["id"],
            )
        )
        costs.append(
            dict(
                job_id=j["id"], **common, parameters=100, batch1_ms_mean=0.1, batch1_ms_median=0.09
            )
        )
    pd.DataFrame(registry).to_csv(run / "fit_registry.csv", index=False)
    pd.DataFrame(costs).to_csv(run / "compute_cost.csv", index=False)
    fixture_metrics().to_csv(run / "metrics_per_seed.csv", index=False)
    (run / "training.complete.json").write_text(
        json.dumps(dict(status="complete", planned=len(jobs()), completed=len(jobs())))
    )
    (run / "selections.lock.json").write_text(
        json.dumps(
            dict(
                test_loaded=False,
                checkpoints_sha256={r["job_id"]: r["checkpoint_sha256"] for r in registry},
            )
        )
    )
    return run


def test_incomplete_run_has_no_formal_report(tmp_path):
    api = reporter()
    with pytest.raises((ValueError, FileNotFoundError)):
        api.render_report(tmp_path)
    assert not (tmp_path / "report.md").exists()


def test_full_fixture_exports_replayable_tables_and_figures(tmp_path):
    api = reporter()
    run = write_fixture(tmp_path / "SYNTHETIC")
    result = api.render_report(run)
    for name in (
        "summary.csv",
        "module_paired_deltas.csv",
        "window_paired_deltas.csv",
        "factorial_interactions.csv",
        "validation_selected_windows.csv",
        "compute_summary.csv",
    ):
        assert (run / name).stat().st_size > 100
        assert name in result["outputs"]
    report = (run / "report.md").read_text(encoding="utf-8")
    for phrase in (
        "合成",
        "alimu_only",
        "pooled",
        "penncosyvio",
        "样本标准差",
        "独立录制",
        "不能",
        "320",
        "验证",
        "容量",
        "历史",
    ):
        assert phrase in report
    assert "置信区间" in report and "Raw" in report
    index = (run / "figures/figure_index.md").read_text(encoding="utf-8")
    assert "window_context_curves.svg" in index and str(run) not in index
    for stem in (
        "window_context_curves",
        "module_delta_320",
        "control_delta_320",
        "compute_alimu_only",
        "compute_pooled",
    ):
        for ext in ("png", "svg", "pdf"):
            assert (run / "figures" / f"{stem}.{ext}").stat().st_size > 1000
    svg = (run / "figures/window_context_curves.svg").read_text(encoding="utf-8")
    assert "rad/s" in svg and "m/s" in svg
    receipt = json.loads((run / "reporting_provenance.json").read_text(encoding="utf-8"))
    assert (
        receipt["inputs"]["metrics_per_seed.csv"]
        == hashlib.sha256((run / "metrics_per_seed.csv").read_bytes()).hexdigest()
    )
    assert receipt["training_performed"] is False


def test_partial_evaluation_fails_closed_even_with_complete_training(tmp_path):
    api = reporter()
    run = write_fixture(tmp_path / "SYNTHETIC_partial")
    metrics = pd.read_csv(run / "metrics_per_seed.csv")
    metrics = metrics[~(metrics.variant.eq("mlp_only") & metrics.seed.eq(11))]
    metrics.to_csv(run / "metrics_per_seed.csv", index=False)
    with pytest.raises(ValueError):
        api.render_report(run, make_figures=False)
    assert not (run / "report.md").exists()


def test_missing_reference_and_forged_balanced_score_are_rejected():
    api = reporter()
    frame = api.prepare_metrics(fixture_metrics())
    broken = frame[~(frame.variant.eq("cnn_bilstm") & frame.window.eq(40) & frame.seed.eq(11))]
    with pytest.raises(ValueError, match="Missing reference"):
        api.paired_deltas(broken)
    frame.loc[frame.aggregation.eq("overall"), "normalized_score"] = 10
    with pytest.raises(ValueError, match="normalized scores"):
        api.prepare_metrics(frame)


def test_undefined_relative_deltas_do_not_break_signed_plots(tmp_path, monkeypatch):
    api = reporter()
    frame = fixture_metrics()
    mask = frame.variant.eq("cnn_bilstm") & frame.seed.eq(11)
    frame.loc[mask, [c for c in frame if c.endswith(("_rmse", "_mae", "_bias_norm"))]] = 0
    prepared = api.prepare_metrics(frame)
    delta = api.paired_deltas(prepared)
    assert delta.loc[delta.seed.eq(11), "relative_delta_pct"].isna().all()
    from virtual_imu.viz import plots_cnn_ablation as plots

    observed = []

    def capture(fig, directory, stem):
        for ax in fig.axes:
            observed.extend(bar.get_width() for bar in ax.patches)
        plots.plt.close(fig)
        return []

    monkeypatch.setattr(plots, "save_figure", capture)
    plots._delta_bars(
        delta, tmp_path, ["bilstm_only", "cnn_bilstm_dense"], "synthetic", "synthetic"
    )
    assert min(observed) < 0 < max(observed)


def test_aggregation_rows_cannot_silently_disappear():
    api = reporter()
    frame = fixture_metrics()
    with pytest.raises(ValueError):
        api.prepare_metrics(frame[~frame.aggregation.eq("overall")])


def test_guard_diagnostic_is_never_a_formal_reporting_input(tmp_path):
    api = reporter()
    diagnostic = tmp_path / "cnn_bilstm_shared_ablation_20261106_guard_diagnostic"
    with pytest.raises(ValueError, match="guard_diagnostic"):
        api.render_report(diagnostic)
    assert api.main(["--run-dir", str(diagnostic)]) == 2
    assert not diagnostic.exists()


def test_archived_diagnostic_job_directory_is_rejected(tmp_path):
    api = reporter()
    run = write_fixture(tmp_path / "SYNTHETIC_mislinked")
    planned = jobs()
    planned[0]["directory"] = str(tmp_path / "some_guard_diagnostic" / "fits" / planned[0]["id"])
    (run / "planned_jobs.json").write_text(json.dumps(planned), encoding="utf-8")
    with pytest.raises(ValueError, match="guard_diagnostic"):
        api.render_report(run, make_figures=False)
    assert not (run / "report.md").exists()
