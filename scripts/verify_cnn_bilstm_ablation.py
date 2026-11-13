"""Independently replay saved common supports, checkpoint selection and metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
LOW = [a + "_low" for a in AXES]
HIGH = [a + "_high" for a in AXES]
KEYS = ["dataset_id", "pair_id", "sequence_id", "timestamp_s"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def support_hash(frame):
    return hashlib.sha256(frame[KEYS].to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def check_close(actual, expected, description, rtol=1e-9, atol=1e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(description + ": " + str(actual) + " != " + str(expected))


def replay_row(frame, pred, row):
    mask = np.ones(len(frame), dtype=bool)
    for col in ["dataset_id", "pair_id", "original_sequence_id"]:
        if row[col] != "__all__":
            mask &= frame[col].eq(row[col]).to_numpy()
    assert mask.sum() == row["samples"]
    error = pred[mask] - frame.loc[mask, HIGH].to_numpy(np.float64)
    metrics = {}
    for name, channels in [("gyro", [0, 1, 2]), ("accel", [3, 4, 5])]:
        e = error[:, channels]
        metrics[name + "_rmse"] = np.sqrt(np.average(e * e))
        metrics[name + "_vector_rmse"] = np.sqrt(np.average(np.sum(e * e, axis=1)))
        metrics[name + "_mae"] = np.average(np.abs(e))
        metrics[name + "_bias_norm"] = np.sqrt(np.sum(np.average(e, axis=0) ** 2))
    for i, axis in enumerate(AXES):
        metrics[axis + "_rmse"] = np.sqrt(np.average(error[:, i] ** 2))
    for metric, value in metrics.items():
        check_close(value, row[metric], "prediction replay " + metric)


def verify(run_dir):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    prov = read_json(run_dir / "provenance.json")
    jobs = read_json(run_dir / "planned_jobs.json")
    complete = read_json(run_dir / "training.complete.json")
    lock = read_json(run_dir / "selections.lock.json")
    assert complete["completed"] == complete["planned"] == len(jobs) == 138
    assert len(lock["checkpoints_sha256"]) == len(jobs)
    assert lock["test_loaded"] is False
    assert sha(run_dir / "config.yaml") == prov["config_sha256"]
    assert sha(ROOT / cfg["data"]["manifest"]) == prov["manifest_sha256"]
    for rel, expected in prov["source_sha256"].items():
        assert sha(ROOT / rel) == expected, rel
        assert sha(run_dir / "source_snapshot" / rel) == expected, rel
    for name, expected in prov["historical_sha256"].items():
        assert sha(name) == expected
    analysis_plan = read_json(run_dir / "analysis_plan.json")
    assert analysis_plan["test_unlocked"] is False
    for rel, expected in analysis_plan["source_sha256"].items():
        assert sha(ROOT / rel) == expected, rel
        assert sha(run_dir / "analysis_source" / rel) == expected, rel
    torch.set_num_threads(1)
    supports, stats_by_protocol, train_dataset_counts, anchor_datasets = {}, {}, {}, {}
    full_frames, full_starts, primary_positions = {}, {}, {}
    input_hashes = {}
    for protocol in cfg["protocols"]:
        directory = run_dir / "supports" / protocol
        records = read_json(directory / "records.json")
        physical = {role: {r["physical_group_id"] for r in rows} for role, rows in records.items()}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            assert not physical[a] & physical[b]
        for role in ["train", "validation", "test"]:
            raw_frames, expected_frames, datasets = [], [], []
            starts_global, positions_global, offset = [], [], 0
            for record in records[role]:
                path = ROOT / record["path"]
                actual = sha(path)
                assert actual == record["sha256"], str(path)
                input_hashes[record["path"]] = actual
                if record["rows"] < cfg["data"]["max_window"]:
                    continue
                raw = pd.read_parquet(path).sort_values("timestamp_s", kind="stable").reset_index(drop=True)
                if role != "train":
                    raw = raw.iloc[:len(raw) // 40 * 40]
                raw_frames.append(raw)
                # Derive row positions directly, without the production window class.
                count = 1 + (len(raw) - 320) // 40
                positions = (np.arange(count)[:, None] * 40 + 140 + np.arange(40)[None, :]).ravel()
                expected_frames.append(raw.iloc[positions])
                starts_global.extend((np.arange(count)*40+offset).tolist())
                positions_global.extend((positions+offset).tolist())
                offset += len(raw)
                if role == "train":
                    datasets.extend([record["dataset_id"]] * count)
            expected = pd.concat(expected_frames, ignore_index=True)
            saved = pd.read_parquet(directory / (role + ".parquet"))
            pd.testing.assert_frame_equal(saved[KEYS], expected[KEYS], check_dtype=False)
            check_close(saved[LOW + HIGH], expected[LOW + HIGH], "original source support", atol=0, rtol=0)
            assert not saved.duplicated(KEYS).any()
            supports[(protocol, role)] = saved
            if role in ["validation", "test"]:
                full_frames[(protocol, role)] = pd.concat(raw_frames, ignore_index=True)
                full_starts[(protocol, role)] = np.array(starts_global, dtype=np.int64)
                primary_positions[(protocol, role)] = np.array(positions_global, dtype=np.int64)
            if role == "train":
                raw_train = pd.concat(raw_frames, ignore_index=True)
                low = raw_train[LOW].to_numpy(float)
                res = raw_train[HIGH].to_numpy(float) - low
                stats = read_json(directory / "statistics.json")
                for name, values in [("feature", low), ("residual", res)]:
                    check_close(values.mean(0), stats[name + "_mean"], "TRAIN-only mean")
                    check_close(np.where(values.std(0) < 1e-8, 1, values.std(0)), stats[name + "_std"], "TRAIN-only std")
                stats_by_protocol[protocol] = stats
                anchor_datasets[protocol] = np.array(datasets)
                train_dataset_counts[protocol] = len(datasets)

    init_receipts, order_sequences = {}, {}
    checkpoint_count = 0
    for job in jobs:
        directory = Path(job["directory"])
        fit = read_json(directory / "fit.json")
        saved_job = read_json(directory / "job.json")
        assert saved_job == job
        assert fit["job_id"] == job["id"]
        assert sha(directory / "best.pt") == fit["checkpoint_sha256"] == lock["checkpoints_sha256"][job["id"]]
        checkpoint = torch.load(directory / "best.pt", map_location="cpu")
        assert all(torch.isfinite(t).all() for t in checkpoint["model"].values())
        history = pd.read_csv(directory / "history.csv")
        assert len(history) == fit["actual_epochs"] <= cfg["training"]["epochs"]
        expected_epoch = int(history.loc[history.val_score.idxmin(), "epoch"])
        assert fit["best_epoch"] == checkpoint["epoch"] == expected_epoch
        check_close(history.val_score.min(), fit["best_val_score"], "selected VALID minimum")
        assert checkpoint["stats"] == stats_by_protocol[job["protocol"]]
        assert checkpoint["train_support_sha256"] == support_hash(supports[(job["protocol"], "train")])
        assert checkpoint["validation_support_sha256"] == support_hash(supports[(job["protocol"], "validation")])
        assert fit["amp_skipped_steps"] == history.amp_skipped_steps.sum()
        init = read_json(directory / "initialization.json")
        assert init == checkpoint["metadata"]
        key = (job["variant"], job["seed"])
        if key in init_receipts:
            assert init == init_receipts[key]
        init_receipts[key] = init
        order_key = (job["protocol"], job["seed"])
        if order_key in order_sequences:
            n = min(len(history), len(order_sequences[order_key]))
            assert history.order_sha256[:n].tolist() == order_sequences[order_key][:n]
        if len(history) > len(order_sequences.get(order_key, [])):
            order_sequences[order_key] = history.order_sha256.tolist()
        checkpoint_count += 1
    # Verify sampler independently once per protocol/seed/observed epoch.
    for (protocol, seed), digests in order_sequences.items():
        labels = anchor_datasets[protocol]
        counts = {d: (labels == d).sum() for d in set(labels)}
        weights = torch.tensor([1 / counts[d] for d in labels], dtype=torch.float64)
        for epoch, expected in enumerate(digests, 1):
            order = torch.multinomial(weights, len(labels), replacement=True,
                                      generator=torch.Generator().manual_seed(seed + epoch * 1000003)).numpy()
            assert hashlib.sha256(order.tobytes()).hexdigest() == expected
    metrics = pd.read_csv(run_dir / "metrics_per_seed.csv")
    replayed, predictions = 0, 0
    for protocol in cfg["protocols"]:
        for role in ["validation", "test"]:
            frame = supports[(protocol, role)]
            z = metrics[metrics.protocol.eq(protocol) & metrics.split.eq(role) & metrics.variant.eq("raw")]
            for _, row in z.iterrows():
                replay_row(frame, frame[LOW].to_numpy(float), row)
                replayed += 1
    for job in jobs:
        for role in ["validation", "test"]:
            frame = supports[(job["protocol"], role)]
            path = run_dir / "predictions" / (job["id"] + "__" + role + ".npz")
            with np.load(path) as archive:
                pred = archive["prediction"]
                assert str(archive["support_sha256"]) == support_hash(frame)
            z = metrics[metrics.job_id.eq(job["id"]) & metrics.split.eq(role)]
            assert len(z) > 0
            for _, row in z.iterrows():
                replay_row(frame, pred, row)
                replayed += 1
            predictions += 1
    overlap_files, overlap_rows = 0, 0
    if (run_dir / "overlap_metrics_per_seed.csv").is_file():
        overlap_receipt = read_json(run_dir / "overlap_diagnostic_receipt.json")
        assert overlap_receipt["added_training_fits"] == 0
        assert overlap_receipt["checkpoint_selection_changed"] is False
        for path, expected in overlap_receipt["inputs_sha256"].items():
            assert sha(path) == expected
        for path, expected in overlap_receipt["outputs_sha256"].items():
            assert sha(run_dir / path) == expected
        overlap_metrics = pd.read_csv(run_dir / "overlap_metrics_per_seed.csv")
        selected_jobs = [j for j in jobs if j["window"]==320 and j["variant"] in ["cnn_bilstm", "cnn_bilstm_dense"]]
        assert len(selected_jobs) == 12
        for job in selected_jobs:
            stats = stats_by_protocol[job["protocol"]]
            for role in ["validation", "test"]:
                context = (job["protocol"], role)
                frame, raw = supports[context], full_frames[context]
                starts, chosen = full_starts[context], primary_positions[context]
                path = run_dir / "predictions" / ("overlap__"+job["id"]+"__"+role+".npz")
                with np.load(path) as archive:
                    prediction = archive["prediction"]
                    normals = archive["normalized_windows"]
                    counts = archive["overlap_counts"]
                    assert str(archive["support_sha256"]) == support_hash(frame)
                assert normals.shape == (len(starts),6,320)
                positions = starts[:,None] + np.arange(320)[None,:]
                physical = normals.transpose(0,2,1).astype(np.float64) * stats["residual_std"] + stats["residual_mean"]
                physical += raw[LOW].to_numpy(float)[positions]
                total = np.zeros((len(raw),6),dtype=np.float64)
                count = np.zeros(len(raw),dtype=np.int64)
                np.add.at(total, positions.ravel(), physical.reshape(-1,6))
                np.add.at(count, positions.ravel(), 1)
                check_close(counts, count[chosen], "overlap contribution counts", atol=0,rtol=0)
                expected = total[chosen] / count[chosen,None]
                check_close(prediction, expected, "independent overlap average", atol=1e-12,rtol=1e-12)
                with np.load(run_dir/"predictions"/(job["id"]+"__"+role+".npz")) as archive:
                    center = archive["prediction"]
                selected = overlap_metrics[overlap_metrics.job_id.eq(job["id"]) & overlap_metrics.split.eq(role)]
                for _, row in selected.iterrows():
                    replay_row(frame, center if row.aggregation_mode=="center" else prediction, row)
                    overlap_rows += 1
                overlap_files += 1
        assert overlap_files == 24
    # Derive scores from independently replayed pair errors, then check all
    # reporting aggregations and paired comparisons without the reporting API.
    metrics["normalized_score"] = np.nan
    raw_rows = metrics[metrics.variant.eq("raw") & metrics.aggregation.eq("pair")]
    raw_lookup = {(r.protocol, r.split, r.dataset_id, r.pair_id): (r.gyro_rmse, r.accel_rmse) for r in raw_rows.itertuples()}
    condition = ["protocol", "split", "variant", "window", "seed"]
    for key, indexes in metrics.groupby(condition, sort=False).groups.items():
        group = metrics.loc[indexes]
        per_dataset = {}
        for ix, row in group[group.aggregation.eq("pair")].iterrows():
            g, a = raw_lookup[(row.protocol, row.split, row.dataset_id, row.pair_id)]
            value = .5 * (row.gyro_rmse / max(g, 1e-12) + row.accel_rmse / max(a, 1e-12))
            metrics.loc[ix, "normalized_score"] = value
            per_dataset.setdefault(row.dataset_id, []).append(value)
        for ix, row in group[group.aggregation.eq("dataset")].iterrows():
            metrics.loc[ix, "normalized_score"] = np.average(per_dataset[row.dataset_id])
        metrics.loc[group[group.aggregation.eq("overall")].index, "normalized_score"] = np.average([np.average(values) for values in per_dataset.values()])
    ids = condition + ["aggregation", "dataset_id", "pair_id", "original_sequence_id"]
    metric_columns = ["gyro_rmse", "gyro_vector_rmse", "gyro_mae", "gyro_bias_norm", "accel_rmse", "accel_vector_rmse", "accel_mae", "accel_bias_norm", *[a+"_rmse" for a in AXES], "normalized_score"]
    long = metrics.melt(id_vars=[*ids, "samples"], value_vars=metric_columns, var_name="metric", value_name="value").dropna(subset=["value"])
    scalar_lookup = {tuple(row[c] for c in ids) + (row["metric"],): row["value"] for _, row in long.iterrows()}
    summary = pd.read_csv(run_dir / "summary.csv")
    summary_keys = [c for c in ids if c != "seed"] + ["metric"]
    expected_groups = long.groupby(summary_keys, sort=False)
    assert len(summary) == len(expected_groups)
    for _, row in summary.iterrows():
        group = expected_groups.get_group(tuple(row[c] for c in summary_keys))
        values = group.value.to_numpy()
        assert row["n"] == group.seed.nunique()
        check_close(row["mean"], np.average(values), "summary seed mean")
        check_close(row["sd"], np.std(values, ddof=1) if len(values) > 1 else np.nan, "summary seed SD")
        assert row["samples"] == group.samples.iloc[0]
    delta_count = 0
    for name in ["module_paired_deltas.csv", "window_paired_deltas.csv"]:
        table = pd.read_csv(run_dir / name)
        for _, row in table.iterrows():
            key = tuple(row[c] for c in ids) + (row["metric"],)
            reference = row.copy()
            reference["variant"] = row["reference_variant"]
            reference["window"] = row["reference_window"]
            refkey = tuple(reference[c] for c in ids) + (row["metric"],)
            v, ref = scalar_lookup[key], scalar_lookup[refkey]
            check_close(row["value"], v, "paired value")
            check_close(row["reference_value"], ref, "paired reference")
            check_close(row["delta"], v-ref, "paired delta")
            check_close(row["relative_delta_pct"], 100 * (v-ref) / ref if ref != 0 else np.nan, "paired relative delta", atol=1e-9)
            assert row["seed"] == row["reference_seed"]
            delta_count += 1
    interactions = pd.read_csv(run_dir / "factorial_interactions.csv")
    for _, row in interactions.iterrows():
        values = {}
        for variant in ["cnn_bilstm", "bilstm_only", "cnn_only", "mlp_only"]:
            record = row.copy()
            record["variant"] = variant
            key = tuple(record[c] for c in ids) + (row["metric"],)
            values[variant] = scalar_lookup[key]
            check_close(row[variant], values[variant], "factorial component")
        check_close(row["interaction"], values["cnn_bilstm"]-values["bilstm_only"]-values["cnn_only"]+values["mlp_only"], "factorial interaction")
    selected = pd.read_csv(run_dir / "validation_selected_windows.csv")
    for row in selected.itertuples():
        z = metrics[metrics.protocol.eq(row.protocol) & metrics.variant.eq(row.variant) & metrics.split.eq("validation") & metrics.aggregation.eq("overall")]
        means = z.groupby("window").normalized_score.mean()
        best = min(means.index, key=lambda window: (means.loc[window], window))
        assert best == row.selected_window
        check_close(row.score_mean, means.loc[best], "VALID-only window choice")
    outputs = {p.relative_to(run_dir).as_posix(): sha(p) for p in run_dir.rglob("*")
               if p.is_file() and p.suffix != ".log" and p.name not in {"verification.json", "independent_review.json"}}
    receipt = dict(status="PASS", fits=checkpoint_count, prediction_files=predictions+overlap_files, metric_rows_replayed=replayed+overlap_rows,
                   data_files=len(input_hashes), data_sha256=input_hashes, source_files=len(prov["source_sha256"]),
                   support_recomputed_from_original_sources=True, train_only_statistics=True,
                   common_minibatch_orders=True, exact_initialization_across_windows_and_protocols=True,
                   checkpoint_selection_from_VALID_minima=True, seed_summary_rows=len(summary), paired_delta_rows=delta_count,
                   factorial_rows=len(interactions), validation_selected_windows=len(selected),
                   overlap_prediction_files=overlap_files, overlap_metric_rows=overlap_rows,
                   pre_TEST_analysis_source_files=len(analysis_plan["source_sha256"]),
                   output_sha256=outputs, verifier_sha256=sha(__file__))
    (run_dir / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("WRITES " + str(run_dir / "verification.json"))
    print("PASS fits=%d predictions=%d metric_rows=%d" % (checkpoint_count, predictions+overlap_files, replayed+overlap_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/cnn_bilstm_shared_ablation_20261106")
    args = parser.parse_args()
    print("READS completed locked artifacts and original paired sources; WRITES verification receipt; SKIPS training, model reselection and original-run edits")
    verify(args.run_dir.resolve())
