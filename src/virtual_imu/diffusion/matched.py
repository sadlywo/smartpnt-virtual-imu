"""Matched per-device search, seed replication, ablation and locked evaluation."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from virtual_imu.config import deep_update, load_config, save_config
from virtual_imu.diffusion.data import PairedSource, PairedWindows, sha256
from virtual_imu.diffusion.evaluation import comparison_table, physical_prediction
from virtual_imu.diffusion.matched_io import write_json
from virtual_imu.diffusion.matched_training import fit_job, load_model, sampling_draws
from virtual_imu.diffusion.pipeline import capture_provenance
from virtual_imu.diffusion.training import choose_device
from virtual_imu.models.baseline_affine import AffineCalibration

LOG = logging.getLogger(__name__)
AXES = ("gx", "gy", "gz", "ax", "ay", "az")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def job_config(config, method, candidate, seed):
    result = deep_update(config, config["search"]["trials"][candidate])
    result["method"], result["seed"] = method, int(seed)
    result["model"]["max_length"] = int(config["data"]["window_size"])
    return result


def verify_frozen_inputs(root, config_path, config, context):
    if sha256(config_path) != context["config_sha256"]:
        raise ValueError("Frozen config changed during experiment")
    if sha256(root / config["data"]["manifest"]) != context["data_manifest_sha256"]:
        raise ValueError("Frozen manifest changed during experiment")
    for relative, digest in context["source_sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Frozen source changed during experiment: {relative}")


def execute_jobs(jobs, workers, progress_path):
    results, pending = {}, []
    for job in jobs:
        directory = Path(job["directory"])
        summary = directory / "summary.json"
        if summary.exists():
            try:
                result = read_json(summary)
            except (json.JSONDecodeError, UnicodeDecodeError):
                result = {}
            required = {
                "config",
                "checkpoint",
                "checkpoint_sha256",
                "validation_score",
                "best_epoch",
                "seconds",
                "pair_id",
                "method",
                "candidate",
                "seed",
                "parameters",
                "initial_backbone_sha256",
                "epochs_ran",
            }
            if isinstance(result, dict) and required <= result.keys():
                if (
                    result["config"] != job["config"]
                    or sha256(Path(result["checkpoint"])) != result["checkpoint_sha256"]
                ):
                    raise ValueError(f"Completed job config/checkpoint changed: {directory}")
                LOG.info("SKIPS completed fit %s", directory)
                results[job["id"]] = result
                continue
            LOG.info("SKIPS incomplete summary marker %s", summary)
        if directory.exists():
            archived = directory.with_name(
                directory.name + f"_interrupted_{datetime.now():%H%M%S_%f}"
            )
            directory.rename(archived)
            LOG.info("WRITES interrupted attempt archive %s", archived)
        pending.append(job)

    def accept(job, result):
        results[job["id"]] = result
        write_json(
            progress_path,
            {"complete": len(results), "expected": len(jobs), "completed_ids": sorted(results)},
        )
        LOG.info(
            "WRITES fit complete %d/%d id=%s best_epoch=%d score=%.5f seconds=%.1f",
            len(results),
            len(jobs),
            job["id"],
            result["best_epoch"],
            result["validation_score"],
            result["seconds"],
        )

    if workers == 1:
        for job in pending:
            accept(job, fit_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fit_job, j): j for j in pending}
            for future in as_completed(futures):
                accept(futures[future], future.result())
    return results


def latency(model, condition, config, pair, method, seed, parameters):
    device = condition.device
    draws = sampling_draws(config)
    rows = []
    with torch.no_grad():
        for batch in (1, int(config["data"]["batch_size"])):
            x = condition[:1].expand(batch * draws, -1, -1).contiguous()
            generator = torch.Generator(device=device).manual_seed(seed + 30000)

            def call(x=x, generator=generator, batch=batch):
                with torch.autocast(
                    device_type=device.type,
                    enabled=bool(config["training"]["amp"]) and device.type == "cuda",
                ):
                    output = model.sample(x, generator=generator)
                    return output.reshape(draws, batch, 6, x.shape[-1]).mean(0)

            for _ in range(int(config["evaluation"].get("latency_warmup", 3))):
                call()
            times = []
            for _ in range(int(config["evaluation"].get("latency_repeats", 10))):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                call()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                times.append(1000 * (time.perf_counter() - start))
            rows.append(
                dict(
                    pair_id=pair,
                    method=method,
                    seed=seed,
                    batch_size=batch,
                    window_points=x.shape[-1],
                    draws=draws,
                    reverse_steps=config["diffusion"]["timesteps"]
                    if config["method"] == "ddpm"
                    else 0,
                    mean_batch_ms=float(np.mean(times)),
                    p50_batch_ms=float(np.median(times)),
                    mean_window_ms=float(np.mean(times) / batch),
                    parameters=parameters,
                )
            )
    return rows


def metric_rows(low, high, predictions, pair, seed):
    table = comparison_table(low, high, predictions)
    table.insert(0, "pair_id", pair)
    table.insert(1, "seed", seed)
    table["vector_rmse"] = table.rmse * np.where(
        table.component.isin(["gyro", "accel"]), np.sqrt(3), 1
    )
    return table


def run_matched_experiment(config_path, root, run_id=None, resume=False):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = load_config(config_path)
    if config["seed"] != config["seeds"][0] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Search seed must be the first unique final seed")
    if not {"ddpm", "no_diffusion"} <= set(config["methods"]):
        raise ValueError("Matched experiment requires DDPM and its no-diffusion ablation")
    identifier = run_id or f"{config['experiment_id']}_{datetime.now():%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Invalid run id")
    run = root / config["output"]["runs_dir"] / identifier
    if run.exists() and not resume:
        raise FileExistsError(run)
    run.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        LOG.info("READS %s; WRITES %s; SKIPS old runs and outputs/latest", config_path, run)
        device = choose_device(config["training"]["device"])
        torch.set_num_threads(int(config["training"].get("cpu_threads", 1)))
        context_path = run / "provenance.json"
        if context_path.exists():
            context = read_json(context_path)
            if context["config_sha256"] != sha256(config_path):
                raise ValueError("Resume config changed")
            if context["data_manifest_sha256"] != sha256(root / config["data"]["manifest"]):
                raise ValueError("Resume manifest changed")
            for relative, digest in context["source_sha256"].items():
                if sha256(root / relative) != digest:
                    raise ValueError(f"Resume source changed: {relative}")
        else:
            context = capture_provenance(root, run, config_path, device)
            save_config(config, run / "config.yaml")
        if (run / "complete.json").exists():
            for relative, digest in read_json(run / "complete.json")["artifacts_sha256"].items():
                if sha256(run / relative) != digest:
                    raise ValueError(f"Completed artifact changed: {relative}")
            LOG.info("SKIPS completed, source-pinned experiment %s", run)
            return run
        sources, trains, vals = {}, {}, {}
        for pair in config["data"]["pairs"]:
            source = PairedSource(root, config["data"], pair)
            train = PairedWindows(
                source.load("train"), config["data"]["window_size"], config["data"]["stride"]
            )
            val = PairedWindows(
                source.load("validation"),
                config["data"]["window_size"],
                config["data"]["eval_stride"],
                train.stats,
            )
            sources[pair], trains[pair], vals[pair] = source, train, val
            pair_dir = run / pair
            pair_dir.mkdir(exist_ok=True)
            write_json(
                pair_dir / "data_manifest.json",
                dict(
                    records=source.records,
                    manifest_sha256=context["data_manifest_sha256"],
                    provenance="../provenance.json",
                ),
            )
            write_json(pair_dir / "normalization.json", train.stats)

        def make_job(pair, method, candidate, seed, suffix):
            directory = run / "fits" / pair / method / suffix
            return dict(
                id=f"{pair}/{method}/{suffix}",
                root=str(root),
                directory=str(directory),
                pair_id=pair,
                candidate=candidate,
                config=job_config(config, method, candidate, seed),
                context={**context, "pair_id": pair, "phase": suffix},
            )

        search_jobs = [
            make_job(pair, method, i, config["seed"], f"search_trial_{i:03d}")
            for pair in sources
            for method in config["methods"]
            for i in range(len(config["search"]["trials"]))
        ]
        # Seeded job order only affects scheduling; every job has an isolated RNG stream.
        np.random.default_rng(config["seed"]).shuffle(search_jobs)
        workers = int(config["training"].get("workers", 1))
        search = execute_jobs(search_jobs, workers, run / "search_progress.json")
        pd.json_normalize(list(search.values())).to_csv(run / "tuning_trials.csv", index=False)
        selected = {}
        for pair in sources:
            for method in config["methods"]:
                candidates = [
                    r for r in search.values() if r["pair_id"] == pair and r["method"] == method
                ]
                selected[pair, method] = min(
                    candidates, key=lambda x: (x["validation_score"], x["candidate"])
                )
        lock = dict(
            test_used_for_selection=False,
            selection_rule="minimum common stitched validation relative RMSE; candidate index breaks ties",
            selections=[
                dict(
                    pair_id=p,
                    method=m,
                    **{k: v for k, v in r.items() if k not in {"pair_id", "method"}},
                )
                for (p, m), r in selected.items()
            ],
        )
        if (run / "selections.lock.json").exists() and read_json(
            run / "selections.lock.json"
        ) != lock:
            raise ValueError("Locked selections changed")
        write_json(run / "selections.lock.json", lock)
        # Prepare all additional final-seed fits before evaluating any TEST signal.
        extra = []
        final = {}
        for (pair, method), best in selected.items():
            final[pair, method, config["seed"]] = best
            for seed in config["seeds"][1:]:
                job = make_job(pair, method, best["candidate"], seed, f"seed_{seed}")
                extra.append(job)
        for pair in sources:
            candidate = selected[pair, "ddpm"]["candidate"]
            for seed in config["seeds"]:
                if candidate == selected[pair, "no_diffusion"]["candidate"]:
                    continue
                if seed == config["seed"]:
                    final[pair, "no_diffusion_matched", seed] = search[
                        f"{pair}/no_diffusion/search_trial_{candidate:03d}"
                    ]
                else:
                    extra.append(
                        make_job(pair, "no_diffusion", candidate, seed, f"matched_ddpm_seed_{seed}")
                    )
        fitted = execute_jobs(extra, workers, run / "final_progress.json")
        for job in extra:
            label = (
                "no_diffusion_matched"
                if "matched_ddpm_seed" in job["id"]
                else job["config"]["method"]
            )
            final[job["pair_id"], label, job["config"]["seed"]] = fitted[job["id"]]
        for pair in sources:
            for seed in config["seeds"]:
                if (pair, "no_diffusion_matched", seed) not in final:
                    final[pair, "no_diffusion_matched", seed] = final[pair, "no_diffusion", seed]
        pd.json_normalize(
            [
                dict(reported_pair=p, reported_method=m, reported_seed=s, **r)
                for (p, m, s), r in final.items()
            ]
        ).to_csv(run / "final_fits.csv", index=False)
        write_json(
            run / "training.complete.json",
            dict(status="complete", fits=len(search) + len(extra), final_slots=len(final)),
        )
        verify_frozen_inputs(root, config_path, config, context)
        tables, latency_rows = [], []
        test_windows = {}
        for pair in sources:
            test_windows[pair] = PairedWindows(
                sources[pair].load("test"),
                config["data"]["window_size"],
                config["data"]["eval_stride"],
                trains[pair].stats,
            )
        for (pair, method, seed), fit in final.items():
            destination = run / "evaluations" / pair / method / f"seed_{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            if sha256(Path(fit["checkpoint"])) != fit["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed before evaluation")
            model, checkpoint = load_model(Path(fit["checkpoint"]), device)
            if checkpoint["stats"] != trains[pair].stats:
                raise ValueError("Train normalization changed")
            for role, windows in [("validation", vals[pair]), ("test", test_windows[pair])]:
                pred, mask = physical_prediction(
                    model,
                    windows,
                    device,
                    config["data"]["batch_size"],
                    seed + 20000,
                    sampling_draws(fit["config"]),
                    bool(config["training"]["amp"]) and device.type == "cuda",
                )
                if not mask.all():
                    raise ValueError("Matched evaluation requires complete common sample support")
                low, high = windows.low.astype(float), windows.high.astype(float)
                frame = windows.frame.copy()
                for i, axis in enumerate(AXES):
                    frame[f"{axis}_low"], frame[f"{axis}_high"], frame[f"{axis}_virtual"] = (
                        low[:, i],
                        high[:, i],
                        pred[:, i],
                    )
                frame.to_parquet(destination / f"{role}_predictions.parquet", index=False)
                metrics = metric_rows(low, high, {method: pred}, pair, seed)
                metrics.to_csv(destination / f"{role}_metrics.csv", index=False)
                if role == "test":
                    tables.append(metrics)
                    per_record = []
                    for seq, group in frame.groupby("sequence_id", sort=False):
                        idx = group.index.to_numpy()
                        t = metric_rows(low[idx], high[idx], {method: pred[idx]}, pair, seed)
                        t.insert(0, "sequence_id", seq)
                        per_record.append(t)
                    pd.concat(per_record).to_csv(
                        destination / "test_per_record_metrics.csv", index=False
                    )
            if seed == config["seed"]:
                x = test_windows[pair][0][0][None].to(device)
                latency_rows += latency(
                    model, x, fit["config"], pair, method, seed, fit["parameters"]
                )
            write_json(
                destination / "context.json",
                dict(
                    checkpoint=fit["checkpoint"],
                    checkpoint_sha256=fit["checkpoint_sha256"],
                    config=fit["config"],
                    selection_lock="../../../../selections.lock.json",
                    initial_backbone_sha256=fit["initial_backbone_sha256"],
                ),
            )
            del model
            LOG.info("WRITES held-out evaluation pair=%s method=%s seed=%s", pair, method, seed)
        # Deterministic controls are fitted once per pair, never counted as three stochastic seeds.
        for pair, windows in test_windows.items():
            affine = AffineCalibration().fit(trains[pair].low, trains[pair].high)
            write_json(
                run / pair / "affine_state.json",
                dict(matrix=affine.matrix_.tolist(), bias=affine.bias_.tolist()),
            )
            tables.append(
                metric_rows(
                    windows.low.astype(float),
                    windows.high.astype(float),
                    {"raw": windows.low.astype(float), "affine": affine.predict(windows.low)},
                    pair,
                    config["seed"],
                )
            )
        methods = [*config["methods"], "no_diffusion_matched"]
        for method in methods:
            for seed in config["seeds"]:
                frames = [
                    pd.read_parquet(
                        run
                        / "evaluations"
                        / p
                        / method
                        / f"seed_{seed}"
                        / "test_predictions.parquet",
                        columns=[f"{a}_{s}" for s in ("low", "high", "virtual") for a in AXES],
                    )
                    for p in sources
                ]
                f = pd.concat(frames, ignore_index=True)
                tables.append(
                    metric_rows(
                        f[[f"{a}_low" for a in AXES]].to_numpy(),
                        f[[f"{a}_high" for a in AXES]].to_numpy(),
                        {method: f[[f"{a}_virtual" for a in AXES]].to_numpy()},
                        "__all_pairs__",
                        seed,
                    )
                )
        controls = []
        for pair, windows in test_windows.items():
            affine = AffineCalibration().fit(trains[pair].low, trains[pair].high)
            controls.append(
                (windows.low.astype(float), windows.high.astype(float), affine.predict(windows.low))
            )
        low, high, aff = (
            np.concatenate([x[0] for x in controls]),
            np.concatenate([x[1] for x in controls]),
            np.concatenate([x[2] for x in controls]),
        )
        tables.append(
            metric_rows(low, high, {"raw": low, "affine": aff}, "__all_pairs__", config["seed"])
        )
        metrics = pd.concat(tables, ignore_index=True)
        metrics.to_csv(run / "metrics_per_seed.csv", index=False)
        summary = (
            metrics.groupby(["pair_id", "method", "component", "unit"], sort=False)
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_sd=("rmse", "std"),
                vector_rmse_mean=("vector_rmse", "mean"),
                vector_rmse_sd=("vector_rmse", "std"),
                n=("seed", "size"),
                samples=("samples", "first"),
            )
            .reset_index()
        )
        summary.to_csv(run / "summary.csv", index=False)
        pd.DataFrame(latency_rows).to_csv(run / "latency.csv", index=False)
        deltas = []
        for pair in sources:
            for seed in config["seeds"]:
                ddpm, ablation = (
                    final[pair, "ddpm", seed],
                    final[pair, "no_diffusion_matched", seed],
                )
                match = ddpm["initial_backbone_sha256"] == ablation["initial_backbone_sha256"]
                if not match:
                    raise ValueError("Strict ablation initialization differs from DDPM")
                for group in ("gyro", "accel"):
                    subset = metrics[
                        (metrics.pair_id == pair)
                        & (metrics.seed == seed)
                        & (metrics.component == group)
                    ].set_index("method")
                    a, b = subset.loc["ddpm", "rmse"], subset.loc["no_diffusion_matched", "rmse"]
                    deltas.append(
                        dict(
                            pair_id=pair,
                            seed=seed,
                            component=group,
                            ddpm_rmse=a,
                            no_diffusion_rmse=b,
                            ddpm_change_pct=100 * (a / b - 1),
                            initial_backbone_matches=match,
                        )
                    )
        pd.DataFrame(deltas).to_csv(run / "matched_seed_deltas.csv", index=False)
        from virtual_imu.viz.plots_matched import render_matched_report

        verify_frozen_inputs(root, config_path, config, context)
        render_matched_report(run, config)
        suffixes = {
            ".csv",
            ".md",
            ".json",
            ".yaml",
            ".pt",
            ".parquet",
            ".png",
            ".pdf",
            ".svg",
            ".py",
            ".patch",
        }
        paths = [
            p
            for p in run.rglob("*")
            if p.is_file() and p.suffix in suffixes and p.name != "complete.json"
        ]
        write_json(
            run / "complete.json",
            dict(
                status="complete",
                final_slots=len(final),
                test_used_for_selection=False,
                artifacts_sha256={str(p.relative_to(run)): sha256(p) for p in paths},
            ),
        )
        return run
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
