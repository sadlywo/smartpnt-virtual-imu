"""Freeze, fit and evaluate a complete shared CNN-BiLSTM ablation matrix."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.ablations.data import CommonWindows, SharedSource, physical_metrics, grouped_score, LOW, HIGH
from virtual_imu.ablations.training import configure, fit_job, predict_normalized, sha, write_json

LOG = logging.getLogger(__name__)


def jobs_for_config(root, cfg, run_dir):
    jobs = []
    for protocol in cfg["protocols"]:
        treatments = [(v, w) for v in cfg["variants"] for w in cfg["windows"]]
        treatments += [(v, cfg["data"]["max_window"]) for v in cfg["supplemental"]]
        for variant, window in treatments:
            for seed in cfg["seeds"]:
                name = protocol + "__" + variant + "__w" + str(window) + "__s" + str(seed)
                jobs.append(dict(id=name, root=str(root), config=cfg, protocol=protocol, variant=variant, window=window,
                                 seed=seed, directory=str(run_dir / "fits" / name)))
    indexes = np.random.default_rng(cfg["seed"]).permutation(len(jobs))
    return [jobs[i] for i in indexes]


def freeze(root, config_path, cfg, run_dir, jobs):
    if (run_dir / "provenance.json").is_file():
        old = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if old["config_sha256"] != sha(config_path):
            raise ValueError("Cannot resume a different experiment configuration")
        for rel, expected in old["source_sha256"].items():
            if sha(root / rel) != expected:
                raise ValueError("Frozen training source changed: " + rel)
        LOG.info("SKIPS rewriting frozen source/config")
        return old
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "source_snapshot"
    source_paths = [*[root / "src/virtual_imu/ablations" / n for n in ["__init__.py", "data.py", "models.py", "training.py", "pipeline.py"]],
                    root / "src/virtual_imu/models/matched.py", root / "src/virtual_imu/models/ddpm.py",
                    root / "src/virtual_imu/models/transformer.py", root / "src/virtual_imu/fir_arx_research/data.py",
                    root / "src/virtual_imu/utils/seed.py", root / "scripts/run_cnn_bilstm_ablation.py", config_path]
    source_hashes = {}
    for path in source_paths:
        relative = path.relative_to(root)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[relative.as_posix()] = sha(path)
    hroot = Path(cfg["historical"]["root"])
    historical = {}
    for protocol in cfg["protocols"]:
        name = protocol + "__cnn_bilstm__final_tuned__trial_004__rna__s20261010.yaml"
        path = hroot / cfg["historical"]["config_directory"] / name
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert all(source["model"][key] == value for key, value in cfg["model"].items())
        for key in ["lr", "weight_decay", "epochs", "early_stopping_patience", "warmup_epochs", "min_lr"]:
            assert source["training"][key] == cfg["training"][key]
        dest = run_dir / "historical_sources" / name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        historical[str(path)] = sha(path)
    hmodel = hroot / cfg["historical"]["source_file"]
    assert sha(hmodel) == cfg["historical"]["source_sha256"]
    shutil.copy2(hmodel, run_dir / "historical_sources/advanced.py")
    historical[str(hmodel)] = sha(hmodel)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=True).stdout
    shutil.copy2(config_path, run_dir / "config.yaml")
    manifest = root / cfg["data"]["manifest"]
    assert sha(manifest) == cfg["data"]["manifest_sha256"]
    shutil.copy2(manifest, run_dir / "data_manifest.json")
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), git_commit=git, git_status=status,
                      config_sha256=sha(config_path), manifest_sha256=sha(manifest), source_sha256=source_hashes,
                      historical_sha256=historical, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      torch=torch.__version__, planned_fits=len(jobs), test_used_for_selection=False,
                      support_convention="same centered40 target blocks in same320 anchors; dense supplemental changes TRAIN loss only")
    write_json(run_dir / "provenance.json", provenance)
    write_json(run_dir / "planned_jobs.json", jobs)
    return provenance


def prepare_support(root, cfg, run_dir):
    result = {}
    for protocol in cfg["protocols"]:
        source = SharedSource(root, cfg["data"], protocol)
        options = {k: cfg["data"][k] for k in ["max_window", "block_size", "stride"]}
        train = CommonWindows(source.load("train"), **options)
        val = CommonWindows(source.load("validation"), stats=train.stats, **options)
        directory = run_dir / "supports" / protocol
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "statistics.json", train.stats)
        write_json(directory / "records.json", source.records)
        for role, data in [("train", train), ("validation", val)]:
            data.target_frame.to_parquet(directory / (role + ".parquet"), index=False)
        result[protocol] = {role: {"anchors": len(data), "samples": len(data.target_frame), "support_sha256": data.support_sha256,
                                  "datasets": data.target_frame.dataset_id.value_counts().to_dict()} for role, data in [("train", train), ("validation", val)]}
        assert "test" not in source.loaded_roles
    write_json(run_dir / "supports.json", result)
    LOG.info("WRITES common TRAIN/VALID supports; SKIPS TEST numerical payloads")


def execute(jobs, workers, run_dir):
    records, failures = [], []
    begin = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_jobs = {pool.submit(fit_job, job): job for job in jobs}
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                record = future.result()
                records.append(record)
                pd.DataFrame(records).to_csv(run_dir / "fit_registry.csv", index=False)
                LOG.info("COMPLETED %d/%d %s epochs=%d valid=%.6g", len(records), len(jobs), job["id"], record["actual_epochs"], record["best_val_score"])
            except Exception as error:
                failures.append({"job": job["id"], "error": repr(error)})
                LOG.exception("Fit failed: %s", job["id"])
            write_json(run_dir / "progress.json", dict(phase="training", completed=len(records), planned=len(jobs), failed=failures,
                                                     elapsed_seconds=time.perf_counter()-begin, last_job=job["id"]))
    if failures:
        raise RuntimeError("Failed fits: " + json.dumps(failures))
    return records


def evaluate(root, cfg, run_dir, jobs, records):
    from virtual_imu.ablations.models import build_ablation_model
    from virtual_imu.ablations.training import json_digest
    if len(records) != len(jobs) or {r["job_id"] for r in records} != {j["id"] for j in jobs}:
        raise ValueError("All planned fit identities must complete before TEST is opened")
    by_id = {r["job_id"]: r for r in records}
    for job in jobs:
        record = by_id[job["id"]]
        if record["job_config_sha256"] != json_digest(job):
            raise ValueError("Fit configuration does not match the frozen plan")
        checkpoint = Path(job["directory"]) / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if sha(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum differs before TEST unlock")
        saved = torch.load(checkpoint, map_location="cpu")
        if saved["job"] != job or saved["epoch"] != record["best_epoch"] or not np.isclose(saved["val_score"], record["best_val_score"], rtol=1e-12):
            raise ValueError("Checkpoint selection identity differs before TEST unlock")
    lock = {r["job_id"]: r["checkpoint_sha256"] for r in records}
    assert len(lock) == len(jobs)
    write_json(run_dir / "selections.lock.json", dict(test_loaded=False, checkpoints_sha256=lock))
    device = configure(cfg)
    metrics, costs = [], []
    amp = cfg["training"]["amp"] and device.type == "cuda"
    for protocol in cfg["protocols"]:
        source = SharedSource(root, cfg["data"], protocol)
        options = {k: cfg["data"][k] for k in ["max_window", "block_size", "stride"]}
        stats = json.loads((run_dir / "supports" / protocol / "statistics.json").read_text(encoding="utf-8"))
        data = {role: CommonWindows(source.load(role), stats=stats, **options) for role in ["validation", "test"]}
        inputs = {role: torch.from_numpy(d.arrays(cfg["data"]["max_window"])[0]).to(device) for role, d in data.items()}
        directory = run_dir / "supports" / protocol
        data["test"].target_frame.to_parquet(directory / "test.parquet", index=False)
        write_json(directory / "test_support.json", dict(samples=len(data["test"].target_frame), anchors=len(data["test"]), support_sha256=data["test"].support_sha256))
        for role, d in data.items():
            baseline = d.target_frame[LOW].to_numpy(float)
            metrics.extend(dict(protocol=protocol, split=role, variant="raw", window=0, seed=-1, **r) for r in physical_metrics(d.target_frame, baseline))
        for j, job in enumerate([job for job in jobs if job["protocol"] == protocol]):
            checkpoint = Path(job["directory"]) / "best.pt"
            assert sha(checkpoint) == lock[job["id"]]
            saved = torch.load(checkpoint, map_location=device)
            model, meta = build_ablation_model(job["variant"], job["seed"], cfg["model"])
            model.load_state_dict(saved["model"])
            model = model.to(device).eval()
            dest = run_dir / "predictions"
            dest.mkdir(exist_ok=True)
            for role, d in data.items():
                norm = predict_normalized(model, inputs[role], window=job["window"], max_window=d.max_window, block_size=d.block_size,
                                          batch_size=cfg["evaluation"]["batch_size"], amp=amp)
                pred = d.physical(norm)
                np.savez_compressed(dest / (job["id"] + "__" + role + ".npz"), prediction=pred, support_sha256=np.array(d.support_sha256))
                metrics.extend(dict(job_id=job["id"], protocol=protocol, split=role, variant=job["variant"], window=job["window"], seed=job["seed"], **r) for r in physical_metrics(d.target_frame, pred))
                if role == "validation":
                    assert np.isclose(grouped_score(d.target_frame, pred), saved["val_score"], rtol=1e-7, atol=1e-8)
            inp = inputs["test"][:1, :, (d.max_window-job["window"])//2:(d.max_window+job["window"])//2]
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                for _ in range(cfg["evaluation"]["latency_warmup"]):
                    model(inp)
                if device.type == "cuda": torch.cuda.synchronize()
                timings = []
                for _ in range(cfg["evaluation"]["latency_repeats"]):
                    tick = time.perf_counter()
                    model(inp)
                    if device.type == "cuda": torch.cuda.synchronize()
                    timings.append(1000 * (time.perf_counter()-tick))
            costs.append(dict(job_id=job["id"], protocol=protocol, variant=job["variant"], window=job["window"], seed=job["seed"],
                              parameters=sum(p.numel() for p in model.parameters()), batch1_ms_mean=np.mean(timings), batch1_ms_median=np.median(timings)))
            pd.DataFrame(metrics).to_csv(run_dir / "metrics_per_seed.csv", index=False)
            pd.DataFrame(costs).to_csv(run_dir / "compute_cost.csv", index=False)
            write_json(run_dir / "evaluation_progress.json", dict(protocol=protocol, completed_in_protocol=j+1, planned_in_protocol=sum(q["protocol"]==protocol for q in jobs)))
            LOG.info("EVALUATED %s", job["id"])
    return pd.DataFrame(metrics)


def run(root, config_path, run_dir=None, mode="all"):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    default_name = cfg["experiment_id"] + ("_preflight" if mode == "preflight" else "")
    run_dir = run_dir or root / cfg["output"]["runs_dir"] / default_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")], force=True)
    LOG.info("READS %s; WRITES %s; SKIPS previous experiment artifacts and outputs/latest", config_path, run_dir)
    jobs = jobs_for_config(root, cfg, run_dir)
    if mode == "preflight":
        cfg["training"]["epochs"] = 2
        jobs = jobs_for_config(root, cfg, run_dir)
        selected = next(j for j in jobs if j["protocol"]=="alimu_only" and j["variant"]=="cnn_bilstm" and j["window"]==320)
        result = fit_job(selected)
        write_json(run_dir / "preflight.json", result)
        return
    freeze(root, config_path, cfg, run_dir, jobs)
    if mode != "evaluate":
        prepare_support(root, cfg, run_dir)
        records = execute(jobs, cfg["training"]["workers"], run_dir)
    else:
        records = [json.loads((Path(j["directory"]) / "fit.json").read_text(encoding="utf-8")) for j in jobs]
    write_json(run_dir / "training.complete.json", dict(status="complete", planned=len(jobs), completed=len(records), fits=records))
    if mode != "train":
        evaluate(root, cfg, run_dir, jobs, records)
        LOG.info("WRITES fixed TEST predictions and metrics; SKIPS TEST model reselection")
