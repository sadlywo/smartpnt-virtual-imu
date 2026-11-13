"""Frozen 48-fit / 18-import module comparison with a complete TEST gate."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.ablations.data import (
    LOW,
    CommonWindows,
    SharedSource,
    grouped_score,
    physical_metrics,
    support_digest,
)
from virtual_imu.ablations.pipeline import prepare_support
from virtual_imu.ablations.training import json_digest, sha, write_json
from virtual_imu.replacements.reuse import (
    CONTROL_MAP,
    check_checkpoint_identity,
    read_json,
    reuse_control,
    validate_import,
)
from virtual_imu.replacements.training import configure, fit_job, predict_normalized

LOG = logging.getLogger(__name__)
VARIANTS = (
    "cnn_lstm64",
    "cnn_bilstm32",
    "cnn_identity",
    "cnn_gru",
    "cnn_bigru",
    "cnn_transformer",
    "cnn_tcn",
    "linear_lstm64",
    "mlp_lstm64",
    "gru_lstm64",
    "transformer_lstm64",
)
ANALYSIS = [
    "scripts/report_module_replacement.py",
    "scripts/verify_module_replacement.py",
    "src/virtual_imu/viz/plots_module_replacement.py",
    "src/virtual_imu/viz/style.py",
    "scripts/report_cnn_bilstm_ablation.py",
    "scripts/verify_cnn_bilstm_ablation.py",
]


def validate_config(cfg):
    for key in ["variants", "seeds", "protocols"]:
        if not cfg[key] or len(cfg[key]) != len(set(cfg[key])):
            raise ValueError("Empty or duplicate " + key)
    if set(cfg["variants"]) != set(VARIANTS) or cfg["windows"] != [320]:
        raise ValueError("Require all11 registered variants at fixed320")
    if cfg["reference_variant"] != "cnn_lstm64" or cfg["reuse"]["variants"] != CONTROL_MAP:
        raise ValueError("Unregistered reference or reuse mapping")
    if set(cfg["protocols"]) != {"alimu_only", "pooled"} or len(cfg["seeds"]) != 3:
        raise ValueError("Require two scopes and three optimization seeds")
    if any(type(s) is not int or s < 0 for s in cfg["seeds"]):
        raise ValueError("Seeds must be nonnegative integers")
    if (
        cfg["data"]["max_window"] != 320
        or cfg["data"]["stride"] != cfg["data"]["block_size"]
        or cfg["data"]["block_size"] != 40
    ):
        raise ValueError("Require common320 inputs and disjoint center40 outputs")
    if cfg["training"]["optimizer"] != "AdamW" or cfg["training"]["epochs"] != 80:
        raise ValueError("Require fixed AdamW/80-epoch recipe")


def jobs_for_config(root, cfg, run_dir):
    jobs = []
    for protocol in cfg["protocols"]:
        for variant in cfg["variants"]:
            for seed in cfg["seeds"]:
                name = f"{protocol}__{variant}__w320__s{seed}"
                jobs.append(
                    dict(
                        id=name,
                        root=str(root),
                        config=cfg,
                        protocol=protocol,
                        variant=variant,
                        window=320,
                        seed=seed,
                        directory=str(run_dir / "fits" / name),
                        reused=variant in cfg["reuse"]["variants"],
                        source_variant=cfg["reuse"]["variants"].get(variant),
                    )
                )
    return [jobs[i] for i in np.random.default_rng(cfg["seed"]).permutation(len(jobs))]


def freeze(root, config_path, cfg, run_dir, jobs):
    if (run_dir / "provenance.json").is_file():
        previous = read_json(run_dir / "provenance.json")
        if previous["config_sha256"] != sha(config_path):
            raise ValueError("Cannot resume a different replacement configuration")
        for rel, expected in previous["source_sha256"].items():
            if sha(root / rel) != expected or sha(run_dir / "source_snapshot" / rel) != expected:
                raise ValueError("Frozen source changed: " + rel)
        if (
            read_json(run_dir / "planned_jobs.json") != jobs
            or sha(run_dir / "config.yaml") != previous["config_sha256"]
            or sha(run_dir / "data_manifest.json") != previous["manifest_sha256"]
        ):
            raise ValueError("Frozen configuration, manifest or planned identities changed")
        LOG.info("SKIPS rewriting frozen replacement config/source")
        return previous
    source_paths = [
        *sorted((root / "src/virtual_imu/replacements").glob("*.py")),
        *sorted((root / "src/virtual_imu/ablations").glob("*.py")),
        root / "src/virtual_imu/models/matched.py",
        root / "src/virtual_imu/models/ddpm.py",
        root / "src/virtual_imu/models/transformer.py",
        root / "src/virtual_imu/fir_arx_research/data.py",
        root / "src/virtual_imu/utils/seed.py",
        root / "scripts/run_module_replacement.py",
        config_path,
    ]
    source_hashes = {}
    for path in source_paths:
        rel = path.relative_to(root)
        dest = run_dir / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[rel.as_posix()] = sha(path)
    manifest = root / cfg["data"]["manifest"]
    if sha(manifest) != cfg["data"]["manifest_sha256"]:
        raise ValueError("Data manifest differs from pinned source")
    shutil.copy2(config_path, run_dir / "config.yaml")
    shutil.copy2(manifest, run_dir / "data_manifest.json")
    old = root / cfg["reuse"]["run_dir"]
    prov = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        config_sha256=sha(config_path),
        manifest_sha256=sha(manifest),
        source_sha256=source_hashes,
        reused_run=str(old),
        reused_run_complete_sha256=sha(old / "complete.json"),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        torch=torch.__version__,
        planned_conditions=len(jobs),
        planned_new_fits=sum(not j["reused"] for j in jobs),
        planned_reused_controls=sum(j["reused"] for j in jobs),
        test_used_for_selection=False,
        support_convention="identical center40 targets in same320 anchors",
        previous_TEST_was_seen=True,
        all_replacements_fixed_before_new_TEST=True,
    )
    write_json(run_dir / "provenance.json", prov)
    write_json(run_dir / "planned_jobs.json", jobs)
    return prov


def freeze_analysis(root, run_dir):
    target = run_dir / "analysis_plan.json"
    if target.is_file():
        plan = read_json(target)
        for rel, expected in plan["source_sha256"].items():
            if sha(root / rel) != expected or sha(run_dir / "analysis_source" / rel) != expected:
                raise ValueError("Pre-TEST analysis source changed: " + rel)
        return plan
    hashes = {}
    for rel in ANALYSIS:
        path, dest = root / rel, run_dir / "analysis_source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[rel] = sha(path)
    plan = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,
        test_unlocked=False,
        reference="cnn_lstm64",
        window=320,
        selection="rank mean normalized VALID across three seeds within training scope; TEST descriptive only",
        paired_delta="exact same protocol/split/support/window/seed vs reference; positive means worse",
        seed_SD="optimization variation, not recording CI",
        reuse_count=18,
        new_fit_count=48,
        capacity="non-deletion replacements approximately within5% reference parameters; dimensions frozen before fitting",
        context="effective receptive field and future information differ by architecture; no causal-operator isolation",
    )
    write_json(target, plan)
    return plan


def execute(jobs, workers, run_dir):
    records, failures = [], []
    begin = time.perf_counter()

    def save_progress(last):
        pd.DataFrame(records).to_csv(run_dir / "fit_registry.csv", index=False)
        write_json(
            run_dir / "progress.json",
            dict(
                phase="training",
                completed=len(records),
                planned=len(jobs),
                new_completed=sum(not r["reused"] for r in records),
                reused_completed=sum(r["reused"] for r in records),
                failed=failures,
                elapsed_seconds=time.perf_counter() - begin,
                last_job=last,
            ),
        )

    for job in jobs:
        if job["reused"]:
            records.append(reuse_control(job))
            save_progress(job["id"])
    fresh = [j for j in jobs if not j["reused"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_job, job): job for job in fresh}
        for future in as_completed(futures):
            job = futures[future]
            try:
                records.append(future.result())
                LOG.info(
                    "COMPLETED %d/%d; new=%d/48 %s",
                    len(records),
                    len(jobs),
                    sum(not r["reused"] for r in records),
                    job["id"],
                )
            except Exception as error:
                failures.append(dict(job=job["id"], error=repr(error)))
                LOG.exception("Replacement fit failed: %s", job["id"])
            save_progress(job["id"])
    if failures:
        raise RuntimeError("Failed replacement fits: " + json.dumps(failures))
    return records


def validate_scientific_checkpoint(job, record, saved, run_dir):
    """Check the frozen initialization and TRAIN/VALID identity before TEST."""
    from virtual_imu.replacements.models import build_replacement_model

    cfg, directory = job["config"], Path(job["directory"])
    if read_json(directory / "job.json") != job:
        raise ValueError("Job artifact differs from planned identity")
    model, meta = build_replacement_model(
        job["variant"], job["seed"], cfg["model"], cfg["replacement_models"]
    )
    if meta != read_json(directory / "initialization.json"):
        raise ValueError("Initialization receipt differs from frozen factory")
    if job["reused"]:
        validate_import(job, record)
        expected_meta = read_json(directory / "source_initialization.json")
        if expected_meta["init_state_sha256"] != meta["init_state_sha256"]:
            raise ValueError("Reused initialization state differs")
    else:
        expected_meta = meta
    if saved["metadata"] != expected_meta:
        raise ValueError("Checkpoint initialization metadata differs")
    if not all(torch.isfinite(value).all() for value in saved["model"].values()):
        raise ValueError("Nonfinite checkpoint state")
    model.load_state_dict(saved["model"], strict=True)
    support_dir = run_dir / "supports" / job["protocol"]
    stats = read_json(support_dir / "statistics.json")
    if saved["stats"] != stats or read_json(directory / "stats.json") != stats:
        raise ValueError("Checkpoint TRAIN statistics differ")
    for role in ["train", "validation"]:
        expected = support_digest(pd.read_parquet(support_dir / (role + ".parquet")))
        if saved[role + "_support_sha256"] != expected:
            raise ValueError("Checkpoint " + role + " support differs")
    history = pd.read_csv(directory / "history.csv")
    if (
        len(history) != record["actual_epochs"]
        or len(history) > cfg["training"]["epochs"]
        or history.epoch.tolist() != list(range(1, len(history) + 1))
    ):
        raise ValueError("Checkpoint epoch history count differs")


def preserve_json(path, data):
    """Create once; accept an identical existing lock without rewriting it."""
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        if read_json(path) != data:
            raise ValueError("Existing immutable lock differs: " + str(path)) from None


def record_test_access(run_dir):
    path = run_dir / "test_access.json"
    digest = sha(run_dir / "selections.lock.json")
    if path.is_file():
        if read_json(path)["selection_lock_sha256"] != digest:
            raise ValueError("TEST access already bound to another checkpoint lock")
        return
    preserve_json(
        path,
        dict(
            first_access_utc=datetime.now(timezone.utc).isoformat(),
            selection_lock_sha256=digest,
            numerical_TEST_access=True,
        ),
    )


def lock_selections(root, cfg, run_dir, jobs, records):
    if len(records) != len(jobs) or {r["job_id"] for r in records} != {j["id"] for j in jobs}:
        raise ValueError("All planned identities must complete before TEST unlock")
    lookup = {r["job_id"]: r for r in records}
    locks = {}
    for job in jobs:
        directory, record = Path(job["directory"]), lookup[job["id"]]
        checkpoint = directory / "best.pt"
        if sha(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint bytes differ before TEST unlock")
        saved = torch.load(checkpoint, map_location="cpu")
        receipt = read_json(directory / "import.json") if job["reused"] else None
        check_checkpoint_identity(job, record, saved, receipt)
        validate_scientific_checkpoint(job, record, saved, run_dir)
        history = pd.read_csv(directory / "history.csv")
        best = history.loc[history.val_score.idxmin()]
        if int(best.epoch) != record["best_epoch"] or not np.isclose(
            best.val_score, record["best_val_score"], rtol=1e-12
        ):
            raise ValueError("Checkpoint is not the minimum VALID epoch")
        if not np.isfinite(history[["val_score", "train_loss"]]).all().all():
            raise ValueError("Nonfinite fitting history")
        locks[job["id"]] = sha(checkpoint)
    if yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) != cfg:
        raise ValueError("Checkpoint lock configuration differs")
    preserve_json(
        run_dir / "selections.lock.json",
        dict(
            test_loaded=False,
            checkpoints_sha256=locks,
            initialization_sha256={
                j["id"]: sha(Path(j["directory"]) / "initialization.json") for j in jobs
            },
            config_sha256=sha(run_dir / "config.yaml"),
            jobs_sha256=json_digest(jobs),
            analysis_plan_sha256=sha(run_dir / "analysis_plan.json"),
            new_fits=sum(not r["reused"] for r in records),
            reused_controls=sum(r["reused"] for r in records),
        ),
    )
    return locks


def evaluate(root, cfg, run_dir, jobs, records):
    from virtual_imu.replacements.models import build_replacement_model

    freeze_analysis(root, run_dir)
    locks = lock_selections(root, cfg, run_dir, jobs, records)
    device, by_id = configure(cfg), {r["job_id"]: r for r in records}
    metrics, costs, evaluated = [], [], 0
    amp = cfg["training"]["amp"] and device.type == "cuda"
    record_test_access(run_dir)
    for protocol in cfg["protocols"]:
        source = SharedSource(root, cfg["data"], protocol)
        options = {key: cfg["data"][key] for key in ["max_window", "block_size", "stride"]}
        stats = read_json(run_dir / "supports" / protocol / "statistics.json")
        data = {
            role: CommonWindows(source.load(role), stats=stats, **options)
            for role in ["validation", "test"]
        }
        inputs = {role: torch.from_numpy(d.arrays(320)[0]).to(device) for role, d in data.items()}
        data["test"].target_frame.to_parquet(
            run_dir / "supports" / protocol / "test.parquet", index=False
        )
        write_json(
            run_dir / "supports" / protocol / "test_support.json",
            dict(
                samples=len(data["test"].target_frame),
                anchors=len(data["test"]),
                support_sha256=data["test"].support_sha256,
            ),
        )
        for role, d in data.items():
            metrics.extend(
                dict(
                    protocol=protocol,
                    split=role,
                    variant="raw",
                    window=0,
                    seed=-1,
                    reused=False,
                    **r,
                )
                for r in physical_metrics(d.target_frame, d.target_frame[LOW].to_numpy(float))
            )
        for job in [j for j in jobs if j["protocol"] == protocol]:
            directory, record = Path(job["directory"]), by_id[job["id"]]
            assert sha(directory / "best.pt") == locks[job["id"]]
            saved = torch.load(directory / "best.pt", map_location=device)
            model, _ = build_replacement_model(
                job["variant"], job["seed"], cfg["model"], cfg["replacement_models"]
            )
            model.load_state_dict(saved["model"], strict=True)
            model = model.to(device).eval()
            dest = run_dir / "predictions"
            dest.mkdir(exist_ok=True)
            for role, d in data.items():
                norm = predict_normalized(
                    model,
                    inputs[role],
                    window=320,
                    max_window=320,
                    block_size=40,
                    batch_size=cfg["evaluation"]["batch_size"],
                    amp=amp,
                )
                pred = d.physical(norm)
                np.savez_compressed(
                    dest / (job["id"] + "__" + role + ".npz"),
                    prediction=pred,
                    support_sha256=np.array(d.support_sha256),
                    job_id=np.array(job["id"]),
                    split=np.array(role),
                    checkpoint_sha256=np.array(locks[job["id"]]),
                    initialization_sha256=np.array(sha(directory / "initialization.json")),
                    config_sha256=np.array(sha(run_dir / "config.yaml")),
                )
                metrics.extend(
                    dict(
                        job_id=job["id"],
                        protocol=protocol,
                        split=role,
                        variant=job["variant"],
                        window=320,
                        seed=job["seed"],
                        reused=job["reused"],
                        **r,
                    )
                    for r in physical_metrics(d.target_frame, pred)
                )
                if role == "validation" and not np.isclose(
                    grouped_score(d.target_frame, pred), saved["val_score"], rtol=1e-7, atol=1e-8
                ):
                    raise ValueError("VALID replay differs from selected checkpoint")
                if job["reused"]:
                    origin = root / cfg["reuse"]["run_dir"]
                    previous = (
                        origin / "predictions" / (record["source_job_id"] + "__" + role + ".npz")
                    )
                    expected = read_json(origin / "artifact_manifest.json")["files_sha256"][
                        previous.relative_to(origin).as_posix()
                    ]
                    if sha(previous) != expected:
                        raise ValueError("Reused source prediction hash mismatch")
                    with np.load(previous) as archive:
                        if str(archive["support_sha256"]) != d.support_sha256 or not np.allclose(
                            pred, archive["prediction"], rtol=1e-7, atol=1e-10
                        ):
                            raise ValueError("Reused control predictions differ from source")
            timings = []
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                example = inputs["test"][:1]
                for _ in range(cfg["evaluation"]["latency_warmup"]):
                    model(example)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                for _ in range(cfg["evaluation"]["latency_repeats"]):
                    begin = time.perf_counter()
                    model(example)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timings.append(1000 * (time.perf_counter() - begin))
            costs.append(
                dict(
                    job_id=job["id"],
                    protocol=protocol,
                    variant=job["variant"],
                    window=320,
                    seed=job["seed"],
                    parameters=sum(p.numel() for p in model.parameters()),
                    batch1_ms_mean=np.mean(timings),
                    batch1_ms_median=np.median(timings),
                    reused=job["reused"],
                    latency_origin="current_run_uniform_measurement",
                )
            )
            pd.DataFrame(metrics).to_csv(run_dir / "metrics_per_seed.csv", index=False)
            pd.DataFrame(costs).to_csv(run_dir / "compute_cost.csv", index=False)
            evaluated += 1
            write_json(
                run_dir / "evaluation_progress.json",
                dict(completed=evaluated, planned=len(jobs), last_job=job["id"]),
            )
            LOG.info("EVALUATED %d/%d %s reused=%s", evaluated, len(jobs), job["id"], job["reused"])
    write_json(
        run_dir / "evaluation.complete.json",
        dict(status="complete", conditions=len(jobs), prediction_files=2 * len(jobs)),
    )


def run(root, config_path, run_dir=None, mode="all"):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    run_dir = run_dir or root / cfg["output"]["runs_dir"] / (
        cfg["experiment_id"] + ("_preflight" if mode == "preflight" else "")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8"),
        ],
        force=True,
    )
    LOG.info(
        "READS %s and sealed controls; WRITES %s; SKIPS prior artifacts and outputs/latest",
        config_path,
        run_dir,
    )
    if mode == "preflight":
        from virtual_imu.replacements.preflight import preflight

        return preflight(root, cfg, run_dir)
    jobs = jobs_for_config(root, cfg, run_dir)
    freeze(root, config_path, cfg, run_dir, jobs)
    if mode != "train":
        freeze_analysis(root, run_dir)
    if mode != "evaluate":
        prepare_support(root, cfg, run_dir)
        records = execute(jobs, cfg["training"]["workers"], run_dir)
    else:
        records = [read_json(Path(j["directory"]) / "fit.json") for j in jobs]
    write_json(
        run_dir / "training.complete.json",
        dict(
            status="complete",
            planned=len(jobs),
            completed=len(records),
            new_fits=sum(not r["reused"] for r in records),
            reused_controls=sum(r["reused"] for r in records),
            fits=records,
        ),
    )
    if mode != "train":
        evaluate(root, cfg, run_dir, jobs, records)
