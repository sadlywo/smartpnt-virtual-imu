"""Replacement fits using the frozen shared optimizer, supports and validation recipe."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from virtual_imu.ablations.data import CommonWindows, SharedSource, balanced_order, grouped_score
from virtual_imu.replacements.reuse import (
    completed_job,
    reuse_control,
)
from virtual_imu.utils.seed import seed_everything

LOG = logging.getLogger(__name__)
_CACHE = {}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(config):
    torch.set_num_threads(int(config["training"]["cpu_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(config["training"].get("deterministic", True))
    torch.backends.cuda.matmul.allow_tf32 = False
    if config["training"]["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested local CUDA is unavailable")
    return torch.device(config["training"]["device"])


def data_for_job(job, device):
    cfg = job["config"]
    key = (job["root"], job["protocol"], json_digest(cfg["data"]), str(device))
    if key not in _CACHE:
        source = SharedSource(job["root"], cfg["data"], job["protocol"])
        if source.dataset.manifest_sha256 != cfg["data"]["manifest_sha256"]:
            raise ValueError("Frozen data manifest changed")
        options = {k: cfg["data"][k] for k in ["max_window", "block_size", "stride"]}
        train = CommonWindows(source.load("train"), **options)
        val = CommonWindows(source.load("validation"), stats=train.stats, **options)
        tx, ty = train.arrays(cfg["data"]["max_window"])
        vx, vy = val.arrays(cfg["data"]["max_window"])
        dense_y = np.stack([train.y[s : s + train.max_window].T for s in train.starts])
        tensors = tuple(torch.from_numpy(v).to(device) for v in [tx, ty, vx, vy, dense_y])
        _CACHE[key] = (source, train, val, tensors)
    return _CACHE[key]


def predict_normalized(model, inputs, *, window, max_window, block_size, batch_size, amp):
    offset, center = (max_window - window) // 2, (window - block_size) // 2
    pieces = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            x = inputs[start : start + batch_size, :, offset : offset + window]
            with torch.autocast(device_type=inputs.device.type, enabled=amp):
                pred = model(x)[..., center : center + block_size]
            pieces.append(pred.float().cpu().numpy())
    result = np.concatenate(pieces)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite validation/test predictions")
    return result


def fit_job(job):
    from virtual_imu.replacements.models import build_replacement_model

    cfg, directory = job["config"], Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "fit.json"
    previous = completed_job(job)
    if previous is not None:
        return previous
    if job.get("reused", False):
        return reuse_control(job)
    handler = logging.FileHandler(directory / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    try:
        device = configure(cfg)
        _source, train, val, (tx, ty, vx, _vy, dense_y) = data_for_job(job, device)
        seed_everything(job["seed"])
        model, meta = build_replacement_model(
            job["variant"], job["seed"], cfg["model"], cfg["replacement_models"]
        )
        model = model.to(device)
        tr = cfg["training"]
        amp = bool(tr["amp"]) and device.type == "cuda"
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tr["lr"]),
            weight_decay=float(tr["weight_decay"]),
            betas=tuple(tr["betas"]),
        )
        epochs, warmup = tr["epochs"], min(tr["warmup_epochs"], tr["epochs"] - 1)
        minimum = tr["min_lr"] / tr["lr"]

        def schedule(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, max(0.0, (step - warmup) / max(1, epochs - warmup - 1)))
            return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        write_json(directory / "job.json", job)
        write_json(directory / "initialization.json", meta)
        write_json(directory / "stats.json", train.stats)
        LOG.info(
            "READS TRAIN=%d VALID=%d common anchors; WRITES %s; SKIPS TEST",
            len(train),
            len(val),
            directory,
        )
        best, patience_best, stale, best_epoch = float("inf"), float("inf"), 0, 0
        history, order_sha = [], hashlib.sha256()
        begin = time.perf_counter()
        batch_size, window, block = (
            cfg["data"]["batch_size"],
            job["window"],
            cfg["data"]["block_size"],
        )
        offset, center = (train.max_window - window) // 2, (window - block) // 2
        for epoch in range(1, epochs + 1):
            model.train()
            start, loss_sum, skipped = time.perf_counter(), 0.0, 0
            order = balanced_order(train, job["seed"], epoch)
            order_sha.update(order.tobytes())
            epoch_digest = hashlib.sha256(order.tobytes()).hexdigest()
            for i in range(0, len(order), batch_size):
                ix = torch.as_tensor(order[i : i + batch_size], device=device)
                x = tx[ix, :, offset : offset + window]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    pred = model(x)
                    if job["variant"] == "cnn_bilstm_dense":
                        loss = F.smooth_l1_loss(pred, dense_y[ix])
                    else:
                        loss = F.smooth_l1_loss(pred[..., center : center + block], ty[ix])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tr["grad_clip_norm"], error_if_nonfinite=not amp
                )
                previous = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped += int(scaler.get_scale() < previous)
                loss_sum += loss.detach().item() * len(ix)
            normal = predict_normalized(
                model,
                vx,
                window=window,
                max_window=val.max_window,
                block_size=block,
                batch_size=cfg["evaluation"]["batch_size"],
                amp=amp,
            )
            score = grouped_score(val.target_frame, val.physical(normal))
            elapsed = time.perf_counter() - start
            history.append(
                dict(
                    epoch=epoch,
                    train_loss=loss_sum / len(train),
                    val_score=score,
                    lr=opt.param_groups[0]["lr"],
                    epoch_seconds=elapsed,
                    amp_skipped_steps=skipped,
                    order_sha256=epoch_digest,
                )
            )
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            LOG.info(
                "%s epoch=%d/%d loss=%.6g val=%.6g seconds=%.2f",
                job["id"],
                epoch,
                epochs,
                loss_sum / len(train),
                score,
                elapsed,
            )
            if score < best:
                best, best_epoch = score, epoch
                torch.save(
                    dict(
                        model=model.state_dict(),
                        job=job,
                        stats=train.stats,
                        epoch=epoch,
                        val_score=score,
                        metadata=meta,
                        train_support_sha256=train.support_sha256,
                        validation_support_sha256=val.support_sha256,
                    ),
                    directory / "best.pt",
                )
            if score < patience_best - tr["min_delta"]:
                patience_best, stale = score, 0
            else:
                stale += 1
            scheduler.step()
            if stale >= tr["early_stopping_patience"]:
                LOG.info("SKIPS remaining epochs: early stopping at %d", epoch)
                break
        result = dict(
            job_id=job["id"],
            protocol=job["protocol"],
            variant=job["variant"],
            window=window,
            seed=job["seed"],
            best_epoch=best_epoch,
            actual_epochs=len(history),
            best_val_score=best,
            train_anchors=len(train),
            validation_anchors=len(val),
            elapsed_seconds=time.perf_counter() - begin,
            amp_skipped_steps=sum(r["amp_skipped_steps"] for r in history),
            order_sha256=order_sha.hexdigest(),
            train_support_sha256=train.support_sha256,
            validation_support_sha256=val.support_sha256,
            checkpoint_sha256=sha(directory / "best.pt"),
            job_config_sha256=json_digest(job),
            directory=str(directory),
            parameters=sum(p.numel() for p in model.parameters()),
            reused=False,
            fit_kind="new_fit",
            source_job_id=None,
            source_run=None,
            training_cost_origin="current_run",
        )
        write_json(complete, result)
        return result
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
