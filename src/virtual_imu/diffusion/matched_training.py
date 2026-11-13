"""Common seeded Adam training for matched direct and diffusion models."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from virtual_imu.config import save_config
from virtual_imu.diffusion.data import PairedSource, PairedWindows, sha256
from virtual_imu.diffusion.evaluation import physical_prediction, relative_score
from virtual_imu.diffusion.matched_io import write_json
from virtual_imu.diffusion.training import choose_device, validation_subset
from virtual_imu.models.matched import build_matched_model
from virtual_imu.utils.seed import seed_everything

LOG = logging.getLogger(__name__)
_CACHE = {}


def state_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def sampling_draws(config):
    return int(config["evaluation"]["draws"]) if config["method"] == "ddpm" else 1


def data_for_job(root, config, pair_id, device):
    key = (
        str(root),
        config["data"]["manifest"],
        pair_id,
        config["data"]["window_size"],
        config["data"]["stride"],
        str(device),
    )
    if key not in _CACHE:
        source = PairedSource(Path(root), config["data"], pair_id)
        train = PairedWindows(
            source.load("train"), config["data"]["window_size"], config["data"]["stride"]
        )
        val = PairedWindows(
            source.load("validation"),
            config["data"]["window_size"],
            config["data"]["eval_stride"],
            train.stats,
        )
        # Small contiguous window tensors avoid repeated CPU slicing/copies.
        samples = [train[i] for i in range(len(train))]
        tensors = (
            torch.stack([sample[0] for sample in samples]).to(device),
            torch.stack([sample[1] for sample in samples]).to(device),
        )
        _CACHE[key] = (train, val, tensors)
    return _CACHE[key]


def fit_job(job):
    """Top-level process-pool entry; it never reads TEST signals."""
    cfg, directory = job["config"], Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(directory / "training.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    try:
        expected_manifest = job["context"].get("data_manifest_sha256")
        if (
            expected_manifest
            and sha256(Path(job["root"]) / cfg["data"]["manifest"]) != expected_manifest
        ):
            raise ValueError("Training manifest differs from the frozen experiment")
        torch.set_num_threads(int(cfg["training"].get("cpu_threads", 1)))
        device = choose_device(cfg["training"]["device"])
        train, val, (train_x, train_y) = data_for_job(job["root"], cfg, job["pair_id"], device)
        seed = int(cfg["seed"])
        seed_everything(seed)
        torch.backends.cudnn.benchmark = False
        model = build_matched_model(cfg).to(device)
        initial = state_hash(model.denoiser) if hasattr(model, "denoiser") else state_hash(model)
        tr = cfg["training"]
        amp = bool(tr.get("amp", True)) and device.type == "cuda"
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(tr["lr"]),
            betas=tuple(tr.get("betas", [0.9, 0.999])),
            weight_decay=float(tr.get("weight_decay", 0)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        generator = torch.Generator().manual_seed(seed)
        subset = validation_subset(val, int(tr["validation_windows"]))
        save_config(cfg, directory / "config.yaml")
        write_json(directory / "validation_windows.json", subset.indices)
        LOG.info(
            "READS TRAIN=%d VALID=%d windows; WRITES %s; SKIPS TEST",
            len(train),
            len(subset),
            directory,
        )
        best, patience_best, stale, history = float("inf"), float("inf"), 0, []
        order_digest = hashlib.sha256()
        begin = time.perf_counter()
        batch_size = int(cfg["data"]["batch_size"])
        for epoch in range(1, int(tr["epochs"]) + 1):
            model.train()
            total = 0.0
            skipped_steps = 0
            order = torch.randperm(len(train), generator=generator)
            order_digest.update(order.numpy().tobytes())
            start = time.perf_counter()
            for offset in range(0, len(order), batch_size):
                indexes = order[offset : offset + batch_size].to(device)
                condition, clean = train_x[indexes], train_y[indexes]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss = model.training_loss(clean, condition)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # AMP records overflow during unscale_ and skips that optimizer
                # step. Raising here prevents its normal scale backoff/recovery.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(tr["grad_clip_norm"]), error_if_nonfinite=not amp
                )
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped_steps += int(scaler.get_scale() < previous_scale)
                total += float(loss.detach()) * len(indexes)
            model.eval()
            prediction, mask = physical_prediction(
                model,
                val,
                device,
                batch_size,
                seed + 10000,
                sampling_draws(cfg),
                amp,
                subset.indices,
            )
            score = relative_score(prediction[mask], val.low[mask], val.high[mask])
            if not np.isfinite(score):
                raise FloatingPointError("Nonfinite validation metric")
            history.append(
                dict(
                    epoch=epoch,
                    train_loss=total / len(train),
                    validation_score=score,
                    epoch_seconds=time.perf_counter() - start,
                    amp_skipped_steps=skipped_steps,
                )
            )
            pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
            LOG.info(
                "epoch=%d/%d loss=%.6g valid=%.6g seconds=%.2f amp_skipped_steps=%d",
                epoch,
                tr["epochs"],
                history[-1]["train_loss"],
                score,
                history[-1]["epoch_seconds"],
                skipped_steps,
            )
            if score < patience_best - float(tr["min_delta"]):
                patience_best, stale = score, 0
            else:
                stale += 1
            if score < best:
                best, best_epoch = score, epoch
                torch.save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),
                        config=cfg,
                        stats=train.stats,
                        epoch=epoch,
                        validation_score=score,
                        context=job["context"],
                        initial_backbone_sha256=initial,
                    ),
                    directory / "best.pt",
                )
            if stale >= int(tr["early_stopping_patience"]):
                LOG.info("SKIPS remaining epochs: patience exhausted")
                break
        result = dict(
            method=cfg["method"],
            pair_id=job["pair_id"],
            seed=seed,
            candidate=job["candidate"],
            best_epoch=best_epoch,
            epochs_ran=epoch,
            validation_score=best,
            parameters=sum(p.numel() for p in model.parameters()),
            seconds=time.perf_counter() - begin,
            checkpoint=str((directory / "best.pt").resolve()),
            checkpoint_sha256=sha256(directory / "best.pt"),
            config=cfg,
            initial_backbone_sha256=initial,
            minibatch_order_sha256=order_digest.hexdigest(),
        )
        write_json(directory / "summary.json", result)
        del model, optimizer, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_matched_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint
