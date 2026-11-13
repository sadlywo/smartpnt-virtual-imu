"""Adam, mixed precision, fixed-seed validation sampling and early stopping."""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from virtual_imu.config import deep_update, save_config
from virtual_imu.diffusion.evaluation import physical_prediction, relative_score
from virtual_imu.models.ddpm import build_ddpm
from virtual_imu.utils.seed import seed_everything

LOG = logging.getLogger(__name__)


def choose_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
    )


def validation_subset(windows, count: int):
    indexes = np.unique(np.linspace(0, len(windows) - 1, min(count, len(windows)), dtype=int))
    return Subset(windows, indexes.tolist())


def fit_trial(config: dict, trial: dict, train, validation, run_dir: Path, context: dict) -> dict:
    cfg = deep_update(config, trial)
    cfg["model"]["max_length"] = int(cfg["data"]["window_size"])
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = choose_device(cfg["training"]["device"])
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    torch.backends.cudnn.benchmark = False
    model = build_ddpm(cfg).to(device)
    tr = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(tr["lr"]),
        betas=tuple(tr.get("betas", [0.9, 0.999])),
        weight_decay=float(tr.get("weight_decay", 0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    train_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=train_generator,
    )
    subset = validation_subset(validation, int(tr["validation_windows"]))
    run_dir.mkdir(parents=True, exist_ok=False)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "validation_windows.json").write_text(json.dumps(subset.indices), encoding="utf-8")
    LOG.info(
        "WRITES trial=%s config, history.csv, best.pt; READS train=%d validation=%d windows; "
        "SKIPS test during tuning",
        run_dir,
        len(train),
        len(subset),
    )
    best = float("inf")
    patience_best = float("inf")
    stale = 0
    history = []
    begin = time.perf_counter()
    parameters = sum(p.numel() for p in model.parameters())
    for epoch in range(1, int(tr["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        epoch_start = time.perf_counter()
        for condition, clean in loader:
            condition = condition.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model.training_loss(clean, condition)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite DDPM training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(tr["grad_clip_norm"]), error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(clean)
            samples += len(clean)
        # A fixed random stream makes stochastic validation comparable between epochs/trials.
        prediction, mask = physical_prediction(
            model,
            validation,
            device,
            cfg["data"]["batch_size"],
            seed + 10000,
            int(cfg["evaluation"]["draws"]),
            amp,
            window_indices=subset.indices,
        )
        score = relative_score(prediction[mask], validation.low[mask], validation.high[mask])
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / samples,
            "validation_score": score,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        LOG.info(
            "%s epoch=%d/%d loss=%.5f val_relative_rmse=%.5f seconds=%.1f",
            run_dir.name,
            epoch,
            tr["epochs"],
            row["train_loss"],
            score,
            row["epoch_seconds"],
        )
        if score < patience_best - float(tr["min_delta"]):
            patience_best = score
            stale = 0
        else:
            stale += 1
        # Persist the actual minimum, including improvements below the patience threshold.
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                    "stats": train.stats,
                    "epoch": epoch,
                    "validation_score": score,
                    "context": context,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else [],
                    "loader_rng_state": train_generator.get_state(),
                },
                run_dir / "best.pt",
            )
        if stale >= int(tr["early_stopping_patience"]):
            LOG.info("SKIPS remaining epochs: early stopping %s best_epoch=%d", run_dir, best_epoch)
            break
    summary = {
        "trial": run_dir.name,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "validation_score": best,
        "parameters": parameters,
        "seconds": time.perf_counter() - begin,
        "early_stopped": epoch < tr["epochs"],
        "checkpoint": str((run_dir / "best.pt").resolve()),
        "model": copy.deepcopy(cfg["model"]),
        "lr": float(tr["lr"]),
        "prediction_type": cfg["diffusion"]["prediction_type"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def load_trained(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_ddpm(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint
