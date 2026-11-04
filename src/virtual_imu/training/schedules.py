"""Opt-in optimizer schedules for validation-only hyperparameter searches."""

from __future__ import annotations

import math

import torch


def warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Warm up linearly, then decay to the explicit minimum learning rate."""
    epochs = int(config["epochs"])
    warmup = int(config.get("warmup_epochs", min(5, epochs // 4)))
    minimum = float(config.get("min_lr", 1e-6))
    if epochs < 1 or not 0 <= warmup < epochs or minimum < 0:
        raise ValueError("Require epochs>0, 0<=warmup_epochs<epochs and min_lr>=0")
    if any(minimum > float(group["lr"]) for group in optimizer.param_groups):
        raise ValueError("Minimum LR exceeds the initial LR")
    initial = [float(group["lr"]) for group in optimizer.param_groups]

    def factor(step: int, base: float):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, epochs - warmup - 1)))
        return minimum / base + (1 - minimum / base) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda step, base=base: factor(step, base) for base in initial]
    )
