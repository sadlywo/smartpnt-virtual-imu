"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> Path:
    """Save a PyTorch checkpoint with parent creation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    return out


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint with a clear missing-file error."""
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return torch.load(ckpt, map_location=map_location)
