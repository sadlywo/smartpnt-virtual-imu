"""Training utilities with lazy imports for read-only experiment planning."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "SmartPNTGTWindowDataset",
    "make_smartpnt_gt_dataloader",
    "make_smartpnt_gt_dataloaders",
    "make_smartpnt_gt_split_dataloaders",
    "train_gt_supervised_model",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = "gt_trainer" if name == "train_gt_supervised_model" else "gt_supervised_dataset"
    return getattr(import_module(f"virtual_imu.training.{module}"), name)
