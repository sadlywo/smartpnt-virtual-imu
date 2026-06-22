"""Model factory and model exports."""

from __future__ import annotations

import torch.nn as nn

from virtual_imu.models.gru import GRUModel
from virtual_imu.models.tcn import TCN
from virtual_imu.models.transformer import TransformerIMU
from virtual_imu.models.unet1d import UNet1D


def make_model(config: dict) -> nn.Module:
    """Create a model from a config dictionary."""
    model_type = str(config.get("type", "tcn")).lower()
    if model_type == "tcn":
        return TCN(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_channels=int(config.get("hidden_channels", 48)),
            levels=int(config.get("levels", 4)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.05)),
            zero_init_head=bool(config.get("zero_init_head", False)),
        )
    if model_type == "gru":
        return GRUModel(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if model_type == "transformer":
        return TransformerIMU(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            d_model=int(config.get("d_model", 64)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_type in {"unet", "unet1d"}:
        return UNet1D(
            in_channels=int(config.get("in_channels", 7)),
            out_channels=int(config.get("out_channels", 6)),
            base_channels=int(config.get("base_channels", 32)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
