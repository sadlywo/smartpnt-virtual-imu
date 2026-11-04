"""Transformer encoder model for IMU windows."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerIMU(nn.Module):
    """Transformer encoder mapping low-grade IMU windows to high-grade windows."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        norm_first: bool = False,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.input = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
            dim_feedforward=dim_feedforward, norm_first=norm_first, activation=activation,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        encoded = self.encoder(self.pos(self.input(seq)))
        return self.head(encoded).transpose(1, 2)
