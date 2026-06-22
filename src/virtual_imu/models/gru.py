"""GRU sequence model."""

from __future__ import annotations

import torch
from torch import nn


class GRUModel(nn.Module):
    """GRU model with channel-first input/output."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        y, _ = self.gru(seq)
        return self.head(y).transpose(1, 2)
