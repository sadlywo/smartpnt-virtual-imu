"""Temporal convolutional network for IMU sequence regression."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right padding to keep causal sequence length."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual dilated convolution block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TCN(nn.Module):
    """Causal TCN mapping low-grade IMU windows to high-grade IMU windows."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 48,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.05,
        zero_init_head: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        channels = [in_channels, *([hidden_channels] * levels)]
        for level in range(levels):
            blocks.append(
                TemporalBlock(
                    channels[level],
                    channels[level + 1],
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*blocks)
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))
