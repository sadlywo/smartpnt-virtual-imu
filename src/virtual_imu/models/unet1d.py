"""Small 1D U-Net for sequence-to-sequence IMU generation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two-layer 1D convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet1D(nn.Module):
    """Compact 1D U-Net."""

    def __init__(self, in_channels: int = 7, out_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 3, base_channels)
        self.head = nn.Conv1d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        b = self.bottleneck(F.avg_pool1d(e2, 2))
        u2 = F.interpolate(b, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        return self.head(self.dec1(torch.cat([u1, e1], dim=1)))
