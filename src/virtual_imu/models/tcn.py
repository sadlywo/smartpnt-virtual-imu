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


class FiLMTCN(nn.Module):
    """TCN whose hidden features are modulated by repeated context channels."""

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 6,
        context_channels: int = 5,
        hidden_channels: int = 48,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.05,
        zero_init_head: bool = False,
    ) -> None:
        super().__init__()
        if context_channels <= 0:
            raise ValueError("context_channels must be positive")
        if context_channels >= in_channels:
            raise ValueError("context_channels must be smaller than in_channels")
        self.context_channels = int(context_channels)
        self.signal_channels = int(in_channels) - self.context_channels
        self.input = nn.Conv1d(self.signal_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TemporalBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
                for level in range(levels)
            ]
        )
        self.conditioner = nn.Sequential(
            nn.Linear(self.context_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
        )
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = x[:, : self.signal_channels]
        context = x[:, self.signal_channels :].mean(dim=-1)
        gamma, beta = self.conditioner(context).chunk(2, dim=1)
        gamma = 1.0 + gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        h = self.input(signal)
        for block in self.blocks:
            h = block(h)
            h = gamma * h + beta
        return self.head(h)


class BiasDriftTCN(nn.Module):
    """TCN with separate short-window bias and long-static drift branches."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        static_channels: int = 7,
        hidden_channels: int = 48,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.05,
        use_bias_branch: bool = True,
        use_drift_branch: bool = True,
        bias_branch_mode: str = "offset",
        branch_init_scale: float = 0.1,
        zero_init_head: bool = False,
    ) -> None:
        super().__init__()
        self.use_bias_branch = bool(use_bias_branch)
        self.use_drift_branch = bool(use_drift_branch)
        self.bias_branch_mode = str(bias_branch_mode).lower()
        self.last_aux: dict[str, torch.Tensor] = {}
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
        self.main = nn.Sequential(*blocks)
        self.bias_branch: nn.Module | None
        if not self.use_bias_branch:
            self.bias_branch = None
        elif self.bias_branch_mode == "offset":
            self.bias_branch = nn.Sequential(
                nn.Conv1d(6, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
            )
        elif self.bias_branch_mode in {"hidden", "feature"}:
            self.bias_branch = nn.Sequential(
                nn.Conv1d(6, hidden_channels, kernel_size=1),
                nn.ReLU(),
                TemporalBlock(hidden_channels, hidden_channels, kernel_size=kernel_size, dilation=1, dropout=dropout),
            )
        else:
            raise ValueError(f"Unsupported bias_branch_mode: {bias_branch_mode}")
        self.drift_branch = (
            nn.Sequential(
                nn.Conv1d(static_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            if self.use_drift_branch
            else None
        )
        self.bias_scale = nn.Parameter(torch.tensor(float(branch_init_scale)))
        self.drift_scale = nn.Parameter(torch.tensor(float(branch_init_scale)))
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, static_context: torch.Tensor | None = None) -> torch.Tensor:
        self.last_aux = {}
        h = self.main(x)
        bias_offset = None
        if self.bias_branch is not None and self.bias_branch_mode == "offset":
            bias_offset = self.bias_scale * self.bias_branch(x[:, :6])
            self.last_aux["bias_offset"] = bias_offset
        elif self.bias_branch is not None:
            h = h + self.bias_scale * self.bias_branch(x[:, :6])
        if self.drift_branch is not None:
            if static_context is None:
                drift = torch.zeros_like(h[:, :, :1])
            else:
                drift = self.drift_branch(static_context)
            h = h + self.drift_scale * drift.expand(-1, -1, h.shape[-1])
        pred = self.head(self.fusion(h))
        if bias_offset is not None:
            pred = pred + bias_offset.expand(-1, -1, pred.shape[-1])
        return pred
