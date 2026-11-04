"""Advanced sequence mappers for low-to-high IMU supervised baselines."""

from __future__ import annotations

import torch
from torch import nn


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return ((kernel_size - 1) * dilation) // 2


class ConvNormAct(nn.Module):
    """Small Conv1d block that preserves sequence length for odd kernels."""

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=_same_padding(kernel_size, dilation),
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock(nn.Module):
    """Residual Conv1d block used by RoNIN-style sequence adapters."""

    def __init__(self, channels: int, *, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=_same_padding(kernel_size, dilation),
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class BrossardDilatedCNN(nn.Module):
    """Dilated CNN baseline inspired by inertial-learning denoising front ends."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        levels: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for level in range(levels):
            layers.append(
                ConvNormAct(
                    channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            channels = hidden_channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))


class CalibNetCNN(nn.Module):
    """Compact CNN calibration baseline for local low-to-high IMU correction."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        kernel_size: int = 7,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvNormAct(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout)]
        for _ in range(max(0, depth - 1)):
            layers.append(ConvNormAct(hidden_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))


class RoNINResNet(nn.Module):
    """RoNIN-inspired residual temporal CNN adapter for IMU signal mapping."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        blocks: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvNormAct(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout)]
        for block_idx in range(blocks):
            layers.append(
                ResidualConvBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2 ** (block_idx % 4),
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))


class ALiMUCNNBiLSTM(nn.Module):
    """ALiMU-inspired CNN + BiLSTM mapper for synchronized low/high IMU pairs."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        lstm_hidden: int = 64,
        conv_depth: int = 2,
        lstm_layers: int = 1,
        kernel_size: int = 5,
        dropout: float = 0.05,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = [
            ConvNormAct(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout)
        ]
        for _ in range(max(0, conv_depth - 1)):
            conv_layers.append(ConvNormAct(hidden_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout))
        self.feature_extractor = nn.Sequential(*conv_layers)
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out),
            nn.Linear(lstm_out, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x).transpose(1, 2)
        encoded, _ = self.lstm(features)
        return self.head(encoded).transpose(1, 2)


class SixDOFCNNBiLSTM(nn.Module):
    """CNN + two-layer BiLSTM backbone adapted from 6-DOF inertial odometry."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        lstm_hidden: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout),
            ConvNormAct(hidden_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x).transpose(1, 2)
        encoded, _ = self.lstm(features)
        return self.head(encoded).transpose(1, 2)


class FSDADenoiser(nn.Module):
    """FSDA-inspired embedding/reconstruction/generator denoising mapper."""

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        hidden_channels: int = 64,
        latent_channels: int = 48,
        kernel_size: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embedding = nn.Sequential(
            ConvNormAct(in_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout),
            ResidualConvBlock(hidden_channels, kernel_size=kernel_size, dilation=1, dropout=dropout),
            ResidualConvBlock(hidden_channels, kernel_size=kernel_size, dilation=2, dropout=dropout),
            nn.Conv1d(hidden_channels, latent_channels, kernel_size=1),
            nn.GELU(),
        )
        self.restructor = nn.Sequential(
            ConvNormAct(latent_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
        )
        self.generator = nn.Sequential(
            ConvNormAct(latent_channels, hidden_channels, kernel_size=kernel_size, dropout=dropout),
            ResidualConvBlock(hidden_channels, kernel_size=kernel_size, dilation=4, dropout=dropout),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )
        self.last_reconstruction: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.embedding(x)
        self.last_reconstruction = self.restructor(latent)
        return self.generator(latent)
