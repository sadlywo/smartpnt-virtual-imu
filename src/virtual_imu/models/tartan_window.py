"""Navigation-only window adaptation of the TartanIMU ResNet-LSTM backbone.

Reference: Zhao et al., "Tartan IMU: A Light Foundation Model for Inertial
Positioning in Robotics", CVPR 2025. The public implementation inspected is
https://github.com/superxslam/TartanIMU at commit
8a8025000f85d2c6b0692b88e15c2c91e761f7d7, specifically
``tartan_imu/model/lstm/trunks.py``, ``tartan_imu/model/common/blocks.py`` and
``config/resnet_lstm_multihead.yaml``. The upstream source attributes Copyright
2026 Shibo Zhao and is licensed under Apache-2.0:
https://github.com/superxslam/TartanIMU/blob/8a8025000f85d2c6b0692b88e15c2c91e761f7d7/LICENSE

This independent architecture implementation uses a single IMU window as one
LSTM token. It predicts one three-dimensional navigation displacement instead
of the original multi-window, multi-platform velocity/covariance outputs. It
has no recurrent context across windows, uses PyTorch's default initialization,
and does not reproduce the original training, pretrained model, or full method.
The flattened feature size uses ceil(window_size / 16) so exactly 80 frames
work. The original formula adds an extra position for lengths divisible by 16.
"""

from __future__ import annotations

import torch
from torch import nn


class _ResidualBlock1D(nn.Module):
    """Two 3-tap convolutions with a projection when dimensions change."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.convs(x) + self.shortcut(x))


class TartanIMUWindowAdapter(nn.Module):
    """Map a six-axis IMU window to a single 3D navigation displacement.

    The output has shape ``(batch, 3, window_size)`` for the shared model
    interface. Every temporal position exposes the same window prediction;
    these positions are not independently estimated displacements or IMU data.
    This adapter is restricted to the navigation task.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        window_size: int = 80,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        head_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if in_channels != 6 or out_channels != 3:
            raise ValueError(
                "TartanIMUWindowAdapter is navigation-only (in_channels=6, out_channels=3)"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size

        self.input_block = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        groups: list[nn.Module] = []
        previous_channels = 64
        for channels, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            groups.append(
                nn.Sequential(
                    _ResidualBlock1D(previous_channels, channels, stride),
                    _ResidualBlock1D(channels, channels, 1),
                )
            )
            previous_channels = channels
        self.residual_groups = nn.Sequential(*groups)
        self.resnet_post_pro = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
        )

        encoded_length = (window_size + 15) // 16
        self.lstm = nn.LSTM(
            input_size=128 * encoded_length,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.window_size:
            raise ValueError(
                f"expected input shape (batch, {self.in_channels}, {self.window_size}), "
                f"got {tuple(x.shape)}"
            )
        features = self.resnet_post_pro(self.residual_groups(self.input_block(x)))
        tokens = features.flatten(start_dim=1).unsqueeze(1)
        encoded, _ = self.lstm(tokens)
        displacement = self.head(encoded[:, -1])
        return displacement.unsqueeze(-1).expand(-1, -1, self.window_size)
