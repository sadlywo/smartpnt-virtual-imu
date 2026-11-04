"""Independently implemented published backbones with common-task output adapters.

These classes reproduce the specified convolutional backbones, not the complete
published methods. They consume already normalized channel-first windows and
return channel-first sequences for the repository's residual/last-head protocol.
No upstream source code, pretrained weights, or training pipelines are imported.
"""

from __future__ import annotations

import torch
from torch import nn

DEEPGYRO_SOURCE_COMMIT = "9948b6646882c547d5ba6eda19a95ea4cba5e89e"
DEEPGYRO_SOURCE_URL = (
    "https://github.com/mbrossar/denoise-imu-gyro/blob/"
    f"{DEEPGYRO_SOURCE_COMMIT}/src/networks.py"
)
TLIO_SOURCE_COMMIT = "d7051c4ff14834b93931984b75b40f894e39654d"
TLIO_SOURCE_URL = (
    "https://github.com/CathIAS/TLIO/blob/"
    f"{TLIO_SOURCE_COMMIT}/src/network/model_resnet.py"
)


class DeepGyroBackboneAdapter(nn.Module):
    """DeepGyro's fixed dilated CNN backbone with a common-task channel head.

    Specification: official ``BaseNet`` at commit
    9948b6646882c547d5ba6eda19a95ea4cba5e89e (``DEEPGYRO_SOURCE_URL``).
    Widths are 16/32/64/128, kernels 7/7/7/7/1 and dilations 1/4/16/64/1.
    Each of the first four convolutions has BatchNorm(momentum=0.1), GELU
    and dropout(0.1 by default). The receptive field is 511 samples.

    The original gyro method uses a 6-to-3 network, its own normalization,
    gyro scaling, a learned 3x3 gyro calibration and orientation training loss.
    Those wrappers and losses are absent here: input/output channels and loss
    follow the common benchmark. Causality applies to evaluation with frozen
    BatchNorm statistics; training BatchNorm uses batch/time statistics.
    """

    receptive_field = 511
    left_padding = 510

    def __init__(
        self, in_channels: int = 6, out_channels: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_channels = in_channels
        for channels, dilation in zip((16, 32, 64, 128), (1, 4, 16, 64), strict=True):
            layers.extend(
                [
                    nn.Conv1d(previous_channels, channels, kernel_size=7, dilation=dilation),
                    nn.BatchNorm1d(channels, momentum=0.1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_channels = channels
        layers.append(nn.Conv1d(previous_channels, out_channels, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] == 0:
            raise ValueError("Expected a nonempty (batch, channels, time) window")
        # ReplicationPad1d CUDA backward rejects strict determinism in Torch2.4.
        # Expansion reduces gradients to the first sample without a pad kernel.
        prefix = x[:, :, :1].expand(-1, -1, self.left_padding)
        return self.network(torch.cat((prefix, x), dim=-1))


class _ResidualBlock1D(nn.Module):
    """Two 3-tap convolutions and a projected skip when resolution changes."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.skip: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.activation(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(hidden)) + residual)


class ResNet18SequenceAdapter(nn.Module):
    """TLIO ResNet-18 backbone with a dense sequence output adapter.

    Specification: TLIO ``ResNet1D(BasicBlock1D, ..., [2,2,2,2], ...)`` at
    commit d7051c4ff14834b93931984b75b40f894e39654d (``TLIO_SOURCE_URL`` and
    the adjacent ``model_factory.py``). A 7-tap/stride-2 64-channel stem and
    3-tap/stride-2 max-pool precede four two-block stages. Stage widths are
    64/128/256/512 and first-block strides are 1/2/2/2. Backbone convolutions
    use Kaiming normal initialization, fan_out, ReLU, as in the source.

    The original dual fully connected displacement/uncertainty heads are
    replaced with nearest temporal expansion and a 1x1 channel head, so the
    current trainer can use dense residuals or the final output sample. This
    is a backbone adapter: TLIO's uncertainty loss, heading/frame processing
    and stochastic-cloning filter are not reproduced. Centered convolutions
    and pooling make the per-step sequence output noncausal.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        previous_channels = 64
        for channels, stride in zip((64, 128, 256, 512), (1, 2, 2, 2), strict=True):
            stages.append(
                nn.Sequential(
                    _ResidualBlock1D(previous_channels, channels, stride),
                    _ResidualBlock1D(channels, channels, 1),
                )
            )
            previous_channels = channels
        self.stages = nn.Sequential(*stages)
        for module in (*self.stem.modules(), *self.stages.modules()):
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.head = nn.Conv1d(512, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] == 0:
            raise ValueError("Expected a nonempty (batch, channels, time) window")
        length = x.shape[-1]
        features = self.stages(self.stem(x))
        # This is the floor-index convention of nearest interpolation. Torch2.4
        # provides deterministic CUDA index_select backward when strict=True.
        indices = torch.div(
            torch.arange(length, device=x.device) * features.shape[-1],
            length,
            rounding_mode="floor",
        )
        return self.head(features.index_select(-1, indices))
