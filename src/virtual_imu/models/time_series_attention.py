"""Independent, full-window time-series encoder adapters for IMU tasks.

These are common-task adapters, not complete reproductions of forecasting
methods or claims of state of the art in IMU calibration. Inputs and outputs
are externally normalized (B, channels, time) tensors. No target tokens,
calendar features, pretrained weights, or forecasting losses are used.

Primary specifications (audited 2026-10-24):
* Informer, AAAI 2021: https://arxiv.org/abs/2012.07436
  https://github.com/zhouhaoyi/Informer2020/tree/
  29f2a739226a509202a092b464163da81fa74960 (Apache-2.0).
* PatchTST, ICLR 2023: https://arxiv.org/abs/2211.14730
  https://github.com/yuqinie98/PatchTST/tree/
  204c21efe0b39603ad6e2ca640ef5896646ab1a9 (Apache-2.0).
* iTransformer, ICLR 2024: https://arxiv.org/abs/2310.06625
  https://github.com/thuml/iTransformer/tree/
  c2426e68ca13f74aaec08045c5c724d8ad328124 (MIT).

The code is independently implemented from the documented topology. Output
alignment/channel mixing replace the original forecast heads. All models
are noncausal within their input window, including Informer query selection.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from virtual_imu.models.transformer import PositionalEncoding


def _validate_configuration(
    in_channels: int,
    out_channels: int,
    window_size: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> None:
    if min(in_channels, out_channels, window_size, d_model, nhead, num_layers, dim_feedforward) < 1:
        raise ValueError(
            "Channels, window size, model dimensions, heads and layers must be positive"
        )
    if d_model % nhead:
        raise ValueError("d_model must be divisible by nhead")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def _validate_input(x: torch.Tensor, in_channels: int, window_size: int) -> None:
    if x.ndim != 3 or x.shape[1:] != (in_channels, window_size):
        raise ValueError(f"Expected (batch, {in_channels}, {window_size}) input")


class _CircularConv3(nn.Module):
    """Three-tap circular convolution using deterministic index selection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(-1, x.shape[-1] + 1, device=x.device).remainder(x.shape[-1])
        return self.conv(x.index_select(-1, indices))


class ProbSparseAttention(nn.Module):
    """Informer unmasked attention: exact attention for O(log L) queries.

    Key samples per query and active queries are capped at
    ``factor * ceil(log L)``. As in official ``models/attn.py``, sparsity is
    sampled max minus sampled sum divided by the FULL key length, and
    unselected queries receive mean(V). The original sparse implementation
    does not drop attention probabilities; encoder residual/FF dropout is
    separate. Unlike the original per-forward RNG calls, samples are drawn
    once with a local CPU generator and persisted in the state dict. This
    deliberate adaptation guarantees evaluation repeatability and avoids
    inference-batch/call-order dependent query selection.
    """

    def __init__(self, sequence_length: int, factor: int = 5, sampling_seed: int = 0) -> None:
        super().__init__()
        if sequence_length < 1 or factor < 1:
            raise ValueError("sequence_length and factor must be positive")
        self.sequence_length = sequence_length
        self.sample_count = min(
            sequence_length, factor * max(1, math.ceil(math.log(sequence_length)))
        )
        self.query_count = self.sample_count
        generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
        samples = torch.randint(
            sequence_length, (sequence_length, self.sample_count), generator=generator
        )
        self.register_buffer("sample_indices", samples)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Attend projected (B, heads, L, head_dimension) self-attention inputs."""
        if (
            q.ndim != 4
            or q.shape != k.shape
            or q.shape != v.shape
            or q.shape[-2] != self.sequence_length
        ):
            raise ValueError(
                "ProbSparseAttention requires equally shaped fixed-length self-attention inputs"
            )
        sampled_keys = k[:, :, self.sample_indices, :]
        sampled_scores = (q.unsqueeze(-2) * sampled_keys).sum(dim=-1)
        sparsity = sampled_scores.amax(dim=-1) - sampled_scores.sum(dim=-1) / k.shape[-2]
        selected = sparsity.topk(self.query_count, dim=-1, sorted=False).indices
        selected_q = q.gather(-2, selected.unsqueeze(-1).expand(*selected.shape, q.shape[-1]))
        logits = selected_q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        exact_context = logits.softmax(dim=-1) @ v
        initial = v.mean(dim=-2, keepdim=True)
        # A one-hot routing matrix avoids scatter CUDA backward kernels;
        # each query is unique in topk. Memory remains O(L log L).
        routing = F.one_hot(selected, num_classes=q.shape[-2]).to(q.dtype).transpose(-1, -2)
        return initial + routing @ (exact_context - initial)


class _FullAttention(nn.Module):
    """Explicit matmul attention, avoiding fused-kernel determinism differences."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch, length, width = query.shape
        q = self.q(query).view(batch, length, self.nhead, -1).transpose(1, 2)
        k = self.k(memory).view(batch, memory.shape[1], self.nhead, -1).transpose(1, 2)
        v = self.v(memory).view(batch, memory.shape[1], self.nhead, -1).transpose(1, 2)
        weights = self.dropout((q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(dim=-1))
        result = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.output(result)


class _PostNormEncoderLayer(nn.Module):
    """Informer/iTransformer postnorm encoder; sparse or full attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        sparse_attention: ProbSparseAttention | None = None,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.sparse_attention = sparse_attention
        if sparse_attention is None:
            self.attention = _FullAttention(d_model, nhead, dropout)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.attention_output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse_attention is None:
            attention = self.attention(x, x)
        else:
            batch, length, width = x.shape
            q, k, v = (
                self.qkv(x).view(batch, length, 3, self.nhead, -1).permute(2, 0, 3, 1, 4).unbind(0)
            )
            context = self.sparse_attention(q, k, v).transpose(1, 2).reshape(batch, length, width)
            attention = self.attention_output(context)
        x = self.norm1(x + self.dropout(attention))
        return self.norm2(x + self.dropout(self.ff(x)))


class _Distill(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _CircularConv3(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


class _AlignedHead(nn.Module):
    """Input tokens query distilled memory; no target-derived decoder tokens."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.attention = _FullAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = self.norm1(query + self.dropout(self.attention(query, memory)))
        return self.norm2(x + self.dropout(self.ff(x)))


class InformerIMU(nn.Module):
    """Informer ProbSparse/distillation encoder with a low-input alignment head.

    Circular token convolution + sinusoidal positions replace value/calendar
    embedding. Distillation follows the official Conv3/BN/ELU/MaxPool3
    topology between encoder layers. Low-input tokens query final memory to
    retain L dense aligned outputs. This replaces the published masked
    forecasting decoder, so its complete O(L log L) efficiency claim does
    not apply to this full-attention alignment head.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 6,
        window_size: int = 80,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        factor: int = 5,
        distil: bool = True,
        sampling_seed: int = 0,
    ) -> None:
        super().__init__()
        _validate_configuration(
            in_channels,
            out_channels,
            window_size,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )
        self.in_channels, self.window_size = in_channels, window_size
        self.token_embedding = _CircularConv3(in_channels, d_model)
        self.position = PositionalEncoding(d_model, max_len=window_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.distillation = nn.ModuleList()
        length = window_size
        for index in range(num_layers):
            sparse = ProbSparseAttention(length, factor=factor, sampling_seed=sampling_seed + index)
            self.encoder.append(
                _PostNormEncoderLayer(d_model, nhead, dim_feedforward, dropout, sparse)
            )
            if distil and index < num_layers - 1:
                self.distillation.append(_Distill(d_model))
                length = (length + 1) // 2
        self.final_norm = nn.LayerNorm(d_model)
        self.alignment = _AlignedHead(d_model, nhead, dim_feedforward, dropout)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.in_channels, self.window_size)
        query = self.dropout(self.position(self.token_embedding(x).transpose(1, 2)))
        memory = query
        for index, layer in enumerate(self.encoder):
            memory = layer(memory)
            if index < len(self.distillation):
                memory = self.distillation[index](memory)
        return self.head(self.alignment(query, self.final_norm(memory))).transpose(1, 2)


class _PatchEncoderLayer(nn.Module):
    """PatchTST shared-channel postnorm BN block with residual attention logits."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attention_output = nn.Linear(d_model, d_model)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self, x: torch.Tensor, previous_scores: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x).view(batch, length, 3, self.nhead, -1).permute(2, 0, 3, 1, 4).unbind(0)
        )
        scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        if previous_scores is not None:
            scores = scores + previous_scores
        # Official supervised PatchTST defaults to zero attention dropout;
        # model dropout still applies to projection/residual/FF branches.
        context = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = self.norm1(
            (x + self.dropout(self.attention_output(context))).transpose(1, 2)
        ).transpose(1, 2)
        x = self.norm2((x + self.dropout(self.ff(x))).transpose(1, 2)).transpose(1, 2)
        return x, scores


class PatchTSTIMU(nn.Module):
    """Channel-independent patch encoder plus learned dense/channel output heads.

    Patches share a Linear(patch_len,d_model), learned position embeddings,
    and encoder weights across input channels. End replication adds one
    stride, as in the supervised official backbone. Flattened patch tokens
    map linearly to every output time, followed by pointwise channel mixing.
    This replaces the same-channel future forecast head. RevIN is omitted:
    constant reference indicators must survive and input/output physical
    residual scales differ. External training-split normalization is used.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 6,
        window_size: int = 80,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        patch_len: int = 16,
        stride: int = 8,
    ) -> None:
        super().__init__()
        _validate_configuration(
            in_channels,
            out_channels,
            window_size,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )
        if patch_len < 1 or stride < 1 or patch_len > window_size or stride > patch_len:
            raise ValueError("Require 1 <= stride <= patch_len <= window_size")
        self.in_channels, self.window_size = in_channels, window_size
        self.patch_len, self.stride = patch_len, stride
        self.patch_count = (window_size - patch_len) // stride + 2
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(self.patch_count, d_model).uniform_(-0.02, 0.02))
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList(
            [
                _PatchEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )
        self.temporal_head = nn.Linear(d_model * self.patch_count, window_size)
        self.channel_mixer = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.in_channels, self.window_size)
        # Expanded boundary concatenation has deterministic CUDA backward.
        padded = torch.cat((x, x[:, :, -1:].expand(-1, -1, self.stride)), dim=-1)
        patches = padded.unfold(-1, self.patch_len, self.stride)
        embedded = self.patch_embedding(patches)
        tokens = self.dropout(
            embedded.reshape(-1, self.patch_count, embedded.shape[-1]) + self.position
        )
        scores = None
        for layer in self.encoder:
            tokens, scores = layer(tokens, scores)
        # Match the official flatten order: model dimension then patch index.
        flat = tokens.transpose(1, 2).reshape(x.shape[0], self.in_channels, -1)
        dense = self.dropout(self.temporal_head(flat))
        return self.channel_mixer(dense)


class ITransformerIMU(nn.Module):
    """Actual variate-token attention with a common-task dense/channel head.

    Each entire input time series is projected Linear(window_size,d_model)
    into a variate token; attention runs across input channels, not time.
    Postnorm encoder layers and final LayerNorm precede Linear(d_model,L)
    per input channel and a pointwise channel mixer. Internal nonstationary
    normalization and input-statistic inverse normalization are deliberately
    omitted: input channels and residual/displacement outputs do not share
    the original forecaster's same-variate units. Use external train-only
    normalization. No positional encoding is added to variate tokens.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 6,
        window_size: int = 80,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_configuration(
            in_channels,
            out_channels,
            window_size,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )
        self.in_channels, self.window_size = in_channels, window_size
        self.variate_embedding = nn.Linear(window_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            *[
                _PostNormEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ],
            nn.LayerNorm(d_model),
        )
        self.temporal_head = nn.Linear(d_model, window_size)
        self.channel_mixer = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.in_channels, self.window_size)
        tokens = self.encoder(self.dropout(self.variate_embedding(x)))
        return self.channel_mixer(self.temporal_head(tokens))
