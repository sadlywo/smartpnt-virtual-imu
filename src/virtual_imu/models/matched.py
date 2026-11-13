"""Matched adapters predicting externally standardized HIGH-minus-LOW residuals.

Historical source directory (verbatim relevant classes/helpers only)::

    G:/IMU_Research_Results/smartpnt-virtual-imu/runs/
    remote_results_reviewer_hpo_20261024/mirror/server/core_project/src/virtual_imu/models/

Source file SHA256:
* advanced.py (ALiMUCNNBiLSTM, ConvNormAct, _same_padding):
  770dcc46009875cbe5663661de6363ab16a7a25dba0e53e9af359b911566b822
* time_series_attention.py (InformerIMU and its required helpers):
  fd7a59399da06dcbd4e38d5619302b0cb54d7adaaf325a0d9d01a689c66cb152

CNN mapping: d_model sets hidden_channels AND lstm_hidden; num_layers sets
conv_depth. Keep the historical kernel_size=5, lstm_layers=1, bidirectional=True.
Attention-only nhead/dim_feedforward and positional max_length do not change
that CNN topology. max_length maps to the Informer's fixed window_size.

Informer retains ProbSparse selection, frozen locally seeded key samples,
postnorm encoders, distillation and its dense LOW-input alignment head. As in
the historical adapter, this is a full-window IMU adapter, not the complete
forecasting architecture; the dense alignment head prevents claiming overall
O(L log L) complexity. No unrelated forecasting models are ported.

The strict no-diffusion model instantiates the original DiffusionTransformer,
including its entire input projection and time MLP, with no prior RNG draws.
Only its execution changes: zero noisy channels, zero timestep, unchanged LOW
condition. Transfer weights with ablation.denoiser.load_state_dict(ddpm.denoiser.state_dict()).
All direct losses are MSE on already normalized residuals; normalization and
switching to eval() for deterministic inference remain the runner's responsibility.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from virtual_imu.models.ddpm import DiffusionTransformer, build_ddpm
from virtual_imu.models.transformer import PositionalEncoding


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


class _DirectResidual(nn.Module):
    """Shared direct MSE and single-pass sampling API; no diffusion RNG."""

    def training_loss(
        self, clean: torch.Tensor, condition: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return (self(condition).float() - clean.float()).square().mean()

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self(condition)


class _ResidualRegressor(_DirectResidual):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.network(condition)


class _NoDiffusion(_DirectResidual):
    def __init__(self, model: dict) -> None:
        super().__init__()
        self.denoiser = DiffusionTransformer(**model)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        noisy = condition.new_zeros((len(condition), 6, condition.shape[-1]))
        time = torch.zeros(len(condition), dtype=torch.long, device=condition.device)
        return self.denoiser(noisy, time, condition)


class _DirectTransformer(nn.Module):
    """LOW-only projection, sinusoidal positions, GELU pre-norm encoder, head."""

    def __init__(
        self, condition_channels: int, d_model: int, nhead: int, num_layers: int,
        dim_feedforward: int, dropout: float, max_length: int,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.input = nn.Linear(condition_channels, d_model)
        self.position = PositionalEncoding(d_model, max_len=max_length)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[-1] > self.max_length:
            raise ValueError("Window exceeds configured max_length")
        tokens = self.position(self.input(condition.transpose(1, 2)))
        return self.head(self.encoder(tokens)).transpose(1, 2)


def build_matched_model(config: dict) -> nn.Module:
    """Build a matched model from config['method'], ['model'], ['diffusion'].

    Common model defaults: condition_channels=7, d_model=64, nhead=4,
    num_layers=2, dim_feedforward=128, dropout=0.1, max_length=80.
    Informer additionally accepts factor, distil, sampling_seed and
    ff_multiplier (used only when dim_feedforward is absent). Its frozen
    sampling seed defaults to config['seed'], or zero if no seed is supplied.
    CNN depth/width mapping is documented in the module provenance above.
    The input config is never mutated.
    """
    method = config["method"]
    if method not in {"ddpm", "cnn_bilstm", "informer", "transformer", "no_diffusion"}:
        raise ValueError(f"Unknown matched model method: {method}")
    options = dict(config.get("model", {}))
    model = {
        "condition_channels": 7, "d_model": 64, "nhead": 4, "num_layers": 2,
        "dim_feedforward": 128, "dropout": 0.1, "max_length": 80,
    }
    for key in model:
        if key in options:
            model[key] = options[key]
    if method == "informer" and "dim_feedforward" not in options:
        model["dim_feedforward"] = model["d_model"] * options.get("ff_multiplier", 2)
    _validate_configuration(
        model["condition_channels"], 6, model["max_length"], model["d_model"],
        model["nhead"], model["num_layers"], model["dim_feedforward"], model["dropout"],
    )
    if method == "ddpm":
        return build_ddpm({**config, "model": model, "diffusion": config.get("diffusion", {})})
    if method == "no_diffusion":
        return _NoDiffusion(model)
    if method == "transformer":
        return _ResidualRegressor(_DirectTransformer(**model))
    if method == "cnn_bilstm":
        network = ALiMUCNNBiLSTM(
            in_channels=model["condition_channels"], out_channels=6,
            hidden_channels=model["d_model"], lstm_hidden=model["d_model"],
            conv_depth=model["num_layers"], lstm_layers=1, kernel_size=5,
            dropout=model["dropout"], bidirectional=True,
        )
    else:
        network = InformerIMU(
            in_channels=model["condition_channels"], out_channels=6,
            window_size=model["max_length"], d_model=model["d_model"],
            nhead=model["nhead"], num_layers=model["num_layers"],
            dim_feedforward=model["dim_feedforward"], dropout=model["dropout"],
            factor=options.get("factor", 5), distil=options.get("distil", True),
            sampling_seed=options.get("sampling_seed", config.get("seed", 0)),
        )
    return _ResidualRegressor(network)
