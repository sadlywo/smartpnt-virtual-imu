"""Front/back replacements of the seeded legacy CNN-LSTM64 regressor.

The wide legacy ablation is the initialization reference, including its
original construction order. Unchanged modules retain its names and tensors.
Only a replacement block is initialized from a SHA-derived CPU seed. Builds
return CPU models without advancing CPU, CUDA, NumPy or Python RNG streams.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from virtual_imu.ablations.models import (
    _reference_config,
    _state_sha256,
    build_ablation_model,
)

REPLACEMENT_VARIANTS = (
    "cnn_lstm64", "cnn_bilstm32", "cnn_identity", "cnn_gru", "cnn_bigru",
    "cnn_transformer", "cnn_tcn", "linear_lstm64", "mlp_lstm64",
    "gru_lstm64", "transformer_lstm64",
)
_CONTROLS = {
    "cnn_lstm64": "cnn_lstm_wide", "cnn_bilstm32": "cnn_bilstm",
    "cnn_identity": "cnn_only",
}
_FRONT = {"linear_lstm64", "mlp_lstm64", "gru_lstm64", "transformer_lstm64"}
_REPLACEMENT_DEFAULTS = {
    "gru_hidden": 72,
    "bigru_hidden": 40,
    "transformer_heads": 4,
    "backend_ff": 128,
    "frontend_ff": 192,
    "frontend_gru_hidden": 104,
    "frontend_mlp_hidden": 176,
    "tcn_hidden": 32,
    "tcn_dilations": [1, 2, 4, 8],
}


def _merge_options(defaults: dict, supplied: Mapping | None, label: str) -> dict:
    if supplied is not None and not isinstance(supplied, Mapping):
        raise ValueError(f"{label} must be a mapping")
    overrides = dict(supplied) if supplied is not None else {}
    unknown = overrides.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(map(str, unknown))}")
    return copy.deepcopy({**defaults, **overrides})


def _model_options(model_config: Mapping | None) -> dict:
    options = _merge_options(_reference_config(None), model_config, "model_config")
    dimensions = (
        "in_channels", "out_channels", "hidden_channels", "lstm_hidden",
        "conv_depth", "lstm_layers", "kernel_size",
    )
    if any(type(options[key]) is not int or options[key] < 1 for key in dimensions):
        raise ValueError("model_config dimensions must be positive integers")
    dropout = options["dropout"]
    if type(dropout) not in (int, float) or not 0 <= dropout < 1:
        raise ValueError("model_config dropout must be a finite number in [0, 1)")
    options = _reference_config(options)
    if options["hidden_channels"] != 2 * options["lstm_hidden"]:
        raise ValueError("model_config hidden_channels must equal twice lstm_hidden")
    return options


def _replacement_options(replacement_config: Mapping | None, width: int) -> dict:
    options = _merge_options(_REPLACEMENT_DEFAULTS, replacement_config, "replacement_config")
    for key in options.keys() - {"tcn_dilations"}:
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"replacement_config {key} must be a positive integer")
    if width % options["transformer_heads"]:
        raise ValueError("replacement_config transformer_heads must divide hidden_channels")
    dilations = options["tcn_dilations"]
    if (
        not isinstance(dilations, (list, tuple)) or not dilations
        or any(type(value) is not int or value < 1 for value in dilations)
    ):
        raise ValueError("replacement_config tcn_dilations must be nonempty positive integers")
    options["tcn_dilations"] = list(dilations)
    return options


class _SequenceFrontend(nn.Module):
    """Expose a token-wise/sequence module with the reference BCL interface."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)


class _LinearNormGELU(nn.Module):
    """Pointwise Linear followed by channel BN and GELU; input/output BCL."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x.transpose(1, 2)).transpose(1, 2)
        return self.activation(self.norm(projected))


class _GRUProjection(nn.Module):
    """One GRU plus a learned projection back to the common token width."""

    def __init__(
        self, in_channels: int, out_channels: int, hidden: int, *, bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            in_channels, hidden, batch_first=True, bidirectional=bidirectional,
            num_layers=1, dropout=0.0,
        )
        self.projection = nn.Linear(hidden * (2 if bidirectional else 1), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.gru(x)
        return self.projection(features)


class _SinusoidalTransformer(nn.Module):
    """Fixed sin/cos positions and one unmasked, postnorm GELU encoder layer."""

    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.width = width
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float64) * (-math.log(10000.0) / width)
        )
        self.register_buffer("frequencies", frequencies)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate positions for the actual length; there is no learned table
        # or arbitrary maximum length. The buffer follows model.to(device).
        positions = torch.arange(
            x.shape[1], device=x.device, dtype=self.frequencies.dtype,
        ).unsqueeze(1)
        angles = positions * self.frequencies
        encoding = torch.empty(
            x.shape[1], self.width, device=x.device, dtype=self.frequencies.dtype,
        )
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :self.width // 2].cos()
        return self.encoder(x + encoding.to(dtype=x.dtype))


class _TCNBottleneck(nn.Module):
    """One symmetric k3 dilated convolution per residual bottleneck."""

    def __init__(self, width: int, hidden: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv1d(hidden, width, 1),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The exact Conv1d filter tensors and initialization are retained. Use
        # token matrix products to avoid slow small dilated CUDA convolutions.
        # A k3 filter at dilation d reads [t-d, t, t+d], with zero padding.
        first, middle, last = self.net[0], self.net[2], self.net[4]
        hidden = self.net[1](F.linear(x, first.weight[:, :, 0], first.bias))
        dilation, length = middle.dilation[0], x.shape[1]
        padded = F.pad(hidden, (0, 0, dilation, dilation))
        taps = torch.cat([
            padded[:, :length], padded[:, dilation:dilation + length],
            padded[:, 2 * dilation:2 * dilation + length],
        ], dim=-1)
        weight = middle.weight.permute(0, 2, 1).reshape(middle.out_channels, -1)
        hidden = self.net[3](F.linear(taps, weight, middle.bias))
        residual = F.linear(hidden, last.weight[:, :, 0], last.bias)
        return self.norm(x + residual)


def _frontend(variant: str, width: int, options: dict) -> nn.Module:
    if variant == "linear_lstm64":
        return _SequenceFrontend(nn.Linear(6, width))
    if variant == "mlp_lstm64":
        hidden = options["frontend_mlp_hidden"]
        return nn.Sequential(
            _LinearNormGELU(6, hidden),
            _LinearNormGELU(hidden, hidden),
            _LinearNormGELU(hidden, width),
        )
    if variant == "gru_lstm64":
        return _SequenceFrontend(_GRUProjection(6, width, options["frontend_gru_hidden"]))
    return _SequenceFrontend(nn.Sequential(
        nn.Linear(6, width),
        _SinusoidalTransformer(width, options["transformer_heads"], options["frontend_ff"]),
    ))


def _backend(variant: str, width: int, options: dict) -> nn.Module:
    if variant == "cnn_gru":
        return _GRUProjection(width, width, options["gru_hidden"])
    if variant == "cnn_bigru":
        return _GRUProjection(width, width, options["bigru_hidden"], bidirectional=True)
    if variant == "cnn_transformer":
        return _SinusoidalTransformer(width, options["transformer_heads"], options["backend_ff"])
    return nn.Sequential(*(
        _TCNBottleneck(width, options["tcn_hidden"], dilation)
        for dilation in options["tcn_dilations"]
    ))


class _ReplacementRegressor(nn.Module):
    def __init__(self, reference: nn.Module, variant: str, width: int, options: dict) -> None:
        super().__init__()
        if variant in _FRONT:
            self.feature_extractor = _frontend(variant, width, options)
            self.lstm = copy.deepcopy(reference.lstm)
        else:
            self.feature_extractor = copy.deepcopy(reference.feature_extractor)
            self.backend = _backend(variant, width, options)
        self.head = copy.deepcopy(reference.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x).transpose(1, 2)
        if hasattr(self, "lstm"):
            features, _ = self.lstm(features)
        else:
            features = self.backend(features)
        return self.head(features).transpose(1, 2)


def _effective_context(variant: str, model_options: dict, replacement_options: dict) -> dict:
    cnn_field = 1 + model_options["conv_depth"] * (model_options["kernel_size"] - 1)
    frontend_field = 1 if variant in _FRONT else cnn_field
    future = (cnn_field - 1) // 2
    backend_field = "prefix"
    field = "prefix_plus_cnn_lookahead"
    if variant == "cnn_identity":
        backend_field, field = 1, cnn_field
    elif variant == "cnn_tcn":
        backend_field = 1 + 2 * sum(replacement_options["tcn_dilations"])
        field = cnn_field + backend_field - 1
        future = (field - 1) // 2
    elif variant in {"cnn_bilstm32", "cnn_bigru", "cnn_transformer", "transformer_lstm64"}:
        field, future = "full_window", "full_window"
        if variant == "transformer_lstm64":
            frontend_field = "full_window"
        else:
            backend_field = "full_window"
    elif variant in _FRONT:
        if variant == "gru_lstm64":
            frontend_field = "prefix"
        field, future = "prefix", 0
    has_bn = variant not in {"linear_lstm64", "gru_lstm64", "transformer_lstm64"}
    notes = (
        "Context describes eval-mode inference and is clipped to the supplied input window. "
        "Prefix means all available samples up to the current token. "
    )
    if has_bn:
        notes += (
            "BN uses fixed running statistics in eval mode; training BN mixes statistics "
            "across batch and time, so the inference receptive field does not describe training. "
        )
    if variant == "cnn_tcn":
        notes += "Symmetric CNN and TCN kernels give 21 past and 21 future samples at default settings."
    return {
        "frontend_receptive_field": frontend_field,
        "backend_receptive_field": backend_field,
        "inference_receptive_field": field,
        "future_samples": future,
        "notes": notes.strip(),
    }


def build_replacement_model(
    variant: str, seed: int, model_config: dict | None = None,
    replacement_config: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Build a CPU B6L regressor and a JSON initialization/context receipt.

    Default model options are the legacy CNN64/depth3/k5, BiLSTM32/depth1
    constructor options; the wide reference doubles the recurrent hidden
    width to 64 and makes it unidirectional. Controls call their native old
    factories, preserving every original tensor and state-dict key. Small
    model overrides retain six-channel IO and hidden_channels=2*lstm_hidden.
    Replacement overrides use the YAML ``replacement_models`` keys.

    For a new block, the first eight SHA256 bytes of UTF-8
    ``cnn-lstm-module-replacement:new:{seed}:{variant}`` are interpreted as a
    little-endian unsigned CPU seed. Only the forked default CPU generator
    is seeded; torch.manual_seed (which also seeds CUDA) is never called.

    All state digests use the legacy sorted-name/dtype/shape/raw-byte format.
    ``retained_keys``/``copied_keys`` cover exactly the unchanged wide modules,
    including BN buffers. Counts refer to these state tensors, not just
    parameters. Every call owns its tensors and configuration snapshots.
    """
    if not isinstance(variant, str) or variant not in REPLACEMENT_VARIANTS:
        raise ValueError(f"Unknown replacement variant: {variant!r}")
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError("seed must be an integer in the torch CPU seed range")
    model_options = _model_options(model_config)
    replacement_options = _replacement_options(
        replacement_config, model_options["hidden_channels"],
    )

    reference, _ = build_ablation_model("cnn_lstm_wide", seed, model_options)
    reference_state = reference.state_dict()
    replacement_seed = None
    if variant == "cnn_lstm64":
        model = reference
    elif variant in _CONTROLS:
        model, _ = build_ablation_model(_CONTROLS[variant], seed, model_options)
    else:
        seed_bytes = hashlib.sha256(
            f"cnn-lstm-module-replacement:new:{seed}:{variant}".encode()
        ).digest()
        replacement_seed = int.from_bytes(seed_bytes[:8], byteorder="little")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(replacement_seed)
            model = _ReplacementRegressor(
                reference, variant, model_options["hidden_channels"], replacement_options,
            )

    if variant == "cnn_lstm64":
        prefixes = ("feature_extractor.", "lstm.", "head.")
    elif variant in _FRONT:
        prefixes = ("lstm.", "head.")
    else:
        prefixes = ("feature_extractor.", "head.")
    retained_keys = sorted(key for key in reference_state if key.startswith(prefixes))
    model.load_state_dict({key: reference_state[key] for key in retained_keys}, strict=False)
    initial_state = model.state_dict()
    shared_state = {key: initial_state[key] for key in retained_keys}
    metadata = {
        "variant": variant,
        "seed": seed,
        "model_config": copy.deepcopy(model_options),
        "replacement_config": copy.deepcopy(replacement_options),
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "init_state_sha256": _state_sha256(initial_state),
        "reference_state_sha256": _state_sha256(reference_state),
        "shared_state_sha256": _state_sha256(shared_state),
        "retained_keys": retained_keys,
        "copied_keys": list(retained_keys),
        "shared_state_count": len(retained_keys),
        "shared_state_numel": sum(tensor.numel() for tensor in shared_state.values()),
        "family": "reference" if variant == "cnn_lstm64" else (
            "front" if variant in _FRONT else "back"
        ),
        "replacement_seed": replacement_seed,
        "effective_context": _effective_context(variant, model_options, replacement_options),
    }
    return model, metadata
