"""Module ablations of the frozen shared six-channel CNN-BiLSTM.

Every build first initializes the original reference class at the supplied seed.
Variants inherit every same-named, same-shaped parameter AND buffer. Newly
shaped blocks use a separate deterministic CPU RNG fork; construction leaves
the caller's CPU/CUDA, NumPy and Python RNG streams untouched. Returned models
are on CPU so the runner can choose a device after recording initialization.
"""

from __future__ import annotations

import copy
import hashlib
import json

import torch
from torch import nn

from virtual_imu.models.matched import ALiMUCNNBiLSTM, ConvNormAct

ABLATION_VARIANTS = (
    "cnn_bilstm", "bilstm_only", "cnn_only", "mlp_only", "cnn_lstm",
    "cnn_lstm_wide", "pointwise_bilstm", "cnn_bilstm_dense",
)

_REFERENCE_CONFIG = {
    "in_channels": 6, "out_channels": 6, "hidden_channels": 64,
    "lstm_hidden": 32, "conv_depth": 3, "lstm_layers": 1,
    "kernel_size": 5, "dropout": 0.0, "bidirectional": True,
}


def _reference_config(model_config: dict | None) -> dict:
    overrides = dict(model_config or {})
    unknown = overrides.keys() - _REFERENCE_CONFIG.keys()
    if unknown:
        raise ValueError(f"Unknown model_config keys: {sorted(unknown)}")
    options = {**_REFERENCE_CONFIG, **overrides}
    dimensions = ("hidden_channels", "lstm_hidden", "conv_depth", "lstm_layers", "kernel_size")
    if any(type(options[key]) is not int or options[key] < 1 for key in dimensions):
        raise ValueError("model_config widths, depths and kernel_size must be positive integers")
    if options["in_channels"] != 6 or options["out_channels"] != 6:
        raise ValueError("model_config must retain six input and output channels")
    if options["bidirectional"] is not True:
        raise ValueError("model_config must define a bidirectional full reference")
    if options["kernel_size"] % 2 != 1:
        raise ValueError("model_config kernel_size must be odd to preserve length")
    if not 0 <= options["dropout"] < 1:
        raise ValueError("model_config dropout must be in [0, 1)")
    return options


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash sorted names, JSON dtype/shape headers, NUL, and contiguous CPU bytes.

    Each header is compact UTF-8 JSON: [name, str(dtype), list(shape)]. Raw
    tensor bytes follow the NUL separator, without a torch.save container.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(header + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _AblationRegressor(nn.Module):
    """Keep reference module names while truly removing the ablated blocks."""

    def __init__(self, reference: ALiMUCNNBiLSTM, variant: str, options: dict) -> None:
        super().__init__()
        width = options["hidden_channels"]
        if variant in {"bilstm_only", "mlp_only"}:
            self.projection = nn.Linear(options["in_channels"], width)
        elif variant == "pointwise_bilstm":
            self.feature_extractor = nn.Sequential(*(
                ConvNormAct(
                    options["in_channels"] if index == 0 else width,
                    width, kernel_size=1, dropout=options["dropout"],
                )
                for index in range(options["conv_depth"])
            ))
        else:
            self.feature_extractor = copy.deepcopy(reference.feature_extractor)

        output_width = width
        if variant not in {"cnn_only", "mlp_only"}:
            hidden = options["lstm_hidden"]
            if variant in {"cnn_lstm", "cnn_lstm_wide"}:
                if variant == "cnn_lstm_wide":
                    hidden *= 2
                self.lstm = nn.LSTM(
                    input_size=width, hidden_size=hidden,
                    num_layers=options["lstm_layers"], batch_first=True,
                    dropout=options["dropout"] if options["lstm_layers"] > 1 else 0.0,
                    bidirectional=False,
                )
                output_width = hidden
            else:
                self.lstm = copy.deepcopy(reference.lstm)
                output_width = hidden * 2

        self.head = copy.deepcopy(reference.head)
        if output_width != reference.head[0].normalized_shape[0]:
            self.head[0] = nn.LayerNorm(output_width)
            self.head[1] = nn.Linear(output_width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "projection"):
            features = self.projection(x.transpose(1, 2))
        else:
            features = self.feature_extractor(x).transpose(1, 2)
        if hasattr(self, "lstm"):
            features, _ = self.lstm(features)
        return self.head(features).transpose(1, 2)


def build_ablation_model(
    variant: str, seed: int, model_config: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Return a seeded treatment and its JSON-serializable initialization receipt.

    model_config accepts the native ALiMUCNNBiLSTM constructor keys. Defaults
    are six inputs/outputs, CNN64/depth3/kernel5, BiLSTM32/depth1, dropout0.
    Six-channel IO and a bidirectional reference are fixed. Width/depth/kernel
    overrides can be used for smoke checks; the unidirectional wide control
    always doubles the reference's per-direction hidden width.
    bilstm_only and mlp_only share the fixed ``pointwise_projection`` seed
    tag, so their newly introduced projections also start identically.

    ``retained_keys`` identifies every same-name/shape reference state tensor;
    ``copied_keys`` records the tensors actually loaded (including BN buffers).
    ``shared_state_count`` counts tensors and ``shared_state_numel`` counts
    their elements. The shared digest covers exactly these copied tensors;
    the init/reference digests cover the complete respective state dicts.
    Dense and full have identical architectures/weights; their loss support
    is selected by the runner, outside this factory.
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    options = _reference_config(model_config)

    # Seed only the CPU generator: torch.manual_seed also reseeds CUDA streams.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        reference = ALiMUCNNBiLSTM(**options)
    reference_state = reference.state_dict()

    if variant in {"cnn_bilstm", "cnn_bilstm_dense"}:
        model = reference
    else:
        seed_tag = "pointwise_projection" if variant in {"bilstm_only", "mlp_only"} else variant
        seed_bytes = hashlib.sha256(f"cnn-bilstm-ablation:new:{seed}:{seed_tag}".encode()).digest()
        new_block_seed = int.from_bytes(seed_bytes[:8], byteorder="little")
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(new_block_seed)
            model = _AblationRegressor(reference, variant, options)

    initial_state = model.state_dict()
    retained_keys = sorted(
        key for key in initial_state
        if key in reference_state and initial_state[key].shape == reference_state[key].shape
    )
    shared_state = {key: reference_state[key] for key in retained_keys}
    model.load_state_dict(shared_state, strict=False)
    initial_state = model.state_dict()
    copied_state = {key: initial_state[key] for key in retained_keys}
    metadata = {
        "variant": variant,
        "seed": seed,
        "model_config": options,
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        "init_state_sha256": _state_sha256(initial_state),
        "reference_state_sha256": _state_sha256(reference_state),
        "shared_state_sha256": _state_sha256(copied_state),
        "shared_state_count": len(retained_keys),
        "shared_state_numel": sum(tensor.numel() for tensor in copied_state.values()),
        "retained_keys": retained_keys,
        "copied_keys": list(retained_keys),
    }
    return model, metadata
