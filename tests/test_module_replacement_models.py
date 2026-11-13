"""CPU contracts for the independently seeded front/back replacements."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from virtual_imu.ablations.models import build_ablation_model

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "cnn_lstm64", "cnn_bilstm32", "cnn_identity", "cnn_gru", "cnn_bigru",
    "cnn_transformer", "cnn_tcn", "linear_lstm64", "mlp_lstm64",
    "gru_lstm64", "transformer_lstm64",
)
CONTROLS = {
    "cnn_lstm64": "cnn_lstm_wide", "cnn_bilstm32": "cnn_bilstm",
    "cnn_identity": "cnn_only",
}
FRONT = {"linear_lstm64", "mlp_lstm64", "gru_lstm64", "transformer_lstm64"}
NEW = tuple(name for name in VARIANTS if name not in CONTROLS)
PARAMETERS = {
    "cnn_lstm64": 81414, "cnn_bilstm32": 73222, "cnn_identity": 48134,
    "cnn_gru": 82614, "cnn_bigru": 78758, "cnn_transformer": 81606,
    "cnn_tcn": 77830, "linear_lstm64": 38406, "mlp_lstm64": 82502,
    "gru_lstm64": 79622, "transformer_lstm64": 80134,
}


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(previous)


def factory():
    path = ROOT / "src/virtual_imu/replacements/models.py"
    assert path.is_file(), "The module replacement model factory has not been implemented"
    module = importlib.import_module("virtual_imu.replacements.models")
    assert callable(getattr(module, "build_replacement_model", None))
    return module.build_replacement_model


def state_digest(state):
    """Replay the existing name/dtype/shape/bytes provenance format independently."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(header + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def inputs(length=320, *, batch=2, seed=51, requires_grad=False):
    return torch.randn(
        batch, 6, length, generator=torch.Generator().manual_seed(seed),
        requires_grad=requires_grad,
    )


@pytest.mark.parametrize("seed", (20261010, 20261011, 20261012))
@pytest.mark.parametrize("variant,legacy", CONTROLS.items())
def test_controls_have_exact_legacy_cpu_states_and_train_eval_outputs(seed, variant, legacy):
    actual, metadata = factory()(variant, seed)
    expected, old_metadata = build_ablation_model(legacy, seed)
    assert type(actual) is type(expected)
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[name], tensor), name
    assert metadata["init_state_sha256"] == old_metadata["init_state_sha256"]
    for training in (True, False):
        actual.train(training)
        expected.train(training)
        for length in (41, 320):
            x = inputs(length)
            with torch.no_grad():
                assert torch.equal(actual(x), expected(x))


@pytest.mark.parametrize("variant", VARIANTS)
def test_retained_modules_are_exact_wide_copies_and_all_receipt_hashes_replay(variant):
    reference, reference_metadata = build_ablation_model("cnn_lstm_wide", 20261010)
    reference_state = reference.state_dict()
    model, metadata = factory()(variant, 20261010)
    actual = model.state_dict()
    if variant == "cnn_lstm64":
        kept = ("feature_extractor.", "lstm.", "head.")
    elif variant in FRONT:
        kept = ("lstm.", "head.")
    else:
        kept = ("feature_extractor.", "head.")
    retained = sorted(name for name in reference_state if name.startswith(kept))
    assert metadata["retained_keys"] == metadata["copied_keys"] == retained
    assert metadata["shared_state_count"] == len(retained)
    assert metadata["shared_state_numel"] == sum(reference_state[k].numel() for k in retained)
    for name in retained:
        assert torch.equal(actual[name], reference_state[name]), name
    assert metadata["reference_state_sha256"] == reference_metadata["init_state_sha256"]
    assert metadata["reference_state_sha256"] == state_digest(reference_state)
    assert metadata["init_state_sha256"] == state_digest(actual)
    assert metadata["shared_state_sha256"] == state_digest({k: actual[k] for k in retained})
    assert metadata["num_params"] == sum(p.numel() for p in model.parameters())
    assert metadata["variant"] == variant and metadata["seed"] == 20261010
    assert metadata["model_config"] == reference_metadata["model_config"]
    expected_family = "reference" if variant == "cnn_lstm64" else (
        "front" if variant in FRONT else "back"
    )
    assert metadata["family"] == expected_family
    assert metadata["effective_context"]["notes"]
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert all(t.device.type == "cpu" for t in actual.values())
    # Receipts are snapshots: updating running statistics cannot rewrite them.
    before = copy.deepcopy(metadata)
    with torch.no_grad():
        model(inputs(40))
    assert metadata == before


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("length", (40, 81, 320))
def test_every_variant_preserves_b6l_and_center_supervision_reaches_all_parameters(variant, length):
    model, _ = factory()(variant, 20261011)
    x = inputs(length, requires_grad=True)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    start = (length - 40) // 2
    y[:, :, start:start + 40].square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name, module in model.named_children():
        gradients = [p.grad for p in module.parameters()]
        if gradients:
            assert any(torch.count_nonzero(g) > 0 for g in gradients), name
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_single_sample_inference_retains_sequence_length(variant):
    model, _ = factory()(variant, 4)
    with torch.no_grad():
        y = model.eval()(inputs(1, batch=1))
    assert y.shape == (1, 6, 1) and torch.isfinite(y).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_default_parameter_counts_and_non_deletion_replacement_budgets(variant):
    model, metadata = factory()(variant, 20261010)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[variant]
    assert metadata["num_params"] == PARAMETERS[variant]
    if variant in NEW and variant != "linear_lstm64":
        assert abs(metadata["num_params"] / 81414 - 1) <= 0.05


@pytest.mark.parametrize("variant", ("linear_lstm64", "mlp_lstm64"))
def test_pointwise_frontends_cannot_mix_time_or_batch_during_inference(variant):
    model, metadata = factory()(variant, 20261012)
    frontend = model.eval().feature_extractor
    assert not any(isinstance(m, nn.Conv1d) for m in frontend.modules())
    x = inputs(81)
    changed = x.clone()
    changed[0, :, 40] += torch.arange(6) + 0.5
    with torch.no_grad():
        first, second = frontend(x), frontend(changed)
    assert first.shape == (2, 64, 81)
    keep = torch.arange(81) != 40
    assert torch.equal(first[0, :, keep], second[0, :, keep])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0, :, 40], second[0, :, 40])
    assert metadata["effective_context"]["future_samples"] == 0
    linears = [m for m in frontend.modules() if isinstance(m, nn.Linear)]
    norms = [m for m in frontend.modules() if isinstance(m, nn.BatchNorm1d)]
    if variant == "linear_lstm64":
        assert [(m.in_features, m.out_features) for m in linears] == [(6, 64)]
        assert not norms
    else:
        assert [(m.in_features, m.out_features) for m in linears] == [
            (6, 176), (176, 176), (176, 64),
        ]
        assert [m.num_features for m in norms] == [176, 176, 64]
        assert sum(isinstance(m, nn.GELU) for m in frontend.modules()) == 3


def test_tcn_has_four_single_dilated_bottlenecks_and_exact_43_sample_inference_support():
    model, metadata = factory()("cnn_tcn", 20261010)
    backend = model.backend
    convolutions = [m for m in backend.modules() if isinstance(m, nn.Conv1d)]
    dilated = [m for m in convolutions if m.kernel_size != (1,)]
    assert len(convolutions) == 12 and len(dilated) == 4
    assert [m.dilation[0] for m in dilated] == [1, 2, 4, 8]
    assert all(m.in_channels == m.out_channels == 32 and m.kernel_size == (3,) for m in dilated)
    pointwise = [m for m in convolutions if m.kernel_size == (1,)]
    assert [(m.in_channels, m.out_channels) for m in pointwise] == [(64, 32), (32, 64)] * 4
    norms = [m for m in backend.modules() if isinstance(m, nn.LayerNorm)]
    assert len(norms) == 4 and all(m.normalized_shape == (64,) for m in norms)
    assert not any(isinstance(m, nn.BatchNorm1d) for m in backend.modules())
    theoretical = 1 + 3 * (5 - 1) + sum((m.kernel_size[0] - 1) * m.dilation[0] for m in dilated)
    assert theoretical == 43
    assert metadata["effective_context"]["inference_receptive_field"] == theoretical
    assert metadata["effective_context"]["future_samples"] == 21
    # BN must use running statistics: training BN introduces global time/batch dependence.
    model.double().eval()
    x = inputs(97, batch=1).double().requires_grad_()
    model(x)[0, 0, 48].backward()
    support = torch.nonzero(x.grad.abs().sum(dim=(0, 1)) > 0).flatten().tolist()
    assert support == list(range(48 - 21, 48 + 22))


@pytest.mark.parametrize("variant,hidden,bidirectional", (
    ("cnn_gru", 72, False), ("cnn_bigru", 40, True), ("gru_lstm64", 104, False),
))
def test_gru_replacements_have_documented_width_direction_and_projection(variant, hidden, bidirectional):
    model, _ = factory()(variant, 20261010)
    block = model.feature_extractor if variant in FRONT else model.backend
    gru, = (m for m in block.modules() if isinstance(m, nn.GRU))
    assert gru.hidden_size == hidden and gru.bidirectional is bidirectional
    assert gru.batch_first and gru.num_layers == 1 and gru.dropout == 0
    assert gru.input_size == (6 if variant in FRONT else 64)
    projection, = (m for m in block.modules() if isinstance(m, nn.Linear))
    assert projection.in_features == hidden * (2 if bidirectional else 1)
    assert projection.out_features == 64
    if variant in FRONT:
        assert model.lstm.hidden_size == 64 and not model.lstm.bidirectional
    else:
        assert not any(isinstance(m, nn.LSTM) for m in model.modules())


@pytest.mark.parametrize("variant,ff", (("cnn_transformer", 128), ("transformer_lstm64", 192)))
def test_transformers_use_one_zero_dropout_layer_and_fixed_sinusoidal_positions(variant, ff):
    model, metadata = factory()(variant, 20261010)
    block = model.feature_extractor if variant in FRONT else model.backend
    layer, = (m for m in block.modules() if isinstance(m, nn.TransformerEncoderLayer))
    assert layer.self_attn.embed_dim == 64 and layer.self_attn.num_heads == 4
    assert layer.self_attn.batch_first and layer.self_attn.dropout == 0
    assert layer.linear1.out_features == ff and layer.linear2.in_features == ff
    assert all(m.p == 0 for m in block.modules() if isinstance(m, nn.Dropout))
    assert metadata["effective_context"]["future_samples"] == "full_window"
    tokens = []
    handle = layer.register_forward_pre_hook(lambda _module, args: tokens.append(args[0].detach()))
    block.eval()
    try:
        # Constant input produces identical pre-position features at each time.
        with torch.no_grad():
            if variant in FRONT:
                block(torch.zeros(1, 6, 321))
            else:
                block(torch.zeros(1, 321, 64))
    finally:
        handle.remove()
    observed = tokens[0][0] - tokens[0][0, :1]
    positions = np.arange(321, dtype=np.float64)[:, None]
    frequencies = np.exp(np.arange(0, 64, 2) * (-np.log(10000.0) / 64))
    expected = np.empty((321, 64))
    expected[:, 0::2] = np.sin(positions * frequencies)
    expected[:, 1::2] = np.cos(positions * frequencies)
    np.testing.assert_allclose(observed.numpy(), expected - expected[:1], atol=3e-5, rtol=1e-5)


def test_factory_preserves_cpu_numpy_python_rng_and_never_calls_global_reseed(monkeypatch):
    build = factory()
    cpu = torch.get_rng_state().clone()
    numpy = np.random.get_state()
    python = random.getstate()

    def forbidden(*_args, **_kwargs):
        pytest.fail("The factory must seed only its forked CPU default generator")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    for variant in VARIANTS:
        build(variant, 20261010)
        assert torch.equal(torch.get_rng_state(), cpu)
        actual_numpy = np.random.get_state()
        assert numpy[0] == actual_numpy[0]
        np.testing.assert_array_equal(numpy[1], actual_numpy[1])
        assert numpy[2:] == actual_numpy[2:]
        assert random.getstate() == python


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_builds_preserve_every_cuda_rng_stream_without_fitting_on_gpu():
    states = torch.cuda.get_rng_state_all()
    for variant in VARIANTS:
        model, _ = factory()(variant, 20261010)
        assert all(p.device.type == "cpu" for p in model.parameters())
        actual = torch.cuda.get_rng_state_all()
        assert len(actual) == len(states)
        assert all(torch.equal(actual[index], state) for index, state in enumerate(states))


def test_seeded_builds_are_order_independent_and_do_not_alias_state():
    build = factory()
    first = {variant: build(variant, 20261011)[1] for variant in VARIANTS}
    torch.rand(29)
    for variant in reversed(VARIANTS):
        model, metadata = build(variant, 20261011)
        assert metadata == first[variant]
        assert build(variant, 20261012)[1]["init_state_sha256"] != metadata["init_state_sha256"]
        pristine, receipt = build(variant, 20261011)
        with torch.no_grad():
            for tensor in model.parameters():
                tensor.add_(1)
        assert state_digest(pristine.state_dict()) == receipt["init_state_sha256"]


@pytest.mark.parametrize("variant", NEW)
def test_new_block_initialization_is_independent_of_reference_rng_draw_count(variant):
    build = factory()
    first, _ = build(variant, 20261010)
    second, _ = build(variant, 20261010, {"conv_depth": 1})
    name = "feature_extractor" if variant in FRONT else "backend"
    first_state = getattr(first, name).state_dict()
    second_state = getattr(second, name).state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(value, second_state[key]) for key, value in first_state.items())


def test_yaml_options_reproduce_defaults_and_metadata_does_not_alias_configs():
    with (ROOT / "configs/ablations/cnn_lstm_module_replacement.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert tuple(config["variants"]) == VARIANTS
    original = copy.deepcopy(config)
    for variant in VARIANTS:
        _, default = factory()(variant, 20261010)
        _, explicit = factory()(variant, 20261010, config["model"], config["replacement_models"])
        assert default == explicit
        assert explicit["replacement_config"] == config["replacement_models"]
        explicit["replacement_config"]["tcn_dilations"].append(99)
        explicit["model_config"]["conv_depth"] = 100
        assert config == original


def test_valid_small_configs_keep_wide_reference_copies_and_native_control_equivalence():
    options = {
        "hidden_channels": 16, "lstm_hidden": 8, "conv_depth": 2,
        "lstm_layers": 2, "kernel_size": 3, "dropout": 0.0,
    }
    replacement = {
        "gru_hidden": 7, "bigru_hidden": 5, "transformer_heads": 4,
        "backend_ff": 24, "frontend_ff": 40, "frontend_gru_hidden": 9,
        "frontend_mlp_hidden": 12, "tcn_hidden": 7, "tcn_dilations": [1, 3],
    }
    original = copy.deepcopy((options, replacement))
    reference, _ = build_ablation_model("cnn_lstm_wide", 19, options)
    for variant in VARIANTS:
        model, metadata = factory()(variant, 19, options, replacement)
        assert model.eval()(inputs(41)).shape == (2, 6, 41)
        assert metadata["reference_state_sha256"] == state_digest(reference.state_dict())
        if variant in CONTROLS:
            expected, _ = build_ablation_model(CONTROLS[variant], 19, options)
            assert state_digest(model.state_dict()) == state_digest(expected.state_dict())
            with torch.no_grad():
                assert torch.equal(model(inputs(41)), expected.eval()(inputs(41)))
    assert (options, replacement) == original


@pytest.mark.parametrize("variant", ("unknown", "cnn_lstm_wide", "", None, []))
def test_unknown_names_fail_clearly_without_consuming_rng(variant):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="variant"):
        factory()(variant, 7)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("config", (
    {"kernel_size": 4}, {"conv_depth": 0}, {"lstm_layers": 0},
    {"hidden_channels": 0}, {"lstm_hidden": 0}, {"dropout": -0.1},
    {"dropout": 1.0}, {"dropout": float("nan")}, {"dropout": "0"},
    {"in_channels": 7}, {"out_channels": 7}, {"bidirectional": False},
    {"conv_depth": True}, {"hidden_channels": 64.0}, {"hidden_channels": 32},
    {"misspelled_width": 64}, [], "bad config",
))
def test_invalid_model_configs_are_rejected_before_rng_changes(config):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="model_config"):
        factory()("cnn_lstm64", 7, model_config=config)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("config", (
    {"gru_hidden": 0}, {"bigru_hidden": -1}, {"frontend_gru_hidden": 3.5},
    {"frontend_mlp_hidden": True}, {"transformer_heads": 3},
    {"transformer_heads": 0}, {"backend_ff": 0}, {"frontend_ff": "192"},
    {"tcn_hidden": 0}, {"tcn_dilations": []}, {"tcn_dilations": [0, 1]},
    {"tcn_dilations": [1, True]}, {"tcn_dilations": [1, 2.5]},
    {"tcn_dilations": "1,2,4,8"}, {"dropout": 0.1}, {"typo": 4}, [], "bad config",
))
def test_invalid_replacement_configs_are_rejected_before_rng_changes(config):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="replacement_config"):
        factory()("cnn_gru", 7, replacement_config=config)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("seed", (True, 3.5, "7", None, -(2**63) - 1, 2**64))
def test_invalid_seeds_are_rejected_before_rng_changes(seed):
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="seed"):
        factory()("cnn_gru", seed)
    assert torch.equal(before, torch.get_rng_state())
