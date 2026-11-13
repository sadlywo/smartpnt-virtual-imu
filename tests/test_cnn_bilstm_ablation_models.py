"""Contracts for the seeded, shared-initialization CNN-BiLSTM treatments."""

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
from torch import nn

from virtual_imu.models.matched import ALiMUCNNBiLSTM, ConvNormAct

VARIANTS = (
    "cnn_bilstm", "bilstm_only", "cnn_only", "mlp_only", "cnn_lstm",
    "cnn_lstm_wide", "pointwise_bilstm", "cnn_bilstm_dense",
)
REFERENCE_CONFIG = {
    "in_channels": 6, "out_channels": 6, "hidden_channels": 64,
    "lstm_hidden": 32, "conv_depth": 3, "lstm_layers": 1,
    "kernel_size": 5, "dropout": 0.0, "bidirectional": True,
}


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def ablation_module():
    path = Path(__file__).resolve().parents[1] / "src/virtual_imu/ablations/models.py"
    assert path.is_file(), "The seeded CNN-BiLSTM ablation factory has not been implemented"
    return importlib.import_module("virtual_imu.ablations.models")


def direct_reference(seed, **options):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ALiMUCNNBiLSTM(**{**REFERENCE_CONFIG, **options})


def state_digest(state):
    """Independently replay the documented name/dtype/shape/byte digest."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
        ).encode("utf-8")
        digest.update(header + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("seed", (20261010, 20261011, 20261012))
def test_baseline_is_the_exact_direct_reference_in_train_and_eval(seed):
    expected = direct_reference(seed)
    actual, metadata = ablation_module().build_ablation_model("cnn_bilstm", seed)
    assert type(actual) is ALiMUCNNBiLSTM
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for key, value in expected.state_dict().items():
        assert torch.equal(actual.state_dict()[key], value), key
    assert metadata["num_params"] == sum(p.numel() for p in expected.parameters())
    for training in (True, False):
        actual.train(training)
        expected.train(training)
        for length in (40, 80, 160, 320):
            x = torch.randn(2, 6, length, generator=torch.Generator().manual_seed(length))
            with torch.no_grad():
                assert torch.equal(actual(x), expected(x))


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_matching_reference_tensor_is_copied_and_hashes_are_replayable(variant):
    seed = 20261010
    reference = direct_reference(seed).state_dict()
    model, metadata = ablation_module().build_ablation_model(variant, seed)
    actual = model.state_dict()
    retained = sorted(
        key for key in actual if key in reference and actual[key].shape == reference[key].shape
    )
    assert retained, "Each treatment must retain shared reference tensors"
    assert metadata["retained_keys"] == retained
    assert metadata["copied_keys"] == retained
    assert metadata["shared_state_count"] == len(retained)
    assert metadata["shared_state_numel"] == sum(actual[key].numel() for key in retained)
    assert metadata["num_params"] == sum(p.numel() for p in model.parameters())
    assert metadata["reference_state_sha256"] == state_digest(reference)
    assert metadata["init_state_sha256"] == state_digest(actual)
    assert metadata["shared_state_sha256"] == state_digest({k: reference[k] for k in retained})
    for key in retained:
        assert torch.equal(actual[key], reference[key]), key
    # The receipt must be suitable for the run's JSON provenance records.
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("length", (40, 80, 160, 320))
def test_all_treatments_preserve_shape_and_have_finite_center_block_gradients(variant, length):
    model, _ = ablation_module().build_ablation_model(variant, 20261011)
    x = torch.randn(2, 6, length, generator=torch.Generator().manual_seed(6), requires_grad=True)
    prediction = model(x)
    assert prediction.shape == x.shape
    assert torch.isfinite(prediction).all()
    start = (length - 40) // 2
    prediction[:, :, start:start + 40].square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dense_control_is_identical_to_full_including_initialization_receipt():
    factory = ablation_module().build_ablation_model
    full, full_metadata = factory("cnn_bilstm", 20261012)
    dense, dense_metadata = factory("cnn_bilstm_dense", 20261012)
    assert type(full) is type(dense) is ALiMUCNNBiLSTM
    assert full.state_dict().keys() == dense.state_dict().keys()
    assert all(torch.equal(value, dense.state_dict()[key]) for key, value in full.state_dict().items())
    for key in (
        "num_params", "init_state_sha256", "shared_state_sha256", "reference_state_sha256",
        "shared_state_count", "shared_state_numel", "retained_keys", "copied_keys",
    ):
        assert full_metadata[key] == dense_metadata[key]
    x = torch.randn(2, 6, 320, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        assert torch.equal(full(x), dense(x))


@pytest.mark.parametrize("variant", VARIANTS)
def test_treatment_modules_are_removed_and_capacity_controls_have_the_right_direction(variant):
    model, metadata = ablation_module().build_ablation_model(variant, 20261010)
    recurrent = [m for m in model.modules() if isinstance(m, nn.LSTM)]
    convolutions = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    blocks = [m for m in model.modules() if isinstance(m, ConvNormAct)]
    norms = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if variant in {"bilstm_only", "mlp_only"}:
        assert not convolutions and not blocks and not norms
        projections = [
            m for m in model.modules()
            if isinstance(m, nn.Linear) and m.in_features == 6 and m.out_features == 64
        ]
        assert len(projections) == 1
        assert not any(key.startswith("feature_extractor.") for key in model.state_dict())
    else:
        kernel = (1,) if variant == "pointwise_bilstm" else (5,)
        assert len(convolutions) == len(blocks) == len(norms) == 3
        assert all(m.kernel_size == kernel for m in convolutions)
        assert convolutions[0].in_channels == 6
        assert all(m.out_channels == 64 for m in convolutions)
    if variant in {"cnn_only", "mlp_only"}:
        assert not recurrent
        assert not any(key.startswith("lstm.") for key in model.state_dict())
        assert model.head[0].normalized_shape == (64,)
        assert model.head[1].in_features == model.head[1].out_features == 64
    else:
        lstm, = recurrent
        assert lstm.input_size == 64 and lstm.num_layers == 1 and lstm.batch_first
        if variant in {"cnn_lstm", "cnn_lstm_wide"}:
            assert not lstm.bidirectional
            assert not any("reverse" in key for key in model.state_dict())
            assert lstm.hidden_size == (64 if variant == "cnn_lstm_wide" else 32)
        else:
            assert lstm.bidirectional and lstm.hidden_size == 32
    assert all(m.p == 0.0 for m in model.modules() if isinstance(m, nn.Dropout))
    if variant == "cnn_lstm_wide":
        reference = direct_reference(20261010)
        assert sum(p.numel() for p in model.lstm.parameters()) > sum(
            p.numel() for p in reference.lstm.parameters()
        )
        assert metadata["num_params"] > sum(p.numel() for p in reference.parameters())
        assert model.head[0].normalized_shape == (64,)


def test_mlp_output_at_other_times_cannot_depend_on_a_changed_sample():
    model, _ = ablation_module().build_ablation_model("mlp_only", 20261010)
    x = torch.randn(2, 6, 80, generator=torch.Generator().manual_seed(4))
    changed = x.clone()
    changed[:, :, 30] += torch.arange(6).view(1, 6) + 0.5
    with torch.no_grad():
        a, b = model(x), model(changed)
    keep = torch.arange(80) != 30
    assert torch.equal(a[:, :, keep], b[:, :, keep])
    assert not torch.equal(a[:, :, 30], b[:, :, 30])


def test_job_order_and_unrelated_random_draws_do_not_change_any_initial_state():
    factory = ablation_module().build_ablation_model
    first = {variant: factory(variant, 20261011)[1] for variant in VARIANTS}
    torch.rand(17)
    for variant in reversed(VARIANTS):
        _, metadata = factory(variant, 20261011)
        assert metadata == first[variant]
        _, another_seed = factory(variant, 20261012)
        assert metadata["init_state_sha256"] != another_seed["init_state_sha256"]
    projection_a = factory("bilstm_only", 20261011)[0].projection.weight
    projection_b = factory("bilstm_only", 20261012)[0].projection.weight
    assert not torch.equal(projection_a, projection_b)


@pytest.mark.parametrize("seed", (20261010, 20261011, 20261012))
def test_pointwise_projection_is_shared_between_bilstm_only_and_mlp_only(seed):
    factory = ablation_module().build_ablation_model
    recurrent, _ = factory("bilstm_only", seed)
    pointwise, _ = factory("mlp_only", seed)
    for key, tensor in recurrent.projection.state_dict().items():
        assert torch.equal(tensor, pointwise.projection.state_dict()[key]), key


def test_builds_preserve_torch_numpy_and_python_global_rng_states():
    factory = ablation_module().build_ablation_model
    cpu_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    for variant in VARIANTS:
        factory(variant, 20261010)
        assert torch.equal(torch.get_rng_state(), cpu_state)
        actual_numpy = np.random.get_state()
        assert numpy_state[0] == actual_numpy[0]
        np.testing.assert_array_equal(numpy_state[1], actual_numpy[1])
        assert numpy_state[2:] == actual_numpy[2:]
        assert random.getstate() == python_state


def test_factory_does_not_cache_or_alias_parameters_across_jobs():
    factory = ablation_module().build_ablation_model
    first, _ = factory("cnn_bilstm", 20261010)
    second, metadata = factory("cnn_only", 20261010)
    before = state_digest(second.state_dict())
    with torch.no_grad():
        for parameter in first.parameters():
            parameter.add_(1.0)
    assert state_digest(second.state_dict()) == before == metadata["init_state_sha256"]


def test_native_reference_config_overrides_are_applied_without_mutating_the_config():
    config = {
        "hidden_channels": 16, "lstm_hidden": 8, "conv_depth": 2,
        "lstm_layers": 2, "kernel_size": 3, "dropout": 0.1,
    }
    original = copy.deepcopy(config)
    expected = direct_reference(17, **config).eval()
    factory = ablation_module().build_ablation_model
    baseline, metadata = factory("cnn_bilstm", 17, config)
    assert config == original
    assert metadata["reference_state_sha256"] == state_digest(expected.state_dict())
    x = torch.randn(2, 6, 40, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(baseline.eval()(x), expected(x))
    for variant in VARIANTS:
        model, receipt = factory(variant, 17, config)
        assert model.eval()(x).shape == x.shape
        assert receipt["reference_state_sha256"] == metadata["reference_state_sha256"]
    assert config == original


def test_unknown_variant_is_rejected_without_consuming_rng():
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="variant"):
        ablation_module().build_ablation_model("not_an_ablation", 7)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("config", (
    {"kernel_size": 4}, {"conv_depth": 0}, {"lstm_layers": 0},
    {"hidden_channels": 0}, {"lstm_hidden": 0}, {"dropout": -0.1},
    {"dropout": 1.0}, {"in_channels": 7}, {"out_channels": 7},
    {"bidirectional": False}, {"misspelled_width": 64},
))
def test_invalid_reference_config_is_rejected_without_consuming_rng(config):
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="config"):
        ablation_module().build_ablation_model("cnn_bilstm", 7, config)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_builds_do_not_reseed_any_cuda_rng_stream():
    factory = ablation_module().build_ablation_model
    states = torch.cuda.get_rng_state_all()
    for variant in VARIANTS:
        model, _ = factory(variant, 20261010)
        assert all(p.device.type == "cpu" for p in model.parameters())
        assert all(torch.equal(a, b) for a, b in zip(states, torch.cuda.get_rng_state_all()))
