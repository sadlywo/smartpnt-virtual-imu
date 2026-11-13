"""Contracts for the matched normalized-residual model adapters."""

from __future__ import annotations

import copy
import importlib
import importlib.util

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from virtual_imu.models.ddpm import ConditionalDDPM, DiffusionTransformer, build_ddpm

DIRECT_METHODS = ("cnn_bilstm", "informer", "transformer", "no_diffusion")


@pytest.fixture(scope="module", autouse=True)
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def matched_module():
    # Fail with an explicit missing-feature assertion before the module exists.
    assert importlib.util.find_spec("virtual_imu.models.matched") is not None, (
        "The matched model adapter has not been implemented"
    )
    return importlib.import_module("virtual_imu.models.matched")


def config(method, **model_options):
    return {
        "method": method,
        "seed": 20261106,
        "model": {
            "condition_channels": 7,
            "d_model": 16,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 40,
            "dropout": 0.0,
            "max_length": 80,
            **model_options,
        },
        "diffusion": {"timesteps": 8, "prediction_type": "x0", "schedule": "cosine"},
    }


@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_direct_models_predict_residuals_and_backpropagate(method):
    matched = matched_module()
    torch.manual_seed(42)
    model = matched.build_matched_model(config(method))
    condition = torch.randn(2, 7, 80)
    clean = torch.randn(2, 6, 80)
    assert isinstance(model, nn.Module)
    prediction = model(condition)
    assert prediction.shape == clean.shape
    assert torch.isfinite(prediction).all()
    loss = model.training_loss(clean, condition)
    torch.testing.assert_close(loss, F.mse_loss(prediction.float(), clean.float()))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_direct_eval_ignores_generators_without_consuming_rng(method):
    matched = matched_module()
    torch.manual_seed(44)
    model = matched.build_matched_model(config(method, dropout=0.2)).eval()
    condition = torch.randn(2, 7, 80)
    first = torch.Generator().manual_seed(7)
    second = torch.Generator().manual_seed(1234)
    states = [g.get_state().clone() for g in (first, second)]
    global_state = torch.get_rng_state().clone()
    a = model.sample(condition, generator=first)
    b = model.sample(condition, generator=second)
    c = model.sample(condition)
    assert not a.requires_grad
    assert torch.equal(a, b) and torch.equal(a, c)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert torch.equal(first.get_state(), states[0])
    assert torch.equal(second.get_state(), states[1])


def test_no_diffusion_keeps_exact_ddpm_initialization_and_accepts_weights():
    matched = matched_module()
    torch.manual_seed(20261106)
    ddpm = matched.build_matched_model(config("ddpm"))
    torch.manual_seed(20261106)
    ablation = matched.build_matched_model(config("no_diffusion"))
    assert type(ablation.denoiser) is DiffusionTransformer
    assert sum(p.numel() for p in ddpm.parameters()) == sum(p.numel() for p in ablation.parameters())
    assert list(dict(ddpm.named_parameters())) == list(dict(ablation.named_parameters()))
    expected = ddpm.denoiser.state_dict()
    actual = ablation.denoiser.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    # Check real strict loading after a weight change, rather than only shapes.
    with torch.no_grad():
        ddpm.denoiser.input.weight.add_(0.5)
    ablation.denoiser.load_state_dict(ddpm.denoiser.state_dict(), strict=True)
    assert all(torch.equal(value, ablation.denoiser.state_dict()[key]) for key, value in ddpm.denoiser.state_dict().items())


def test_no_diffusion_calls_denoiser_once_with_zero_noise_time_and_original_condition():
    matched = matched_module()
    model = matched.build_matched_model(config("no_diffusion")).eval()
    condition = torch.randn(2, 7, 80)
    calls = []
    handle = model.denoiser.register_forward_pre_hook(lambda _module, args: calls.append(args))
    prediction = model.sample(condition, generator=torch.Generator().manual_seed(999))
    handle.remove()
    assert len(calls) == 1
    noisy, time, supplied_condition = calls[0]
    assert noisy.shape == (2, 6, 80) and torch.count_nonzero(noisy) == 0
    assert noisy.dtype == condition.dtype and noisy.device == condition.device
    assert time.shape == (2,) and time.dtype == torch.long and torch.count_nonzero(time) == 0
    assert supplied_condition is condition
    with torch.no_grad():
        expected = model.denoiser(condition.new_zeros(2, 6, 80), torch.zeros(2, dtype=torch.long), condition)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


def test_ddpm_factory_delegates_all_model_and_diffusion_settings():
    matched = matched_module()
    cfg = config("ddpm", num_layers=1)
    cfg["diffusion"].update(prediction_type="epsilon", clip_x0=2.5)
    untouched = copy.deepcopy(cfg)
    torch.manual_seed(37)
    expected = build_ddpm(cfg)
    torch.manual_seed(37)
    actual = matched.build_matched_model(cfg)
    assert type(actual) is ConditionalDDPM
    assert cfg == untouched
    assert actual.timesteps == 8 and actual.prediction_type == "epsilon" and actual.clip_x0 == 2.5
    assert actual.state_dict().keys() == expected.state_dict().keys()
    assert all(torch.equal(value, actual.state_dict()[key]) for key, value in expected.state_dict().items())
    clean, condition = torch.randn(2, 6, 80), torch.randn(2, 7, 80)
    loss = actual.training_loss(clean, condition, generator=torch.Generator().manual_seed(4))
    loss.backward()
    assert torch.isfinite(loss)
    actual.eval()
    a = actual.sample(condition, generator=torch.Generator().manual_seed(8))
    b = actual.sample(condition, generator=torch.Generator().manual_seed(8))
    assert a.shape == clean.shape and torch.isfinite(a).all() and torch.equal(a, b)


def test_ordinary_transformer_uses_only_low_input_and_configured_encoder():
    matched = matched_module()
    model = matched.build_matched_model(config("transformer"))
    linear_inputs = [m for m in model.modules() if isinstance(m, nn.Linear) and m.in_features == 7]
    layers = [m for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer)]
    assert len(linear_inputs) == 1 and linear_inputs[0].out_features == 16
    assert len(layers) == 2
    assert all(layer.norm_first and layer.linear1.out_features == 40 for layer in layers)
    probe = torch.tensor([-2.0, 0.0, 1.5])
    torch.testing.assert_close(layers[0].activation(probe), F.gelu(probe))
    assert not any("time_mlp" in name or "denoiser" in name for name, _ in model.named_parameters())


def test_cnn_bilstm_maps_common_depth_to_convolutions_with_historical_lstm():
    matched = matched_module()
    model = matched.build_matched_model(config("cnn_bilstm", num_layers=3))
    convolutions = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    lstm, = [m for m in model.modules() if isinstance(m, nn.LSTM)]
    assert len(convolutions) == 3 and all(m.kernel_size == (5,) for m in convolutions)
    assert convolutions[0].in_channels == 7 and all(m.out_channels == 16 for m in convolutions)
    assert lstm.hidden_size == 16 and lstm.input_size == 16
    assert lstm.num_layers == 1 and lstm.bidirectional


def test_informer_preserves_sparse_queries_distillation_and_frozen_samples():
    matched = matched_module()
    model = matched.build_matched_model(config("informer", dropout=0.1)).eval()
    sparse = [m for m in model.modules() if isinstance(m, matched.ProbSparseAttention)]
    assert [m.sequence_length for m in sparse] == [80, 40]
    assert all(m.query_count < m.sequence_length for m in sparse)
    assert len([m for m in model.modules() if isinstance(m, nn.MaxPool1d)]) == 1
    assert len([key for key in model.state_dict() if key.endswith("sample_indices")]) == 2
    attention = sparse[0]
    state = torch.get_rng_state().clone()
    matched.ProbSparseAttention(80, sampling_seed=17)
    assert torch.equal(state, torch.get_rng_state())
    q, k, v = (torch.randn(2, 2, 80, 8) for _ in range(3))
    sampled_scores = (q.unsqueeze(-2) * k[:, :, attention.sample_indices, :]).sum(-1)
    selected = (sampled_scores.amax(-1) - sampled_scores.sum(-1) / 80).topk(attention.query_count, sorted=False).indices
    expected = v.mean(-2, keepdim=True).expand_as(v).clone()
    active_q = q.gather(-2, selected.unsqueeze(-1).expand(-1, -1, -1, 8))
    exact = (active_q @ k.transpose(-1, -2) / 8**0.5).softmax(-1) @ v
    expected.scatter_(-2, selected.unsqueeze(-1).expand_as(exact), exact)
    torch.testing.assert_close(attention(q, k, v), expected)
    # Saved samples restore exactly and inference does not depend on call ordering.
    restored = matched.build_matched_model(config("informer", dropout=0.1)).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    condition = torch.randn(2, 7, 80)
    first = model.sample(condition)
    model.sample(condition + 1)
    assert torch.equal(model.sample(condition), first)
    assert torch.equal(restored.sample(condition), first)


def test_informer_ff_multiplier_and_explicit_feedforward_precedence():
    matched = matched_module()
    cfg = config("informer")
    del cfg["model"]["dim_feedforward"]
    cfg["model"]["ff_multiplier"] = 3
    network = matched.build_matched_model(cfg).network
    assert network.encoder[0].ff[0].out_features == 48
    cfg["model"]["dim_feedforward"] = 52
    assert matched.build_matched_model(cfg).network.encoder[0].ff[0].out_features == 52


def test_unknown_method_is_rejected():
    matched = matched_module()
    with pytest.raises(ValueError, match="method"):
        matched.build_matched_model(config("dense_labelled_informer"))
