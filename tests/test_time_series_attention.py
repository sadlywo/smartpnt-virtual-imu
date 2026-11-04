"""Behavioral checks for common-task time-series attention adapters."""

from __future__ import annotations

import io
import math

import pytest
import torch

from virtual_imu.models.time_series_attention import (
    InformerIMU,
    ITransformerIMU,
    PatchTSTIMU,
    ProbSparseAttention,
)


@pytest.mark.parametrize("model_class", [InformerIMU, PatchTSTIMU, ITransformerIMU])
@pytest.mark.parametrize("channels,length", [(9, 80), (6, 160)])
def test_attention_adapters_align_outputs_and_have_finite_gradients(model_class, channels, length):
    torch.manual_seed(17)
    model = model_class(
        in_channels=channels,
        out_channels=6 if channels == 9 else 3,
        window_size=length,
        d_model=16,
        nhead=2,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
    )
    x = torch.randn(2, channels, length, requires_grad=True)
    output = model(x)
    assert output.shape == (2, 6 if channels == 9 else 3, length)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_probsparse_selects_only_logarithmic_number_of_queries():
    torch.manual_seed(21)
    length = 80
    attention = ProbSparseAttention(length, factor=1, sampling_seed=9)
    q, k, v = [torch.randn(2, 3, length, 4) for _ in range(3)]
    result = attention(q, k, v)
    initial = v.mean(dim=-2, keepdim=True).expand_as(result)
    updated = (result - initial).abs().amax(dim=-1) > 1e-6
    assert attention.sample_indices.shape == (length, math.ceil(math.log(length)))
    assert torch.all(updated.sum(dim=-1) == math.ceil(math.log(length)))
    assert (attention.sample_indices >= 0).all()
    assert (attention.sample_indices < length).all()


def test_probsparse_becomes_exact_full_attention_when_all_queries_selected():
    torch.manual_seed(25)
    q, k, v = [torch.randn(2, 2, 11, 4) for _ in range(3)]
    attention = ProbSparseAttention(11, factor=100, sampling_seed=4)
    actual = attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-1, -2) / 2.0, dim=-1) @ v
    torch.testing.assert_close(actual, expected)


def test_probsparse_matches_reference_query_selection_and_exact_selected_context():
    torch.manual_seed(26)
    q, k, v = [torch.randn(1, 2, 32, 4) for _ in range(3)]
    attention = ProbSparseAttention(32, factor=1, sampling_seed=7)
    dense_scores = q @ k.transpose(-1, -2)
    full_context = (dense_scores / 2.0).softmax(dim=-1) @ v
    expected = v.mean(dim=-2, keepdim=True).expand_as(v).clone()
    for head in range(2):
        sparsity = []
        for query in range(32):
            samples = dense_scores[0, head, query, attention.sample_indices[query]]
            # Official implementation uses L_K here, not sampled-key count.
            sparsity.append(samples.max() - samples.sum() / 32)
        selected = torch.stack(sparsity).topk(attention.query_count, sorted=False).indices
        expected[0, head, selected] = full_context[0, head, selected]
    torch.testing.assert_close(attention(q, k, v), expected)


@pytest.mark.parametrize("model_class", [InformerIMU, PatchTSTIMU, ITransformerIMU])
def test_evaluation_is_repeatable_independent_of_rng_and_state_roundtrip(model_class):
    kwargs = dict(
        in_channels=9,
        out_channels=6,
        window_size=80,
        d_model=16,
        nhead=2,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.1,
    )
    model = model_class(**kwargs).eval()
    x = torch.randn(2, 9, 80)
    with torch.no_grad():
        first = model(x)
        torch.manual_seed(900)
        torch.randn(1000)
        torch.testing.assert_close(first, model(x), rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = model_class(**kwargs).eval()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(first, restored(x), rtol=0, atol=0)
        # Results must also be independent of inference batch partitioning.
        torch.testing.assert_close(
            first, torch.cat([model(x[:1]), model(x[1:])]), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("model_class", [InformerIMU, PatchTSTIMU, ITransformerIMU])
def test_invalid_attention_configuration_and_window_are_rejected(model_class):
    with pytest.raises(ValueError, match="divisible"):
        model_class(d_model=15, nhead=4)
    model = model_class(in_channels=9, window_size=80)
    with pytest.raises(ValueError, match="80"):
        model(torch.randn(2, 9, 79))


def test_informer_distillation_halves_memory_but_preserves_dense_output_alignment():
    model = InformerIMU(
        in_channels=9,
        out_channels=6,
        window_size=80,
        d_model=16,
        nhead=2,
        num_layers=3,
        dim_feedforward=32,
        distil=True,
    )
    observed = []
    handle = model.alignment.register_forward_pre_hook(
        lambda module, args: observed.append(args[1].shape[1])
    )
    assert model(torch.randn(2, 9, 80)).shape == (2, 6, 80)
    handle.remove()
    assert observed == [20]


def test_patchtst_preserves_reference_conditioning_and_mixes_channels():
    model = PatchTSTIMU(
        in_channels=9,
        out_channels=6,
        window_size=80,
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        dropout=0.0,
    ).eval()
    x = torch.randn(2, 9, 80)
    x[:, 6:] = 0
    x[:, 6] = 1
    alternate = x.clone()
    alternate[:, 6] = 0
    alternate[:, 7] = 1
    with torch.no_grad():
        assert not torch.allclose(model(x), model(alternate))


@pytest.mark.parametrize("model_class", [InformerIMU, PatchTSTIMU, ITransformerIMU])
def test_adapters_support_320_sample_context(model_class):
    model = model_class(
        in_channels=6,
        out_channels=3,
        window_size=320,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    ).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 6, 320)).shape == (1, 3, 320)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("model_class", [InformerIMU, PatchTSTIMU, ITransformerIMU])
def test_cuda_strict_determinism_forward_backward(model_class):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = model_class(
            in_channels=9, out_channels=6, d_model=16, nhead=2, dim_feedforward=32, dropout=0.0
        ).cuda()
        x = torch.randn(2, 9, 80, device="cuda", requires_grad=True)
        model(x).square().mean().backward()
        assert torch.isfinite(x.grad).all()
    finally:
        torch.use_deterministic_algorithms(previous)
