"""Scientific contracts for paper-backbone adapters under the common tasks."""

from __future__ import annotations

import importlib
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def _adapter(name: str, **kwargs) -> nn.Module:
    module = importlib.import_module("virtual_imu.models.published_backbones")
    return getattr(module, name)(**kwargs)


@pytest.mark.parametrize("name", ["DeepGyroBackboneAdapter", "ResNet18SequenceAdapter"])
@pytest.mark.parametrize("in_channels,out_channels", [(9, 6), (6, 3)])
def test_common_task_shapes_and_first_order_gradients(name, in_channels, out_channels):
    torch.manual_seed(20261010)
    model = _adapter(name, in_channels=in_channels, out_channels=out_channels)
    model.train()
    x = torch.randn(2, in_channels, 80)

    output = model(x)
    assert output.shape == (2, out_channels, 80)
    prediction = output if out_channels == 6 else output[:, :, -1]
    target = torch.randn_like(prediction)
    loss = F.smooth_l1_loss(prediction, target)
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_deepgyro_matches_pinned_convolution_and_normalization_specification():
    model = _adapter("DeepGyroBackboneAdapter", in_channels=9, out_channels=6)
    convolutions = [m for m in model.modules() if isinstance(m, nn.Conv1d)]
    assert [(m.in_channels, m.out_channels) for m in convolutions] == [
        (9, 16), (16, 32), (32, 64), (64, 128), (128, 6),
    ]
    assert [m.kernel_size[0] for m in convolutions] == [7, 7, 7, 7, 1]
    assert [m.dilation[0] for m in convolutions] == [1, 4, 16, 64, 1]
    assert [m.padding[0] for m in convolutions] == [0] * 5
    assert [m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm1d)] == [0.1] * 4
    assert sum(isinstance(m, nn.GELU) for m in model.modules()) == 4
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [0.1] * 4
    assert not any(isinstance(m, nn.ReplicationPad1d) for m in model.modules())


def test_deepgyro_deterministic_padding_matches_replication_values_and_input_gradients():
    torch.manual_seed(20261010)
    model = _adapter("DeepGyroBackboneAdapter", in_channels=6, out_channels=3).eval()
    x = torch.randn(2, 6, 80, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)

    actual = model(x)
    reference = model.network(F.pad(reference_x, (510, 0), mode="replicate"))
    actual_grad, = torch.autograd.grad(actual.sum(), x)
    reference_grad, = torch.autograd.grad(reference.sum(), reference_x)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-5, atol=1e-6)


def test_deepgyro_eval_does_not_use_future_samples():
    torch.manual_seed(20261010)
    model = _adapter("DeepGyroBackboneAdapter", in_channels=9, out_channels=6).eval()
    x = torch.randn(2, 9, 80)
    changed = x.clone()
    changed[:, :, 40:] += 100 * torch.randn_like(changed[:, :, 40:])

    with torch.no_grad():
        output, changed_output = model(x), model(changed)

    torch.testing.assert_close(output[:, :, :40], changed_output[:, :, :40], rtol=0, atol=0)
    assert not torch.equal(output[:, :, 40:], changed_output[:, :, 40:])


def test_deepgyro_last_prediction_has_exact_511_sample_receptive_field():
    model = _adapter("DeepGyroBackboneAdapter", in_channels=1, out_channels=1).eval()
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Conv1d):
                layer.weight.fill_(0.05)
                layer.bias.zero_()
    x = torch.zeros(1, 1, 600, requires_grad=True)

    model(x)[0, 0, -1].backward()

    assert torch.count_nonzero(x.grad[0, 0, :89]) == 0
    assert torch.count_nonzero(x.grad[0, 0, 89:]) == 511


def test_resnet18_uses_four_full_basic_block_stages():
    model = _adapter("ResNet18SequenceAdapter", in_channels=9, out_channels=6)
    assert len(model.stages) == 4
    assert [len(stage) for stage in model.stages] == [2, 2, 2, 2]
    assert [stage[-1].conv2.out_channels for stage in model.stages] == [64, 128, 256, 512]
    assert [stage[0].conv1.stride[0] for stage in model.stages] == [1, 2, 2, 2]
    assert sum(isinstance(m, nn.Conv1d) for m in model.modules()) == 21
    assert not any(isinstance(m, nn.Linear) for m in model.modules())


@pytest.mark.parametrize("length", [79, 81])
def test_resnet18_dense_head_preserves_nonmultiple_of_32_lengths(length):
    model = _adapter("ResNet18SequenceAdapter", in_channels=6, out_channels=3).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 6, length))
    assert output.shape == (2, 3, length)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("name", ["DeepGyroBackboneAdapter", "ResNet18SequenceAdapter"])
def test_checkpoint_roundtrip_preserves_batchnorm_statistics_and_eval_output(name):
    torch.manual_seed(20261010)
    model = _adapter(name, in_channels=9, out_channels=6)
    model(torch.randn(2, 9, 80))
    model.eval()
    x = torch.randn(2, 9, 80)
    state = model.state_dict()
    assert any(key.endswith("running_mean") for key in state)
    assert any(key.endswith("num_batches_tracked") for key in state)
    payload = io.BytesIO()
    torch.save(state, payload)
    payload.seek(0)
    restored = _adapter(name, in_channels=9, out_channels=6).eval()
    restored.load_state_dict(torch.load(payload, weights_only=True), strict=True)

    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA strict-backward preflight")
@pytest.mark.parametrize("name", ["DeepGyroBackboneAdapter", "ResNet18SequenceAdapter"])
def test_strict_cuda_first_order_backward(name, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261010)
        model = _adapter(name, in_channels=9, out_channels=6).cuda().train()
        output = model(torch.randn(2, 9, 80, device="cuda"))
        output.square().mean().backward()
        torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    finally:
        torch.use_deterministic_algorithms(previous)
