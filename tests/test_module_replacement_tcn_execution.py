"""Equivalent TCN filters avoid slow small CUDA convolution launches."""

import copy

import pytest
import torch

from virtual_imu.replacements.models import _TCNBottleneck


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_token_execution_equals_native_filters_and_gradients_without_conv_launches(dilation):
    torch.set_num_threads(1)
    torch.manual_seed(971)
    fast = _TCNBottleneck(8, 5, dilation).double().eval()
    native = copy.deepcopy(fast)
    x = torch.randn(2, 19, 8, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    launched = []
    handles = [
        m.register_forward_hook(lambda *args: launched.append(True))
        for m in fast.modules()
        if isinstance(m, torch.nn.Conv1d)
    ]
    actual = fast(x)
    expected = native.norm(y + native.net(y.transpose(1, 2)).transpose(1, 2))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-11)
    for (na, pa), (nb, pb) in zip(fast.named_parameters(), native.named_parameters()):
        assert na == nb
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-9, atol=1e-11)
    for h in handles:
        h.remove()
    assert not launched, (
        "Use the same filter weights as token matrix products to avoid native Conv1d launches"
    )
