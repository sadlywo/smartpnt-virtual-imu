import os

import pytest
import torch

from virtual_imu.models.losses import preintegration_consistency_loss


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_strict_deterministic_integral_matches_cpu_cumsum_value_and_gradient(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA regression runs on the training server")
    old_flag = torch.are_deterministic_algorithms_enabled()
    old_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(4, 6, 80, generator=generator, dtype=dtype)
    targets = torch.randn(4, 6, 80, generator=generator, dtype=dtype)
    reference = values.clone().requires_grad_()
    expected = torch.nn.functional.smooth_l1_loss(reference.cumsum(-1) * 0.025, targets.cumsum(-1) * 0.025)
    expected.backward()
    try:
        torch.use_deterministic_algorithms(True)
        predicted = values.to(device).requires_grad_()
        result = preintegration_consistency_loss(predicted, targets.to(device), dt=0.025)
        result.backward()
        torch.testing.assert_close(result.cpu(), expected.detach(), rtol=2e-5, atol=1e-8)
        torch.testing.assert_close(predicted.grad.cpu(), reference.grad, rtol=2e-5, atol=1e-8)
    finally:
        torch.use_deterministic_algorithms(old_flag)
        if old_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_workspace
