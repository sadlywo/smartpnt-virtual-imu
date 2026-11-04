from __future__ import annotations

import pytest
import torch


def test_tartan_window_predicts_one_navigation_displacement_for_the_window():
    from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

    torch.manual_seed(20261024)
    model = TartanIMUWindowAdapter(
        in_channels=6,
        out_channels=3,
        window_size=80,
        lstm_hidden=128,
        lstm_layers=2,
        lstm_dropout=0.1,
        head_dropout=0.3,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(2, 6, 80))

    assert prediction.shape == (2, 3, 80)
    torch.testing.assert_close(prediction, prediction[:, :, -1:].expand_as(prediction))


def test_tartan_window_navigation_loss_backpropagates_finite_gradients():
    from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

    torch.manual_seed(20261024)
    model = TartanIMUWindowAdapter(window_size=80)
    prediction = model(torch.randn(2, 6, 80))[:, :, -1]
    loss = torch.nn.functional.mse_loss(prediction, torch.randn(2, 3))
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("window_size", [0, -1, 80.5, True])
def test_tartan_window_rejects_invalid_configured_window_sizes(window_size):
    from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        TartanIMUWindowAdapter(window_size=window_size)


@pytest.mark.parametrize("shape", [(2, 6, 79), (2, 9, 80), (2, 6, 80, 1)])
def test_tartan_window_rejects_inputs_outside_the_navigation_window_contract(shape):
    from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

    model = TartanIMUWindowAdapter(window_size=80)
    with pytest.raises(ValueError, match="expected input shape"):
        model(torch.randn(shape))


def test_tartan_window_rejects_the_paired_imu_mapping_task():
    from virtual_imu.models.tartan_window import TartanIMUWindowAdapter

    with pytest.raises(ValueError, match="navigation-only"):
        TartanIMUWindowAdapter(in_channels=9, out_channels=6)
