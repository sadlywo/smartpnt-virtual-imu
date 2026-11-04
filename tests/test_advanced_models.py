from __future__ import annotations

import torch

from virtual_imu.models import make_model


def test_advanced_model_factory_outputs_channel_first_windows():
    configs = [
        {"type": "alimu_cnn_bilstm", "in_channels": 7, "out_channels": 6, "hidden_channels": 16, "lstm_hidden": 12},
        {"type": "calibnet_cnn", "in_channels": 7, "out_channels": 6, "hidden_channels": 16},
        {"type": "ronin_resnet", "in_channels": 7, "out_channels": 6, "hidden_channels": 16},
        {"type": "brossard_cnn", "in_channels": 7, "out_channels": 6, "hidden_channels": 16},
        {"type": "sixdof_cnn_bilstm", "in_channels": 7, "out_channels": 6, "hidden_channels": 16, "lstm_hidden": 12},
        {"type": "fsda_denoiser", "in_channels": 7, "out_channels": 6, "hidden_channels": 16, "latent_channels": 12},
    ]
    x = torch.randn(2, 7, 32)
    for cfg in configs:
        model = make_model(cfg)
        y = model(x)
        assert tuple(y.shape) == (2, 6, 32)
