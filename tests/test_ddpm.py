from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.diffusion.data import PairedWindows, validate_record_splits
from virtual_imu.models.ddpm import ConditionalDDPM, DiffusionTransformer


def tiny_model(prediction_type="x0"):
    return ConditionalDDPM(
        DiffusionTransformer(
            d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.0, max_length=32
        ),
        timesteps=8,
        prediction_type=prediction_type,
    )


def test_forward_diffusion_and_posterior_endpoints():
    model = tiny_model()
    clean = torch.randn(2, 6, 12)
    noise = torch.randn_like(clean)
    t = torch.tensor([0, 7])
    noisy = model.q_sample(clean, t, noise)
    a = model.alpha_bar[t, None, None]
    torch.testing.assert_close(noisy, a.sqrt() * clean + (1 - a).sqrt() * noise)
    torch.testing.assert_close(
        model.predict_x0(noisy, t, noise, "epsilon"), clean, atol=2e-4, rtol=2e-4
    )
    mean, variance = model.q_posterior(clean, noisy, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(mean, clean)
    assert (variance == 0).all()
    assert model.alpha_bar[-1] < 0.001


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0"])
def test_train_and_full_reverse_sampling_are_finite_and_seeded(prediction_type):
    torch.manual_seed(12)
    model = tiny_model(prediction_type)
    condition = torch.randn(2, 7, 12)
    target = torch.randn(2, 6, 12)
    loss = model.training_loss(target, condition)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    model.eval()
    a = model.sample(condition, generator=torch.Generator().manual_seed(7))
    b = model.sample(condition, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a, b)
    assert a.shape == target.shape and torch.isfinite(a).all()
    changed = model.sample(condition + 2, generator=torch.Generator().manual_seed(7))
    assert not torch.allclose(a, changed)


def test_recording_leakage_rejected_even_across_sensor_pairs():
    records = [
        {"split": "train", "physical_group_id": "recording-1"},
        {"split": "test", "physical_group_id": "recording-1"},
        {"split": "validation", "physical_group_id": "recording-2"},
    ]
    with pytest.raises(ValueError, match="leak"):
        validate_record_splits(records)


def frame(n=23):
    data = {"timestamp_s": np.arange(n) / 40, "sequence_id": "a"}
    for i, axis in enumerate(["gx", "gy", "gz", "ax", "ay", "az"]):
        data[f"{axis}_low"] = np.arange(n) + i
        data[f"{axis}_high"] = np.arange(n) + i + 0.2
    return pd.DataFrame(data)


def test_windows_cover_tail_do_not_cross_sequences_and_reuse_train_stats():
    train = PairedWindows(frame(), window_size=8, stride=6)
    validation = PairedWindows(
        frame().assign(gx_low=1000), window_size=8, stride=6, stats=train.stats
    )
    assert train.starts[-1] == 15
    assert train.stats == validation.stats
    predictions = np.stack([np.full((6, 8), 0.2) for _ in train.starts])
    stitched, covered = train.stitch(predictions)
    assert covered.all()
    np.testing.assert_allclose(stitched, 0.2)
    two = pd.concat([frame(), frame().assign(sequence_id="b")], ignore_index=True)
    windows = PairedWindows(two, window_size=8, stride=6)
    assert all(len(set(windows.frame.iloc[s : s + 8].sequence_id)) == 1 for s in windows.starts)


def test_subset_stitching_counts_each_timestamp_once():
    windows = PairedWindows(frame(), window_size=8, stride=4)
    predictions = np.stack([np.ones((6, 8)), np.full((6, 8), 3.0)])
    stitched, covered = windows.stitch(predictions, indexes=[0, 1])
    assert covered.sum() == 12
    np.testing.assert_allclose(stitched[:4], 1)
    np.testing.assert_allclose(stitched[4:8], 2)
    np.testing.assert_allclose(stitched[8:12], 3)


def test_linear_schedule_rejects_non_gaussian_terminal_marginal():
    with pytest.raises(ValueError, match="Terminal alpha_bar"):
        ConditionalDDPM(torch.nn.Identity(), timesteps=100, schedule="linear")
    model = ConditionalDDPM(torch.nn.Identity(), timesteps=1000, schedule="linear")
    assert model.alpha_bar[-1] < 0.001
