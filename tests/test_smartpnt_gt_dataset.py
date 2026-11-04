from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SmartPNTGTWindowDataset,
    make_smartpnt_gt_dataloader,
    make_smartpnt_gt_split_dataloaders,
)


def _smartpnt_frame(sequence_id: str = "traj") -> pd.DataFrame:
    n = 6
    return pd.DataFrame(
        {
            "timestamp_s": np.arange(n, dtype=float) * 0.01,
            "sequence_id": [sequence_id] * n,
            "gx_radps": np.arange(n, dtype=float),
            "gy_radps": np.arange(n, dtype=float) + 10,
            "gz_radps": np.arange(n, dtype=float) + 20,
            "ax_mps2": np.arange(n, dtype=float) + 30,
            "ay_mps2": np.arange(n, dtype=float) + 40,
            "az_mps2": np.arange(n, dtype=float) + 50,
            "gt_v_east_mps": np.arange(n, dtype=float) + 100,
            "gt_v_north_mps": np.arange(n, dtype=float) + 200,
            "gt_v_up_mps": np.arange(n, dtype=float) + 300,
            "gt_x_local_m": np.arange(n, dtype=float) * 2.0,
            "gt_y_local_m": np.arange(n, dtype=float) * 3.0,
            "gt_z_local_m": np.arange(n, dtype=float) * 4.0,
            "gt_heading_deg": [350.0, 355.0, 2.0, 5.0, 8.0, 11.0],
            "gt_pitch_deg": np.arange(n, dtype=float) * 0.1,
            "gt_roll_deg": -np.arange(n, dtype=float) * 0.2,
        }
    )


def test_smartpnt_gt_dataset_returns_imu_input_and_velocity_position_sequence():
    df = _smartpnt_frame()

    dataset = SmartPNTGTWindowDataset(
        df,
        window_size=4,
        stride=2,
        label_mode="velocity_position",
        normalize_features=False,
        normalize_targets=False,
    )

    assert len(dataset) == 2
    x, y, meta = dataset[0]
    assert x.shape == torch.Size([6, 4])
    assert y.shape == torch.Size([6, 4])
    assert meta["sequence_id"] == "traj"
    assert meta["start_index"] == 0
    assert meta["end_index"] == 3
    np.testing.assert_allclose(x[:, 0].numpy(), df.loc[0, SMARTPNT_GT_IMU_COLUMNS].to_numpy(float))
    np.testing.assert_allclose(
        y[:, -1].numpy(),
        df.loc[3, ["gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps", "gt_x_local_m", "gt_y_local_m", "gt_z_local_m"]]
        .to_numpy(float),
    )


def test_smartpnt_gt_dataset_returns_short_window_pose_delta_with_angle_wrap():
    df = _smartpnt_frame()

    dataset = SmartPNTGTWindowDataset(
        df,
        window_size=4,
        stride=2,
        label_mode="pose_delta",
        normalize_features=False,
        normalize_targets=False,
    )

    _, y, meta = dataset[0]
    assert y.shape == torch.Size([6])
    assert meta["target_columns"] == [
        "delta_x_local_m",
        "delta_y_local_m",
        "delta_z_local_m",
        "delta_heading_deg",
        "delta_pitch_deg",
        "delta_roll_deg",
    ]
    np.testing.assert_allclose(y.numpy(), [6.0, 9.0, 12.0, 15.0, 0.3, -0.6], atol=1e-5)


def test_smartpnt_gt_dataset_returns_heading_aligned_body_pose_delta_and_integration_aux():
    df = _smartpnt_frame()
    df["gt_heading_deg"] = 90.0
    df["gt_x_local_m"] = np.arange(len(df), dtype=float) * 2.0
    df["gt_y_local_m"] = np.zeros(len(df), dtype=float)
    df["gt_z_local_m"] = np.arange(len(df), dtype=float) * 0.5
    df["gt_v_east_mps"] = 200.0
    df["gt_v_north_mps"] = 0.0
    df["gt_v_up_mps"] = 50.0

    dataset = SmartPNTGTWindowDataset(
        df,
        window_size=4,
        stride=2,
        label_mode="body_pose_delta",
        normalize_features=False,
        normalize_targets=False,
    )

    _, y, meta = dataset[0]
    assert meta["target_columns"] == [
        "delta_forward_m",
        "delta_right_m",
        "delta_up_m",
        "delta_heading_deg",
        "delta_pitch_deg",
        "delta_roll_deg",
    ]
    np.testing.assert_allclose(y.numpy()[:3], [6.0, 0.0, 1.5], atol=1e-5)
    assert meta["integrated_delta_forward_m"] == pytest.approx(6.0, abs=1e-5)
    assert meta["integrated_delta_right_m"] == pytest.approx(0.0, abs=1e-5)
    assert meta["integrated_delta_up_m"] == pytest.approx(1.5, abs=1e-5)


def test_smartpnt_gt_dataloader_batches_tensor_outputs():
    df = pd.concat([_smartpnt_frame("a"), _smartpnt_frame("b")], ignore_index=True)

    loader = make_smartpnt_gt_dataloader(
        df,
        window_size=3,
        stride=3,
        batch_size=2,
        label_mode="velocity",
        normalize_features=False,
        normalize_targets=False,
        shuffle=False,
    )

    x, y, meta = next(iter(loader))
    assert x.shape == torch.Size([2, 6, 3])
    assert y.shape == torch.Size([2, 3, 3])
    assert meta["sequence_id"] == ["a", "a"]


def test_smartpnt_gt_dataset_rejects_missing_required_columns():
    df = _smartpnt_frame().drop(columns=["az_mps2"])

    with pytest.raises(ValueError, match="Missing required SmartPNT GT columns"):
        SmartPNTGTWindowDataset(df, window_size=4, stride=1)


def test_smartpnt_gt_split_dataloaders_share_train_normalization_stats():
    df = pd.concat(
        [_smartpnt_frame("train_seq"), _smartpnt_frame("val_seq"), _smartpnt_frame("test_seq")],
        ignore_index=True,
    )

    loaders = make_smartpnt_gt_split_dataloaders(
        df,
        split_sequences={
            "train": ["train_seq"],
            "val": ["val_seq"],
            "test": ["test_seq"],
        },
        window_size=3,
        stride=3,
        batch_size=2,
        label_mode="position_delta",
    )

    assert set(loaders) == {"train", "val", "test"}
    train_stats = loaders["train"].dataset.feature_stats
    assert loaders["val"].dataset.feature_stats == train_stats
    assert loaders["test"].dataset.feature_stats == train_stats
