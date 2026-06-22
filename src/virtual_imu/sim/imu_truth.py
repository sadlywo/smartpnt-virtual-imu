"""High-grade IMU truth generation entry points."""

from __future__ import annotations

import pandas as pd

from virtual_imu.sim.trajectory import generate_motion_truth


def make_high_grade_imu(
    duration_s: float,
    rate_hz: float,
    seed: int,
    sequence_id: str,
    motion_cfg: dict | None = None,
) -> pd.DataFrame:
    """Generate high-grade IMU truth samples."""
    return generate_motion_truth(duration_s, rate_hz, seed, sequence_id, motion_cfg)
