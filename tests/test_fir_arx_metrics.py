from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.metrics import identity_normalized_score, signal_metrics


def test_signal_metrics_zero_for_perfect_prediction() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(400, 6))
    frame = pd.DataFrame({"sequence_id": ["s"] * 400})
    metrics = signal_metrics(reference, reference, frame=frame)
    assert metrics.gyro_vector_rmse == 0.0
    assert metrics.accel_vector_rmse == 0.0
    assert metrics.gyro_highpass_rmse < 1e-10
    assert metrics.accel_highpass_rmse < 1e-10


def test_identity_score_is_one() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(400, 6))
    identity = reference + rng.normal(scale=0.1, size=reference.shape)
    frame = pd.DataFrame({"sequence_id": ["s"] * 400})
    score = identity_normalized_score(identity, reference, identity, frame=frame)
    assert abs(score - 1.0) < 1e-12


def test_reference_and_prediction_shapes_are_checked() -> None:
    with np.testing.assert_raises(ValueError):
        signal_metrics(np.zeros((10, 6)), np.zeros((10, 5)))

