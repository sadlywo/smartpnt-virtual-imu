from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.data import LOW
from virtual_imu.fir_arx_research.minimal import (
    MinimalConfig,
    SceneRouter,
    annotate_scene,
    causal_band_split,
    scene_windows,
)


def _frame(length: int = 640) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    t = np.arange(length) / 40.0
    values = np.column_stack([
        np.sin(2 * np.pi * (0.1 + 2.0 * (index % 2)) * t) + 0.01 * rng.normal(size=length)
        for index in range(6)
    ])
    return pd.DataFrame({
        "sequence_id": ["s"] * length,
        "timestamp_s": t,
        **{name: values[:, index] for index, name in enumerate(LOW)},
        "gx_high": values[:, 0], "gy_high": values[:, 1], "gz_high": values[:, 2],
        "ax_high": values[:, 3], "ay_high": values[:, 4], "az_high": values[:, 5],
    })


def test_causal_band_split_preserves_sum_and_shape() -> None:
    frame = _frame()
    bank = causal_band_split(frame, MinimalConfig())
    assert bank.values.shape == (len(frame), 18)
    np.testing.assert_allclose(bank.values[:, :6], bank.values[:, 6:12] + bank.values[:, 12:18], atol=1e-10)


def test_scene_windows_and_router_are_deterministic() -> None:
    frame = _frame()
    bank = causal_band_split(frame, MinimalConfig())
    windows, features = scene_windows(frame, bank, MinimalConfig())
    assert len(windows) == 4
    assert features.shape[0] == 4
    router = SceneRouter.fit(features, MinimalConfig(n_scenes=2))
    labels = router.predict(features)
    assert set(labels).issubset({0, 1})
    annotated = annotate_scene(frame, windows, labels)
    assert "scene" in annotated.columns
    assert annotated["scene"].nunique() == len(set(labels))


def test_causal_future_does_not_change_past() -> None:
    config = MinimalConfig()
    frame = _frame(320)
    changed = frame.copy()
    changed.loc[200:, LOW] *= 100.0
    first = causal_band_split(frame, config).values
    second = causal_band_split(changed, config).values
    np.testing.assert_allclose(first[:200, 6:], second[:200, 6:], atol=0.0)
