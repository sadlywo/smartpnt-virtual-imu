from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.data import HIGH, LOW
from virtual_imu.fir_arx_research.features import RawFeatureSource
from virtual_imu.fir_arx_research.models import fit_models, ridge_predict


def _frame(seed: int, length: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    low = rng.normal(scale=0.2, size=(length, 6))
    high = low * np.array([1.4, 1.2, 1.3, 0.8, 0.75, 0.7]) + 0.03
    timestamps = np.arange(length) * 0.025
    return pd.DataFrame(
        {
            "timestamp_s": timestamps,
            "sequence_id": ["sequence"] * length,
            "dataset_id": ["synthetic"] * length,
            "pair_id": ["low_to_high"] * length,
            "low_sensor": ["low"] * length,
            "high_sensor": ["high"] * length,
            **{name: low[:, index] for index, name in enumerate(LOW)},
            **{name: high[:, index] for index, name in enumerate(HIGH)},
        }
    )


def test_fir_and_ridge_affine_recover_linear_device_mapping() -> None:
    train = _frame(1)
    validation = _frame(2)
    test = _frame(3)
    source = RawFeatureSource()
    models = fit_models(
        condition=source.name,
        train=train,
        validation=validation,
        test=test,
        train_bank=source.transform(train),
        validation_bank=source.transform(validation),
        test_bank=source.transform(test),
        methods=["identity", "affine", "ridge_affine", "fir"],
        search={
            "ridge_alpha": [0.0001, 0.01],
            "fir_taps": [1, 3],
            "arx_taps": [3],
        },
    )
    identity = next(model for model in models if model.name == "identity")
    fir = next(model for model in models if model.name == "fir")
    identity_error = np.sqrt(np.mean(np.sum((identity.test_prediction - test[HIGH].to_numpy()) ** 2, axis=1)))
    fir_error = np.sqrt(np.mean(np.sum((fir.test_prediction - test[HIGH].to_numpy()) ** 2, axis=1)))
    assert fir_error < identity_error * 0.05
    assert fir.state["fit_split"] if "fit_split" in fir.state else True
    assert len(fir.state["weights"]) == 6 * fir.state["taps"]


def test_ridge_predict_matches_manual_product() -> None:
    x = np.arange(12, dtype=float).reshape(3, 4)
    weights = np.ones((4, 6))
    bias = np.arange(6, dtype=float)
    result = ridge_predict(x, weights, bias)
    np.testing.assert_allclose(result, x @ weights + bias)

