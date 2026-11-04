from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virtual_imu.fir_arx_research.features import ComponentBank, lag_matrix


def _bank(values: np.ndarray, names: tuple[str, ...] = ("a", "b")) -> ComponentBank:
    return ComponentBank(
        names=names,
        values=values,
        families=tuple("raw" for _ in names),
        channel_labels=names,
        metadata={},
    )


def test_lag_matrix_resets_at_sequence_boundary() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1", "s2", "s2"],
            "timestamp_s": [0.0, 0.025, 0.05, 0.0, 0.025],
        }
    )
    values = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    result = lag_matrix(frame, _bank(values), taps=2, include_delta=False)
    expected = np.array(
        [
            [1, 10, 1, 10],
            [2, 20, 1, 10],
            [3, 30, 2, 20],
            [4, 40, 4, 40],
            [5, 50, 4, 40],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(result, expected)


def test_arx_delta_does_not_cross_sequence_boundary() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s2", "s2"],
            "timestamp_s": [0.0, 0.025, 0.0, 0.025],
        }
    )
    values = np.array([[1, 0], [2, 1], [20, 10], [23, 14]], dtype=float)
    result = lag_matrix(frame, _bank(values), taps=1, include_delta=True)
    expected_delta = np.array([[0, 0], [1, 1], [0, 0], [3, 4]], dtype=float)
    np.testing.assert_allclose(result[:, 2:], expected_delta)


def test_component_subset_preserves_names_and_values() -> None:
    bank = _bank(np.arange(12, dtype=float).reshape(6, 2))
    subset = bank.subset([1])
    assert subset.names == ("b",)
    np.testing.assert_allclose(subset.values[:, 0], bank.values[:, 1])
    assert subset.metadata["subset_indices"] == [1]


def test_component_bank_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        _bank(np.array([[1.0, np.nan]]))

