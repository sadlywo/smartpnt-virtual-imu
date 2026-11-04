"""Causal FIR/ARX features over named input components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.data import LOW


@dataclass(frozen=True)
class ComponentBank:
    """Named per-sample components aligned exactly with a frame's row order."""

    names: tuple[str, ...]
    values: np.ndarray
    families: tuple[str, ...]
    channel_labels: tuple[str, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError("Component bank shape does not match component names")
        if len(self.families) != len(self.names) or len(self.channel_labels) != len(self.names):
            raise ValueError("Every component needs family and channel metadata")
        if not np.isfinite(values).all():
            raise ValueError("Component bank contains nonfinite values")

    def subset(self, indices: list[int] | tuple[int, ...] | np.ndarray) -> "ComponentBank":
        idx = np.asarray(indices, dtype=int)
        if idx.ndim != 1 or len(idx) == 0 or len(np.unique(idx)) != len(idx):
            raise ValueError("Component subset must be nonempty and unique")
        if idx.min() < 0 or idx.max() >= len(self.names):
            raise ValueError("Component subset is out of range")
        return ComponentBank(
            names=tuple(self.names[i] for i in idx),
            values=self.values[:, idx],
            families=tuple(self.families[i] for i in idx),
            channel_labels=tuple(self.channel_labels[i] for i in idx),
            metadata={**self.metadata, "subset_indices": idx.tolist()},
        )


class RawFeatureSource:
    """Identity preprocessing: the six low-grade measurements themselves."""

    name = "raw"

    def transform(self, frame: pd.DataFrame) -> ComponentBank:
        return ComponentBank(
            names=tuple(LOW),
            values=frame[LOW].to_numpy(dtype=np.float64),
            families=tuple("raw" for _ in LOW),
            channel_labels=tuple(LOW),
            metadata={"condition": self.name, "decomposition": None},
        )


def _lag_values(values: np.ndarray, taps: int, include_delta: bool) -> np.ndarray:
    padded = np.pad(values, ((taps - 1, 0), (0, 0)), mode="edge")
    pieces = [padded[taps - 1 - lag : taps - 1 - lag + len(values)] for lag in range(taps)]
    if include_delta:
        pieces.append(np.diff(values, axis=0, prepend=values[[0]]))
    return np.concatenate(pieces, axis=1)


def lag_matrix(
    frame: pd.DataFrame,
    bank: ComponentBank,
    *,
    taps: int,
    include_delta: bool,
) -> np.ndarray:
    """Build sequence-aware causal lag/difference features without boundary leakage."""
    if taps <= 0:
        raise ValueError("taps must be positive")
    if len(frame) != len(bank.values):
        raise ValueError("Frame and component bank lengths differ")
    width = len(bank.names) * (taps + int(include_delta))
    result = np.empty((len(frame), width), dtype=np.float64)
    for positions in frame.groupby("sequence_id", sort=False, dropna=False).indices.values():
        timestamps = frame.iloc[positions]["timestamp_s"].to_numpy(dtype=float)
        ordered = positions[np.argsort(timestamps, kind="stable")]
        values = bank.values[ordered]
        result[ordered] = _lag_values(values, taps, include_delta)
    return result


def lag_feature_names(bank: ComponentBank, *, taps: int, include_delta: bool) -> list[str]:
    names = [f"{name}@lag{lag}" for lag in range(taps) for name in bank.names]
    if include_delta:
        names.extend(f"{name}@delta" for name in bank.names)
    return names


def current_component_matrix(bank: ComponentBank) -> np.ndarray:
    return np.asarray(bank.values, dtype=np.float64)

