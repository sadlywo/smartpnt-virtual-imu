"""Scenario-aware minimal causal component selection for FIR/ARX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from virtual_imu.fir_arx_research.data import LOW
from virtual_imu.fir_arx_research.features import ComponentBank, lag_matrix
from virtual_imu.fir_arx_research.metrics import identity_normalized_score
from virtual_imu.fir_arx_research.models import ridge_predict


@dataclass(frozen=True)
class MinimalConfig:
    band_cutoff_hz: float = 0.5
    rate_hz: float = 40.0
    scene_window_rows: int = 160
    scene_stride_rows: int = 160
    n_scenes: int = 2
    random_state: int = 20261105
    accuracy_tolerance: float = 0.02
    min_scene_rows: int = 1000


def causal_band_split(frame: pd.DataFrame, config: MinimalConfig) -> ComponentBank:
    values = frame[LOW].to_numpy(dtype=np.float64)
    low = np.empty_like(values)
    sos = butter(4, config.band_cutoff_hz, btype="lowpass", fs=config.rate_hz, output="sos")
    for positions in frame.groupby("sequence_id", sort=False).indices.values():
        ordered = positions[np.argsort(frame.iloc[positions]["timestamp_s"].to_numpy(float), kind="stable")]
        low[ordered] = sosfilt_causal(sos, values[ordered])
    high = values - low
    names = tuple([f"raw_{axis}" for axis in LOW] + [f"low_{axis}" for axis in LOW] + [f"high_{axis}" for axis in LOW])
    families = tuple(["raw"] * 6 + ["low"] * 6 + ["high"] * 6)
    return ComponentBank(
        names=names,
        values=np.column_stack([values, low, high]),
        families=families,
        channel_labels=names,
        metadata={"band_cutoff_hz": config.band_cutoff_hz, "rate_hz": config.rate_hz},
    )


def sosfilt_causal(sos: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply a causal SOS filter independently to each column."""
    result = np.empty_like(values, dtype=np.float64)
    for channel in range(values.shape[1]):
        result[:, channel] = lfilter(sos[0, :3], sos[0, 3:], values[:, channel])
        for section in range(1, sos.shape[0]):
            result[:, channel] = lfilter(
                sos[section, :3],
                sos[section, 3:],
                result[:, channel],
            )
    return result


def scene_windows(frame: pd.DataFrame, bank: ComponentBank, config: MinimalConfig) -> tuple[list[pd.DataFrame], np.ndarray]:
    windows: list[pd.DataFrame] = []
    features: list[np.ndarray] = []
    for sequence_id, positions in frame.groupby("sequence_id", sort=False).indices.items():
        ordered = positions[np.argsort(frame.iloc[positions]["timestamp_s"].to_numpy(float), kind="stable")]
        count = len(ordered) // config.scene_window_rows
        for index in range(count):
            start = index * config.scene_window_rows
            stop = start + config.scene_window_rows
            indices = ordered[start:stop]
            block = frame.iloc[indices].copy().reset_index(drop=True)
            block["window_id"] = f"{sequence_id}__scene{index:05d}"
            block["sequence_id"] = block["window_id"]
            raw = bank.values[indices, 0:6]
            low = bank.values[indices, 6:12]
            high = bank.values[indices, 12:18]
            total_energy = float(np.mean(raw**2)) + 1e-12
            low_energy = float(np.mean(low**2))
            high_energy = float(np.mean(high**2))
            gyro_energy = float(np.mean(raw[:, :3] ** 2))
            accel_energy = float(np.mean(raw[:, 3:] ** 2))
            features.append(
                np.asarray(
                    [
                        np.log1p(total_energy),
                        low_energy / total_energy,
                        high_energy / total_energy,
                        np.log1p(gyro_energy),
                        np.log1p(accel_energy),
                    ],
                    dtype=np.float64,
                )
            )
            windows.append(block)
    if not windows:
        raise ValueError("No full scenario windows are available")
    return windows, np.vstack(features)


@dataclass
class SceneRouter:
    scaler: StandardScaler
    kmeans: KMeans
    high_ratio_index: int = 2

    @classmethod
    def fit(cls, features: np.ndarray, config: MinimalConfig) -> "SceneRouter":
        if len(features) < config.n_scenes:
            raise ValueError("Too few windows for scenario clustering")
        scaler = StandardScaler().fit(features)
        kmeans = KMeans(
            n_clusters=config.n_scenes,
            random_state=config.random_state,
            n_init=10,
        ).fit(scaler.transform(features))
        raw_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        order = np.argsort(raw_centers[:, 2])
        relabel = {int(old): int(new) for new, old in enumerate(order)}
        kmeans.labels_ = np.asarray([relabel[int(label)] for label in kmeans.labels_], dtype=int)
        return cls(scaler, kmeans, high_ratio_index=2)

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw_labels = self.kmeans.predict(self.scaler.transform(features))
        # Recover the stable low-high mapping from sorted cluster high-frequency ratios.
        raw_centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        order = np.argsort(raw_centers[:, self.high_ratio_index])
        relabel = {int(old): int(new) for new, old in enumerate(order)}
        return np.asarray([relabel[int(label)] for label in raw_labels], dtype=int)


def annotate_scene(frame: pd.DataFrame, windows: list[pd.DataFrame], labels: np.ndarray, column: str = "scene") -> pd.DataFrame:
    annotated = []
    for window, label in zip(windows, labels, strict=True):
        block = window.copy()
        block[column] = int(label)
        annotated.append(block)
    result = pd.concat(annotated, ignore_index=True)
    if result.empty:
        raise ValueError("Scenario annotation is empty")
    return result


def fit_component_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    test_bank: ComponentBank,
    *,
    taps: int,
    alpha: float,
    include_delta: bool,
) -> tuple[dict[str, Any], np.ndarray, float]:
    from virtual_imu.fir_arx_research.data import HIGH
    from virtual_imu.fir_arx_research.models import ridge_fit_from_gram

    train_x = lag_matrix(train, train_bank, taps=taps, include_delta=include_delta)
    validation_x = lag_matrix(validation, validation_bank, taps=taps, include_delta=include_delta)
    test_x = lag_matrix(test, test_bank, taps=taps, include_delta=include_delta)
    weights, bias = ridge_fit_from_gram(train_x, train[HIGH].to_numpy(float), alpha)
    validation_prediction = ridge_predict(validation_x, weights, bias)
    score = identity_normalized_score(
        validation_prediction,
        validation[HIGH].to_numpy(float),
        validation[LOW].to_numpy(float),
        frame=validation,
    )
    test_prediction = ridge_predict(test_x, weights, bias)
    state = {
        "kind": "minimal_component_fir_arx",
        "components": list(train_bank.names),
        "taps": taps,
        "include_delta": include_delta,
        "alpha": float(alpha),
        "weights": np.asarray(weights).tolist(),
        "bias": np.asarray(bias).tolist(),
        "validation_score": float(score),
    }
    return state, test_prediction, float(score)


def rank_single_components(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    config: MinimalConfig,
) -> list[int]:
    scores = []
    for index, name in enumerate(train_bank.names):
        family = train_bank.families[index]
        # Scenario priors reduce unnecessary candidates before validation.
        prior = 0.0 if family in {"raw", "low", "high"} else 0.0
        subset_train = train_bank.subset([index])
        subset_validation = validation_bank.subset([index])
        _, _, score = fit_component_model(
            train,
            validation,
            validation,
            subset_train,
            subset_validation,
            subset_validation,
            taps=9,
            alpha=0.01,
            include_delta=False,
        )
        scores.append((score + prior, index))
    return [index for _, index in sorted(scores)]


def select_minimal_prefix(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    test_bank: ComponentBank,
    *,
    config: MinimalConfig,
    search: dict[str, Any],
) -> dict[str, Any]:
    order = rank_single_components(train, validation, train_bank, validation_bank, config)
    prefixes = []
    for count in range(1, len(order) + 1):
        selected = order[:count]
        best_candidate = None
        for include_delta, taps_grid in (
            (False, [int(v) for v in search["fir_taps"]]),
            (True, [int(v) for v in search["arx_taps"]]),
        ):
            for taps in taps_grid:
                for alpha in [float(v) for v in search["ridge_alpha"]]:
                    state, prediction, score = fit_component_model(
                        train,
                        validation,
                        test,
                        train_bank.subset(selected),
                        validation_bank.subset(selected),
                        test_bank.subset(selected),
                        taps=taps,
                        alpha=alpha,
                        include_delta=include_delta,
                    )
                    candidate = {
                        "component_count": count,
                        "selected_indices": selected,
                        "components": [train_bank.names[i] for i in selected],
                        "state": state,
                        "validation_score": score,
                        "test_prediction": prediction,
                        "taps": taps,
                        "alpha": alpha,
                        "include_delta": include_delta,
                    }
                    if best_candidate is None or score < best_candidate["validation_score"]:
                        best_candidate = candidate
        prefixes.append(best_candidate)
    best_score = min(candidate["validation_score"] for candidate in prefixes)
    threshold = best_score * (1.0 + config.accuracy_tolerance)
    selected = next(candidate for candidate in prefixes if candidate["validation_score"] <= threshold)
    selected["frontier"] = [
        {
            "component_count": candidate["component_count"],
            "validation_score": candidate["validation_score"],
            "components": candidate["components"],
            "taps": candidate["taps"],
            "alpha": candidate["alpha"],
            "include_delta": candidate["include_delta"],
        }
        for candidate in prefixes
    ]
    return selected
