"""Deterministic linear calibration, FIR, and ARX fitting for paired IMUs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.data import HIGH, LOW
from virtual_imu.fir_arx_research.features import (
    ComponentBank,
    lag_feature_names,
    lag_matrix,
)
from virtual_imu.fir_arx_research.metrics import identity_normalized_score, signal_metrics


@dataclass
class FittedModel:
    name: str
    condition: str
    state: dict[str, Any]
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_score: float
    candidates: list[dict[str, Any]]


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(x, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if design.ndim != 2 or target.ndim != 2 or len(design) != len(target):
        raise ValueError("Ridge design/target shapes differ")
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("Ridge design/target contain nonfinite values")
    augmented = np.column_stack([design, np.ones(len(design))])
    penalty = float(alpha) * np.eye(augmented.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ target)
    return coefficients[:-1], coefficients[-1]


def ridge_fit_from_gram(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.column_stack([x, np.ones(len(x))])
    gram = augmented.T @ augmented
    rhs = augmented.T @ y
    penalty = float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return coefficients[:-1], coefficients[-1]


def ridge_predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + np.asarray(
        bias, dtype=np.float64
    )


def _best_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_frame: pd.DataFrame,
    alphas: Iterable[float],
) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    augmented = np.column_stack([train_x, np.ones(len(train_x))])
    gram = augmented.T @ augmented
    rhs = augmented.T @ train_y
    validation_design = np.column_stack([validation_x, np.ones(len(validation_x))])
    identity = validation_frame[LOW].to_numpy(dtype=np.float64)
    best = None
    history = []
    for alpha in alphas:
        penalty = float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(gram + penalty, rhs)
        weights, bias = coefficients[:-1], coefficients[-1]
        prediction = validation_design @ coefficients
        score = identity_normalized_score(
            prediction,
            validation_y,
            identity,
            frame=validation_frame,
        )
        history.append({"alpha": float(alpha), "validation_score": score})
        if best is None or score < best[0]:
            best = (score, float(alpha), weights, bias)
    assert best is not None
    return best[1], best[2], best[3], history


def _select_lagged(
    name: str,
    condition: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    test_bank: ComponentBank,
    *,
    taps_grid: list[int],
    alphas: list[float],
    include_delta: bool,
) -> tuple[FittedModel, list[dict[str, Any]]]:
    train_y = train[HIGH].to_numpy(dtype=np.float64)
    validation_y = validation[HIGH].to_numpy(dtype=np.float64)
    best: dict[str, Any] | None = None
    trials = []
    for taps in taps_grid:
        tx = lag_matrix(train, train_bank, taps=taps, include_delta=include_delta)
        vx = lag_matrix(validation, validation_bank, taps=taps, include_delta=include_delta)
        alpha, weights, bias, alpha_history = _best_alpha(
            tx,
            train_y,
            vx,
            validation_y,
            validation,
            alphas,
        )
        validation_prediction = ridge_predict(vx, weights, bias)
        score = identity_normalized_score(
            validation_prediction,
            validation_y,
            validation[LOW].to_numpy(dtype=np.float64),
            frame=validation,
        )
        trial = {
            "model": name,
            "condition": condition,
            "taps": int(taps),
            "include_delta": bool(include_delta),
            "alpha": alpha,
            "validation_score": score,
            "alpha_history": alpha_history,
            "feature_count": int(tx.shape[1]),
        }
        trials.append(trial)
        if best is None or score < best["validation_score"]:
            best = {
                "validation_score": score,
                "taps": int(taps),
                "alpha": alpha,
                "weights": weights,
                "bias": bias,
                "trial": trial,
            }
    assert best is not None
    taps = best["taps"]
    test_x = lag_matrix(test, test_bank, taps=taps, include_delta=include_delta)
    test_prediction = ridge_predict(test_x, best["weights"], best["bias"])
    validation_x = lag_matrix(validation, validation_bank, taps=taps, include_delta=include_delta)
    validation_prediction = ridge_predict(validation_x, best["weights"], best["bias"])
    state = {
        "model": name,
        "condition": condition,
        "kind": "lagged_ridge",
        "taps": taps,
        "include_delta": bool(include_delta),
        "alpha": float(best["alpha"]),
        "components": list(train_bank.names),
        "feature_names": lag_feature_names(train_bank, taps=taps, include_delta=include_delta),
        "weights": np.asarray(best["weights"]).tolist(),
        "bias": np.asarray(best["bias"]).tolist(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_score": float(best["validation_score"]),
    }
    return FittedModel(name, condition, state, validation_prediction, test_prediction, float(best["validation_score"]), trials), trials


def _fit_simple(
    name: str,
    condition: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    bank_train: ComponentBank,
    bank_validation: ComponentBank,
    bank_test: ComponentBank,
    alphas: list[float],
) -> tuple[FittedModel, list[dict[str, Any]]]:
    x_train = bank_train.values
    x_validation = bank_validation.values
    x_test = bank_test.values
    y_train = train[HIGH].to_numpy(dtype=np.float64)
    y_validation = validation[HIGH].to_numpy(dtype=np.float64)
    identity = validation[LOW].to_numpy(dtype=np.float64)
    if name == "identity":
        validation_prediction = x_validation.copy()
        test_prediction = x_test.copy()
        state = {"model": name, "condition": condition, "kind": "identity"}
        score = 1.0
        trials = [{"model": name, "condition": condition, "validation_score": score}]
    elif name == "mean_bias":
        bias = np.mean(y_train - x_train, axis=0)
        validation_prediction = x_validation + bias
        test_prediction = x_test + bias
        state = {
            "model": name,
            "condition": condition,
            "kind": "mean_bias",
            "bias": bias.tolist(),
        }
        score = identity_normalized_score(validation_prediction, y_validation, identity, frame=validation)
        trials = [{"model": name, "condition": condition, "validation_score": score}]
    elif name in {"affine", "ridge_affine"}:
        if name == "affine":
            weights, bias = ridge_fit(x_train, y_train, alpha=0.0)
            alpha_history = [{"alpha": 0.0}]
            selected_alpha = 0.0
        else:
            selected_alpha, weights, bias, alpha_history = _best_alpha(
                x_train,
                y_train,
                x_validation,
                y_validation,
                validation,
                alphas,
            )
        validation_prediction = ridge_predict(x_validation, weights, bias)
        test_prediction = ridge_predict(x_test, weights, bias)
        score = identity_normalized_score(validation_prediction, y_validation, identity, frame=validation)
        state = {
            "model": name,
            "condition": condition,
            "kind": "affine",
            "alpha": float(selected_alpha),
            "components": list(bank_train.names),
            "weights": np.asarray(weights).tolist(),
            "bias": np.asarray(bias).tolist(),
        }
        trials = [
            {
                "model": name,
                "condition": condition,
                "alpha": float(selected_alpha),
                "validation_score": score,
                "alpha_history": alpha_history,
            }
        ]
    else:
        raise ValueError(f"Unknown simple model {name}")
    return FittedModel(name, condition, state, validation_prediction, test_prediction, float(score), trials), trials


def fit_models(
    *,
    condition: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    test_bank: ComponentBank,
    methods: list[str],
    search: dict[str, Any],
) -> list[FittedModel]:
    models = []
    alphas = [float(value) for value in search["ridge_alpha"]]
    for method in methods:
        if method in {"identity", "mean_bias", "affine", "ridge_affine"}:
            model, _ = _fit_simple(
                method,
                condition,
                train,
                validation,
                test,
                train_bank,
                validation_bank,
                test_bank,
                alphas,
            )
        elif method == "fir":
            model, _ = _select_lagged(
                method,
                condition,
                train,
                validation,
                test,
                train_bank,
                validation_bank,
                test_bank,
                taps_grid=[int(value) for value in search["fir_taps"]],
                alphas=alphas,
                include_delta=False,
            )
        elif method == "arx":
            model, _ = _select_lagged(
                method,
                condition,
                train,
                validation,
                test,
                train_bank,
                validation_bank,
                test_bank,
                taps_grid=[int(value) for value in search["arx_taps"]],
                alphas=alphas,
                include_delta=True,
            )
        else:
            raise ValueError(f"Unknown method {method}")
        models.append(model)
    return models


def model_metric_rows(
    models: list[FittedModel],
    protocol: str,
    identity: tuple[str, str, str, str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    identity_tuple = dict(zip(["dataset_id", "pair_id", "low_sensor", "high_sensor"], identity, strict=True))
    identity_model = next(model for model in models if model.name == "identity")
    baseline_test = signal_metrics(
        identity_model.test_prediction,
        test[HIGH].to_numpy(dtype=np.float64),
        frame=test,
    )
    rows = []
    for model in models:
        validation_metrics = signal_metrics(
            model.validation_prediction,
            validation[HIGH].to_numpy(dtype=np.float64),
            frame=validation,
        )
        test_metrics = signal_metrics(
            model.test_prediction,
            test[HIGH].to_numpy(dtype=np.float64),
            frame=test,
        )
        rows.append(
            {
                "protocol": protocol,
                **identity_tuple,
                "condition": model.condition,
                "model": model.name,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "validation_score": model.validation_score,
                **{f"validation_{key}": value for key, value in validation_metrics.__dict__.items()},
                **{f"test_{key}": value for key, value in test_metrics.__dict__.items()},
                "test_gyro_improvement_vs_identity_pct": float(
                    100.0
                    * (baseline_test.gyro_vector_rmse - test_metrics.gyro_vector_rmse)
                    / max(baseline_test.gyro_vector_rmse, 1e-12)
                ),
                "test_accel_improvement_vs_identity_pct": float(
                    100.0
                    * (baseline_test.accel_vector_rmse - test_metrics.accel_vector_rmse)
                    / max(baseline_test.accel_vector_rmse, 1e-12)
                ),
                "taps": model.state.get("taps"),
                "alpha": model.state.get("alpha"),
                "include_delta": model.state.get("include_delta"),
                "component_count": len(model.state.get("components", LOW)),
                "feature_count": len(model.state.get("feature_names", LOW)),
            }
        )
    return rows
