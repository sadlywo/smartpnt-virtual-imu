"""Affine calibration baseline."""

from __future__ import annotations

import numpy as np


class AffineCalibration:
    """Fit y = A x + b with least squares."""

    def __init__(self) -> None:
        self.matrix_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> AffineCalibration:
        """Fit affine parameters."""
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D arrays")
        design = np.column_stack([x, np.ones(len(x))])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.matrix_ = coef[:-1].T
        self.bias_ = coef[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict calibrated output."""
        if self.matrix_ is None or self.bias_ is None:
            raise RuntimeError("AffineCalibration must be fitted before predict")
        return x @ self.matrix_.T + self.bias_


def identity_baseline(x: np.ndarray) -> np.ndarray:
    """Return low-grade IMU unchanged."""
    return np.asarray(x)
