"""Simple calibration placeholders."""

from __future__ import annotations

import numpy as np


def affine_apply(x: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply y = A x + b to an array of 3D or 6D samples."""
    return x @ matrix.T + bias
