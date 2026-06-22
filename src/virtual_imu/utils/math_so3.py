"""Small SO(3) and quaternion helpers used by simulation and evaluation."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3-vector."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to a rotation matrix."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3) + skew(rotvec)
    axis = rotvec / theta
    k = skew(axis)
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def integrate_gyro(gyro: np.ndarray, dt: float) -> np.ndarray:
    """Integrate angular-rate samples into rotation matrices."""
    rotations = np.zeros((len(gyro), 3, 3))
    rot = np.eye(3)
    for idx, sample in enumerate(gyro):
        rot = rot @ rotvec_to_matrix(sample * dt)
        rotations[idx] = rot
    return rotations


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a wxyz quaternion."""
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s]
        )
    idx = int(np.argmax(np.diag(rot)))
    if idx == 0:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return np.array([(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s])
    if idx == 1:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return np.array([(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s])
    s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return np.array([(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s])
