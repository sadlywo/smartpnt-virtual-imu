"""SO(3) propagation against independent Penn optical rig poses.

Inputs are the already calibrated rig-body IMU streams. ``timestamp_s`` is
relative physical signal time. The known source clock origin maps it back to
the official GT clock; availability time is latency metadata, never a GT clock.
Ground truth initializes each predefined window once and is read thereafter
only for scoring. This is an oracle-initialized inertial diagnostic, not VIO.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.spatial.transform import Rotation, Slerp

ROTATION_COLUMNS = [f"r{i}{j}" for i in range(3) for j in range(3)]
POSITION_COLUMNS = ["px_m", "py_m", "pz_m"]
GYRO_AXES = ["gx", "gy", "gz"]


def _rotation_array(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=float)
    if values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError("rotation must contain finite 3x3 matrices")
    orthogonality = np.linalg.norm(np.swapaxes(values, -1, -2) @ values - np.eye(3), axis=(-2, -1))
    if np.any(orthogonality > 1e-5) or np.any(np.abs(np.linalg.det(values) - 1) > 1e-5):
        raise ValueError("rotation must be proper SO(3), without reflection or scale")
    return values


def read_penn_pose(path: str | Path) -> pd.DataFrame:
    """Read official time + row-major 3x4 body-to-world pose, without assumptions."""
    logger.info("READS independent optical Penn pose {}", path)
    values = np.loadtxt(path, dtype=float, ndmin=2)
    if values.shape[1] != 13 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Penn pose needs at least two finite rows of exactly 13 columns")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Penn pose timestamps must be strictly increasing")
    transforms = values[:, 1:].reshape(-1, 3, 4)
    rotations = _rotation_array(transforms[:, :, :3])
    frame = pd.DataFrame(rotations.reshape(-1, 9), columns=ROTATION_COLUMNS)
    frame.insert(0, "timestamp_s", values[:, 0])
    frame[POSITION_COLUMNS] = transforms[:, :, 3]
    frame.attrs.update(
        {"pose_direction": "world_from_rig_body", "position_unit": "m", "source": str(path)}
    )
    return frame


def align_pose_ground_truth(
    timestamps_s: np.ndarray,
    truth: pd.DataFrame,
    *,
    max_gap_s: float = 0.1,
) -> pd.DataFrame:
    """SLERP rotation and linearly interpolate position only inside valid GT support."""
    query = np.asarray(timestamps_s, dtype=float)
    times = truth["timestamp_s"].to_numpy(dtype=float)
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query timestamps must be finite and one-dimensional")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("truth timestamps must be finite and strictly increasing")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    rotations = _rotation_array(truth[ROTATION_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3))
    right = np.clip(np.searchsorted(times, query, side="left"), 0, len(times) - 1)
    exact = np.isclose(times[right], query, rtol=0, atol=1e-9)
    left = np.maximum(right - 1, 0)
    bracketed = (query >= times[0]) & (query <= times[-1])
    valid = bracketed & (exact | ((times[right] - times[left]) <= max_gap_s))
    result = pd.DataFrame({"timestamp_s": query, "gt_valid": valid})
    result[ROTATION_COLUMNS + POSITION_COLUMNS] = np.nan
    if valid.any():
        interpolated = Slerp(times, Rotation.from_matrix(rotations))(query[valid]).as_matrix()
        result.loc[valid, ROTATION_COLUMNS] = interpolated.reshape(-1, 9)
        positions = truth[POSITION_COLUMNS].to_numpy(dtype=float)
        for axis, column in enumerate(POSITION_COLUMNS):
            result.loc[valid, column] = np.interp(query[valid], times, positions[:, axis])
    return result


def _imu_arrays(timestamps_s: np.ndarray, gyro_radps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(timestamps_s, dtype=float)
    gyro = np.asarray(gyro_radps, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "IMU timestamps must be finite and strictly increasing, with at least two rows"
        )
    if gyro.shape != (len(times), 3) or not np.isfinite(gyro).all():
        raise ValueError("gyro must have finite shape (rows, 3)")
    return times, gyro


def integrate_gyro_so3(
    timestamps_s: np.ndarray,
    gyro_radps: np.ndarray,
    initial_world_from_body: np.ndarray,
) -> np.ndarray:
    """Compose body-frame midpoint increments R[k+1] = R[k] Exp(mean(omega) dt)."""
    times, gyro = _imu_arrays(timestamps_s, gyro_radps)
    initial = _rotation_array(initial_world_from_body)
    if initial.shape != (3, 3):
        raise ValueError("initial rotation must be one 3x3 matrix")
    increments = Rotation.from_rotvec(0.5 * (gyro[:-1] + gyro[1:]) * np.diff(times)[:, None])
    result = np.empty((len(times), 3, 3), dtype=float)
    current = Rotation.from_matrix(initial)
    result[0] = current.as_matrix()
    for index, increment in enumerate(increments, start=1):
        current = current * increment
        result[index] = current.as_matrix()
    return result


def integrate_strapdown(
    timestamps_s: np.ndarray,
    gyro_radps: np.ndarray,
    specific_force_mps2: np.ndarray,
    initial_world_from_body: np.ndarray,
    initial_position_m: np.ndarray,
    initial_velocity_mps: np.ndarray,
    gravity_world_mps2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Midpoint SO(3), trapezoid world acceleration; explicit initial state/gravity.

    Callers must independently validate accelerometer sign, world gravity and
    body origin before interpreting position. No GT is used during propagation.
    """
    times, gyro = _imu_arrays(timestamps_s, gyro_radps)
    force = np.asarray(specific_force_mps2, dtype=float)
    if force.shape != gyro.shape or not np.isfinite(force).all():
        raise ValueError("specific force must have finite shape (rows, 3)")
    vectors = [
        np.asarray(value, dtype=float)
        for value in (initial_position_m, initial_velocity_mps, gravity_world_mps2)
    ]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors):
        raise ValueError("position, velocity and gravity must be finite three-vectors")
    rotations = integrate_gyro_so3(times, gyro, initial_world_from_body)
    position = np.empty_like(force)
    velocity = np.empty_like(force)
    position[0], velocity[0] = vectors[:2]
    gravity = vectors[2]
    for index, dt in enumerate(np.diff(times), start=1):
        acceleration = (
            0.5 * (rotations[index - 1] @ force[index - 1] + rotations[index] @ force[index])
            + gravity
        )
        position[index] = (
            position[index - 1] + velocity[index - 1] * dt + 0.5 * acceleration * dt**2
        )
        velocity[index] = velocity[index - 1] + acceleration * dt
    return {"rotation_world_from_body": rotations, "position_m": position, "velocity_mps": velocity}


def _orientation_errors(predicted: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    error = np.swapaxes(reference, 1, 2) @ predicted
    angle = np.degrees(Rotation.from_matrix(error).magnitude())
    # World vertical expressed in body coordinates isolates gravity-direction error.
    pred_vertical = predicted[:, 2, :]
    ref_vertical = reference[:, 2, :]
    tilt = np.degrees(np.arccos(np.clip(np.sum(pred_vertical * ref_vertical, axis=1), -1, 1)))
    # Heading of body x projected on world xy; undefined near a vertical body x.
    pred_x, ref_x = predicted[:, :2, 0], reference[:, :2, 0]
    heading_valid = (np.linalg.norm(pred_x, axis=1) > 1e-6) & (np.linalg.norm(ref_x, axis=1) > 1e-6)
    heading = np.full(len(angle), np.nan)
    delta = np.arctan2(pred_x[:, 1], pred_x[:, 0]) - np.arctan2(ref_x[:, 1], ref_x[:, 0])
    heading[heading_valid] = np.abs(
        np.degrees(np.arctan2(np.sin(delta[heading_valid]), np.cos(delta[heading_valid])))
    )
    return {"geodesic_deg": angle, "tilt_deg": tilt, "heading_deg": heading}


def source_clock_origin_s(frame: pd.DataFrame) -> float:
    """Read the recorded source-to-unified clock origin; never fit a time lag."""
    if "source_timestamp_s" not in frame:
        raise ValueError("source_timestamp_s is required to verify the source clock origin")
    relative = frame["timestamp_s"].to_numpy(dtype=float)
    source = frame["source_timestamp_s"].to_numpy(dtype=float)
    if not len(source) or not np.isfinite(source).all() or not np.isfinite(relative).all():
        raise ValueError("source clock timestamps must be nonempty and finite")
    offsets = source - relative
    origin = float(offsets[0])
    if np.any(np.abs(offsets - origin) > 1e-6):
        raise ValueError(
            "source clock origin must be constant within the supplied physical recording"
        )
    identities = (
        frame["sequence_id"].astype(str).to_numpy()
        if "sequence_id" in frame
        else np.full(len(frame), "recording")
    )
    for identity in np.unique(identities):
        if np.any(np.diff(source[identities == identity]) <= 0):
            raise ValueError("source clock timestamps must be strictly increasing per sequence")
    return origin


def _resolve_gt_origin(frame: pd.DataFrame, explicit: float | None) -> float:
    recorded = source_clock_origin_s(frame) if "source_timestamp_s" in frame else None
    if explicit is None:
        return recorded if recorded is not None else 0.0
    if not np.isfinite(explicit):
        raise ValueError("source clock origin must be finite")
    if recorded is not None and abs(explicit - recorded) > 1e-6:
        raise ValueError("explicit source clock origin disagrees with recorded timestamp mapping")
    return float(explicit)


def _supported_inputs(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    prefixes: Sequence[str],
    axes: Sequence[str],
    rate_hz: float,
    max_gt_gap_s: float,
    gt_timestamp_origin_s: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, list[np.ndarray]]:
    if len(frame) < 2 or not prefixes or len(set(prefixes)) != len(prefixes):
        raise ValueError("at least two rows and unique method prefixes are required")
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be finite and positive")
    columns = [f"{axis}_{prefix}" for prefix in prefixes for axis in axes]
    missing = set(["timestamp_s", *columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"missing IMU columns: {sorted(missing)}")
    times = frame["timestamp_s"].to_numpy(dtype=float)
    aligned = align_pose_ground_truth(times + gt_timestamp_origin_s, truth, max_gap_s=max_gt_gap_s)
    common_valid = aligned["gt_valid"].to_numpy() & np.isfinite(
        frame[columns].to_numpy(dtype=float)
    ).all(axis=1)
    identity = (
        frame["sequence_id"].astype(str).to_numpy()
        if "sequence_id" in frame
        else np.full(len(frame), "recording")
    )
    breaks = np.ones(len(frame), dtype=bool)
    differences = np.diff(times)
    breaks[1:] = (
        (identity[1:] != identity[:-1])
        | (differences <= 0)
        | (differences > 1.5 / rate_hz)
        | ~common_valid[:-1]
        | ~common_valid[1:]
    )
    labels = np.cumsum(breaks)
    segments = [
        np.flatnonzero((labels == label) & common_valid)
        for label in np.unique(labels[common_valid])
    ]
    return (
        times,
        aligned,
        identity,
        common_valid,
        [indices for indices in segments if len(indices) >= 2],
    )


def evaluate_attitude_stream(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    prefixes: Sequence[str] = ("low", "virtual", "high"),
    window_samples: Sequence[int] = (80, 200, 400, 800),
    rate_hz: float = 40.0,
    max_gt_gap_s: float = 0.1,
    gt_timestamp_origin_s: float | None = None,
) -> dict[str, Any]:
    """Score fixed nonoverlapping horizons and each full continuous supported segment.

    Every supplied method shares exactly the same valid samples and reset times.
    The caller must assemble neural overlap predictions before invoking this;
    overlapping displacement predictions must never be summed into a trajectory.
    """
    sizes = tuple(int(size) for size in window_samples)
    if any(size < 2 or size != value for size, value in zip(sizes, window_samples, strict=True)):
        raise ValueError("window_samples must be integer counts of at least two")
    origin_s = _resolve_gt_origin(frame, gt_timestamp_origin_s)
    times, aligned, identity, common_valid, segments = _supported_inputs(
        frame, truth, prefixes, GYRO_AXES, rate_hz, max_gt_gap_s, origin_s
    )
    accumulators: dict[tuple[str, int], list[dict[str, Any]]] = {
        (prefix, size): [] for prefix in prefixes for size in sizes
    }
    full: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for segment_number, indices in enumerate(segments):
        segment_times = times[indices]
        reference = aligned.iloc[indices][ROTATION_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
        for prefix in prefixes:
            gyro = frame.iloc[indices][[f"{axis}_{prefix}" for axis in GYRO_AXES]].to_numpy(
                dtype=float
            )
            predicted = integrate_gyro_so3(segment_times, gyro, reference[0])
            errors = _orientation_errors(predicted, reference)
            full.append(
                {
                    "method": prefix,
                    "sequence_id": str(identity[indices[0]]),
                    "segment": segment_number,
                    "rows": len(indices),
                    "start_s": float(segment_times[0]),
                    "end_s": float(segment_times[-1]),
                    "duration_s": float(segment_times[-1] - segment_times[0]),
                    "endpoint_deg": float(errors["geodesic_deg"][-1]),
                    "rmse_deg": float(np.sqrt(np.mean(errors["geodesic_deg"] ** 2))),
                    "tilt_rmse_deg": float(np.sqrt(np.mean(errors["tilt_deg"] ** 2))),
                }
            )
            for size in sizes:
                for start in range(0, len(indices) - size + 1, size):
                    stop = start + size
                    prediction = integrate_gyro_so3(
                        segment_times[start:stop], gyro[start:stop], reference[start]
                    )
                    errors = _orientation_errors(prediction, reference[start:stop])
                    detail = {
                        "method": prefix,
                        "sequence_id": str(identity[indices[0]]),
                        "segment": segment_number,
                        "samples": size,
                        "start_s": float(segment_times[start]),
                        "end_s": float(segment_times[stop - 1]),
                        "endpoint_deg": float(errors["geodesic_deg"][-1]),
                        "sum_squared_deg": float(np.sum(errors["geodesic_deg"] ** 2)),
                        "tilt_endpoint_deg": float(errors["tilt_deg"][-1]),
                        "heading_endpoint_deg": float(errors["heading_deg"][-1])
                        if np.isfinite(errors["heading_deg"][-1])
                        else None,
                    }
                    accumulators[(prefix, size)].append(detail)
                    details.append(detail)
    windows = []
    for (prefix, size), entries in accumulators.items():
        endpoints = np.asarray([entry["endpoint_deg"] for entry in entries])
        heading = [
            entry["heading_endpoint_deg"]
            for entry in entries
            if entry["heading_endpoint_deg"] is not None
        ]
        windows.append(
            {
                "method": prefix,
                "samples": size,
                "horizon_s": (size - 1) / rate_hz,
                "count": len(entries),
                "endpoint_rmse_deg": float(np.sqrt(np.mean(endpoints**2)))
                if len(entries)
                else None,
                "endpoint_median_deg": float(np.median(endpoints)) if len(entries) else None,
                "endpoint_p95_deg": float(np.quantile(endpoints, 0.95)) if len(entries) else None,
                "all_samples_rmse_deg": float(
                    np.sqrt(
                        sum(entry["sum_squared_deg"] for entry in entries) / (size * len(entries))
                    )
                )
                if entries
                else None,
                "tilt_endpoint_rmse_deg": float(
                    np.sqrt(np.mean([entry["tilt_endpoint_deg"] ** 2 for entry in entries]))
                )
                if entries
                else None,
                "heading_endpoint_rmse_deg": float(np.sqrt(np.mean(np.asarray(heading) ** 2)))
                if heading
                else None,
                "heading_valid_windows": len(heading),
            }
        )
    return {
        "protocol": {
            "algorithm": "midpoint_body_gyro_SO3_composition",
            "ground_truth": "independent_optical_Penn_AprilTag_rig_pose",
            "frame": "rig_body_at_rig_origin",
            "timestamp_column": "timestamp_s",
            "gt_timestamp_origin_s": origin_s,
            "gt_time_mapping": "raw_official_gt_time = timestamp_s + constant_recorded_source_clock_origin",
            "initialization": "GT_orientation_once_at_each_predefined_window_or_continuous_segment_start",
            "ground_truth_during_propagation": False,
            "window_policy": "nonoverlapping_fixed_sample_counts_starting_at_common_segment_start",
            "rate_hz": rate_hz,
            "max_gt_gap_s": max_gt_gap_s,
            "gravity_or_magnetometer_updates": False,
            "interpretation": "oracle_initialized_gyro_propagation_diagnostic_not_deployable_VIO; full_recordings_are_individual_continuous_supported_segments",
        },
        "support": {
            "input_rows": len(frame),
            "common_valid_rows": int(common_valid.sum()),
            "skipped_rows": int((~common_valid).sum()),
            "continuous_segments": len(segments),
            "common_methods": list(prefixes),
        },
        "windows": windows,
        "window_details": details,
        "full_recordings": full,
    }


def _past_initial_velocity(
    truth: pd.DataFrame, start_s: float, span_s: float, max_gap_s: float
) -> np.ndarray | None:
    times = truth["timestamp_s"].to_numpy(dtype=float)
    last = int(np.searchsorted(times, start_s + 1e-10, side="right") - 1)
    if last < 1:
        return None
    prior_s = times[last] - span_s
    if prior_s < times[0] - 1e-10:
        return None
    first = max(0, int(np.searchsorted(times, prior_s, side="right") - 1))
    if np.any(np.diff(times[first : last + 1]) > max_gap_s):
        return None
    positions = truth[POSITION_COLUMNS].to_numpy(dtype=float)
    prior = np.array(
        [
            np.interp(prior_s, times[first : last + 1], positions[first : last + 1, axis])
            for axis in range(3)
        ]
    )
    return (positions[last] - prior) / span_s


def evaluate_strapdown_stream(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    prefixes: Sequence[str] = ("low", "virtual", "high"),
    window_samples: Sequence[int] = (80, 200, 400, 800),
    rate_hz: float = 40.0,
    max_gt_gap_s: float = 0.1,
    gt_timestamp_origin_s: float | None = None,
    initial_velocity_span_s: float = 0.1,
    gravity_world_mps2: Sequence[float] = (0.0, 0.0, -9.80665),
) -> dict[str, Any]:
    """Diagnostic six-axis propagation against real independent optical positions.

    Initial q/p come from optical GT once. Initial velocity uses a fixed 0.1s
    position difference ending at the latest optical pose no later than start.
    Missing past support skips that fixed window without moving its start time.
    This does not estimate a deployable initial state, gravity, or sensor biases.
    """
    if not np.isfinite(initial_velocity_span_s) or initial_velocity_span_s <= 0:
        raise ValueError("initial_velocity_span_s must be positive")
    sizes = tuple(int(size) for size in window_samples)
    if any(size < 2 or size != value for size, value in zip(sizes, window_samples, strict=True)):
        raise ValueError("window_samples must be integer counts of at least two")
    gravity = np.asarray(gravity_world_mps2, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity_world_mps2 must be a finite three-vector")
    origin_s = _resolve_gt_origin(frame, gt_timestamp_origin_s)
    times, aligned, identity, common_valid, segments = _supported_inputs(
        frame, truth, prefixes, [*GYRO_AXES, "ax", "ay", "az"], rate_hz, max_gt_gap_s, origin_s
    )
    accumulators: dict[tuple[str, int], list[dict[str, Any]]] = {
        (prefix, size): [] for prefix in prefixes for size in sizes
    }
    details: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    skipped_windows, skipped_segments = 0, 0
    for segment_number, indices in enumerate(segments):
        reference_rotation = (
            aligned.iloc[indices][ROTATION_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
        )
        reference_position = aligned.iloc[indices][POSITION_COLUMNS].to_numpy(dtype=float)
        specs = [(0, len(indices), None)] + [
            (start, start + size, size)
            for size in sizes
            for start in range(0, len(indices) - size + 1, size)
        ]
        for start, stop, size in specs:
            start_time = float(times[indices[start]])
            velocity = _past_initial_velocity(
                truth, start_time + origin_s, initial_velocity_span_s, max_gt_gap_s
            )
            if velocity is None:
                skipped_segments += int(size is None)
                skipped_windows += int(size is not None)
                continue
            for prefix in prefixes:
                part = frame.iloc[indices[start:stop]]
                predicted = integrate_strapdown(
                    times[indices[start:stop]],
                    part[[f"{axis}_{prefix}" for axis in GYRO_AXES]].to_numpy(dtype=float),
                    part[[f"{axis}_{prefix}" for axis in ("ax", "ay", "az")]].to_numpy(dtype=float),
                    reference_rotation[start],
                    reference_position[start],
                    velocity,
                    gravity,
                )
                error = predicted["position_m"] - reference_position[start:stop]
                error_norm = np.linalg.norm(error, axis=1)
                horizontal = np.linalg.norm(error[:, :2], axis=1)
                entry = {
                    "method": prefix,
                    "sequence_id": str(identity[indices[0]]),
                    "segment": segment_number,
                    "samples": stop - start,
                    "start_s": start_time,
                    "end_s": float(times[indices[stop - 1]]),
                    "duration_s": float(times[indices[stop - 1]] - start_time),
                    "position_endpoint_m": float(error_norm[-1]),
                    "horizontal_endpoint_m": float(horizontal[-1]),
                    "position_rmse_m": float(np.sqrt(np.mean(error_norm**2))),
                    "position_sum_squared_m": float(np.sum(error_norm**2)),
                    "initial_velocity_mps": velocity.tolist(),
                }
                if size is None:
                    full.append(entry)
                else:
                    accumulators[(prefix, size)].append(entry)
                    details.append(entry)
    windows = []
    for (prefix, size), entries in accumulators.items():
        endpoints = np.asarray([entry["position_endpoint_m"] for entry in entries])
        horizontal = np.asarray([entry["horizontal_endpoint_m"] for entry in entries])
        windows.append(
            {
                "method": prefix,
                "samples": size,
                "horizon_s": (size - 1) / rate_hz,
                "count": len(entries),
                "position_endpoint_rmse_m": float(np.sqrt(np.mean(endpoints**2)))
                if entries
                else None,
                "horizontal_endpoint_rmse_m": float(np.sqrt(np.mean(horizontal**2)))
                if entries
                else None,
                "position_endpoint_median_m": float(np.median(endpoints)) if entries else None,
                "position_endpoint_p95_m": float(np.quantile(endpoints, 0.95)) if entries else None,
                "all_samples_position_rmse_m": float(
                    np.sqrt(
                        sum(entry["position_sum_squared_m"] for entry in entries)
                        / (size * len(entries))
                    )
                )
                if entries
                else None,
            }
        )
    return {
        "protocol": {
            "algorithm": "midpoint_body_gyro_SO3_trapezoid_world_acceleration",
            "ground_truth": "independent_optical_Penn_AprilTag_rig_pose",
            "frame": "rig_body_at_rig_origin",
            "timestamp_column": "timestamp_s",
            "gt_timestamp_origin_s": origin_s,
            "gt_time_mapping": "raw_official_gt_time = timestamp_s + constant_recorded_source_clock_origin",
            "initialization": "GT_orientation_position_velocity_once_at_fixed_window_or_segment_start",
            "initial_velocity": f"preceding_{initial_velocity_span_s:g}s_optical_position_difference_once",
            "initial_velocity_timestamp": "latest_optical_pose_at_or_before_window_start; entire difference uses past optical poses",
            "ground_truth_during_propagation": False,
            "gravity_world_mps2": gravity.tolist(),
            "window_policy": "nonoverlapping_fixed_sample_counts_starting_at_common_segment_start",
            "rate_hz": rate_hz,
            "max_gt_gap_s": max_gt_gap_s,
            "interpretation": "oracle_initialized_inertial_diagnostic_not_deployable_VIO; position_accuracy_limited_by_optical_GT_and_0.1s_initial_velocity_difference",
        },
        "support": {
            "input_rows": len(frame),
            "common_valid_rows": int(common_valid.sum()),
            "skipped_rows": int((~common_valid).sum()),
            "continuous_segments": len(segments),
            "window_initializations_skipped": skipped_windows,
            "full_segment_initializations_skipped": skipped_segments,
            "common_methods": list(prefixes),
        },
        "windows": windows,
        "window_details": details,
        "full_recordings": full,
    }
