"""Attach navigation truth to native IMU without pre-decimating its signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def native_smartpnt_frame(imu: pd.DataFrame, groundtruth: pd.DataFrame) -> pd.DataFrame:
    """Use a fixed ENU origin and reject truth gaps and non-Fixed brackets.

    IMU rates are the existing parser's delta-angle/delta-velocity divided by
    packet duration. That interpretation remains an explicit source assumption.
    Navigation truth is interpolated; measured IMU signals are never interpolated
    or downsampled here. The common-rate stage performs their anti-alias filtering.
    """
    required = [
        "timestamp_s",
        "latitude_deg",
        "longitude_deg",
        "x_ecef_m",
        "y_ecef_m",
        "z_ecef_m",
        "vx_ecef_mps",
        "vy_ecef_mps",
        "vz_ecef_mps",
        "heading_deg",
        "pitch_deg",
        "roll_deg",
        "ambiguity_status",
    ]
    missing = sorted(set(required) - set(groundtruth))
    if missing:
        raise ValueError(f"Native SmartPNT truth missing fields: {missing}")
    gt = groundtruth.drop_duplicates("timestamp_s", keep="first").reset_index(drop=True)
    times = gt.timestamp_s.to_numpy(dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Native navigation timestamps must be finite and increasing")
    position = gt[["x_ecef_m", "y_ecef_m", "z_ecef_m"]].to_numpy(dtype=float)
    velocity = gt[["vx_ecef_mps", "vy_ecef_mps", "vz_ecef_mps"]].to_numpy(dtype=float)
    lat, lon = np.deg2rad(gt.iloc[0][["latitude_deg", "longitude_deg"]].to_numpy(dtype=float))
    rotation = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)],
            [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)],
        ]
    )
    origin = position[0].copy()
    if not np.isfinite(np.r_[origin, lat, lon]).all():
        raise ValueError("Native navigation ENU origin must be finite")
    local_position, local_velocity = (position - origin) @ rotation.T, velocity @ rotation.T
    out = imu.loc[imu.timestamp_s.between(times[0], times[-1])].reset_index(drop=True).copy()
    sample_times = out.timestamp_s.to_numpy(dtype=float)
    if not len(out) or not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) < 0):
        raise ValueError("Native IMU timestamps must overlap navigation truth and increase")
    right = np.searchsorted(times, sample_times, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    exact = np.isclose(times[right], sample_times, atol=1e-6, rtol=0)
    left = np.where(exact, right, left)
    truth_period = float(np.median(np.diff(times)))
    fixed = gt.ambiguity_status.eq("Fixed").to_numpy()
    finite = np.isfinite(
        np.c_[
            position, velocity, gt[["heading_deg", "pitch_deg", "roll_deg"]].to_numpy(dtype=float)
        ]
    ).all(axis=1)
    out["_label_valid"] = (
        fixed[left]
        & fixed[right]
        & finite[left]
        & finite[right]
        & ((times[right] - times[left]) <= 1.5 * truth_period)
    )
    out["gt_ambiguity_status"] = np.where(fixed[left] & fixed[right], "Fixed", "NotFixed")
    for index, axis in enumerate(("x", "y", "z")):
        out[f"gt_{axis}_local_m"] = np.interp(sample_times, times, local_position[:, index])
    for index, axis in enumerate(("east", "north", "up")):
        out[f"gt_v_{axis}_mps"] = np.interp(sample_times, times, local_velocity[:, index])
    for angle in ("heading", "pitch", "roll"):
        truth = gt[f"{angle}_deg"].to_numpy(dtype=float)
        indices = np.flatnonzero(np.isfinite(truth))
        cuts = (
            np.flatnonzero((np.diff(indices) > 1) | (np.diff(times[indices]) > 1.5 * truth_period))
            + 1
        )
        value = np.full(len(sample_times), np.nan)
        for selected in np.split(indices, cuts):
            if not len(selected):
                continue
            supported = (sample_times >= times[selected[0]]) & (sample_times <= times[selected[-1]])
            unwrapped = np.unwrap(np.deg2rad(truth[selected]))
            value[supported] = np.rad2deg(
                np.interp(sample_times[supported], times[selected], unwrapped)
            )
        out[f"gt_{angle}_deg"] = value % 360 if angle == "heading" else (value + 180) % 360 - 180
    for column in ("x_local_m", "y_local_m", "z_local_m", "v_east_mps", "v_north_mps", "v_up_mps"):
        if column in gt:
            out[f"upstream_{column}"] = np.interp(
                sample_times, times, gt[column].to_numpy(dtype=float)
            )
    for column in ("quality", "ins_flag", "numsats"):
        if column in gt:
            out[f"gt_{column}"] = gt[column].to_numpy()[right]
    out.attrs["navigation_frame"] = {
        "convention": "fixed_local_ENU_from_ECEF",
        "origin_ecef_m": origin.tolist(),
        "origin_latitude_deg": float(gt.latitude_deg.iloc[0]),
        "origin_longitude_deg": float(gt.longitude_deg.iloc[0]),
        "position_source": "ECEF position; upstream X-LL/Y-LL/Z-LL retained for audit",
        "velocity_source": "ECEF velocity rotated into the same fixed ENU frame",
        "transform_source": "https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates",
    }
    out.attrs["unit_interpretation"] = {
        "imu_rates": "parsed delta-angle/delta-velocity divided by packet dt",
        "status": "parser interpretation supported by dt/gravity sanity checks; device grade unspecified",
        "time_clock": "shared upstream GPS time representation; not asserted to be Unix time",
    }
    return out
