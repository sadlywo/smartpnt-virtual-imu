from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rosbags.highlevel import AnyReader

GPS_WEEK_SECONDS = 604800.0

GT_USECOLS = [
    "Week",
    "GPSTime",
    "Latitude",
    "Longitude",
    "H-Ell",
    "Q",
    "AmbStatus",
    "X-ECEF",
    "Y-ECEF",
    "Z-ECEF",
    "X-LL",
    "Y-LL",
    "Z-LL",
    "VX-ECEF",
    "VY-ECEF",
    "VZ-ECEF",
    "VEast",
    "VNorth",
    "VUp",
    "Heading",
    "Pitch",
    "Roll",
    "HdngSD",
    "PitchSD",
    "RollSD",
    "NS",
    "GP",
    "GL",
    "BD",
    "GA",
    "QZ",
    "HDOP",
    "VDOP",
    "PDOP",
    "iFlag",
    "L1Rms",
    "E-Sep",
    "N-Sep",
    "H-Sep",
    "HdngSep",
    "PtchSep",
    "RollSep",
]

GT_RENAME = {
    "Week": "gps_week",
    "GPSTime": "gps_sow_s",
    "Latitude": "latitude_deg",
    "Longitude": "longitude_deg",
    "H-Ell": "height_ell_m",
    "Q": "quality",
    "AmbStatus": "ambiguity_status",
    "X-ECEF": "x_ecef_m",
    "Y-ECEF": "y_ecef_m",
    "Z-ECEF": "z_ecef_m",
    "X-LL": "x_local_m",
    "Y-LL": "y_local_m",
    "Z-LL": "z_local_m",
    "VX-ECEF": "vx_ecef_mps",
    "VY-ECEF": "vy_ecef_mps",
    "VZ-ECEF": "vz_ecef_mps",
    "VEast": "v_east_mps",
    "VNorth": "v_north_mps",
    "VUp": "v_up_mps",
    "Heading": "heading_deg",
    "Pitch": "pitch_deg",
    "Roll": "roll_deg",
    "HdngSD": "heading_sd_deg",
    "PitchSD": "pitch_sd_deg",
    "RollSD": "roll_sd_deg",
    "NS": "num_sats",
    "GP": "gps_sats",
    "GL": "glonass_sats",
    "BD": "beidou_sats",
    "GA": "galileo_sats",
    "QZ": "qzss_sats",
    "HDOP": "hdop",
    "VDOP": "vdop",
    "PDOP": "pdop",
    "iFlag": "ins_flag",
    "L1Rms": "l1_rms_m",
    "E-Sep": "east_sep_m",
    "N-Sep": "north_sep_m",
    "H-Sep": "height_sep_m",
    "HdngSep": "heading_sep_deg",
    "PtchSep": "pitch_sep_deg",
    "RollSep": "roll_sep_deg",
}

IMU_COLUMNS = [
    "timestamp_s",
    "dt_s",
    "dtheta_x_rad",
    "dtheta_y_rad",
    "dtheta_z_rad",
    "dv_x_mps",
    "dv_y_mps",
    "dv_z_mps",
    "gx_radps",
    "gy_radps",
    "gz_radps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
    "sequence_id",
]

GT_STRING_COLUMNS = {"ambiguity_status", "ins_flag", "sequence_id"}


def _write_frame(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _counter_update(counter: Counter[str], values: pd.Series) -> None:
    counter.update(str(value) for value in values.dropna().astype(str).to_list())


def prepare_groundtruth(path: Path, out_path: Path, sequence_id: str, chunksize: int) -> dict[str, Any]:
    if out_path.exists():
        out_path.unlink()
    writer: pq.ParquetWriter | None = None
    rows = 0
    start = None
    end = None
    quality_counts: Counter[str] = Counter()
    ambiguity_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    dt_samples: list[np.ndarray] = []

    reader = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=[1],
        usecols=GT_USECOLS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk.rename(columns=GT_RENAME)
        for column in chunk.columns:
            if column not in GT_STRING_COLUMNS:
                chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype(float)
        chunk["gps_week"] = pd.to_numeric(chunk["gps_week"], errors="coerce").astype(float)
        chunk["gps_sow_s"] = pd.to_numeric(chunk["gps_sow_s"], errors="coerce")
        chunk["timestamp_s"] = chunk["gps_week"].astype(float) * GPS_WEEK_SECONDS + chunk["gps_sow_s"]
        chunk["sequence_id"] = sequence_id
        chunk = chunk.dropna(subset=["timestamp_s"])
        ordered_cols = [
            "timestamp_s",
            "gps_week",
            "gps_sow_s",
            "latitude_deg",
            "longitude_deg",
            "height_ell_m",
            "x_ecef_m",
            "y_ecef_m",
            "z_ecef_m",
            "x_local_m",
            "y_local_m",
            "z_local_m",
            "vx_ecef_mps",
            "vy_ecef_mps",
            "vz_ecef_mps",
            "v_east_mps",
            "v_north_mps",
            "v_up_mps",
            "heading_deg",
            "pitch_deg",
            "roll_deg",
            "heading_sd_deg",
            "pitch_sd_deg",
            "roll_sd_deg",
            "quality",
            "ambiguity_status",
            "ins_flag",
            "num_sats",
            "gps_sats",
            "glonass_sats",
            "beidou_sats",
            "galileo_sats",
            "qzss_sats",
            "hdop",
            "vdop",
            "pdop",
            "l1_rms_m",
            "east_sep_m",
            "north_sep_m",
            "height_sep_m",
            "heading_sep_deg",
            "pitch_sep_deg",
            "roll_sep_deg",
            "sequence_id",
        ]
        chunk = chunk[ordered_cols]
        timestamps = chunk["timestamp_s"].to_numpy(float)
        if len(timestamps):
            start = float(timestamps[0]) if start is None else min(start, float(timestamps[0]))
            end = float(timestamps[-1]) if end is None else max(end, float(timestamps[-1]))
            if len(timestamps) > 1 and len(dt_samples) < 8:
                dt_samples.append(np.diff(timestamps[: min(len(timestamps), 2000)]))
        _counter_update(quality_counts, chunk["quality"])
        _counter_update(ambiguity_counts, chunk["ambiguity_status"])
        _counter_update(flag_counts, chunk["ins_flag"])
        writer = _write_frame(writer, chunk, out_path)
        rows += len(chunk)

    if writer is not None:
        writer.close()
    duration = None if start is None or end is None else end - start
    dt = np.concatenate(dt_samples) if dt_samples else np.array([])
    return {
        "sequence_id": sequence_id,
        "source": str(path),
        "output": str(out_path),
        "rows": rows,
        "start_s": start,
        "end_s": end,
        "duration_s": duration,
        "rate_hz_est": None if not duration or duration <= 0 else (rows - 1) / duration,
        "median_dt_s": None if len(dt) == 0 else float(np.median(dt)),
        "quality_counts": dict(quality_counts.most_common()),
        "ambiguity_counts": dict(ambiguity_counts.most_common()),
        "ins_flag_counts": dict(flag_counts.most_common()),
    }


def prepare_imu_bag(
    path: Path,
    out_path: Path,
    sequence_id: str,
    chunk_rows: int,
    field_mode: str,
    default_dt_s: float,
) -> dict[str, Any]:
    if out_path.exists():
        out_path.unlink()
    writer: pq.ParquetWriter | None = None
    rows = 0
    start = None
    end = None
    topic = None
    msgtype = None
    buffer: dict[str, list[Any]] = {name: [] for name in IMU_COLUMNS}
    dt_samples: list[np.ndarray] = []
    last_t: float | None = None

    def flush() -> None:
        nonlocal writer, buffer
        if not buffer["timestamp_s"]:
            return
        frame = pd.DataFrame(buffer)
        writer = _write_frame(writer, frame, out_path)
        buffer = {name: [] for name in IMU_COLUMNS}

    with AnyReader([path]) as reader:
        candidates = [conn for conn in reader.connections if conn.topic == "/imu_raw"]
        if not candidates:
            candidates = [conn for conn in reader.connections if "imu" in conn.topic.lower()]
        if not candidates:
            raise ValueError(f"No IMU-like topic found in {path}")
        conn = candidates[0]
        topic = conn.topic
        msgtype = conn.msgtype
        for _conn, timestamp, rawdata in reader.messages(connections=[conn]):
            msg = reader.deserialize(rawdata, conn.msgtype)
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            if stamp is not None:
                t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            else:
                t = float(timestamp) * 1e-9
            dt_s = default_dt_s if last_t is None else max(t - last_t, 1e-9)
            raw_gx = float(msg.angular_velocity.x)
            raw_gy = float(msg.angular_velocity.y)
            raw_gz = float(msg.angular_velocity.z)
            raw_ax = float(msg.linear_acceleration.x)
            raw_ay = float(msg.linear_acceleration.y)
            raw_az = float(msg.linear_acceleration.z)
            if field_mode == "incremental":
                dtheta_x, dtheta_y, dtheta_z = raw_gx, raw_gy, raw_gz
                dv_x, dv_y, dv_z = raw_ax, raw_ay, raw_az
                gx, gy, gz = raw_gx / dt_s, raw_gy / dt_s, raw_gz / dt_s
                ax, ay, az = raw_ax / dt_s, raw_ay / dt_s, raw_az / dt_s
            else:
                gx, gy, gz = raw_gx, raw_gy, raw_gz
                ax, ay, az = raw_ax, raw_ay, raw_az
                dtheta_x, dtheta_y, dtheta_z = raw_gx * dt_s, raw_gy * dt_s, raw_gz * dt_s
                dv_x, dv_y, dv_z = raw_ax * dt_s, raw_ay * dt_s, raw_az * dt_s
            buffer["timestamp_s"].append(t)
            buffer["dt_s"].append(dt_s)
            buffer["dtheta_x_rad"].append(dtheta_x)
            buffer["dtheta_y_rad"].append(dtheta_y)
            buffer["dtheta_z_rad"].append(dtheta_z)
            buffer["dv_x_mps"].append(dv_x)
            buffer["dv_y_mps"].append(dv_y)
            buffer["dv_z_mps"].append(dv_z)
            buffer["gx_radps"].append(gx)
            buffer["gy_radps"].append(gy)
            buffer["gz_radps"].append(gz)
            buffer["ax_mps2"].append(ax)
            buffer["ay_mps2"].append(ay)
            buffer["az_mps2"].append(az)
            buffer["sequence_id"].append(sequence_id)
            start = t if start is None else min(start, t)
            end = t if end is None else max(end, t)
            if last_t is not None and len(dt_samples) < 4000:
                dt_samples.append(np.array([t - last_t]))
            last_t = t
            rows += 1
            if len(buffer["timestamp_s"]) >= chunk_rows:
                flush()
    flush()
    if writer is not None:
        writer.close()
    duration = None if start is None or end is None else end - start
    dt = np.concatenate(dt_samples) if dt_samples else np.array([])
    return {
        "sequence_id": sequence_id,
        "source": str(path),
        "output": str(out_path),
        "topic": topic,
        "msgtype": msgtype,
        "field_mode": field_mode,
        "default_dt_s": default_dt_s,
        "rows": rows,
        "start_s": start,
        "end_s": end,
        "duration_s": duration,
        "rate_hz_est": None if not duration or duration <= 0 else (rows - 1) / duration,
        "median_dt_s": None if len(dt) == 0 else float(np.median(dt)),
    }


def align_imu_to_gt(imu_path: Path, gt_path: Path, out_path: Path, sequence_id: str) -> dict[str, Any]:
    imu = pd.read_parquet(imu_path).sort_values("timestamp_s")
    gt = pd.read_parquet(gt_path).sort_values("timestamp_s")
    start = max(float(imu["timestamp_s"].min()), float(gt["timestamp_s"].min()))
    end = min(float(imu["timestamp_s"].max()), float(gt["timestamp_s"].max()))
    gt = gt[(gt["timestamp_s"] >= start) & (gt["timestamp_s"] <= end)].copy()
    imu = imu[(imu["timestamp_s"] >= start) & (imu["timestamp_s"] <= end)].copy()
    if gt.empty or imu.empty:
        raise ValueError(f"No overlap between IMU and GT for {sequence_id}")
    imu_t = imu["timestamp_s"].to_numpy(float)
    gt_t = gt["timestamp_s"].to_numpy(float)
    aligned = pd.DataFrame(
        {
            "timestamp_s": gt_t,
            "gx_radps": np.interp(gt_t, imu_t, imu["gx_radps"].to_numpy(float)),
            "gy_radps": np.interp(gt_t, imu_t, imu["gy_radps"].to_numpy(float)),
            "gz_radps": np.interp(gt_t, imu_t, imu["gz_radps"].to_numpy(float)),
            "ax_mps2": np.interp(gt_t, imu_t, imu["ax_mps2"].to_numpy(float)),
            "ay_mps2": np.interp(gt_t, imu_t, imu["ay_mps2"].to_numpy(float)),
            "az_mps2": np.interp(gt_t, imu_t, imu["az_mps2"].to_numpy(float)),
            "gt_x_local_m": gt["x_local_m"].to_numpy(float),
            "gt_y_local_m": gt["y_local_m"].to_numpy(float),
            "gt_z_local_m": gt["z_local_m"].to_numpy(float),
            "gt_v_east_mps": gt["v_east_mps"].to_numpy(float),
            "gt_v_north_mps": gt["v_north_mps"].to_numpy(float),
            "gt_v_up_mps": gt["v_up_mps"].to_numpy(float),
            "gt_heading_deg": gt["heading_deg"].to_numpy(float),
            "gt_pitch_deg": gt["pitch_deg"].to_numpy(float),
            "gt_roll_deg": gt["roll_deg"].to_numpy(float),
            "gt_quality": gt["quality"].astype(str).to_numpy(),
            "gt_ambiguity_status": gt["ambiguity_status"].astype(str).to_numpy(),
            "gt_ins_flag": gt["ins_flag"].astype(str).to_numpy(),
            "sequence_id": sequence_id,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(out_path, index=False, compression="zstd")
    accel_norm = np.linalg.norm(aligned[["ax_mps2", "ay_mps2", "az_mps2"]].to_numpy(float), axis=1)
    gyro_norm = np.linalg.norm(aligned[["gx_radps", "gy_radps", "gz_radps"]].to_numpy(float), axis=1)
    duration = end - start
    return {
        "sequence_id": sequence_id,
        "source_imu": str(imu_path),
        "source_groundtruth": str(gt_path),
        "output": str(out_path),
        "rows": int(len(aligned)),
        "start_s": start,
        "end_s": end,
        "duration_s": duration,
        "rate_hz_est": None if duration <= 0 else (len(aligned) - 1) / duration,
        "accel_norm_median": float(np.median(accel_norm)),
        "accel_norm_p95": float(np.percentile(accel_norm, 95)),
        "gyro_norm_median": float(np.median(gyro_norm)),
        "gyro_norm_p95": float(np.percentile(gyro_norm, 95)),
        "quality_counts": {str(k): int(v) for k, v in aligned["gt_quality"].value_counts().to_dict().items()},
        "ins_flag_counts": {str(k): int(v) for k, v in aligned["gt_ins_flag"].value_counts().to_dict().items()},
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# SmartPNT-MSF Real Data Processing Summary", ""]
    lines.append(f"- Raw root: `{summary['raw_root']}`")
    lines.append(f"- Processed root: `{summary['processed_root']}`")
    lines.append(f"- IMU field mode: `{summary['imu_field_mode']}`")
    lines.append("")
    for section in ["groundtruth", "imu", "aligned"]:
        lines.append(f"## {section.title()}")
        lines.append("")
        if section == "aligned":
            header = "| sequence | rows | duration s | rate Hz | accel norm median | gyro norm median | output |"
            lines.append(header)
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for item in summary[section]:
                lines.append(
                    f"| {item['sequence_id']} | {item['rows']} | {item['duration_s']:.3f} | "
                    f"{item['rate_hz_est']:.3f} | {item['accel_norm_median']:.6g} | "
                    f"{item['gyro_norm_median']:.6g} | `{item['output']}` |"
                )
        else:
            header = "| sequence | rows | duration s | rate Hz | output |"
            lines.append(header)
            lines.append("|---|---:|---:|---:|---|")
            for item in summary[section]:
                lines.append(
                    f"| {item['sequence_id']} | {item['rows']} | {item['duration_s']:.3f} | "
                    f"{item['rate_hz_est']:.3f} | `{item['output']}` |"
                )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- IMU topic parsed from each bag: `/imu_raw`.")
    lines.append("- Alignment uses GT timestamps as the output clock and linear interpolation for converted IMU rate channels.")
    lines.append("- The default SmartPNT-MSF setting treats ROS IMU fields as delta-angle and delta-velocity increments, then divides by raw IMU `dt_s` to produce `gx_radps` and `ax_mps2` columns.")
    lines.append("- This incremental interpretation is inferred from the raw delta-velocity norm being close to `9.81 * dt`; verify against the upstream driver documentation before making final sensor-unit claims.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/smartpnt_msf/01_extracted/SmartPNT-MSF-Database"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/smartpnt_msf"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/smartpnt_msf"))
    parser.add_argument("--gt-chunksize", type=int, default=500_000)
    parser.add_argument("--imu-chunk-rows", type=int, default=200_000)
    parser.add_argument(
        "--imu-field-mode",
        choices=["incremental", "rate"],
        default="incremental",
        help="Interpret ROS IMU fields as delta-angle/delta-velocity increments or physical rates.",
    )
    parser.add_argument("--default-imu-dt-s", type=float, default=0.005)
    args = parser.parse_args()

    raw_root = args.raw_root
    processed_root = args.processed_root
    report_root = args.report_root
    gt_out = processed_root / "groundtruth"
    imu_out = processed_root / "imu_raw"
    aligned_out = processed_root / "imu_gt_aligned"

    summary: dict[str, Any] = {
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "imu_field_mode": args.imu_field_mode,
        "default_imu_dt_s": args.default_imu_dt_s,
        "groundtruth": [],
        "imu": [],
        "aligned": [],
    }
    for traj_dir in sorted(raw_root.glob("Traj*")):
        if not traj_dir.is_dir():
            continue
        sequence_id = traj_dir.name
        gt_files = sorted(traj_dir.glob("groundtruth/Mini_GroundTruth.txt"))
        imu_files = sorted(traj_dir.glob("imu/*.bag"))
        if not gt_files or not imu_files:
            continue
        gt_path = gt_out / f"{sequence_id}_groundtruth.parquet"
        imu_path = imu_out / f"{sequence_id}_imu_raw.parquet"
        aligned_path = aligned_out / f"{sequence_id}_imu_gt_aligned.parquet"
        gt_summary = prepare_groundtruth(gt_files[0], gt_path, sequence_id, args.gt_chunksize)
        imu_summary = prepare_imu_bag(
            imu_files[0],
            imu_path,
            sequence_id,
            args.imu_chunk_rows,
            args.imu_field_mode,
            args.default_imu_dt_s,
        )
        aligned_summary = align_imu_to_gt(imu_path, gt_path, aligned_path, sequence_id)
        summary["groundtruth"].append(gt_summary)
        summary["imu"].append(imu_summary)
        summary["aligned"].append(aligned_summary)

    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "real_data_processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_root / "real_data_processing_summary.md").write_text(_markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
