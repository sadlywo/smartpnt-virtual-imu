"""Independent source-stream adapters for the strict multi-IMU pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

IMU_COLUMNS = [
    "timestamp_s",
    "gx",
    "gy",
    "gz",
    "ax",
    "ay",
    "az",
    "frame_id",
    "record_timestamp_s",
    "source_file",
]


def read_penn_vector_file(path: Path, value_columns: list[str]) -> pd.DataFrame:
    """Read one Penn timestamped vector stream without changing source order."""

    if len(value_columns) != 3 or len(set(value_columns)) != 3:
        raise ValueError("value_columns must contain three unique axis names")
    columns = ["timestamp_s", *value_columns]
    logger.info("READS Penn vector stream {}", path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"Penn vector stream {path} must have {len(columns)} columns, "
            f"found {frame.shape[1]}"
        )
    frame.columns = columns
    _attach_timestamp_diagnostics(frame, path)
    return frame


def read_penn_vi_imu(path: Path) -> pd.DataFrame:
    """Read a Penn VI Sensor ADIS16488 six-axis IMU text stream."""

    columns = ["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"]
    logger.info("READS Penn VI Sensor IMU stream {}", path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, dtype=float)
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"Penn VI IMU stream {path} must have {len(columns)} columns, "
            f"found {frame.shape[1]}"
        )
    frame.columns = columns
    _attach_timestamp_diagnostics(frame, path)
    return frame


def extract_penn_sequence(
    root: Path,
    sequence: str,
    cache_root: Path,
) -> dict[str, Path]:
    """Extract the three independent Penn streams for one shared sequence."""

    _safe_segment(sequence, "Penn sequence")
    tango_dir = _first_existing_directory(
        _penn_sequence_candidates(root, sequence, "tango_top")
    )
    vi_dir = _first_existing_directory(
        _penn_sequence_candidates(root, sequence, "visensor")
    )
    frames = {
        "tango_accel": read_penn_vector_file(
            tango_dir / "accelerometer.txt", ["ax", "ay", "az"]
        ),
        "tango_gyro": read_penn_vector_file(
            tango_dir / "gyroscope.txt", ["gx", "gy", "gz"]
        ),
        "vi_imu": read_penn_vi_imu(vi_dir / "imu.txt"),
    }

    output_dir = cache_root / sequence
    outputs: dict[str, Path] = {}
    for stream_id, frame in frames.items():
        path = output_dir / f"{stream_id}.parquet"
        _write_parquet_atomic(frame, path)
        outputs[stream_id] = path
    return outputs


def _imu_message_row(
    message: Any,
    *,
    record_timestamp_ns: int,
    source_file: Path,
    accel_scale: float = 1.0,
) -> dict[str, Any]:
    """Convert a ROS sensor_msgs/Imu-like object into one canonical SI row."""

    scale = float(accel_scale)
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("accel_scale must be finite and non-zero")
    record_timestamp_s = float(record_timestamp_ns) * 1e-9
    header = getattr(message, "header", None)
    header_timestamp_s = _stamp_seconds(getattr(header, "stamp", None))
    timestamp_s = record_timestamp_s if header_timestamp_s is None else header_timestamp_s
    angular_velocity = message.angular_velocity
    linear_acceleration = message.linear_acceleration
    return {
        "timestamp_s": timestamp_s,
        "gx": float(angular_velocity.x),
        "gy": float(angular_velocity.y),
        "gz": float(angular_velocity.z),
        "ax": float(linear_acceleration.x) * scale,
        "ay": float(linear_acceleration.y) * scale,
        "az": float(linear_acceleration.z) * scale,
        "frame_id": str(getattr(header, "frame_id", "")),
        "record_timestamp_s": record_timestamp_s,
        "source_file": str(source_file),
    }


def extract_rosbag_imu_topics(
    bag_path: Path,
    topic_specs: dict[str, dict[str, Any]],
    cache_root: Path,
    chunk_rows: int,
    message_iterator: Callable[
        [Path, tuple[str, ...]], Iterator[tuple[str, int, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    """Scan a ROS bag once and write bounded-memory Parquet for selected IMU topics."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if not topic_specs:
        raise ValueError("topic_specs must not be empty")
    topics = tuple(topic_specs)
    duration_limits = {
        float(spec["max_duration_s"])
        for spec in topic_specs.values()
        if spec.get("max_duration_s") is not None
    }
    if len(duration_limits) > 1:
        raise ValueError("All topic specs must use the same max_duration_s")
    max_duration_s = next(iter(duration_limits), None)
    if max_duration_s is not None and (not np.isfinite(max_duration_s) or max_duration_s <= 0.0):
        raise ValueError("max_duration_s must be finite and positive")
    output_ids = {
        topic: _safe_segment(str(spec.get("id", _topic_id(topic))), "topic output id")
        for topic, spec in topic_specs.items()
    }
    if len(set(output_ids.values())) != len(output_ids):
        raise ValueError("topic output ids must be unique")
    if message_iterator is None and not bag_path.is_file():
        raise FileNotFoundError(f"ROS bag is missing: {bag_path}")

    cache_root.mkdir(parents=True, exist_ok=True)
    extraction_fingerprint = _extraction_fingerprint(bag_path, topic_specs)
    manifest_path = cache_root / "extraction_manifest.json"
    cached = _verified_extraction_cache(manifest_path, extraction_fingerprint)
    if cached is not None:
        logger.info("SKIPS verified ROS bag topic cache {}", cache_root)
        return cached
    state = {
        topic: _new_topic_state(cache_root / f"{output_ids[topic]}.parquet")
        for topic in topics
    }
    logger.info("READS ROS bag {} for {} IMU topics", bag_path, len(topics))
    iterator = message_iterator or _iter_rosbag_messages
    writers: dict[str, pq.ParquetWriter | None] = {topic: None for topic in topics}
    first_record_timestamp_s: float | None = None

    try:
        for topic, record_timestamp_ns, message in iterator(bag_path, topics):
            if topic not in topic_specs:
                continue
            record_timestamp_s = float(record_timestamp_ns) * 1e-9
            if first_record_timestamp_s is None:
                first_record_timestamp_s = record_timestamp_s
            if (
                max_duration_s is not None
                and record_timestamp_s - first_record_timestamp_s > max_duration_s
            ):
                break
            spec = topic_specs[topic]
            row = _imu_message_row(
                message,
                record_timestamp_ns=record_timestamp_ns,
                source_file=bag_path,
                accel_scale=float(spec.get("accel_scale", 1.0)),
            )
            topic_state = state[topic]
            _update_topic_stats(topic_state, float(row["timestamp_s"]))
            topic_state["buffer"].append(row)
            if len(topic_state["buffer"]) >= chunk_rows:
                writers[topic] = _flush_topic_buffer(topic_state, writers[topic])

        for topic in topics:
            writers[topic] = _flush_topic_buffer(state[topic], writers[topic])
            if writers[topic] is None:
                writers[topic] = _write_empty_topic(state[topic])
            writers[topic].close()
            writers[topic] = None
            output = state[topic]["path"]
            temporary = state[topic]["temporary_path"]
            logger.info("WRITES extracted IMU topic {} to {}", topic, output)
            temporary.replace(output)
    except Exception:
        for writer in writers.values():
            if writer is not None:
                writer.close()
        for topic_state in state.values():
            temporary = topic_state["temporary_path"]
            if temporary.exists():
                logger.warning("SKIPS and removes incomplete topic cache {}", temporary)
                temporary.unlink()
        raise

    topic_summary: dict[str, dict[str, Any]] = {}
    for topic in topics:
        topic_state = state[topic]
        rows = int(topic_state["rows"])
        start = topic_state["start_s"]
        end = topic_state["end_s"]
        duration = None if start is None or end is None else float(end - start)
        topic_summary[output_ids[topic]] = {
            "topic": topic,
            "path": str(topic_state["path"]),
            "rows": rows,
            "start_s": start,
            "end_s": end,
            "duration_s": duration,
            "rate_hz_est": (
                None if duration is None or duration <= 0.0 else float((rows - 1) / duration)
            ),
            "duplicates": int(topic_state["duplicates"]),
            "non_monotonic": int(topic_state["non_monotonic"]),
            "accel_scale": float(topic_specs[topic].get("accel_scale", 1.0)),
        }
    summary = {
        "status": "extracted",
        "fingerprint": extraction_fingerprint,
        "source_file": str(bag_path),
        "scan_count": 1,
        "bounded_duration_s": max_duration_s,
        "topics": topic_summary,
    }
    _write_extraction_manifest(manifest_path, summary)
    return summary


def _iter_rosbag_messages(
    bag_path: Path,
    topics: tuple[str, ...],
) -> Iterator[tuple[str, int, Any]]:
    from rosbags.highlevel import AnyReader

    with AnyReader([bag_path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        found_topics = {connection.topic for connection in connections}
        for missing in sorted(set(topics) - found_topics):
            logger.warning("WARNS requested IMU topic is absent from {}: {}", bag_path, missing)
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            yield (
                connection.topic,
                int(timestamp),
                reader.deserialize(rawdata, connection.msgtype),
            )


def _stamp_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    seconds = getattr(stamp, "sec", None)
    nanoseconds = getattr(stamp, "nanosec", getattr(stamp, "nsec", None))
    if seconds is None or nanoseconds is None:
        return None
    result = float(seconds) + float(nanoseconds) * 1e-9
    return result if np.isfinite(result) else None


def _attach_timestamp_diagnostics(frame: pd.DataFrame, path: Path) -> None:
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    differences = np.diff(timestamps)
    frame.attrs.update(
        {
            "source_file": str(path),
            "rows": len(frame),
            "duplicate_timestamps": int(np.count_nonzero(differences == 0.0)),
            "non_monotonic_timestamps": int(np.count_nonzero(differences < 0.0)),
        }
    )


def _penn_sequence_candidates(root: Path, sequence: str, sensor: str) -> list[Path]:
    archive_name = "penncosyvio_tango_top" if sensor == "tango_top" else "penncosyvio_visensor"
    candidates = [
        root / archive_name / sensor / sequence,
        root / sensor / sequence,
        root / sequence,
        root.parent / archive_name / sensor / sequence,
        root.parent.parent / archive_name / sensor / sequence,
    ]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _first_existing_directory(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Penn sequence directory not found; attempted: {attempted}")


def _safe_segment(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.drive or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a single safe path segment: {value}")
    return path.name


def _topic_id(topic: str) -> str:
    value = topic.strip("/").replace("/", "_")
    if not value:
        raise ValueError("ROS topic must not be empty")
    return value


def _new_topic_state(path: Path) -> dict[str, Any]:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if temporary.exists():
        logger.warning("SKIPS stale incomplete topic cache and removes {}", temporary)
        temporary.unlink()
    return {
        "path": path,
        "temporary_path": temporary,
        "buffer": [],
        "rows": 0,
        "duplicates": 0,
        "non_monotonic": 0,
        "last_timestamp_s": None,
        "start_s": None,
        "end_s": None,
    }


def _update_topic_stats(state: dict[str, Any], timestamp_s: float) -> None:
    previous = state["last_timestamp_s"]
    if previous is not None:
        if timestamp_s == previous:
            state["duplicates"] += 1
        elif timestamp_s < previous:
            state["non_monotonic"] += 1
    state["last_timestamp_s"] = timestamp_s
    state["start_s"] = timestamp_s if state["start_s"] is None else min(state["start_s"], timestamp_s)
    state["end_s"] = timestamp_s if state["end_s"] is None else max(state["end_s"], timestamp_s)
    state["rows"] += 1


def _flush_topic_buffer(
    state: dict[str, Any],
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    rows = state["buffer"]
    if not rows:
        return writer
    frame = pd.DataFrame.from_records(rows, columns=IMU_COLUMNS)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(state["temporary_path"], table.schema, compression="zstd")
    writer.write_table(table)
    state["buffer"] = []
    return writer


def _write_empty_topic(state: dict[str, Any]) -> pq.ParquetWriter:
    frame = pd.DataFrame(
        {
            **{column: pd.Series(dtype="float64") for column in IMU_COLUMNS[:7]},
            **{column: pd.Series(dtype="string") for column in IMU_COLUMNS[7:]},
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = pq.ParquetWriter(state["temporary_path"], table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    logger.info("WRITES Penn stream cache {}", path)
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _extraction_fingerprint(bag_path: Path, topic_specs: dict[str, dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(str(bag_path.resolve()).encode("utf-8"))
    if bag_path.is_file():
        stat = bag_path.stat()
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    else:
        digest.update(b"injected-message-source")
    digest.update(json.dumps(topic_specs, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _verified_extraction_cache(
    manifest_path: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    try:
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if summary.get("fingerprint") != fingerprint:
        return None
    topics = summary.get("topics")
    if not isinstance(topics, dict) or not all(
        isinstance(record, dict) and Path(str(record.get("path"))).is_file()
        for record in topics.values()
    ):
        return None
    return {**summary, "status": "verified_skip", "scan_count": 0}


def _write_extraction_manifest(path: Path, summary: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    logger.info("WRITES ROS bag extraction manifest {}", path)
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)
