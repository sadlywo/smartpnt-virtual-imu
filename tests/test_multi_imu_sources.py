from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from virtual_imu.io.multi_imu_sources import (
    _imu_message_row,
    extract_penn_sequence,
    extract_rosbag_imu_topics,
    read_penn_vector_file,
    read_penn_vi_imu,
)


def test_read_penn_vector_file_preserves_order_and_timestamp_diagnostics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "accelerometer.txt"
    path.write_text(
        "0.00 1 2 3\n0.01 4 5 6\n0.01 7 8 9\n0.005 10 11 12\n",
        encoding="utf-8",
    )

    frame = read_penn_vector_file(path, value_columns=["ax", "ay", "az"])

    assert frame.columns.tolist() == ["timestamp_s", "ax", "ay", "az"]
    assert frame["timestamp_s"].tolist() == [0.0, 0.01, 0.01, 0.005]
    assert frame.attrs["duplicate_timestamps"] == 1
    assert frame.attrs["non_monotonic_timestamps"] == 1


def test_read_penn_vi_imu_has_documented_si_column_order(tmp_path: Path) -> None:
    path = tmp_path / "imu.txt"
    path.write_text("0.005 1 2 3 4 5 6\n", encoding="utf-8")

    frame = read_penn_vi_imu(path)

    assert frame.columns.tolist() == [
        "timestamp_s",
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
    ]
    assert frame.iloc[0].to_dict() == {
        "timestamp_s": 0.005,
        "ax": 1.0,
        "ay": 2.0,
        "az": 3.0,
        "gx": 4.0,
        "gy": 5.0,
        "gz": 6.0,
    }


def test_imu_message_row_prefers_header_time_and_scales_livox_acceleration() -> None:
    message = _imu_message(
        stamp=SimpleNamespace(sec=12, nanosec=250_000_000),
        frame_id="livox_imu",
        accel=(1.0, -2.0, 0.5),
        gyro=(0.1, 0.2, 0.3),
    )

    row = _imu_message_row(
        message,
        record_timestamp_ns=99_000_000_000,
        source_file=Path("sample.bag"),
        accel_scale=9.8,
    )

    assert row == {
        "timestamp_s": 12.25,
        "gx": 0.1,
        "gy": 0.2,
        "gz": 0.3,
        "ax": 9.8,
        "ay": -19.6,
        "az": 4.9,
        "frame_id": "livox_imu",
        "record_timestamp_s": 99.0,
        "source_file": "sample.bag",
    }


def test_imu_message_row_falls_back_to_record_timestamp() -> None:
    message = _imu_message(stamp=None, frame_id="imu", accel=(1, 2, 3), gyro=(4, 5, 6))

    row = _imu_message_row(
        message,
        record_timestamp_ns=1_500_000_000,
        source_file=Path("sample.bag"),
    )

    assert row["timestamp_s"] == 1.5
    assert row["record_timestamp_s"] == 1.5


def test_extract_rosbag_topics_scans_once_and_writes_chunked_topic_files(
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, tuple[str, ...]]] = []
    topic_a = "/stim300/imu/data_raw"
    topic_b = "/livox/imu"

    def messages(path: Path, topics: tuple[str, ...]):
        calls.append((path, topics))
        yield topic_a, 1_000_000_000, _imu_message(stamp=None, frame_id="stim", accel=(1, 2, 3), gyro=(4, 5, 6))
        yield topic_b, 1_100_000_000, _imu_message(stamp=None, frame_id="livox", accel=(1, 0, 0), gyro=(0, 0, 0))
        yield topic_a, 1_000_000_000, _imu_message(stamp=None, frame_id="stim", accel=(7, 8, 9), gyro=(1, 2, 3))
        yield topic_a, 900_000_000, _imu_message(stamp=None, frame_id="stim", accel=(0, 0, 0), gyro=(0, 0, 0))

    summary = extract_rosbag_imu_topics(
        tmp_path / "fake.bag",
        {
            topic_a: {"id": "stim300"},
            topic_b: {"id": "livox", "accel_scale": 9.8},
        },
        tmp_path / "cache",
        chunk_rows=2,
        message_iterator=messages,
    )

    assert calls == [(tmp_path / "fake.bag", (topic_a, topic_b))]
    stim = pd.read_parquet(summary["topics"]["stim300"]["path"])
    assert stim["timestamp_s"].tolist() == [1.0, 1.0, 0.9]
    assert summary["topics"]["stim300"]["duplicates"] == 1
    assert summary["topics"]["stim300"]["non_monotonic"] == 1
    assert pd.read_parquet(summary["topics"]["livox"]["path"])["ax"].tolist() == [9.8]


def test_extract_penn_sequence_writes_three_independent_streams(tmp_path: Path) -> None:
    root = tmp_path / "01_extracted"
    tango = root / "penncosyvio_tango_top" / "tango_top" / "af"
    vi = root / "penncosyvio_visensor" / "visensor" / "af"
    tango.mkdir(parents=True)
    vi.mkdir(parents=True)
    (tango / "accelerometer.txt").write_text("0.0 1 2 3\n", encoding="utf-8")
    (tango / "gyroscope.txt").write_text("0.1 4 5 6\n", encoding="utf-8")
    (vi / "imu.txt").write_text("0.2 7 8 9 10 11 12\n", encoding="utf-8")

    outputs = extract_penn_sequence(root, "af", tmp_path / "cache")

    assert set(outputs) == {"tango_accel", "tango_gyro", "vi_imu"}
    assert pd.read_parquet(outputs["tango_accel"])["timestamp_s"].tolist() == [0.0]
    assert pd.read_parquet(outputs["tango_gyro"])["timestamp_s"].tolist() == [0.1]
    assert pd.read_parquet(outputs["vi_imu"])["timestamp_s"].tolist() == [0.2]


def test_extract_rosbag_topics_rejects_duplicate_output_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        extract_rosbag_imu_topics(
            tmp_path / "fake.bag",
            {"/a": {"id": "imu"}, "/b": {"id": "imu"}},
            tmp_path / "cache",
            chunk_rows=2,
            message_iterator=lambda _path, _topics: iter(()),
        )


def test_extract_rosbag_topics_honors_shared_duration_limit(tmp_path: Path) -> None:
    topic = "/imu"

    def messages(_path: Path, _topics: tuple[str, ...]):
        for seconds in (10.0, 10.5, 11.1, 12.0):
            yield (
                topic,
                int(seconds * 1e9),
                _imu_message(stamp=None, frame_id="imu", accel=(0, 0, 1), gyro=(0, 0, 0)),
            )

    summary = extract_rosbag_imu_topics(
        tmp_path / "fake.bag",
        {topic: {"id": "imu", "max_duration_s": 1.0}},
        tmp_path / "cache",
        chunk_rows=2,
        message_iterator=messages,
    )

    assert summary["bounded_duration_s"] == 1.0
    assert pd.read_parquet(summary["topics"]["imu"]["path"])["timestamp_s"].tolist() == [
        10.0,
        10.5,
    ]


def test_extract_rosbag_topics_verified_cache_skips_second_scan(tmp_path: Path) -> None:
    topic = "/imu"
    calls = 0

    def messages(_path: Path, _topics: tuple[str, ...]):
        nonlocal calls
        calls += 1
        yield topic, 1_000_000_000, _imu_message(
            stamp=None,
            frame_id="imu",
            accel=(0, 0, 1),
            gyro=(0, 0, 0),
        )

    first = extract_rosbag_imu_topics(
        tmp_path / "fake.bag",
        {topic: {"id": "imu"}},
        tmp_path / "cache",
        chunk_rows=2,
        message_iterator=messages,
    )
    second = extract_rosbag_imu_topics(
        tmp_path / "fake.bag",
        {topic: {"id": "imu"}},
        tmp_path / "cache",
        chunk_rows=2,
        message_iterator=messages,
    )

    assert first["status"] == "extracted"
    assert second["status"] == "verified_skip"
    assert calls == 1


def _imu_message(
    *,
    stamp: object | None,
    frame_id: str,
    accel: tuple[float, float, float],
    gyro: tuple[float, float, float],
) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, frame_id=frame_id),
        linear_acceleration=SimpleNamespace(x=accel[0], y=accel[1], z=accel[2]),
        angular_velocity=SimpleNamespace(x=gyro[0], y=gyro[1], z=gyro[2]),
    )
