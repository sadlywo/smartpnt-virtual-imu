from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "freecad"))

from xsens_dot2_watch.artifacts import (  # noqa: E402
    BuildLog,
    create_run_directory,
    sha256_file,
    write_json,
)


def test_create_run_directory_is_timestamped_and_non_overwriting(tmp_path: Path) -> None:
    now = datetime(2026, 7, 14, 21, 30, 45)
    first = create_run_directory(tmp_path, now)
    second = create_run_directory(tmp_path, now)
    assert first.name == "20260714T213045"
    assert second.name == "20260714T213045_01"


def test_build_log_records_read_write_skip(tmp_path: Path) -> None:
    path = tmp_path / "build_log.txt"
    log = BuildLog(path)
    log.read("config.json")
    log.write("part.step")
    log.skip("existing.stl", "already exists")
    text = path.read_text(encoding="utf-8")
    assert "READ config.json" in text
    assert "WRITE part.step" in text
    assert "SKIP existing.stl reason=already exists" in text


def test_json_and_hash_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert sha256_file(path) == sha256_file(path)
