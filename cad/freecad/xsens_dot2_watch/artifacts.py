from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class BuildLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _line(self, category: str, target: str, detail: str = "") -> None:
        suffix = " " + detail if detail else ""
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{category} {target}{suffix}\n")

    def read(self, target: str) -> None:
        self._line("READ", target)

    def write(self, target: str) -> None:
        self._line("WRITE", target)

    def skip(self, target: str, reason: str) -> None:
        self._line("SKIP", target, f"reason={reason}")

    def fail(self, stage: str, reason: str) -> None:
        self._line("FAIL", stage, f"reason={reason}")


def create_run_directory(root: Path, now: datetime) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%Y%m%dT%H%M%S")
    for index in range(100):
        name = stem if index == 0 else f"{stem}_{index:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate timestamped build directory under {root}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
