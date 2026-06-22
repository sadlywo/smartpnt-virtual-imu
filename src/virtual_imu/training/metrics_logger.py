"""CSV metrics logger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Append scalar metrics to a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        """Append one row."""
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({key: row.get(key) for key in self._fieldnames})
