"""Atomic completion markers for recoverable local experiment fits."""

from __future__ import annotations

import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
