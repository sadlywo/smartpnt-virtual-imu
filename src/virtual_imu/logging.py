"""Project logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Configure loguru for console and optional file logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, level=level, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


__all__ = ["logger", "setup_logging"]
