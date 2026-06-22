"""Small helper to rebuild the figure index for exported IMU figures."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", default="outputs/figures")
    args = parser.parse_args()
    root = Path(args.figures)
    rows = []
    for svg in sorted(root.glob("*.svg")):
        stem = svg.stem.replace("_", " ")
        rows.append(f"| `{svg.name}` | {stem} | Caption draft pending. |")
    index = root / "figure_index.md"
    index.write_text(
        "# Figure Index\n\n| Figure | Purpose | Caption Draft |\n|---|---|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
