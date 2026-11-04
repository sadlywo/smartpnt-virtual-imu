from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.smartpnt_reader import scan_dataset


def inspection_markdown(report: dict[str, Any]) -> str:
    lines = ["# Dataset Inspection", ""]
    if not report.get("exists", False):
        return "\n".join([*lines, report.get("message", "Dataset not found."), ""]) + "\n"
    lines.extend(
        [
            f"- Root: `{report.get('root')}`",
            f"- File count: {report.get('file_count')}",
            f"- Total size bytes: {report.get('total_size_bytes')}",
            f"- Max file size bytes: {report.get('max_size_bytes')}",
            "",
            "## Suffix Counts",
            "",
        ]
    )
    for suffix, count in sorted(report.get("suffix_counts", {}).items()):
        lines.append(f"- `{suffix}`: {count}")
    lines.extend(["", "## Directory Tree Preview", ""])
    for item in report.get("tree", [])[:80]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## File Previews", ""])
    for preview in report.get("previews", []):
        lines.append(f"### `{preview.get('path')}`")
        lines.append("")
        lines.append(f"- Kind: {preview.get('kind')}")
        if preview.get("columns"):
            lines.append(f"- Columns: {preview.get('columns')}")
        if preview.get("keys"):
            lines.append(f"- Keys: {preview.get('keys')}")
        if preview.get("note"):
            lines.append(f"- Note: {preview.get('note')}")
        if preview.get("error"):
            lines.append(f"- Error: {preview.get('error')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(
    root: Path = typer.Option(Path("data/raw/smartpnt_msf/01_extracted"), "--root"),
    out: Path | None = typer.Option(None, "--out"),
    preview_limit: int = typer.Option(5, "--preview-limit"),
) -> None:
    report = scan_dataset(root, preview_limit=preview_limit)
    markdown = inspection_markdown(report)
    if out is None:
        typer.echo(markdown)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote inspection report to {out}")


if __name__ == "__main__":
    typer.run(main)