"""Package completed IMU results and linked original TCN artifacts portably."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--original-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    results, original = args.results.resolve(), args.original_results.resolve()
    print(
        f"READS: completed report/result tree {results}, linked original TCN server artifacts {original}",
        flush=True,
    )
    print(f"WRITES: {args.output}, per-file inventory and archive verification receipt", flush=True)
    print(
        "SKIPS: raw/processed datasets, duplicate download archives, transient monitor state, server changes and credentials",
        flush=True,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace an existing delivery archive: {args.output}")
    report = results / "report"
    for path, field in [
        (results / "delivery_verification.json", "status"),
        (report / "scientific_audit.json", "status"),
        (report / "qa/figure_technical_qa.json", "status"),
    ]:
        if json.loads(path.read_text(encoding="utf-8"))[field] != "passed":
            raise ValueError(f"Required verification has not passed: {path}")
    for path in [report / "report_manifest.json", report / "plotting_figure_manifest.json"]:
        if json.loads(path.read_text(encoding="utf-8"))["visual_review"]["status"] != "passed":
            raise ValueError(f"Visual review has not passed: {path}")
    files = set()
    for root in [results, original]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(root).parts
            if parts[0] == "download" or path.suffix.lower() == ".zip":
                continue
            if path.name in {
                "stage_state.json",
                "DELIVERY_MANIFEST.json",
                "portable_archive_receipt.json",
            }:
                continue
            files.add(path.resolve())
    for name in [
        "scripts/analyze_completed_benchmark.py",
        "scripts/visualize_completed_benchmark.py",
        "scripts/build_completed_benchmark_report.py",
        "scripts/build_completed_formal_report.py",
        "scripts/verify_benchmark_delivery.py",
        "scripts/package_benchmark_delivery.py",
        "configs/plot_completed_benchmark.yaml",
        "docs/BENCHMARK_EXPANSION_20261024.md",
    ]:
        files.add(project / name)
    # Preserve project-relative layout so links to the separately identified old
    # TCN result tree still resolve after extraction into any directory.
    for markdown in [report / "完整模型对照报告.md", results / "结果导航.md"]:
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", markdown.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>"))
            if target.startswith(("https:", "http:", "#", "mailto:")):
                continue
            path = (markdown.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Broken deliverable link: {target}")
            if path.is_file():
                files.add(path)
    for path in files:
        if not path.is_relative_to(project) or path.is_symlink():
            raise ValueError(f"Nonportable file: {path}")
        if path.relative_to(project).parts[:2] in {("data", "raw"), ("data", "processed")}:
            raise ValueError(f"Datasets must not be duplicated into the result archive: {path}")
    inventory = [
        {
            "path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(files)
    ]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Complete new benchmark results/report plus separately identified original TCN source artifacts; project-relative paths preserve report links",
        "neural_runs": 45,
        "new_runs": 39,
        "reused_original_runs": 6,
        "files": inventory,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    navigation = (
        "# IMU expanded benchmark results\n\n"
        "Open runs/remote_results_benchmark_20261024/结果导航.md for the Chinese report, complete tables, figures and models.\n\n"
        "Extract the whole archive before following relative links. Keep both result directories: the six original TCN runs retain their original identity under runs/remote_results_formal_20261024/server.\n\n"
        "This archive contains 45 neural runs (39 new and 6 reused), 13 paired methods, 11 navigation methods, all checkpoints/predictions, a 51-page report and 35 scientific figures. Training data are represented by immutable manifests and fingerprints; raw/processed input datasets are not duplicated.\n\n"
        "Paper backbones are common-task adaptations, not complete native-paper SOTA reproductions. Negative results and all seeds are retained. DELIVERY_MANIFEST.json records each source file SHA256.\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "x", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
    ) as archive:
        for entry in inventory:
            archive.write(project / entry["path"], entry["path"])
        archive.writestr("DELIVERY_MANIFEST.json", manifest_bytes)
        archive.writestr("README_RESULTS.md", navigation)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Delivery archive failed CRC verification")
        if len(archive.namelist()) != len(inventory) + 2:
            raise ValueError("Delivery archive member count mismatch")
        for entry in inventory:
            with archive.open(entry["path"]) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != entry["sha256"]:
                raise ValueError(f"Delivery member SHA256 mismatch: {entry['path']}")
    receipt = {
        "status": "passed",
        "archive": str(args.output.resolve()),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": digest(args.output),
        "members": len(inventory) + 2,
        "source_hashes_and_archived_member_hashes_verified": len(inventory),
        "report_pdf_sha256": digest(report / "完整模型对照报告.pdf"),
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }
    (results / "DELIVERY_MANIFEST.json").write_bytes(manifest_bytes)
    (results / "portable_archive_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
