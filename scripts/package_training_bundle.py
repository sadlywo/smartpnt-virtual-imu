"""Package the live source snapshot and curated dataset for a Linux training host."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.training_bundle import DEFAULT_DATASET_ROOT, build_training_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path, default=Path(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/server_bundles"))
    parser.add_argument(
        "--bundle-id",
        help="Optional unique server_bundle_<name>; existing artifacts are never overwritten",
    )
    args = parser.parse_args()
    print(
        f"READS: {args.project_root.resolve()}; curated dataset {args.dataset_root}; current source bytes including untracked code"
    )
    print(
        f"WRITES: {args.output_dir}/<bundle>.tar.gz, .sha256 and runs/<bundle>/ provenance, metrics, report and verification"
    )
    print(
        "SKIPS: raw/diagnostic/old data, duplicate sequences, other runs/checkpoints, outputs, .git, environments, caches, credentials and keys; only selected data build audit files are exported"
    )
    try:
        result = build_training_bundle(
            args.project_root,
            args.dataset_root,
            bundle_id=args.bundle_id,
            output_dir=args.output_dir,
        )
    except (ValueError, OSError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    print(f"ARCHIVE: {result.archive_path}")
    print(f"SHA256: {result.archive_sha256}")
    print(f"RUN: {result.run_dir}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
