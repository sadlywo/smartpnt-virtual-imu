"""Verify a server bundle, its portable data contract, and optional CUDA execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from virtual_imu.io.training_bundle import (
    check_training_environment,
    portable_relative_path,
    verify_training_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--checksums-only",
        action="store_true",
        help="Standard-library verification before dependency installation; skips Parquet schemas and runtime",
    )
    modes.add_argument(
        "--data-only",
        action="store_true",
        help="Verify checksums, Parquet schemas and splits without importing PyTorch",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require CUDA matrix computation and Conv1d forward/backward with finite gradients",
    )
    parser.add_argument(
        "--report", type=Path, help="Optional project-relative runs/<id>/preflight.json"
    )
    args = parser.parse_args()
    if args.require_gpu and (args.checksums_only or args.data_only):
        parser.error("--require-gpu cannot be combined with --checksums-only or --data-only")
    root = args.project_root.resolve()
    print(
        f"READS: {root}/bundle_manifest.json, SHA256SUMS, exported files, dataset manifest/schema/splits"
    )
    print(f"WRITES: {args.report if args.report else 'stdout verification result only'}")
    print(
        f"SKIPS: {'Parquet schemas and training runtime' if args.checksums_only else 'PyTorch runtime' if args.data_only else 'training, raw data and any remote operations'}"
    )
    try:
        report = verify_training_bundle(root, check_schemas=not args.checksums_only)
        if not (args.checksums_only or args.data_only):
            profile = None
            if report.get("environment_profile"):
                import yaml

                profile_path = root / report["environment_profile"]["path"]
                print(f"READS: verified environment profile {profile_path}")
                profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            report["environment"] = check_training_environment(
                require_gpu=args.require_gpu, profile=profile
            )
        if args.report:
            report_arg = args.report
            relative = (
                report_arg.resolve().relative_to(root).as_posix()
                if report_arg.is_absolute()
                else report_arg.as_posix()
            )
            portable = portable_relative_path(relative)
            if portable.parts[0] != "runs":
                raise ValueError("Preflight reports must be written under runs/<id>/")
            destination = root.joinpath(*portable.parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError("Preflight report resolves outside the project")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    print("VERIFIED: bundle integrity, portable paths and requested data/runtime checks")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
