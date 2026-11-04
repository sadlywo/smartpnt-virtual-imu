from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.training.penn_matrix_runner import run_penn_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the strict Penn four-fold, three-seed matrix."
    )
    parser.add_argument("--config", type=Path, required=True, help="Matrix YAML path.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one epoch, a small CPU model, and a short static context.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume the exact run directory written to outputs/active_penn_matrix_run.txt.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        parser.error(f"Matrix config does not exist: {config_path}")
    if args.resume_run is not None and not args.resume_run.is_dir():
        parser.error(f"Resume run directory does not exist: {args.resume_run}")

    print(f"READS: matrix YAML {config_path}")
    print("WRITES: resumable matrix run with 60 neural trials, 16 analytic baselines, and report")
    print("SKIPS: matching completed baseline/trial artifacts on --resume-run; raw data mutation")
    run_dir = run_penn_matrix(
        load_config(config_path),
        config_path=config_path,
        smoke=args.smoke,
        resume_run=args.resume_run.resolve() if args.resume_run is not None else None,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
