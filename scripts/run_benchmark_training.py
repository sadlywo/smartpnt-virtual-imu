"""Run a frozen retrospective multi-model IMU benchmark with honest task adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.training.benchmark_pipeline import run_benchmark_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and first seed for all candidate/task pairs; separate nonformal run, no standalone held-out evaluation",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze expected job configs without reading data/results or initializing CUDA",
    )
    parser.add_argument(
        "--resume-root",
        type=Path,
        help="Continue only untouched planned jobs after rechecking completed results; partial optimization is never resumed/restarted",
    )
    args = parser.parse_args()
    print(
        f"READS: matrix {args.config}, source task YAMLs; frozen manifests and completed artifacts only when execution/resume requires them",
        flush=True,
    )
    print(
        "WRITES: independent runs/benchmark_<plan|smoke|training>_<time>/ configs, checkpoint/evaluation provenance, candidate-aware physical comparison CSVs",
        flush=True,
    )
    print(
        "SKIPS: old formal-run mutation, raw/duplicate data, temperature/static branches, test-fitted preprocessing, silent partial retraining and unsupported published-SOTA claims",
        flush=True,
    )
    run_benchmark_pipeline(
        load_config(args.config),
        config_path=args.config,
        smoke=args.smoke,
        resume_root=args.resume_root,
        plan_only=args.plan_only,
    )


if __name__ == "__main__":
    main()
