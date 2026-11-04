"""Run or resume the strict ALiMU leave-sensor matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from virtual_imu.config import load_config  # noqa: E402
from virtual_imu.training.alimu_matrix_runner import run_alimu_matrix  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _preflight_input_paths(config: dict[str, Any], parser: argparse.ArgumentParser) -> None:
    required_paths = {"base model config": config.get("base_model_config")}
    split = config.get("split")
    if not isinstance(split, dict):
        parser.error("Matrix config requires a split mapping.")
    for name in ("train", "val", "test"):
        required_paths[f"{name} Parquet"] = split.get(f"{name}_path")
    required_paths["split manifest"] = split.get("manifest_path")

    for label, value in required_paths.items():
        if not value:
            parser.error(f"Matrix config is missing {label} path.")
        path = _project_path(value)
        if not path.is_file():
            parser.error(f"Matrix {label} does not exist: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the strict ALiMU MPU9250 leave-sensor matrix."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/alimu_leave_sensor_matrix_3seed.yaml"),
        help="Matrix YAML path.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the runner's reduced CPU smoke settings; preserves the method and seed axes.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Resume a run directory recorded by outputs/active_alimu_matrix_run.txt.",
    )
    args = parser.parse_args()

    config_path = _project_path(args.config).resolve()
    if not config_path.is_file():
        parser.error(f"Matrix config does not exist: {config_path}")
    if args.resume_run is not None and not args.resume_run.is_dir():
        parser.error(f"Resume run directory does not exist: {args.resume_run}")

    config = load_config(config_path)
    _preflight_input_paths(config, parser)
    resume_run = args.resume_run.resolve() if args.resume_run is not None else None

    print(f"READS: matrix YAML {config_path}")
    print("READS: base model YAML and strict ALiMU train/validation/test Parquet plus manifest")
    print("WRITES: one resumable ALiMU matrix run, ALiMU pointers, metrics, and report")
    print("SKIPS: raw data mutation and matching completed baseline/trial artifacts on resume")
    run_dir = run_alimu_matrix(
        config,
        config_path=config_path,
        smoke=args.smoke,
        resume_run=resume_run,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
