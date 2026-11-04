"""Run validation-only reviewer HPO, seed confirmation and selected final refits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.training.reviewer_hpo import run_reviewer_hpo
from virtual_imu.utils.paths import project_root_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reviewer_hpo.yaml"))
    parser.add_argument("--plan-only", action="store_true", help="Freeze candidates/initial configs without dataset reads or CUDA")
    parser.add_argument("--smoke", action="store_true", help="Explicit nonformal workflow: first trial, one epoch, first final seed")
    parser.add_argument("--resume-root", type=Path, help="Verify completed results and run untouched jobs; interrupted optimization is never silently restarted")
    parser.add_argument("--protocol", action="append", help="Restrict to explicitly named protocol(s)")
    parser.add_argument("--cell", action="append", help="Restrict to protocol/candidate identity (repeatable)")
    arguments = parser.parse_args()
    print(f"READS: HPO matrix and source task YAMLs {arguments.config}", flush=True)
    print("WRITES: separate reviewer run, frozen configs, validation-only promotions, selected final artifacts", flush=True)
    print("SKIPS: test-based search, old run mutation, target normalization fitting, silent partial optimization restart", flush=True)
    run_reviewer_hpo(
        load_config(arguments.config), config_path=arguments.config,
        root=project_root_from_file(__file__), plan_only=arguments.plan_only,
        smoke=arguments.smoke, resume_root=arguments.resume_root,
        protocol_filter=arguments.protocol, cell_filter=arguments.cell,
    )


if __name__ == "__main__":
    main()
