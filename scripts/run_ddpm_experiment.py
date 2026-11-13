"""Train/tune/evaluate separate conditional DDPM models from an existing paired manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from virtual_imu.diffusion.pipeline import run_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ddpm/alimu.yaml")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed trials; restart interrupted ones"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    run = run_experiment(args.config, ROOT, args.run_id, args.resume)
    print(f"RUN_DIR={run}", flush=True)


if __name__ == "__main__":
    main()
