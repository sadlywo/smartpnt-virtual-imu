"""Build cross-branch Nature-style figures from frozen experiment tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from virtual_imu.fir_arx_research.branch_viz import plot_branch_effects, plot_branch_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/figures/fir_arx_branch_summary_20261105.yaml")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = Path(config["output"]["root"])
    out_dir = output_root if args.run_id is None else output_root.parent / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    comparison = pd.read_csv(ROOT / config["inputs"]["comparison"])
    summary = pd.read_csv(ROOT / config["inputs"]["summary"])
    records = [plot_branch_effects(comparison, out_dir), plot_branch_summary(summary, out_dir)]
    (out_dir / "figures.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
