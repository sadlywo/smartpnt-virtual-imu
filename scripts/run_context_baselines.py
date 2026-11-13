"""Run or verify matched FIR/ARX baselines without changing the sealed study."""
from pathlib import Path
import argparse
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from virtual_imu.context_study.classical_pipeline import execute, verify


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/analysis/alimu_context_classical.yaml')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.verify:
        cfg = yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
        print(verify(ROOT/cfg['output']['root']/cfg['experiment_id']))
    else:
        print(execute(ROOT, args.config))
