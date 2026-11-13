"""Run the approved ALiMU study; all READ/WRITE/SKIP operations are logged."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.context_study.pipeline import execute
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/ablations/alimu_context_mechanisms.yaml')
    parser.add_argument('--run-id');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();execute(ROOT,args.config,args.run_id,args.preflight)
