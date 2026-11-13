"""Build or seal a presentation-only revision of the completed ALiMU study."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.viz.plots_context_revision import build,seal
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/figures/alimu_context_revision.yaml');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    (seal if args.seal else build)(ROOT,args.config)
