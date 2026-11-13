"""Resume the frozen study with an explicit execution-only four-worker override."""
from pathlib import Path
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.context_study.execution import install_pool_override
from virtual_imu.context_study.pipeline import execute
from virtual_imu.replacements.training import sha,write_json

def main():
    run=ROOT/'runs/alimu_context_mechanisms_20261108'
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),ROOT/'src/virtual_imu/context_study/execution.py']}
    receipt=dict(pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),configured_workers=2,actual_workers=4,cpu_threads_per_worker=1,
                 source_sha256=sources,training_math_changed=False,cache_policy='Release cached supports only when held-device key changes',
                 resume_policy='Frozen per-fit optimizer, scheduler, AMP scaler and Python/NumPy/Torch/CUDA RNG states',
                 timing_policy='Contended training wall times are execution records, not isolated model training speed benchmarks')
    write_json(run/'execution_pool_override.json',receipt)
    (run/'coordinator.pid').write_text(str(os.getpid()),encoding='utf-8')
    print('READ frozen config and resumable per-epoch states; WRITE execution override (four workers); SKIP TEST until all selections lock',flush=True)
    install_pool_override(4)
    execute(ROOT,'configs/ablations/alimu_context_mechanisms.yaml')
if __name__=='__main__':main()
