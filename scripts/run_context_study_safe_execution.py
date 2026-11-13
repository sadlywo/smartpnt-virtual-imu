"""Resume frozen jobs with a documented attention overflow recovery policy."""
from pathlib import Path
import concurrent.futures as futures,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.context_study.numerical_safety import install,initialize_safe_worker
from virtual_imu.replacements.training import sha,write_json

def main():
    run=ROOT/'runs/alimu_context_mechanisms_20261108'
    lock=json.loads((run/'numeric_precision_recovery.lock.json').read_text(encoding='utf-8'))
    for path,digest in lock['sources'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Numeric recovery source changed '+path)
    install(run)
    original=futures.ProcessPoolExecutor
    def safe_pool(*args,**kwargs):
        if args or kwargs.get('initializer') is not None:raise ValueError('Unexpected worker signature')
        kwargs.update(max_workers=4,initializer=initialize_safe_worker);return original(**kwargs)
    futures.ProcessPoolExecutor=safe_pool
    (run/'coordinator.pid').write_text(str(os.getpid()),encoding='utf-8')
    write_json(run/'numeric_precision_execution.json',dict(pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),actual_workers=4,policy_lock_sha256=sha(run/'numeric_precision_recovery.lock.json')))
    print('READ frozen TRAIN/VALID resumes; WRITE conditional FP32 overflow receipts and results; SKIP verified fits and TEST before locks',flush=True)
    from virtual_imu.context_study.pipeline import execute
    execute(ROOT,'configs/ablations/alimu_context_mechanisms.yaml')
if __name__=='__main__':main()
