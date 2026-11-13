"""Run a bounded, disjoint subset of frozen fits ahead of the coordinator."""
from pathlib import Path
import json,logging,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.context_study.training import fit_job
from virtual_imu.replacements.training import sha,write_json

def main():
    run=ROOT/'runs/alimu_context_mechanisms_20261108'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
    preparation=json.loads((run/'preparation.json').read_text(encoding='utf-8'))
    for name,digest in preparation['training_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen training source changed '+name)
    plan=json.loads((run/'planned_jobs.json').read_text(encoding='utf-8'))
    jobs=[j for j in plan['mechanism'] if j['model_options']['depth']==1]
    receipt=dict(pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),planned_ids=[j['id'] for j in jobs],completed_ids=[],source_sha256=sha(Path(__file__)),training_changes=False,execution_override='One additional independent worker on disjoint depth-1 mechanism fits; frozen main coordinator retains two workers.')
    path=run/'execution_assistant.json';write_json(path,receipt)
    logging.info('READ frozen plan/source hashes; WRITE bounded mechanism fits (%d); SKIP all TEST payloads',len(jobs))
    for job in jobs:
        phase=json.loads((run/'progress.json').read_text(encoding='utf-8'))['phase']
        if phase not in ['window','future']:
            logging.info('SKIP remaining helper jobs: coordinator reached %s',phase);break
        if (Path(job['directory'])/'history.csv').exists() and not (Path(job['directory'])/'fit.json').exists():
            raise RuntimeError('Refusing possible concurrent active fit '+job['id'])
        result=fit_job(job)
        receipt['completed_ids'].append(result['job_id']);receipt['updated']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(path,receipt)
        logging.info('WRITE assistant completed=%d/%d job=%s val=%.6f',len(receipt['completed_ids']),len(jobs),job['id'],result['best_val_score'])
    receipt['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write_json(path,receipt)
    logging.info('WRITE assistant finished; SKIP coordinator state mutations')
if __name__=='__main__':main()
