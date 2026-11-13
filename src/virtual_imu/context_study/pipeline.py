"""Frozen protocol selection and run coordinator."""
from __future__ import annotations
import pandas as pd

def select_candidates(rows,expected_candidates):
    if 'role' not in rows or set(rows.role)!= {'validation'}: raise ValueError('Selection requires validation-only rows')
    chosen={}
    for family,group in rows.groupby('family',sort=True):
        if set(group.candidate)!=set(expected_candidates): raise ValueError('Incomplete candidate budget')
        scores=group.groupby('candidate').best_val_score.mean().sort_values(kind='stable')
        chosen[family]=scores.index[0]
    return chosen

import copy
import concurrent.futures as futures
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from virtual_imu.context_study.data import StudySource
from virtual_imu.context_study.training import fit_job
from virtual_imu.replacements.training import sha,write_json,json_digest
LOG=logging.getLogger(__name__)

def _job(cfg,root,run,stage,family,seed,window=320,tag=None,model_options=None,held_sensor=None,candidate='base',training_override=None):
    name='__'.join([stage,tag or family,'w'+str(window),'s'+str(seed)]+([held_sensor] if held_sensor else []))
    tr=copy.deepcopy(cfg['training'])
    if stage!='window':tr.update(cfg['new_stage_training'])
    if training_override:tr.update(training_override)
    return dict(id=name,root=str(root),directory=str(run/'fits'/name),config=cfg,stage=stage,family=family,window=window,seed=seed,
                candidate=candidate,model_options=model_options or {},held_sensor=held_sensor,training=tr,
                diagnostics=stage=='future' and seed==cfg['seeds'][0] and (family in ['gru','bigru','transformer']))

def plan_jobs(cfg,root,run):
    plan={k:[] for k in ['window','future','mechanism','search_pilot','generalization']}
    registries=[]
    for source in cfg['reuse_runs']:
        path=root/source/'fit_registry.csv';LOG.info('READ reusable registry %s',path)
        registries.append((root/source,pd.read_csv(path)))
    for family in cfg['window_families']:
        for window in cfg['windows']:
            for seed in cfg['seeds']:
                job=_job(cfg,root,run,'window',family,seed,window)
                for source,registry in reversed(registries):
                    variant={'cnn_lstm64':'cnn_lstm_wide','cnn_bilstm32':'cnn_bilstm'}.get(family,family) if 'shared_ablation' in source.name else family
                    rows=registry[(registry.protocol=='alimu_only')&(registry.variant==variant)&(registry.window==window)&(registry.seed==seed)]
                    if len(rows):
                        row=rows.iloc[0].to_dict();job['reuse']={'directory':row['directory'],'checkpoint_sha256':row['checkpoint_sha256'],'row':row};break
                plan['window'].append(job)
    common=dict(width=64,depth=3,kernel=5,dilation=1,norm='ln',activation='gelu')
    for backend in cfg['controlled_backends']:
        limits=cfg['lookaheads']+(['full'] if backend in cfg['full_future_backends'] else [])
        for look in limits:
            for seed in cfg['seeds']:
                plan['future'].append(_job(cfg,root,run,'future',backend,seed,tag=backend+'_look'+str(look),model_options={**common,'backend':backend,'lookahead':look}))
    for depth,width,kernel in [(1,68,13),(3,64,5)]:
        for norm in ['bn','ln','none']:
            for act in ['gelu','identity']:
                for seed in cfg['seeds']:
                    opts=dict(backend='gru',lookahead=0,width=width,depth=depth,kernel=kernel,dilation=1,norm=norm,activation=act)
                    job=_job(cfg,root,run,'mechanism','gru',seed,tag='d%d_k%d_w%d_%s_%s'%(depth,kernel,width,norm,act),model_options=opts)
                    if depth==3 and norm=='ln' and act=='gelu':job['reuse_source_id']='future__gru_look0__w320__s'+str(seed)
                    plan['mechanism'].append(job)
    for label,kernel,dilation in [('rf25',5,2),('rf49',5,4),('pointwise',1,1)]:
        for seed in cfg['seeds']:
            plan['mechanism'].append(_job(cfg,root,run,'mechanism','gru',seed,tag=label,model_options={**common,'backend':'gru','lookahead':0,'kernel':kernel,'dilation':dilation}))
    for family in cfg['search_families']:
        for candidate in cfg['candidate_packages']:
            plan['search_pilot'].append(_job(cfg,root,run,'search',family,cfg['seeds'][0],tag=family+'_'+candidate['id'],candidate=candidate['id'],model_options={'dropout':candidate['dropout']},training_override={'lr':candidate['lr']}))
    source=StudySource(root,cfg['data'])
    for sensor in source.split_audit['sensors']['train']:
        for family in cfg['strict_families']:
            for seed in cfg['seeds']:
                plan['generalization'].append(_job(cfg,root,run,'generalization',family,seed,held_sensor=sensor))
    return plan

def freeze(root,run,cfg,config_path,plan):
    run.mkdir(parents=True,exist_ok=True)
    frozen=run/'preparation.json'
    paths=list((root/'src/virtual_imu').rglob('*.py'))+[root/'scripts/run_context_study.py',root/config_path]
    training_prefixes=['src/virtual_imu/context_study/models.py','src/virtual_imu/context_study/data.py','src/virtual_imu/context_study/training.py']
    paths=sorted(set(paths))
    current={p.relative_to(root).as_posix():sha(p) for p in paths}
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        for name,digest in previous['training_sources'].items():
            if sha(root/name)!=digest:raise ValueError('Training source changed after freeze: '+name)
        if previous['config_sha256']!=json_digest(cfg):raise ValueError('Frozen config changed')
        LOG.info('SKIP existing verified preparation %s',run);return
    LOG.info('WRITE frozen run %s',run)
    for p in paths:
        dest=run/'source_snapshot'/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(root/cfg['data']['manifest'],run/'data_manifest.json')
    (run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    write_json(run/'planned_jobs.json',plan)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    env={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'platform':platform.platform()}
    write_json(run/'environment.json',env)
    training={n:h for n,h in current.items() if n in training_prefixes or not n.startswith('src/virtual_imu/context_study/')}
    write_json(frozen,{'created':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'git_commit':commit,'config_sha256':json_digest(cfg),
                      'manifest_sha256':sha(root/cfg['data']['manifest']),'training_sources':training,'all_initial_sources':current,
                      'plan_sha256':json_digest(plan),'prior_sources':cfg['reuse_runs'],'note':'TEST already inspected historically; this run locks new choices before loading TEST.'})

def _resolve(job,all_jobs):
    if 'reuse_source_id' not in job:return job
    job=copy.deepcopy(job);original=all_jobs[job['reuse_source_id']];directory=Path(original['directory']);row=json.loads((directory/'fit.json').read_text())
    job['reuse']={'directory':str(directory),'checkpoint_sha256':row['checkpoint_sha256'],'row':row}
    return job

def run_group(name,jobs,pool,run,all_jobs):
    LOG.info('READ phase %s slots=%d',name,len(jobs))
    pending={}
    for planned in jobs:
        job=_resolve(planned,all_jobs);all_jobs[job['id']]=job
        pending[pool.submit(fit_job,job)]=job
    completed=0
    for done in futures.as_completed(pending):
        job=pending[done]
        try:result=done.result()
        except Exception:
            LOG.exception('FAIL fit %s; completed fits remain resumable',job['id']);raise
        completed+=1
        write_json(run/'progress.json',{'phase':name,'phase_complete':completed,'phase_total':len(jobs),'last_job':job['id'],'updated':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        LOG.info('WRITE phase=%s completed=%d/%d fit=%s val=%.6f epochs=%d reused=%s',name,completed,len(jobs),job['id'],result['best_val_score'],result['actual_epochs'],result['reused'])
    write_json(run/('phase_'+name+'.complete.json'),{'jobs':[j['id'] for j in jobs],'completed':time.strftime('%Y-%m-%dT%H:%M:%S%z')})

def execute(root,config_path,run_override=None,preflight=False):
    root=Path(root).resolve();cfg=yaml.safe_load((root/config_path).read_text(encoding='utf-8'))
    run=root/'runs'/(run_override or cfg['experiment_id']);run.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(run/'pipeline.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)])
    LOG.info('READ config %s; WRITE %s; SKIP sealed run mutations',config_path,run)
    plan=plan_jobs(cfg,root,run)
    if preflight:
        job=copy.deepcopy(next(j for j in plan['window'] if not j.get('reuse')))
        job['training']['epochs']=2;job['training']['early_stopping_patience']=2
        job['directory']=str(run/'preflight'/job['id']);job['id']='preflight_'+job['id'];job['diagnostics']=True
        result=fit_job(job);write_json(run/'cuda_preflight.json',result);return
    freeze(root,run,cfg,config_path,plan)
    all_jobs={j['id']:j for group in plan.values() for j in group}
    with futures.ProcessPoolExecutor(max_workers=int(cfg['training']['workers'])) as pool:
        for stage in ['window','future','mechanism','search_pilot']:
            run_group(stage,plan[stage],pool,run,all_jobs)
        pilots=pd.DataFrame([{**json.loads((Path(j['directory'])/'fit.json').read_text()),'role':'validation'} for j in plan['search_pilot']])
        chosen=select_candidates(pilots,[p['id'] for p in cfg['candidate_packages']])
        finalists=[]
        for family,candidate_id in chosen.items():
            package=next(p for p in cfg['candidate_packages'] if p['id']==candidate_id)
            for seed in cfg['seeds'][1:]:
                job=_job(cfg,root,run,'search',family,seed,tag=family+'_'+candidate_id,candidate=candidate_id,model_options={'dropout':package['dropout']},training_override={'lr':package['lr']})
                finalists.append(job);all_jobs[job['id']]=job
        write_json(run/'hpo_selection.lock.json',{'source_role':'validation','candidate_budget':3,'chosen':chosen,'pilot_rows':pilots.to_dict('records')})
        run_group('search_final',finalists,pool,run,all_jobs)
        run_group('generalization',plan['generalization'],pool,run,all_jobs)
    results=[json.loads((Path(j['directory'])/'fit.json').read_text()) for j in all_jobs.values()]
    pd.DataFrame(results).to_csv(run/'fit_registry.csv',index=False)
    window_rows=pd.DataFrame([r for r in results if r['stage']=='window'])
    selected_windows={f:int(g.groupby('window').best_val_score.mean().idxmin()) for f,g in window_rows.groupby('family')}
    lock={'validation_selected_windows':selected_windows,'hpo_selected_candidates':chosen,'all_fits':{r['job_id']:r['checkpoint_sha256'] for r in results},
          'selection_role':'validation','test_payload_loaded':False,'created':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    if (run/'selections.lock.json').exists():
        previous=json.loads((run/'selections.lock.json').read_text())
        if previous['all_fits']!=lock['all_fits'] or previous['hpo_selected_candidates']!=chosen:raise ValueError('Selection lock changed')
    else:write_json(run/'selections.lock.json',lock)
    write_json(run/'executed_jobs.json',all_jobs)
    LOG.info('WRITE locked all validation selections; now permit TEST evaluation')
    from virtual_imu.context_study.evaluation import evaluate_and_report
    evaluate_and_report(root,run,cfg,all_jobs,lock)
