"""Pre-TEST amendment: two fully active causal GRU branches match BiGRU capacity."""
from __future__ import annotations
import copy
import concurrent.futures as futures
import json
import logging
from pathlib import Path
import pandas as pd
import torch
import yaml
from virtual_imu.context_study.models import ContextModel,build_model
from virtual_imu.replacements.training import write_json,sha,json_digest
LOG=logging.getLogger(__name__)

class DualForwardGRU(ContextModel):
    def __init__(self,**options):
        if options.get('backend')!='bigru' or options.get('lookahead')=='full':raise ValueError('Dual forward control requires two GRUs and finite future')
        super().__init__(**options)

    def encode(self,x):
        z=self.feature_extractor(x).transpose(1,2)
        left=self.forward_gru(z)[0];right=self.backward_gru(z)[0]
        z=self.projection(torch.cat([left,right],dim=-1))
        if self.lookahead:
            indices=(torch.arange(z.shape[1],device=z.device)+self.lookahead).clamp(max=z.shape[1]-1)
            z=z[:,indices]
        return z

class LongFieldTCN(ContextModel):
    def __init__(self,backend_dilations=(3,9,27,45),**options):
        if len(backend_dilations)!=4 or any(not isinstance(d,int) or d<1 for d in backend_dilations):raise ValueError('Require four positive integer TCN dilations')
        if options.get('backend')!='tcn' or options.get('lookahead')=='full':raise ValueError('Long TCN requires causal TCN and finite future')
        super().__init__(**options)
        for block,dilation in zip(self.backend,backend_dilations):block.b.dilation=(dilation,)
        self.total_receptive_field=self.feature_extractor.receptive_field+2*sum(backend_dilations)

def build_study_model(job):
    if job.get('long_field_tcn',False):
        from virtual_imu.utils.seed import seed_everything
        seed_everything(job['seed']);model=LongFieldTCN(backend_dilations=job.get('long_field_dilations',(3,9,27,45)),**job['model_options'])
        return model,dict(implementation='context_study.supplement.LongFieldTCN',parameters=sum(p.numel() for p in model.parameters()),future_points=job['model_options']['lookahead'],total_receptive_field=model.total_receptive_field,model_options=job['model_options'])
    if not job.get('dual_forward',False):return build_model(job)
    from virtual_imu.utils.seed import seed_everything
    seed_everything(job['seed']);model=DualForwardGRU(**job['model_options'])
    return model,dict(implementation='context_study.supplement.DualForwardGRU',parameters=sum(p.numel() for p in model.parameters()),
                      future_points=job['model_options']['lookahead'],fully_active_past_recurrent_branches=2,
                      frontend_receptive_field=model.feature_extractor.receptive_field,model_options=job['model_options'])

def fit_supplement(job):
    # In a separate worker the unchanged, source-hashed trainer calls this explicit builder.
    import virtual_imu.context_study.training as training
    original=training.build_model;training.build_model=build_study_model
    try:return training.fit_job(job)
    finally:training.build_model=original

def complete_pretest_control(root,run,cfg,jobs,base_lock):
    from virtual_imu.context_study.pipeline import _job
    config_path=root/'configs/ablations/alimu_causal_dual_gru.yaml'
    extra=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    source_paths=[config_path,root/'src/virtual_imu/context_study/supplement.py',root/'src/virtual_imu/context_study/training.py',root/'src/virtual_imu/context_study/models.py']
    provenance={p.relative_to(root).as_posix():sha(p) for p in source_paths}
    refinement=json.loads((run/'pretest_control_refinement.lock.json').read_text(encoding='utf-8'))
    for name,digest in refinement['sources'].items():
        if sha(root/name)!=digest:raise ValueError('Pretraining control definition changed '+name)
    freeze_file=run/'dual_gru_pretest_amendment.json'
    definition={'sources':provenance,'config':extra,'base_selection_lock_sha256':sha(run/'selections.lock.json'),'test_payload_loaded':False}
    if freeze_file.exists():
        if json.loads(freeze_file.read_text(encoding='utf-8'))!=definition:raise ValueError('Pretest amendment changed')
    else:write_json(freeze_file,definition)
    new=[]
    for family in extra['families']:
        for look in extra['lookaheads']:
            for seed in extra['seeds']:
                backend='bigru' if family=='dual_gru' else 'tcn'
                opts=dict(width=64,depth=3,kernel=5,dilation=1,norm='ln',activation='gelu',backend=backend,lookahead=look)
                job=_job(cfg,root,run,'future',family,seed,tag=family+'_look'+str(look),model_options=opts,
                         training_override={'epochs':extra['epochs'],'early_stopping_patience':extra['early_stopping_patience']})
                job['dual_forward']=family=='dual_gru';job['long_field_tcn']=family=='tcn_long'
                if family=='tcn_long':job['long_field_dilations']=extra['tcn_long_dilations']
                job['amendment_sha256']=json_digest(definition);job['diagnostics']=False
                new.append(job)
    with futures.ProcessPoolExecutor(max_workers=cfg['training']['workers']) as pool:
        pending={pool.submit(fit_supplement,j):j for j in new}
        for future in futures.as_completed(pending):
            result=future.result();LOG.info('WRITE dual-forward control %s val=%.6f',result['job_id'],result['best_val_score'])
    checkpoint_lock={}
    for job in new:
        result=json.loads((Path(job['directory'])/'fit.json').read_text(encoding='utf-8'))
        if sha(Path(job['directory'])/'best.pt')!=result['checkpoint_sha256']:raise ValueError('Supplement checkpoint changed')
        checkpoint_lock[job['id']]=result['checkpoint_sha256'];jobs[job['id']]=job
    supplementary={'base_selection_lock_sha256':sha(run/'selections.lock.json'),'checkpoint_sha256':checkpoint_lock,'test_payload_loaded':False,'config':extra}
    path=run/'supplementary_selection.lock.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=supplementary:raise ValueError('Supplement lock changed')
    else:write_json(path,supplementary)
    # Preserve the original215-fit lock; the linked supplement lock seals the additional12.
    rows=[json.loads((Path(j['directory'])/'fit.json').read_text(encoding='utf-8')) for j in jobs.values()]
    pd.DataFrame(rows).to_csv(run/'fit_registry.csv',index=False)
    write_json(run/'executed_jobs_with_supplement.json',jobs)
    base_lock['all_fits'].update(checkpoint_lock)
    return jobs
