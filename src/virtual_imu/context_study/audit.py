"""Independent array-level recomputation; no training/scoring helper is imported."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
LOW=['gx_low','gy_low','gz_low','ax_low','ay_low','az_low']
HIGH=['gx_high','gy_high','gz_high','ax_high','ay_high','az_high']

def recompute(frame,prediction):
    high=frame[HIGH].to_numpy(dtype=np.float64);raw=frame[LOW].to_numpy(dtype=np.float64)
    error=prediction-high
    gyro=float(np.sqrt(np.sum(error[:,:3]**2)/(len(error)*3)))
    accel=float(np.sqrt(np.sum(error[:,3:]**2)/(len(error)*3)))
    ratios=[]
    for pair,indices in frame.groupby('pair_id',sort=False).indices.items():
        e=error[indices];r=raw[indices]-high[indices]
        g=np.sqrt(np.sum(e[:,:3]**2)/e[:,:3].size)/max(np.sqrt(np.sum(r[:,:3]**2)/r[:,:3].size),1e-12)
        a=np.sqrt(np.sum(e[:,3:]**2)/e[:,3:].size)/max(np.sqrt(np.sum(r[:,3:]**2)/r[:,3:].size),1e-12)
        ratios.append((g+a)/2)
    return dict(score=float(np.sum(ratios)/len(ratios)),gyro_rmse=gyro,accel_rmse=accel)

def audit_results(run,jobs,cfg,lock):
    run=Path(run);metrics=pd.read_csv(run/'metrics_per_seed.csv');overall=metrics[metrics.aggregation=='overall']
    fits=pd.read_csv(run/'fit_registry.csv');sources={};rechecks=[]
    execution=json.loads((run/'execution_pool_override.json').read_text(encoding='utf-8'))
    repository=Path(next(iter(jobs.values()))['root'])
    for name,digest in execution['source_sha256'].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Runtime execution source changed '+name)
    if execution['training_math_changed'] or execution['actual_workers']!=4:raise AssertionError('Unexpected execution override')
    def data(held,role):
        key=(held,role)
        if key not in sources:sources[key]=pd.read_parquet(run/'supports'/(held+'__'+role+'.parquet'))
        return sources[key]
    def compare(row,values,identity):
        for key,value in values.items():
            if not np.isclose(value,float(row[key]),rtol=1e-10,atol=1e-12):raise AssertionError('Independent '+key+' mismatch '+identity)
        return max(abs(values[key]-float(row[key])) for key in values)
    for identity,job in jobs.items():
        if job['stage']=='search' and job['candidate']!=lock['hpo_selected_candidates'][job['family']]:continue
        held=job.get('held_sensor') or 'shared'
        for role in ['validation','test']:
            frame=data(held,role);prediction=np.load(run/'predictions'/(identity+'__'+role+'.npz'))
            receipt=json.loads((run/'supports'/(held+'__'+role+'.json')).read_text(encoding='utf-8'))
            if str(prediction['support_sha256'])!=receipt['support_sha256']:raise AssertionError('Prediction support hash mismatch')
            rows=overall[(overall.job_id==identity)&(overall.role==role)]
            if len(rows)!=1:raise AssertionError('Ambiguous metric row '+identity)
            error=compare(rows.iloc[0],recompute(frame,prediction['prediction']),identity)
            rechecks.append(dict(job_id=identity,role=role,max_numeric_error=error))
    affine_receipts=[]
    for held in ['shared']+sorted({j['held_sensor'] for j in jobs.values() if j.get('held_sensor')}):
        train=data(held,'train');x=np.column_stack([train[LOW].to_numpy(float),np.ones(len(train))])
        fresh=np.linalg.lstsq(x,train[HIGH].to_numpy(float),rcond=None)[0]
        saved=np.load(run/'supports'/(held+'__affine_train_coefficients.npy'))
        if not np.allclose(fresh,saved,rtol=1e-7,atol=1e-9):raise AssertionError('Affine was not fitted on declared TRAIN support')
        split=json.loads((run/'supports'/(held+'__train.json')).read_text(encoding='utf-8'))['split_audit']
        physical={role:set(group) for role,group in split['physical_groups'].items()}
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            if physical[a]&physical[b]:raise AssertionError('Physical-recording overlap')
        if held!='shared':
            if any(held in split['sensors'][role] for role in ['train','validation']):raise AssertionError('Held sensor in source TRAIN/VALID')
            if split['sensors']['test']!=[held]:raise AssertionError('Unexpected TEST sensor')
        for role in ['validation','test']:
            frame=data(held,role);low=frame[LOW].to_numpy(float)
            for family,pred in [('raw',low),('affine',np.column_stack([low,np.ones(len(low))])@saved)]:
                row=overall[(overall.stage=='baseline')&(overall.family==family)&(overall.held_sensor==held)&(overall.role==role)].iloc[0]
                compare(row,recompute(frame,pred),family+'__'+held)
        affine_receipts.append(dict(held_sensor=held,fit_rows=len(train),train_coefficient_max_error=float(np.max(np.abs(fresh-saved))),recording_and_sensor_isolation=True))
    for _,row in fits.iterrows():
        fit=json.loads((Path(row.directory)/'fit.json').read_text(encoding='utf-8'))
        if fit['source_roles_loaded']!=['train','validation']:raise AssertionError('TEST used in fitting')
    windows=fits[fits.stage=='window']
    for family,group in windows.groupby('family'):
        scores={int(w):float(g.best_val_score.mean()) for w,g in group.groupby('window')}
        if min(scores,key=scores.get)!=int(lock['validation_selected_windows'][family]):raise AssertionError('Window selection is not VALID-only')
    pilot=fits[(fits.stage=='search')&(fits.seed==cfg['seeds'][0])]
    for family,group in pilot.groupby('family'):
        if set(group.candidate)!={c['id'] for c in cfg['candidate_packages']}:raise AssertionError('Unequal search trial budget')
        chosen=group.sort_values(['best_val_score','candidate']).iloc[0].candidate
        if chosen!=lock['hpo_selected_candidates'][family]:raise AssertionError('Wrong HPO VALID selection')
    table=pd.read_csv(run/'window_results.csv')
    for _,row in table.iterrows():
        values=overall[(overall.stage=='window')&(overall.role=='test')&(overall.family==row.family)&(overall.window==row.window)].score.to_numpy(float)
        if len(values)!=3 or abs(values.mean()-row.S)>1e-12 or abs(values.std(ddof=1)-row.SD)>1e-12:raise AssertionError('Incorrect window mean/SD')
    deployment=json.loads((run/'deployment_selection.lock.json').read_text(encoding='utf-8'))['zero_future_configurations']
    reporting=json.loads((run/'causal_configuration_reporting.lock.json').read_text(encoding='utf-8'))
    for name,digest in reporting['sources'].items():
        if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Causal reporting source changed '+name)
    expected_ids=set()
    for identity,job in jobs.items():
        opts=job.get('model_options',{})
        alias=all(opts.get(k,v)==v for k,v in dict(depth=3,width=64,kernel=5,dilation=1,norm='ln',activation='gelu').items())
        if opts.get('lookahead')==0 and (job['stage']=='future' or job['stage']=='mechanism' and opts.get('norm','ln')!='bn' and not alias):expected_ids.add(identity)
    listed=[identity for c in deployment['candidates'] for identity in c['job_ids']]
    if len(listed)!=len(set(listed)) or set(listed)!=expected_ids or len(deployment['candidates'])!=17:raise AssertionError('Causal configuration eligibility mismatch')
    config_table=pd.read_csv(run/'causal_configuration_results.csv')
    for candidate in deployment['candidates']:
        source=fits[fits.job_id.isin(candidate['job_ids'])]
        if sorted(source.seed.tolist())!=sorted(cfg['seeds']):raise AssertionError('Causal candidate missing seeds')
        if abs(float(source.best_val_score.mean())-candidate['validation_score'])>1e-12:raise AssertionError('Causal selection used wrong score')
        row=config_table[config_table.candidate_id==candidate['candidate_id']].iloc[0]
        values=overall[overall.job_id.isin(candidate['job_ids'])&overall.role.eq('test')].score.to_numpy(float)
        if len(values)!=3 or abs(values.mean()-row.S)>1e-12 or abs(values.std(ddof=1)-row.SD)>1e-12:raise AssertionError('Causal configuration mean/SD mismatch')
    chosen=min(deployment['candidates'],key=lambda c:(c['validation_score'],c['candidate_id']))['candidate_id']
    if chosen!=deployment['selected_candidate_id'] or config_table.loc[config_table.VALID_selected,'candidate_id'].tolist()!=[chosen]:raise AssertionError('Causal VALID choice mismatch')
    expected_evaluations=len(jobs)-len(cfg['search_families'])*(len(cfg['candidate_packages'])-1)
    if len(rechecks)!=2*expected_evaluations:raise AssertionError('Missing evaluated model role')
    if len(fits)!=227 or int(fits.reused.sum())!=30:raise AssertionError('Incomplete planned matrix')
    for suffix in ['svg','pdf','png']:
        paths=list((run/'figures').glob('fig*.'+suffix))
        if len(paths)!=9 or any(p.stat().st_size<1000 for p in paths):raise AssertionError('Missing figure exports')
    from virtual_imu.context_study.precision_audit import audit_precision
    precision=audit_precision(run,repository)
    return dict(status='PASS',numeric_precision_recovery=precision,method='Independent NumPy array recomputation, no production scorer import',fits=len(fits),
                new_fits=int((~fits.reused).sum()),exact_reuse=int(fits.reused.sum()),prediction_role_recomputations=len(rechecks),
                maximum_metric_recompute_error=max(r['max_numeric_error'] for r in rechecks),affine_checks=affine_receipts,
                window_mean_sd_checks=len(table),validation_only_selection=True,figures=27,execution_workers=execution['actual_workers'],execution_sources_unchanged=True,source_roles='train and validation only',causal_configurations=17,causal_configuration_selection='VALID only')
