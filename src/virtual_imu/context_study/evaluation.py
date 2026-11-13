"""Evaluate only after every VALID selection and checkpoint has been locked."""
from __future__ import annotations
import json
import logging
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from virtual_imu.ablations.data import grouped_score,physical_metrics,LOW,HIGH
from virtual_imu.context_study.data import StudySource,windows
from virtual_imu.context_study.gradients import eval_forward_for_input_gradients
from virtual_imu.context_study.causal_candidates import eligible_causal_configuration,select_causal_configurations
from virtual_imu.context_study.supplement import build_study_model as build_model
from virtual_imu.replacements.training import configure,predict_normalized,sha,write_json
LOG=logging.getLogger(__name__)

def describe_job(job):
    opts=job.get('model_options',{})
    return dict(job_id=job['id'],stage=job['stage'],family=job['family'],condition=job['id'].split('__')[1],
                window=job['window'],seed=job['seed'],candidate=job.get('candidate','base'),held_sensor=job.get('held_sensor') or 'shared',
                lookahead=str(opts.get('lookahead','legacy')),depth=opts.get('depth',-1),norm=opts.get('norm','legacy'),
                activation=opts.get('activation','legacy'),kernel=opts.get('kernel',-1),dilation=opts.get('dilation',1),width=opts.get('width',64))


def improvement_fields(frame,row):
    raw=frame[LOW].to_numpy(float)-frame[HIGH].to_numpy(float)
    for name,part in [('gyro',slice(0,3)),('accel',slice(3,6))]:
        baseline=float(np.sqrt(np.mean(raw[:,part]**2)))
        row['raw_'+name+'_rmse']=baseline
        row[name+'_improvement_pct']=100*(1-row[name+'_rmse']/baseline) if baseline>1e-12 else float('nan')
    row['relative_error_reduction_pct']=100*(1-row['score'])
    return row

def metric_rows(data,prediction,identity,role):
    rows=[]
    for result in physical_metrics(data.target_frame,prediction):
        if result['aggregation']=='overall':ix=np.arange(len(prediction))
        else:
            valid=np.ones(len(prediction),bool)
            columns={'dataset':['dataset_id'],'pair':['dataset_id','pair_id'],'recording':['dataset_id','pair_id','original_sequence_id']}[result['aggregation']]
            for col in columns:valid&=(data.target_frame[col].to_numpy()==result[col])
            ix=np.flatnonzero(valid)
        result['physical_group_id']='__all__'
        result.update(identity,role=role,score=grouped_score(data.target_frame.iloc[ix],prediction[ix]))
        rows.append(improvement_fields(data.target_frame.iloc[ix],result))
    for physical,ix in data.target_frame.groupby('physical_group_id',sort=False).indices.items():
        local=data.target_frame.iloc[ix].reset_index(drop=True)
        result=physical_metrics(local,prediction[ix])[0]
        result.update(identity,role=role,aggregation='physical_recording',physical_group_id=physical,score=grouped_score(local,prediction[ix]))
        rows.append(improvement_fields(local,result))
    return rows

def _latency(model,x,warmup=5,repeats=20):
    model.eval()
    with torch.inference_mode(),torch.backends.cudnn.flags(allow_tf32=False):
        for _ in range(warmup):model(x)
        torch.cuda.synchronize();begin=time.perf_counter()
        for _ in range(repeats):model(x)
        torch.cuda.synchronize()
    return (time.perf_counter()-begin)/repeats*1000

def verify_causality(run,jobs,device):
    with torch.backends.cudnn.flags(allow_tf32=False):
        return _verify_causality(run,jobs,device)

def _verify_causality(run,jobs,device):
    receipts=[]
    for job in jobs.values():
        controlled=job['stage']=='future' and job['model_options']['lookahead']!='full'
        if job['seed']!=job['config']['seeds'][0] or not (controlled or eligible_causal_configuration(job)):continue
        model,_=build_model(job);checkpoint=torch.load(Path(job['directory'])/'best.pt',map_location='cpu');model.load_state_dict(checkpoint['model']);model.to(device)
        look=job['model_options']['lookahead'];generator=torch.Generator().manual_seed(job['config']['explanations']['seed']+101)
        x=torch.randn(2,6,320,generator=generator).to(device);y=x.clone();y[:,:,160+look+1:]+=10
        for training in [False,True]:
            model.train(training)
            with torch.no_grad():a=model(x)[:,:,160];b=model(y)[:,:,160]
            error=float((a-b).abs().max())
            if error>2e-5:raise AssertionError('Future perturbation leaked '+job['id'])
            receipts.append(dict(job_id=job['id'],training=training,future_bound=look,max_output_difference=error))
        from virtual_imu.context_study.deployment import available_prediction
        model.eval()
        with torch.no_grad():
            full=model(x)
            errors=[]
            for target in [140,160,179]:
                prefix=x[:,:,:target+look+1]
                arrived=available_prediction(model,prefix,look)
                error=float((arrived-full[:,:,target]).abs().max());errors.append(error)
                if error>3e-5:raise AssertionError('Arrived-prefix replay mismatch '+job['id'])
        receipts[-1]['available_prefix_replay_errors']=errors
        model.eval();x.requires_grad_(True);gradient=torch.autograd.grad(eval_forward_for_input_gradients(model,x)[:,:,160].sum(),x)[0]
        bound=float(gradient[:,:,160+look+1:].abs().max())
        if bound>1e-8:raise AssertionError('Future Jacobian leaked '+job['id'])
        receipts[-1]['max_forbidden_future_gradient']=bound
        del model
    write_json(run/'causality_verification.json',{'status':'PASS','tests':receipts,'precision':'FP32; matmul TF32 disabled by configure, cuDNN TF32 disabled for replay audit'})
    return receipts


def fixed_checkpoint_context_diagnostic(run,jobs,device):
    rows=[]
    torch.manual_seed(next(iter(jobs.values()))['config']['explanations']['seed'])
    x=torch.randn(2,6,320,device=device)
    for job in jobs.values():
        if job['seed']!=job['config']['seeds'][0]:continue
        relevant=job['stage']=='window' and job['family']=='cnn_tcn' and job['window']==320 or job['stage']=='future' and job['family']=='tcn'
        if not relevant:continue
        model,_=build_model(job);model.load_state_dict(torch.load(Path(job['directory'])/'best.pt',map_location='cpu')['model']);model.to(device).eval()
        with torch.no_grad():
            full=model(x)[:,:,140:180]
            for width in [80,160]:
                offset=(320-width)//2;center=(width-40)//2
                cropped=model(x[:,:,offset:offset+width])[:,:,center:center+40]
                error=float((full-cropped).abs().max())
                if width==160 and error>3e-5:raise AssertionError('Fixed RF43 unexpectedly sees more than available160 context')
                rows.append(dict(job_id=job['id'],stage=job['stage'],trained_window=320,inference_window=width,
                                 local_total_receptive_field=43,max_center40_difference=error,
                                 interpretation='Fixed weights/eval normalization; 160 already covers RF43. This is not a newly trained short-window result.'))
    pd.DataFrame(rows).to_csv(run/'fixed_checkpoint_context_diagnostic.csv',index=False)
    return rows

def evaluate_and_report(root,run,cfg,jobs,lock):
    if not (run/'selections.lock.json').exists():raise RuntimeError('TEST gate requires selection lock')
    from virtual_imu.context_study.precision_audit import validate_precision_policy
    if validate_precision_policy(run,root) is not None:
        from virtual_imu.context_study.numerical_safety import install
        install(run)
    device=configure(cfg)
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    records=[r for r in manifest['records'] if r.get('dataset_id')=='alimu' and r.get('training_eligible')]
    known=[]
    for record in records:
        values=record.get('preprocessing',{}).get('total_known_latency_by_channel_s',{})
        known.extend(float(values.get(axis,0.)) for axis in LOW)
    latency_audit=dict(max_known_low_preprocessing_delay_s=max(known,default=0.),
                       model_future_points=[0,6],model_additional_delay_s=[0.,6/40],
                       note='Zero future is a network dependency constraint. Existing LOW preprocessing has a documented delay. Prefix replay avoids offline full-window/center-block buffering; unknown upstream delays are not inferred to be zero.',
                       reference_frame='upstream_corresponding_sensor_axes; extrinsics_not_independently_verified')
    write_json(run/'deployment_latency_audit.json',latency_audit)
    # Reject stale/changed training sources before TEST, including uncommitted code.
    frozen=json.loads((run/'preparation.json').read_text(encoding='utf-8'))
    for name,digest in frozen['training_sources'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen source changed: '+name)
    for ident,digest in lock['all_fits'].items():
        if sha(Path(jobs[ident]['directory'])/'best.pt')!=digest:raise ValueError('Locked checkpoint changed '+ident)
    from virtual_imu.context_study.supplement import complete_pretest_control
    jobs=complete_pretest_control(root,run,cfg,jobs,lock)
    registry=pd.read_csv(run/'fit_registry.csv',keep_default_na=False)
    zero_ids=[j['id'] for j in jobs.values() if j['stage']=='future' and j['model_options']['lookahead']==0]
    zero=registry[registry.job_id.isin(zero_ids)].groupby('family').best_val_score.mean().sort_values()
    strict={}
    for sensor,group in registry[registry.stage=='generalization'].groupby('held_sensor'):
        strict[sensor]=group.groupby('family').best_val_score.mean().sort_values().index[0]
    deployment={'role':'validation','criterion':'mean best checkpoint VALID score across the three paired optimization seeds',
                'zero_future_family':zero.index[0],'zero_future_scores':zero.to_dict(),'strict_fold_families':strict,
                'source_registry_sha256':sha(run/'fit_registry.csv'),'test_payload_loaded':False}
    reporting_lock=json.loads((run/'causal_configuration_reporting.lock.json').read_text(encoding='utf-8'))
    for name,digest in reporting_lock['sources'].items():
        if sha(root/name)!=digest:raise ValueError('Pretest configuration reporting rule changed '+name)
    deployment['zero_future_configurations']=select_causal_configurations(jobs,registry.set_index('job_id').best_val_score.to_dict(),cfg['seeds'])
    if len(deployment['zero_future_configurations']['candidates'])!=17:raise ValueError('Unexpected distinct zero-future configuration count')
    deploy_path=run/'deployment_selection.lock.json'
    if deploy_path.exists():
        if json.loads(deploy_path.read_text(encoding='utf-8'))!=deployment:raise ValueError('Deployment selection changed')
    else:write_json(deploy_path,deployment)
    analysis_sources={p.relative_to(root).as_posix():sha(p) for p in list((root/'src/virtual_imu/context_study').glob('*.py'))+[root/'src/virtual_imu/viz/plots_context_study.py',root/'scripts/check_context_study.py',root/'scripts/assist_context_study.py',root/'scripts/assist_context_generalization.py',root/'scripts/run_context_study_execution.py',root/'configs/ablations/alimu_causal_dual_gru.yaml'] if p.exists()}
    write_json(run/'analysis_sources.json',analysis_sources)
    for name in analysis_sources:
        target=run/'analysis_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    write_json(run/'metric_definitions.json',dict(primary_score='Equal-device mean of 0.5*(gyro component RMSE/raw gyro component RMSE + accel component RMSE/raw accel component RMSE); lower is better',gyro_unit='rad/s',accel_unit='m/s^2',component_rmse='sqrt(mean(error^2)) pooled over samples and three axes',relative_error_reduction_pct='100*(1-primary_score)',physical_improvement_pct='100*(1-model_component_RMSE/raw_component_RMSE) on identical support',uncertainty='sample SD over three optimization seeds, not recording confidence intervals',independent_test_recordings=2))
    (run/'supports').mkdir(exist_ok=True);(run/'predictions').mkdir(exist_ok=True)
    rows=[];costs=[];validation_receipts=[]
    sources={};data_cache={};tensors={}
    def get_data(held,role,stats=None):
        key=(held,role)
        if key not in data_cache:
            if held not in sources:sources[held]=StudySource(root,cfg['data'],held)
            data=windows(sources[held],role,stats)
            data_cache[key]=data
            name=(held or 'shared')+'__'+role
            data.target_frame.to_parquet(run/'supports'/(name+'.parquet'),index=False)
            if role!='train':
                x,_=data.arrays(320);tensors[key]=torch.from_numpy(x).to(device)
            write_json(run/'supports'/(name+'.json'),dict(support_sha256=data.support_sha256,samples=len(data.target_frame),anchors=len(data),stats=data.stats,split_audit=sources[held].split_audit))
        return data_cache[key]
    for held in [None]+sorted({j['held_sensor'] for j in jobs.values() if j.get('held_sensor')}):
        train=get_data(held,'train')
        low=train.target_frame[LOW].to_numpy(float);high=train.target_frame[HIGH].to_numpy(float)
        affine=np.linalg.lstsq(np.column_stack([low,np.ones(len(low))]),high,rcond=None)[0]
        np.save(run/'supports'/((held or 'shared')+'__affine_train_coefficients.npy'),affine)
        write_json(run/'supports'/((held or 'shared')+'__affine_fit.json'),dict(role='train',target_support_sha256=train.support_sha256,rows=len(low),method='shared multivariate LOW6 plus intercept -> measured HIGH6 least squares'))
        for role in ['validation','test']:
            data=get_data(held,role,train.stats)
            raw=data.target_frame[LOW].to_numpy(float)
            for name,pred in [('raw',raw),('affine',np.column_stack([raw,np.ones(len(raw))])@affine)]:
                identity=dict(job_id=name+'__'+(held or 'shared'),stage='baseline',family=name,condition=name,window=320,seed=0,candidate='base',held_sensor=held or 'shared',lookahead='0',depth=0,norm='none',activation='identity',kernel=1,dilation=1,width=6)
                rows.extend(metric_rows(data,pred,identity,role))
    LOG.info('READ TEST allowed by sealed selection lock; WRITE fixed baselines and support manifests')
    for number,job in enumerate(jobs.values(),1):
        if job['stage']=='search' and job['candidate']!=lock['hpo_selected_candidates'][job['family']]:
            LOG.info('SKIP TEST for unselected HPO package %s',job['id']);continue
        held=job.get('held_sensor');train=data_cache[(held,'train')]
        checkpoint=torch.load(Path(job['directory'])/'best.pt',map_location='cpu')
        model,meta=build_model(job);model.load_state_dict(checkpoint['model'],strict=True);model.to(device)
        if checkpoint['stats']!=train.stats:raise ValueError('TRAIN-only statistics changed '+job['id'])
        identity=describe_job(job)
        for role in ['validation','test']:
            data=data_cache[(held,role)];x=tensors[(held,role)]
            normal=predict_normalized(model,x,window=job['window'],max_window=320,block_size=40,batch_size=cfg['evaluation']['batch_size'],amp=cfg['training']['amp'])
            physical=data.physical(normal);score=grouped_score(data.target_frame,physical)
            fit=json.loads((Path(job['directory'])/'fit.json').read_text(encoding='utf-8'))
            if role=='validation':
                delta=abs(score-fit['best_val_score'])
                if delta>5e-5:raise AssertionError('VALID checkpoint replay mismatch '+job['id']+' '+str(delta))
                validation_receipts.append(dict(job_id=job['id'],abs_score_replay_error=delta))
            np.savez_compressed(run/'predictions'/(job['id']+'__'+role+'.npz'),prediction=physical,support_sha256=np.array(data.support_sha256),checkpoint_sha256=np.array(fit['checkpoint_sha256']))
            rows.extend(metric_rows(data,physical,identity,role))
        x=tensors[(held,'validation')];offset=(320-job['window'])//2
        costs.append({**identity,'parameters':meta['parameters'],'batch1_latency_ms':_latency(model,x[:1,:,offset:offset+job['window']]),
                      'batch32_latency_ms':_latency(model,x[:32,:,offset:offset+job['window']]),'actual_epochs':fit['actual_epochs'],
                      'training_seconds':fit.get('source_elapsed_seconds',fit['elapsed_seconds']),'incremental_training_seconds':fit['elapsed_seconds'],'latency_precision':'float32','latency_cudnn_tf32':False,'latency_warmup':5,'latency_repeats':20,'reused':fit['reused']})
        look=job.get('model_options',{}).get('lookahead')
        if (job['stage']=='future' or eligible_causal_configuration(job)) and isinstance(look,int):
            prefix_ms=_latency(model,x[:1,:,:161+look])
            costs[-1].update(available_prefix_latency_ms=prefix_ms,model_future_delay_ms=look/40*1000,known_preprocessing_delay_ms=latency_audit['max_known_low_preprocessing_delay_s']*1000,
                             minimum_accounted_latency_ms=prefix_ms+look/40*1000+latency_audit['max_known_low_preprocessing_delay_s']*1000,context_warmup_points=140)
        LOG.info('WRITE evaluation %d/%d %s TEST score=%.6f',number,len(jobs),job['id'],score)
        pd.DataFrame(rows).to_csv(run/'metrics_per_seed.partial.csv',index=False)
        del model
    metrics=pd.DataFrame(rows);metrics.to_csv(run/'metrics_per_seed.csv',index=False)
    pd.DataFrame(costs).to_csv(run/'compute_cost.csv',index=False)
    group=['stage','family','condition','window','held_sensor','lookahead','depth','norm','activation','kernel','dilation','width','role','aggregation','dataset_id','pair_id','original_sequence_id','physical_group_id']
    numerical=['relative_error_reduction_pct','gyro_improvement_pct','accel_improvement_pct','score','gyro_rmse','accel_rmse','gx_rmse','gy_rmse','gz_rmse','ax_rmse','ay_rmse','az_rmse']
    summary=metrics.groupby(group,dropna=False)[numerical].agg(['mean','std','count']).reset_index()
    summary.columns=['_'.join([v for v in col if v]) if isinstance(col,tuple) else col for col in summary.columns]
    summary.to_csv(run/'summary.csv',index=False)
    causality=verify_causality(run,jobs,device)
    context_diagnostic=fixed_checkpoint_context_diagnostic(run,jobs,device)
    from virtual_imu.context_study.explain import collect_diagnostics
    collect_diagnostics(root,run,cfg,jobs,device)
    from virtual_imu.viz.plots_context_study import build_figures_and_report
    build_figures_and_report(run,cfg,jobs,lock)
    from virtual_imu.context_study.audit import audit_results
    independent=audit_results(run,jobs,cfg,lock)
    write_json(run/'independent_audit.json',independent)
    provenance=dict(status='PASS',fit_slots=len(jobs),evaluated_models=len(validation_receipts),
                    validation_replay=validation_receipts,causal_checks=len(causality),fixed_checkpoint_context_checks=context_diagnostic,selection_lock_sha256=sha(run/'selections.lock.json'),
                    all_checkpoints_locked=True,training_sources_unchanged=True,test_selection=False,
                    limitations=['Two physical TEST recordings; three optimization seeds are not independent datasets.',
                                 'Already inspected historical TEST; no new dataset confirmation.',
                                 'LOSO uses fixed predeclared recipe, not all-device-tuned hyperparameters.'])
    write_json(run/'verification.json',provenance)
    import zipfile
    package_files=list((run/'figures').glob('*'))+[run/n for n in ['report.md','answers.json','config.yaml','verification.json','independent_audit.json','summary.csv','overall_per_seed.csv','window_results.csv','future_results.csv','mechanism_results.csv','equal_budget_results.csv','strict_generalization_results.csv','paired_contrasts_summary.csv','factor_effect_digest.csv','compute_cost.csv','deployment_selection.lock.json','causal_configuration_results.csv','causal_configuration_reporting.lock.json','deployment_latency_audit.json','dual_gru_design_amendment.md','data_manifest.json','metric_definitions.json','preparation.json','environment.json','analysis_sources.json','execution_pool_override.json','execution_restart_four_workers.json','execution_assistant.json','execution_generalization_assistant.json','hpo_selection.lock.json','selections.lock.json','supplementary_selection.lock.json','pretest_control_refinement.lock.json','fixed_checkpoint_context_diagnostic.csv','diagnostics/verification.json','diagnostics/shapley_index.csv','analysis_source/configs/ablations/alimu_causal_dual_gru.yaml']]
    package_files+=list((run/'diagnostics').glob('*/*_shapley.npz'))
    package_files += [run/name for name in ['numeric_precision_recovery.lock.json','numeric_precision_execution.json','numeric_precision_audit.json','numeric_precision_recovery.csv','preparation/nonfinite_informer_probe/diagnostic.json','preparation/numeric_recovery_tests.xml']]
    package_files += [run/'analysis_source'/name for name in ['src/virtual_imu/context_study/numerical_safety.py','src/virtual_imu/context_study/precision_audit.py','scripts/run_context_study_safe_execution.py','configs/ablations/alimu_numeric_precision_recovery.yaml']]
    package_files += list((run/'fits').glob('*/numeric_precision_events.jsonl')) + list(run.glob('numeric_precision_events.jsonl'))
    with zipfile.ZipFile(run/'ALiMU_context_mechanisms_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_files:
            if path.is_file():archive.write(path,path.relative_to(run).as_posix())
    # Seal numerical/figure/report artifacts; logs and transient coordinator state are excluded.
    protected={str(p.relative_to(run)).replace('\\','/'):sha(p) for p in run.rglob('*') if p.is_file() and p.suffix not in ['.log'] and p.name not in ['artifact_manifest.json','complete.json','coordinator.pid','progress.json']}
    write_json(run/'artifact_manifest.json',{'files':protected,'policy':'source-of-truth run; preserve after completion'})
    write_json(run/'complete.json',dict(status='PASS',fit_slots=len(jobs),evaluated_models=len(validation_receipts),completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),verification_sha256=sha(run/'verification.json'),artifact_manifest_sha256=sha(run/'artifact_manifest.json')))
    LOG.info('WRITE completed study report and verification %s',run)
