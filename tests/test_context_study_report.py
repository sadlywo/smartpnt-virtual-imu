"""End-to-end reporting on explicitly synthetic analytical values in pytest temp files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from virtual_imu.viz.plots_context_study import build_figures_and_report


def test_report_keeps_selection_and_known_three_seed_statistics(tmp_path):
    run=tmp_path/'synthetic_report_fixture';run.mkdir()
    jobs={};registry=[];metrics=[]
    seeds=[1,2,3];cfg={'seeds':seeds};window_families=['cnn_lstm64','cnn_bilstm32','cnn_gru','cnn_bigru','cnn_transformer','cnn_tcn']
    search_families=['cnn_lstm64','cnn_gru','cnn_bigru','cnn_transformer','cnn_tcn','transformer','informer']
    sensors=['BMX160','LSM9DS1','MPU9250']
    def add(stage,family,seed,tag=None,window=320,opts=None,candidate='base',held='shared',base=.5,test=True):
        opts=opts or {};tag=tag or family
        ident='__'.join([stage,tag,'w'+str(window),'s'+str(seed),held]);directory=run/'fits'/ident;directory.mkdir(parents=True)
        diag=stage=='future' and family in ['gru','bigru','transformer'] and seed==1
        job=dict(id=ident,directory=str(directory),stage=stage,family=family,seed=seed,window=window,
                 model_options=opts,diagnostics=diag,config=cfg,candidate=candidate,held_sensor=None if held=='shared' else held)
        jobs[ident]=job
        registry.append(dict(job_id=ident,stage=stage,family=family,seed=seed,candidate=candidate,reused=False,elapsed_seconds=1.,best_val_score=base))
        hist=pd.DataFrame(dict(epoch=np.arange(1,11),train_loss=np.linspace(.4,.2,10),val_score=np.linspace(.6,.4,10),gradient_norm=np.linspace(.3,.1,10)))
        hist.to_csv(directory/'history.csv',index=False)
        (directory/'stats.json').write_text(json.dumps(dict(feature_std=[1]*6,residual_std=[1]*6)),encoding='utf-8')
        for role in ['validation']+(['test'] if test else []):
            score=base+seed*.001
            metrics.append(dict(job_id=ident,stage=stage,family=family,condition=tag,window=window,seed=seed,candidate=candidate,
                                held_sensor=held,lookahead=str(opts.get('lookahead','legacy')),depth=opts.get('depth',-1),norm=opts.get('norm','legacy'),
                                activation=opts.get('activation','legacy'),kernel=opts.get('kernel',-1),dilation=opts.get('dilation',1),width=opts.get('width',64),
                                role=role,aggregation='overall',score=score,gyro_rmse=score*.001,accel_rmse=score*.2))
        if stage=='future' and seed==1:
            folder=run/'diagnostics'/ident;folder.mkdir(parents=True)
            z=np.tile(np.linspace(-1,1,320)[None,:,None],(8,1,64))
            for checkpoint,epoch in [('epoch_001',1),('epoch_010',10),('best',10)]:
                np.savez(folder/(checkpoint+'_intermediate.npz'),hidden=z,input_gradient=np.ones((8,6,6,320))*.01,epoch=np.array(epoch))
                for group in ['axis','lag']:
                    np.savez(folder/(checkpoint+'_'+group+'_shapley.npz'),residual_shapley=np.ones((8,6,6))*.1,residual_std=np.ones(6))
    for fi,f in enumerate(window_families):
        for w in [40,80,160,320]:
            for seed in seeds:add('window',f,seed,window=w,base=.5+fi*.01-w*.0001)
    for fi,f in enumerate(['lstm','gru','bigru','dual_gru','transformer','tcn','tcn_long']):
        for look in [0,6]+(['full'] if f in ['bigru','transformer'] else []):
            for seed in seeds:add('future',f,seed,tag=f+'_look'+str(look),opts={'lookahead':look},base=.3+fi*.01)
    for depth,width,kernel in [(1,68,13),(3,64,5)]:
        for norm in ['bn','ln','none']:
            for act in ['gelu','identity']:
                for seed in seeds:
                    opts=dict(depth=depth,width=width,kernel=kernel,dilation=1,norm=norm,activation=act,lookahead=0)
                    add('mechanism','gru',seed,tag='d%d_%s_%s'%(depth,norm,act),opts=opts,base=.3+depth*.01)
    for tag,kernel,dilation in [('rf25',5,2),('rf49',5,4),('pointwise',1,1)]:
        for seed in seeds:add('mechanism','gru',seed,tag=tag,opts=dict(depth=3,width=64,kernel=kernel,dilation=dilation,norm='ln',activation='gelu',lookahead=0),base=.3)
    for fi,f in enumerate(search_families):
        for candidate,base in [('c0',.4),('c1',.2),('c2',.3)]:
            for seed in seeds if candidate=='c1' else [1]:add('search',f,seed,tag=f+'_'+candidate,candidate=candidate,base=base+fi*.01,test=candidate=='c1')
    for sensor in sensors:
        for fi,f in enumerate(['cnn_gru','cnn_bigru','informer']):
            for seed in seeds:add('generalization',f,seed,held=sensor,base=.35+fi*.01)
    for held in ['shared']+sensors:
        for family,score in [('raw',1.),('affine',.7)]:
            for role in ['validation','test']:
                metrics.append(dict(stage='baseline',family=family,held_sensor=held,role=role,aggregation='overall',score=score,gyro_rmse=score*.001,accel_rmse=score*.2))
    pd.DataFrame(registry).to_csv(run/'fit_registry.csv',index=False)
    pd.DataFrame(metrics).to_csv(run/'metrics_per_seed.csv',index=False)
    deploy={'zero_future_family':'lstm','strict_fold_families':{s:'cnn_gru' for s in sensors}}
    from virtual_imu.context_study.causal_candidates import select_causal_configurations
    deploy['zero_future_configurations']=select_causal_configurations(jobs,{r['job_id']:r['best_val_score'] for r in registry},seeds)
    (run/'deployment_selection.lock.json').write_text(json.dumps(deploy),encoding='utf-8')
    (run/'deployment_latency_audit.json').write_text(json.dumps({'max_known_low_preprocessing_delay_s':.025}),encoding='utf-8')
    lock={'hpo_selected_candidates':{f:'c1' for f in search_families},'validation_selected_windows':{f:320 for f in window_families}}
    build_figures_and_report(run,cfg,jobs,lock)
    answer=json.loads((run/'answers.json').read_text(encoding='utf-8'))
    assert answer['short_window_observed_winners']['40']['family']=='cnn_lstm64'
    assert answer['short_window_observed_winners']['40']['S']==pytest.approx(.498)
    assert answer['short_window_observed_winners']['40']['SD']==pytest.approx(.001)
    causal=pd.read_csv(run/'causal_configuration_results.csv')
    assert len(causal)==17 and causal.VALID_selected.sum()==1
    assert answer['zero_future_configuration_validation_selected']['candidate_id']=='future__lstm_look0'
    assert answer['equal_budget_observed_best']['family']=='cnn_lstm64'
    assert all(r['family']=='cnn_gru' for r in answer['strict_observed_per_device'].values())
    assert len(list((run/'figures').glob('fig*.png')))==9
    assert len(list((run/'figures').glob('fig*.pdf')))==9
    assert len(list((run/'figures').glob('fig*.svg')))==9
    contrasts=pd.read_csv(run/'paired_contrasts.csv')
    window=contrasts[contrasts.contrast=='window320_minus40_cnn_gru']
    np.testing.assert_allclose(window.score_delta,-.028,atol=1e-12)

import pytest
