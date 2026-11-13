"""Behavioral controls for the ALiMU mechanism study."""
import numpy as np
import pandas as pd
import pytest
import torch
from virtual_imu.context_study.models import ContextModel, build_model
from virtual_imu.context_study.data import strict_fold_records, validate_split_records
from virtual_imu.context_study.explain import exact_group_shapley
from virtual_imu.context_study.pipeline import select_candidates

torch.set_num_threads(1)

@pytest.mark.parametrize("backend", ["lstm","gru","bigru","transformer","tcn"])
@pytest.mark.parametrize("lookahead", [0,6])
@pytest.mark.parametrize("training", [False,True])
def test_declared_future_bound_in_training_and_inference(backend,lookahead,training):
    torch.manual_seed(7)
    m=ContextModel(backend=backend,lookahead=lookahead,width=16,hidden=12,depth=2,kernel=3)
    m.train(training)
    x=torch.randn(2,6,30); altered=x.clone(); altered[:,:,11+lookahead+1:]+=50
    with torch.no_grad():
        a=m(x); b=m(altered)
    torch.testing.assert_close(a[:,:,11],b[:,:,11],atol=2e-5,rtol=2e-5)
    assert a.shape==x.shape

def test_strict_folds_remove_held_device_from_train_and_validation():
    rec={r:[dict(low_sensor=s,physical_group_id=r,sequence_id=r+s) for s in ['a','b','c']] for r in ['train','validation','test']}
    f=strict_fold_records(rec,'c'); validate_split_records(f,held_sensor='c')
    assert {d['low_sensor'] for d in f['test']}=={'c'}
    assert all(d['low_sensor']!='c' for r in ['train','validation'] for d in f[r])
    f['validation'][0]['physical_group_id']='train'
    with pytest.raises(ValueError,match='recording'):
        validate_split_records(f,held_sensor='c')

def test_exact_shapley_of_additive_physical_output():
    x=np.arange(12,dtype=float).reshape(1,3,4); bg=np.zeros((2,3,4))
    groups=np.arange(12).reshape(3,4)//4
    def predict(v):
        return np.column_stack([v.sum((1,2)),(v*np.arange(1,4)[None,:,None]).sum((1,2))])
    phi,base,out=exact_group_shapley(predict,x,bg,groups)
    np.testing.assert_allclose(phi.sum(1)+base,out,atol=1e-9)
    np.testing.assert_allclose(phi[0,:,0],x.sum(-1)[0])
    np.testing.assert_allclose(phi[0,:,1],x.sum(-1)[0]*np.arange(1,4))

def test_selection_refuses_test_rows_and_uses_full_candidate_budget():
    d=pd.DataFrame([dict(family='a',candidate=c,seed=1,best_val_score=v,role='validation') for c,v in [('c0',2),('c1',1),('c2',3)]])
    assert select_candidates(d,expected_candidates=['c0','c1','c2'])['a']=='c1'
    with pytest.raises(ValueError): select_candidates(d.iloc[:2],expected_candidates=['c0','c1','c2'])
    d.loc[0,'role']='test'
    with pytest.raises(ValueError,match='validation'): select_candidates(d,expected_candidates=['c0','c1','c2'])

def test_equal_parameter_dilation_and_approximately_matched_depth():
    def params(m): return sum(p.numel() for p in m.parameters())
    a=ContextModel(backend='gru',lookahead=0,width=64,hidden=72,depth=3,kernel=5)
    b=ContextModel(backend='gru',lookahead=0,width=64,hidden=72,depth=3,kernel=5,dilation=4)
    c=ContextModel(backend='gru',lookahead=0,width=68,hidden=72,depth=1,kernel=13)
    assert params(a)==params(b)
    assert abs(params(c)/params(a)-1)<0.05


def test_short_windows_keep_identical_supervised_targets():
    from virtual_imu.ablations.data import CommonWindows,LOW,HIGH
    n=440; rng=np.random.default_rng(4)
    d=pd.DataFrame(dict(dataset_id=['alimu']*n,pair_id=['a']*n,sequence_id=['seq']*n,timestamp_s=np.arange(n)/40))
    for axis in LOW+HIGH:d[axis]=rng.normal(size=n)
    windows=CommonWindows(d,max_window=320,block_size=40,stride=40)
    supports=[]
    for w in [40,80,160,320]:
        x,y=windows.arrays(w);supports.append(y)
        assert x.shape[-1]==w
    for y in supports:np.testing.assert_array_equal(y,supports[0])

def test_causal_convolution_taps_match_conventional_padding_and_gradient():
    from virtual_imu.context_study.models import CausalConv
    from torch.nn import functional as F
    for dilation in [1,2,4]:
        conv=CausalConv(3,5,5,dilation=dilation)
        x=torch.randn(2,3,30,requires_grad=True)
        actual=conv(x);expected=F.conv1d(F.pad(x,(4*dilation,0)),conv.weight,conv.bias,dilation=dilation)
        torch.testing.assert_close(actual,expected)
        ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
        ge=torch.autograd.grad(expected.square().sum(),x)[0]
        torch.testing.assert_close(ga,ge)


def test_exact_shapley_splits_interactions_symmetrically():
    x=np.array([[[2.,3.]]]);background=np.zeros((1,1,2));groups=np.array([[0,1]])
    phi,base,value=exact_group_shapley(lambda z:(z[:,0,0]*z[:,0,1])[:,None],x,background,groups)
    np.testing.assert_allclose(phi[0,:,0],[3.,3.]);np.testing.assert_allclose(base+phi.sum(1),value)

@pytest.mark.parametrize('device_name',['cpu','cuda'])
def test_interrupted_epoch_resume_reproduces_uninterrupted_fit(tmp_path,monkeypatch,device_name):
    if device_name=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA not present')
    import copy
    import json
    from types import SimpleNamespace
    from virtual_imu.ablations.data import CommonWindows,LOW,HIGH
    import virtual_imu.context_study.training as training
    rng=np.random.default_rng(13);n=200
    d=pd.DataFrame(dict(dataset_id=['alimu']*n,pair_id=['a']*n,sequence_id=['s']*n,timestamp_s=np.arange(n)/40))
    for low,high in zip(LOW,HIGH):
        d[low]=rng.normal(size=n);d[high]=.8*d[low]+rng.normal(size=n)*.1
    tr=CommonWindows(d,max_window=40,block_size=40,stride=40);va=CommonWindows(d,40,40,40,stats=tr.stats)
    tx,ty=tr.arrays(40);vx,_=va.arrays(40)
    monkeypatch.setattr(training,'configure',lambda cfg:torch.device(device_name))
    monkeypatch.setattr(training,'cached_data',lambda job,device:(SimpleNamespace(loaded_roles=['train','validation'],split_audit={}),tr,va,tuple(torch.from_numpy(v).to(device_name) for v in [tx,ty,vx])))
    def builder(job):
        torch.manual_seed(job['seed']);m=torch.nn.Conv1d(6,6,1);return m,{'parameters':42}
    monkeypatch.setattr(training,'build_model',builder)
    recipe=dict(amp=device_name=='cuda',lr=.001,weight_decay=.01,betas=[.9,.999],epochs=4,warmup_epochs=1,min_lr=1e-6,grad_clip_norm=1.,min_delta=.0001,early_stopping_patience=5)
    cfg=dict(data=dict(block_size=40,batch_size=2),evaluation=dict(batch_size=2),explanations=dict(epochs=[]))
    job=dict(id='fixture',directory=str(tmp_path/'whole'),config=cfg,seed=7,stage='future',family='gru',window=40,training=recipe)
    full=training.fit_job(job);reference=torch.load(tmp_path/'whole'/'best.pt',map_location='cpu')['model']
    interrupted=copy.deepcopy(job);interrupted['directory']=str(tmp_path/'resumed')
    original_save=torch.save
    def fail_after_second_epoch_checkpoint(value,path,*args,**kwargs):
        original_save(value,path,*args,**kwargs)
        if str(path).endswith('last.tmp.pt') and value.get('epoch')==2:raise RuntimeError('simulated interruption')
    monkeypatch.setattr(torch,'save',fail_after_second_epoch_checkpoint)
    with pytest.raises(RuntimeError,match='simulated interruption'):training.fit_job(interrupted)
    monkeypatch.setattr(torch,'save',original_save)
    resumed=training.fit_job(interrupted);actual=torch.load(tmp_path/'resumed'/'best.pt',map_location='cpu')['model']
    for key in reference:torch.testing.assert_close(actual[key],reference[key],atol=0,rtol=0)
    assert resumed['best_val_score']==full['best_val_score']
    a=pd.read_csv(tmp_path/'whole'/'history.csv');b=pd.read_csv(tmp_path/'resumed'/'history.csv')
    pd.testing.assert_frame_equal(a.drop(columns='epoch_seconds'),b.drop(columns='epoch_seconds'))


def test_metrics_do_not_count_devices_as_independent_recordings():
    from types import SimpleNamespace
    from virtual_imu.context_study.evaluation import metric_rows
    from virtual_imu.ablations.data import LOW,HIGH
    rows=[]
    for recording in ['r1','r2']:
        for sensor in ['a','b','c']:
            rows.append(dict(dataset_id='alimu',pair_id=sensor,original_sequence_id=recording+sensor,
                             physical_group_id=recording,**{c:float(i+1) for i,c in enumerate(LOW)},**{c:0. for c in HIGH}))
    frame=pd.DataFrame(rows)
    result=pd.DataFrame(metric_rows(SimpleNamespace(target_frame=frame),frame[LOW].to_numpy(),{},'test'))
    assert len(result[result.aggregation=='physical_recording'])==2
    assert len(result[result.aggregation=='recording'])==6
    assert (result.score==1).all()


@pytest.mark.parametrize('backend',['lstm','gru','bigru','transformer','tcn'])
@pytest.mark.parametrize('lookahead',[0,6])
def test_available_prefix_matches_offline_sample_without_trailing_wait(backend,lookahead):
    from virtual_imu.context_study.deployment import available_prediction
    model=ContextModel(backend=backend,lookahead=lookahead,width=16,depth=2,kernel=3).eval()
    x=torch.randn(2,6,70)
    with torch.no_grad():
        offline=model(x)[:,:,30]
        arrived=available_prediction(model,x[:,:,:31+lookahead],lookahead)
    torch.testing.assert_close(offline,arrived,atol=2e-5,rtol=2e-5)


def test_dual_causal_gru_matches_bigrus_initial_weights_and_keeps_both_recurrences_active():
    from virtual_imu.context_study.supplement import DualForwardGRU
    torch.manual_seed(9);a=ContextModel(backend='bigru',lookahead=0)
    torch.manual_seed(9);b=DualForwardGRU(backend='bigru',lookahead=0)
    assert set(a.state_dict())==set(b.state_dict())
    for name,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
    x=torch.randn(2,6,30,requires_grad=True);y=b(x)[:,:,20].square().sum();y.backward()
    assert b.backward_gru.weight_hh_l0.grad.abs().sum()>0
    assert a.backward_gru.weight_hh_l0.numel()==b.backward_gru.weight_hh_l0.numel()
    assert x.grad[:,:,21:].abs().max()==0


@pytest.mark.parametrize('lookahead',[0,6])
def test_dual_gru_arrived_prefix_and_future_bound(lookahead):
    from virtual_imu.context_study.supplement import DualForwardGRU
    from virtual_imu.context_study.deployment import available_prediction
    model=DualForwardGRU(backend='bigru',lookahead=lookahead)
    x=torch.randn(2,6,50);changed=x.clone();changed[:,:,21+lookahead:]+=20
    for train in [False,True]:
        model.train(train)
        with torch.no_grad():
            expected=model(x)[:,:,20]
            torch.testing.assert_close(expected,model(changed)[:,:,20],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(expected,available_prediction(model,x[:,:,:21+lookahead],lookahead),atol=2e-5,rtol=2e-5)


def test_diagnostic_export_preserves_physical_shapley_completeness_and_zero_future(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from virtual_imu.ablations.data import CommonWindows,LOW,HIGH
    import virtual_imu.context_study.data as data_module
    from virtual_imu.context_study.explain import collect_diagnostics
    from virtual_imu.context_study.models import build_model
    import json
    rng=np.random.default_rng(3);n=440
    frame=pd.DataFrame(dict(dataset_id=['alimu']*n,pair_id=['a']*n,sequence_id=['synthetic']*n,
                            original_sequence_id=['synthetic']*n,physical_group_id=['synthetic']*n,timestamp_s=np.arange(n)/40))
    for low,high in zip(LOW,HIGH):
        frame[low]=rng.normal(size=n);frame[high]=.7*frame[low]+rng.normal(size=n)*.1
    train=CommonWindows(frame);valid=CommonWindows(frame,stats=train.stats)
    source=SimpleNamespace(loaded_roles=[])
    monkeypatch.setattr(data_module,'StudySource',lambda root,config:source)
    def role_windows(source,role,stats=None):
        source.loaded_roles.append(role);return train if role=='train' else valid
    monkeypatch.setattr(data_module,'windows',role_windows)
    cfg=dict(seeds=[1,2,3],data={},explanations=dict(seed=7,samples=1,backgrounds=1,epochs=[1,10]))
    run=tmp_path/'synthetic_diagnostics';run.mkdir();directory=run/'fits'/'synthetic';directory.mkdir(parents=True)
    job=dict(id='synthetic',directory=str(directory),stage='future',family='gru',seed=1,diagnostics=True,config=cfg,
             model_options=dict(backend='gru',lookahead=0,width=8,hidden=4,depth=1,kernel=3))
    model,_=build_model(job)
    torch.save(dict(model=model.state_dict(),epoch=1,val_score=.4,stats=train.stats),directory/'best.pt')
    collect_diagnostics(tmp_path,run,cfg,{'synthetic':job},torch.device('cpu'))
    index=pd.read_csv(run/'diagnostics'/'shapley_index.csv')
    assert set(index.grouping)=={'axis','lag'}
    assert index.full_completeness_error.max()<1e-8
    assert index.residual_completeness_error.max()<1e-5
    lag=np.load(run/'diagnostics'/'synthetic'/'best_lag_shapley.npz')
    np.testing.assert_allclose(lag['shapley'][:,5],0,atol=1e-8)
    receipt=json.loads((run/'diagnostics'/'verification.json').read_text(encoding='utf-8'))
    assert receipt['source_roles_loaded']==['train','validation']


def test_long_tcn_changes_only_dilation_and_covers_dense_rf181():
    from virtual_imu.context_study.supplement import LongFieldTCN
    torch.manual_seed(4);short=ContextModel(backend='tcn',lookahead=0)
    torch.manual_seed(4);long=LongFieldTCN(backend='tcn',lookahead=0)
    for name,value in short.state_dict().items():torch.testing.assert_close(value,long.state_dict()[name],atol=0,rtol=0)
    assert long.total_receptive_field==181
    support=set(range(13))
    for d in [3,9,27,45]:support={v+k*d for v in support for k in range(3)}
    assert support==set(range(181))
    x=torch.randn(2,6,60);altered=x.clone();altered[:,:,31:]+=10
    with torch.no_grad():torch.testing.assert_close(long(x)[:,:,30],long(altered)[:,:,30],atol=2e-5,rtol=2e-5)
    long.zero_grad();long(torch.randn(2,6,320))[:,:,140:180].square().mean().backward()
    for block in long.backend:
        assert (block.b.weight.grad.abs().sum((0,1))>0).all(), 'A temporal kernel tap receives only padding'


def test_independent_audit_uses_equal_devices_rather_than_pooled_ratio():
    from virtual_imu.context_study.audit import recompute,LOW,HIGH
    raw=np.array([[1.]*6]+[[2.]*6]*9)
    prediction=np.array([[.2]*6]+[[1.6]*6]*9)
    frame=pd.DataFrame(raw,columns=LOW)
    for c in HIGH:frame[c]=0.
    frame['pair_id']=['small']+['large']*9
    measured=recompute(frame,prediction)
    assert measured['score']==pytest.approx(.5)
    assert measured['gyro_rmse']==pytest.approx(np.sqrt((.2**2+9*1.6**2)/10))
    assert measured['accel_rmse']==pytest.approx(measured['gyro_rmse'])
