"""Exact grouped interventional Shapley (not a gradient SHAP approximation)."""
from __future__ import annotations
import math
import numpy as np

def exact_group_shapley(predict,samples,backgrounds,groups):
    samples,backgrounds=np.asarray(samples),np.asarray(backgrounds)
    groups=np.asarray(groups)
    ids=np.unique(groups)
    if not np.array_equal(ids,np.arange(len(ids))) or groups.shape!=samples.shape[1:]:
        raise ValueError('Require contiguous exhaustive group IDs shaped like each sample')
    if len(ids)>10 or not len(backgrounds): raise ValueError('Exact enumeration limited to ten groups and nonempty TRAIN background')
    n=len(ids); masks=np.arange(1<<n); group_masks=np.array([(mask>>groups)&1 for mask in masks],bool)
    all_values=[]
    for x in samples:
        mixtures=np.where(group_masks[:,None],x[None,None],backgrounds[None])
        values=np.asarray(predict(mixtures.reshape((-1,)+x.shape)))
        if values.ndim==1: values=values[:,None]
        values=values.reshape(1<<n,len(backgrounds),-1).mean(1)
        all_values.append(values)
    values=np.asarray(all_values);phi=np.zeros((len(samples),n,values.shape[-1]),float)
    for group in range(n):
        for mask in masks:
            if mask&(1<<group): continue
            size=int(mask).bit_count() if hasattr(int,'bit_count') else bin(int(mask)).count('1')
            weight=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            phi[:,group]+=weight*(values[:,mask|(1<<group)]-values[:,mask])
    return phi,values[:,0],values[:,-1]


def collect_diagnostics(root,run,cfg,jobs,device):
    import json
    import logging
    from pathlib import Path
    import torch
    import pandas as pd
    from virtual_imu.context_study.data import StudySource,windows
    from virtual_imu.context_study.gradients import eval_forward_for_input_gradients
    from virtual_imu.context_study.supplement import build_study_model as build_model
    from virtual_imu.ablations.data import LOW,HIGH
    from virtual_imu.replacements.training import write_json,sha
    LOG=logging.getLogger(__name__)
    folder=run/'diagnostics';folder.mkdir(exist_ok=True)
    source=StudySource(root,cfg['data']);train=windows(source,'train');valid=windows(source,'validation',train.stats)
    tx,_=train.arrays(320);vx,_=valid.arrays(320)
    rng=np.random.default_rng(cfg['explanations']['seed'])
    def balanced_indices(data,n):
        pairs=data.target_frame.iloc[::40].pair_id.to_numpy()
        ids=[]
        for pair in sorted(set(pairs)):
            eligible=np.flatnonzero(pairs==pair)
            ids.extend(rng.choice(eligible,size=min(len(eligible),int(np.ceil(n/len(set(pairs))))),replace=False).tolist())
        return np.asarray(ids[:n],dtype=int)
    sample_ids=balanced_indices(valid,cfg['explanations']['samples']);background_ids=balanced_indices(train,cfg['explanations']['backgrounds'])
    x=vx[sample_ids];background=tx[background_ids]
    center=160;target_offset=center-(320-40)//2
    sample_frame=valid.target_frame.iloc[sample_ids*40+target_offset].reset_index(drop=True)
    background_frame=train.target_frame.iloc[background_ids*40+target_offset].reset_index(drop=True)
    sample_frame.to_csv(folder/'valid_explained_samples.csv',index=False)
    background_frame.to_csv(folder/'train_background_samples.csv',index=False)
    np.savez_compressed(folder/'inputs.npz',samples=x,backgrounds=background,sample_anchor_indices=sample_ids,background_anchor_indices=background_ids)
    fs=np.array(train.stats['feature_std']);fm=np.array(train.stats['feature_mean']);rs=np.array(train.stats['residual_std']);rm=np.array(train.stats['residual_mean'])
    groups_axis=np.broadcast_to(np.arange(6)[:,None],(6,320))
    t=np.arange(320);lag=np.select([t<80,t<120,t<148,t<160,t==160],[0,1,2,3,4],default=5)
    groups_lag=np.broadcast_to(lag[None],(6,320))
    records=[];shap_records=[]
    for job in jobs.values():
        if not job.get('diagnostics',False) and not (job['stage']=='future' and job['seed']==cfg['seeds'][0]):continue
        dest=folder/job['id'];dest.mkdir(exist_ok=True)
        checkpoint_dir=Path(job['directory'])
        paths=sorted((checkpoint_dir/'snapshots').glob('epoch_*.pt'))+[checkpoint_dir/'best.pt']
        for checkpoint_path in paths:
            checkpoint=torch.load(checkpoint_path,map_location='cpu');model,meta=build_model(job);model.load_state_dict(checkpoint['model']);model.to(device).eval()
            label=checkpoint_path.stem;epoch=int(checkpoint['epoch'])
            tensor=torch.from_numpy(x).to(device)
            with torch.no_grad():
                features=model.feature_extractor(tensor).transpose(1,2).cpu().numpy()
                hidden=model.encode(tensor).cpu().numpy()
                output=model(tensor).cpu().numpy()
            # Independent per-output gradients in standardized residual units; raw skip included.
            tensor.requires_grad_(True)
            normal=eval_forward_for_input_gradients(model,tensor)
            physical=normal*torch.as_tensor(rs[None,:,None],device=device,dtype=normal.dtype)+torch.as_tensor(rm[None,:,None],device=device,dtype=normal.dtype)
            physical=physical+tensor*torch.as_tensor(fs[None,:,None],device=device,dtype=normal.dtype)+torch.as_tensor(fm[None,:,None],device=device,dtype=normal.dtype)
            gradient=[]
            for axis in range(6):
                g=torch.autograd.grad((physical[:,axis,center]/rs[axis]).sum(),tensor,retain_graph=axis<5)[0]
                gradient.append(g.detach().cpu().numpy())
            gradient=np.stack(gradient,axis=1)
            np.savez_compressed(dest/(label+'_intermediate.npz'),frontend=features,hidden=hidden,normalized_residual=output,
                                input_gradient=gradient,epoch=np.array(epoch),valid_score=np.array(checkpoint.get('val_score')),checkpoint_sha256=np.array(sha(checkpoint_path)))
            records.append(dict(job_id=job['id'],checkpoint=label,epoch=epoch,path=str(dest/(label+'_intermediate.npz')),
                                future_points=job['model_options']['lookahead'],family=job['family']))
            anchor_shap=job['family']=='gru' and job['model_options']['lookahead']==0 or job['family']=='bigru' and job['model_options']['lookahead'] in [0,'full']
            extra_best=label=='best' and job['family'] in ['lstm','tcn','tcn_long','transformer','dual_gru'] and job['model_options']['lookahead']==0
            if anchor_shap and (label in ['best','epoch_001','epoch_010']) or extra_best:
                def predict(inputs):
                    values=[]
                    with torch.inference_mode():
                        for start in range(0,len(inputs),64):
                            z=torch.from_numpy(np.ascontiguousarray(inputs[start:start+64],dtype=np.float32)).to(device)
                            y=model(z)[:,:,center].double()
                            y=y*torch.as_tensor(rs[None],device=device)+torch.as_tensor(rm[None],device=device)
                            y=y+z[:,:,center].double()*torch.as_tensor(fs[None],device=device)+torch.as_tensor(fm[None],device=device)
                            values.append(y.cpu().numpy())
                    return np.concatenate(values)
                for group_name,groups in [('axis',groups_axis),('lag',groups_lag)]:
                    phi,base,value=exact_group_shapley(predict,x,background,groups)
                    correction=phi.copy()
                    raw_delta=(x[:,:,center]-background[:,:,center].mean(0))*fs
                    if group_name=='axis':
                        for axis in range(6):correction[:,axis,axis]-=raw_delta[:,axis]
                    else:correction[:,4,:]-=raw_delta
                    raw_base=background[:,:,center].mean(0)*fs+fm
                    raw_sample=x[:,:,center]*fs+fm
                    full_error=float(np.max(np.abs(phi.sum(1)+base-value)))
                    residual_error=float(np.max(np.abs(correction.sum(1)+base-raw_base-(value-raw_sample))))
                    if max(full_error,residual_error)>1e-5:raise AssertionError('Shapley completeness failed')
                    file=dest/(label+'_'+group_name+'_shapley.npz')
                    np.savez_compressed(file,shapley=phi,residual_shapley=correction,expected_value=base,prediction=value,
                                        raw_expected_value=raw_base,raw_prediction=raw_sample,residual_std=rs,groups=groups,
                                        checkpoint_sha256=np.array(sha(checkpoint_path)))
                    shap_records.append(dict(job_id=job['id'],checkpoint=label,epoch=epoch,grouping=group_name,samples=len(x),backgrounds=len(background),coalitions=64,
                                             full_completeness_error=full_error,residual_completeness_error=residual_error,path=str(file),
                                             max_future_shapley=float(np.max(np.abs(phi[:,5]))) if group_name=='lag' else None))
                    LOG.info('WRITE exact %s Shapley %s epoch=%s completeness=%.3g',group_name,job['id'],label,full_error)
            del model
            LOG.info('WRITE training intermediate %s epoch=%s',job['id'],label)
    pd.DataFrame(records).to_csv(folder/'intermediate_index.csv',index=False)
    pd.DataFrame(shap_records).to_csv(folder/'shapley_index.csv',index=False)
    write_json(folder/'verification.json',dict(status='PASS',method='exact grouped interventional Shapley, all 64 coalitions',
        background_role='train',explained_role='validation',seed=cfg['explanations']['seed'],samples=len(x),backgrounds=len(background),
        axis_groups='six measured input axes, each spanning the full window',lag_groups=['<= -81 points','-80..-41','-40..-13','-12..-1','0','+1..+159'],
        explained_output='six physical HIGH predictions at window index160; residual plots subtract exact raw-skip contribution',
        intermediate_checkpoints=len(records),shapley_exports=len(shap_records),source_roles_loaded=source.loaded_roles,gradient_backend='cuDNN gradient-capable zero-dropout recurrent forward; only recurrent flags temporarily enabled and restored; other modules remain eval',
        note='Interventional mixing can leave the joint sensor distribution. Interpret attribution as model explanation, not causal sensor physics.'))
