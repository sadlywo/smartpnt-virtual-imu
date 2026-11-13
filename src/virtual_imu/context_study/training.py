"""CUDA fits with frozen supports, intermediate states and exact resume state."""
from __future__ import annotations
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from virtual_imu.ablations.data import balanced_order,grouped_score,LOW,HIGH
from virtual_imu.replacements.training import configure,predict_normalized,write_json,json_digest,sha
from virtual_imu.context_study.data import StudySource,windows
from virtual_imu.context_study.models import build_model
LOG=logging.getLogger(__name__)
_CACHE={}

def cached_data(job,device):
    key=(job['root'],job.get('held_sensor'),str(device),json_digest(job['config']['data']))
    if key not in _CACHE:
        source=StudySource(job['root'],job['config']['data'],job.get('held_sensor'))
        train=windows(source,'train'); val=windows(source,'validation',train.stats)
        tx,ty=train.arrays(train.max_window);vx,_=val.arrays(val.max_window)
        _CACHE[key]=(source,train,val,tuple(torch.from_numpy(a).to(device) for a in [tx,ty,vx]))
    return _CACHE[key]

def _state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def _restore(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def fit_job(job):
    directory=Path(job['directory']);directory.mkdir(parents=True,exist_ok=True)
    complete=directory/'fit.json';digest=json_digest(job)
    if complete.exists():
        result=json.loads(complete.read_text())
        if result['job_config_sha256']!=digest or result['checkpoint_sha256']!=sha(directory/'best.pt'):
            raise ValueError('Completed fit identity changed: '+job['id'])
        LOG.info('SKIP verified completed fit %s',job['id']);return result
    if job.get('reuse'): return reuse_job(job)
    handler=logging.FileHandler(directory/'training.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler);logging.getLogger().setLevel(logging.INFO)
    try:
        cfg=job['config'];device=configure(cfg);source,train,val,(tx,ty,vx)=cached_data(job,device)
        if source.loaded_roles!=['train','validation']:raise AssertionError('TEST loaded during training')
        model,meta=build_model(job);model.to(device)
        tr=job['training'];amp=bool(tr['amp']) and device.type=='cuda'
        opt=torch.optim.AdamW(model.parameters(),lr=float(tr['lr']),weight_decay=float(tr['weight_decay']),betas=tuple(tr['betas']))
        epochs=tr['epochs'];warmup=min(tr['warmup_epochs'],epochs-1);minimum=tr['min_lr']/tr['lr']
        def schedule(step):
            if warmup and step<warmup:return (step+1)/warmup
            p=min(1.,max(0.,(step-warmup)/max(1,epochs-warmup-1)))
            return minimum+(1-minimum)*.5*(1+math.cos(math.pi*p))
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
        scaler=torch.cuda.amp.GradScaler(enabled=amp)
        write_json(directory/'job.json',job);write_json(directory/'initialization.json',meta)
        write_json(directory/'stats.json',train.stats);write_json(directory/'split_audit.json',source.split_audit)
        best,patience_best,stale,best_epoch=float('inf'),float('inf'),0,0
        history=[];begin=time.perf_counter();previous_seconds=0.;first=1
        resume=directory/'last.pt'
        if resume.exists():
            ck=torch.load(resume,map_location='cpu')
            if ck['job_sha256']!=digest:raise ValueError('Resume config changed')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
            _restore(ck['rng']);history=ck['history'];best=ck['best'];patience_best=ck['patience_best'];stale=ck['stale'];best_epoch=ck['best_epoch'];first=ck['epoch']+1;previous_seconds=ck['elapsed_seconds']
            LOG.info('READ resume %s epoch=%d',resume,ck['epoch'])
        snap=bool(job.get('diagnostics',False));snapshot_epochs=set(cfg['explanations']['epochs'])
        def save_model(path,epoch,score):
            torch.save({'model':model.state_dict(),'job':job,'metadata':meta,'stats':train.stats,'epoch':epoch,'val_score':score,
                       'train_support_sha256':train.support_sha256,'validation_support_sha256':val.support_sha256},path)
        if snap and first==1:
            (directory/'snapshots').mkdir(exist_ok=True);save_model(directory/'snapshots/epoch_000.pt',0,None)
        w=job['window'];block=cfg['data']['block_size'];offset=(train.max_window-w)//2;center=(w-block)//2
        bs=cfg['data']['batch_size']
        for epoch in range(first,epochs+1):
            if stale>=tr['early_stopping_patience']:break
            model.train();start=time.perf_counter();loss_sum=0.;skipped=0;grad_sum=0.;grad_count=0
            order=balanced_order(train,job['seed'],epoch)
            for i in range(0,len(order),bs):
                ix=torch.as_tensor(order[i:i+bs],device=device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=amp):
                    pred=model(tx[ix,:,offset:offset+w])[...,center:center+block]
                    loss=F.smooth_l1_loss(pred,ty[ix])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss '+job['id'])
                scaler.scale(loss).backward();scaler.unscale_(opt)
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),tr['grad_clip_norm'],error_if_nonfinite=not amp)
                if torch.isfinite(grad):grad_sum+=float(grad);grad_count+=1
                previous=scaler.get_scale();scaler.step(opt);scaler.update();skipped+=int(scaler.get_scale()<previous)
                loss_sum+=float(loss.detach())*len(ix)
            normal=predict_normalized(model,vx,window=w,max_window=val.max_window,block_size=block,batch_size=cfg['evaluation']['batch_size'],amp=amp)
            physical=val.physical(normal);score=grouped_score(val.target_frame,physical)
            err=physical-val.target_frame[HIGH].to_numpy(float)
            history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),val_score=score,
                                val_gyro_rmse=float(np.sqrt(np.mean(err[:,:3]**2))),val_accel_rmse=float(np.sqrt(np.mean(err[:,3:]**2))),
                                lr=opt.param_groups[0]['lr'],gradient_norm=grad_sum/max(grad_count,1),
                                epoch_seconds=time.perf_counter()-start,amp_skipped_steps=skipped,
                                order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
            if score<best:
                best,best_epoch=score,epoch;save_model(directory/'best.pt',epoch,score)
            if snap and epoch in snapshot_epochs:save_model(directory/('snapshots/epoch_%03d.pt'%epoch),epoch,score)
            if score<patience_best-tr['min_delta']:patience_best,stale=score,0
            else:stale+=1
            scheduler.step()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                            rng=_state(),history=history,best=best,best_epoch=best_epoch,patience_best=patience_best,stale=stale,
                            epoch=epoch,job_sha256=digest,elapsed_seconds=previous_seconds+time.perf_counter()-begin),directory/'last.tmp.pt')
            (directory/'last.tmp.pt').replace(resume)
            LOG.info('WRITE %s epoch=%d/%d val=%.6f loss=%.5f seconds=%.2f stale=%d',job['id'],epoch,epochs,score,history[-1]['train_loss'],history[-1]['epoch_seconds'],stale)
        elapsed=previous_seconds+time.perf_counter()-begin
        result=dict(job_id=job['id'],stage=job['stage'],family=job['family'],window=w,seed=job['seed'],candidate=job.get('candidate','base'),
                    held_sensor=job.get('held_sensor',''),lookahead=job.get('model_options',{}).get('lookahead','legacy'),
                    best_epoch=best_epoch,actual_epochs=len(history),best_val_score=best,elapsed_seconds=elapsed,
                    parameters=meta['parameters'],train_anchors=len(train),validation_anchors=len(val),
                    train_support_sha256=train.support_sha256,validation_support_sha256=val.support_sha256,
                    checkpoint_sha256=sha(directory/'best.pt'),job_config_sha256=digest,directory=str(directory),
                    reused=False,device=str(device),amp_skipped_steps=sum(r['amp_skipped_steps'] for r in history),
                    source_roles_loaded=list(source.loaded_roles))
        write_json(complete,result);LOG.info('WRITE completed %s seconds=%.1f',job['id'],elapsed);return result
    finally:
        logging.getLogger().removeHandler(handler);handler.close()

def reuse_job(job):
    import shutil,yaml
    origin=Path(job['reuse']['directory']);target=Path(job['directory'])
    old=torch.load(origin/'best.pt',map_location='cpu')
    oldcfg=old['job']['config'];cfg=job['config']
    for key in ['data','model','training']:
        if oldcfg[key]!=cfg[key]:raise ValueError('Reuse recipe mismatch '+key+' '+job['id'])
    if old['job'].get('training',oldcfg['training'])!=job['training']:raise ValueError('Actual reuse training budget differs')
    if old['job'].get('stage') in {'future','mechanism'} and old['job'].get('model_options')!=job.get('model_options'):raise ValueError('Reuse architecture differs')
    if 'replacement_models' in oldcfg and oldcfg['replacement_models']!=cfg['replacement_models']:raise ValueError('Replacement recipe changed')
    if sha(origin/'best.pt')!=job['reuse']['checkpoint_sha256']:raise ValueError('Reuse checkpoint changed')
    device=configure(cfg);source,train,val,_=cached_data(job,device)
    if old['train_support_sha256']!=train.support_sha256 or old['validation_support_sha256']!=val.support_sha256:raise ValueError('Reuse target support differs')
    model,meta=build_model(job);model.load_state_dict(old['model'],strict=True)
    shutil.copy2(origin/'best.pt',target/'best.pt');shutil.copy2(origin/'history.csv',target/'history.csv')
    write_json(target/'job.json',job);write_json(target/'stats.json',train.stats);write_json(target/'initialization.json',meta);write_json(target/'split_audit.json',source.split_audit)
    row=job['reuse']['row'];result=dict(job_id=job['id'],stage=job['stage'],family=job['family'],window=job['window'],seed=job['seed'],candidate=job.get('candidate','base'),held_sensor=job.get('held_sensor',''),lookahead=job.get('model_options',{}).get('lookahead','legacy'),
         best_epoch=int(row['best_epoch']),actual_epochs=int(row['actual_epochs']),best_val_score=float(row['best_val_score']),
         elapsed_seconds=0.,source_elapsed_seconds=float(row['elapsed_seconds']),parameters=meta['parameters'],train_anchors=len(train),validation_anchors=len(val),
         train_support_sha256=train.support_sha256,validation_support_sha256=val.support_sha256,checkpoint_sha256=sha(target/'best.pt'),
         job_config_sha256=json_digest(job),directory=str(target),reused=True,device='cuda_verified_source',source_directory=str(origin),source_roles_loaded=list(source.loaded_roles))
    write_json(target/'fit.json',result);LOG.info('SKIP training: exact verified reuse %s from %s',job['id'],origin);return result
