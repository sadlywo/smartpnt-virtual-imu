"""Publication figures for the predeclared context/mechanism study."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from virtual_imu.viz.style import apply_paper_style,save_figure
LABELS={'cnn_lstm64':'CNN-LSTM64','cnn_bilstm32':'CNN-BiLSTM32','cnn_gru':'CNN-GRU','cnn_bigru':'CNN-BiGRU',
        'cnn_transformer':'CNN-Transformer','cnn_tcn':'CNN-TCN','transformer':'Transformer','informer':'Informer',
        'tcn_long':'TCN (RF181)','dual_gru':'Dual causal GRU','lstm':'LSTM','gru':'GRU','bigru':'BiGRU','tcn':'TCN','raw':'Raw LOW','affine':'Affine (TRAIN only)'}
PALETTE=['#1f77b4','#c95f3f','#2a9d8f','#9467bd','#d18f27','#5b6875','#b34c75']
AXES=['gx','gy','gz','ax','ay','az']

def _style():
    apply_paper_style();plt.rcParams['svg.fonttype']='none'

def _save(fig,out,stem):
    fig.tight_layout(pad=1.1);files=save_figure(fig,out,stem);plt.close(fig);return files

def _panel(ax,letter):
    ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)

def _legend_below(fig,ax,columns=3):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=columns,fontsize=7,frameon=False)

def _mean(group,metric='score'):
    return float(group[metric].mean()),float(group[metric].std(ddof=1)) if len(group)>1 else 0.

def live_training_preview(run):
    run=Path(run);_style();fig,axes=plt.subplots(1,3,figsize=(7.2,2.5))
    files=sorted((run/'fits').glob('*/history.csv'))
    active=[]
    for p in files:
        if (p.parent/'fit.json').exists() or time.time()-p.stat().st_mtime>120:continue
        try:d=pd.read_csv(p)
        except pd.errors.EmptyDataError:continue
        if len(d):active.append((p,d))
    if not active:
        active=[(p,pd.read_csv(p)) for p in files[:2]]
    curve_frames=[]
    for index,(p,d) in enumerate(active[:4]):
        curve_frames.append(d.assign(job_id=p.parent.name))
        job=json.loads((p.parent/'job.json').read_text(encoding='utf-8'));seed=int(job['seed'])%100
        if job['stage']=='mechanism':
            opts=job['model_options'];rf=1+opts['depth']*(opts['kernel']-1)*opts.get('dilation',1);name='Frontend d%d / RF%d / %s+%s / seed %02d'%(opts['depth'],rf,opts['norm'].upper(),opts['activation'],seed)
        elif job['stage']=='generalization':name='Held %s / %s / seed %02d'%(job['held_sensor'],LABELS[job['family']],seed)
        elif job['stage']=='future':name='%s / future %s / seed %02d'%(LABELS[job['family']],job['model_options']['lookahead'],seed)
        elif job['stage']=='search':name='%s / %s / seed %02d'%(LABELS[job['family']],job['candidate'],seed)
        else:name='%s / w%d / seed %02d'%(LABELS.get(job['family'],job['family']),job['window'],seed)
        for ax,metric in zip(axes,['train_loss','val_score','gradient_norm']):
            if metric in d:ax.plot(d.epoch,d[metric],label=name,color=PALETTE[index%len(PALETTE)])
    for i,(ax,label) in enumerate(zip(axes,['TRAIN Huber loss','VALID score (lower better)','Gradient norm (before clip)'])):
        ax.set_xlabel('Epoch');ax.set_ylabel(label);_panel(ax,chr(97+i))
    if active:
        handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=5.5,frameon=False)
    out=run/'progress_figures';fig.tight_layout(pad=1.1);save_figure(fig,out,'training_in_progress');plt.close(fig)
    (out/'figure_index.md').write_text('Training in progress; TRAIN/VALID only. Curves are intermediate states and cannot establish TEST rankings.\n',encoding='utf-8')
    if curve_frames:
        import hashlib
        pd.concat(curve_frames,ignore_index=True).to_csv(out/'curve_data.csv',index=False)
        preparation=json.loads((run/'preparation.json').read_text(encoding='utf-8'))
        receipt=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),roles=['train','validation'],git_commit=preparation['git_commit'],config_sha256=preparation['config_sha256'],manifest_sha256=preparation['manifest_sha256'],curve_data_sha256=hashlib.sha256((out/'curve_data.csv').read_bytes()).hexdigest(),rows=sum(len(d) for d in curve_frames))
        (out/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

def _window_figure(overall,run):
    _style();fig,axs=plt.subplots(2,2,figsize=(7.2,5.0))
    stages=overall[overall.stage=='window']
    for i,family in enumerate(stages.family.drop_duplicates()):
        d=stages[stages.family==family];color=PALETTE[i]
        for ax,role,metric,label in [(axs[0,0],'validation','score','VALID relative error score'),(axs[0,1],'test','score','TEST relative error score'),
                                     (axs[1,0],'test','gyro_rmse','Gyro RMSE [rad/s]'),(axs[1,1],'test','accel_rmse','Accel RMSE [m/s²]')]:
            g=d[d.role==role].groupby('window')[metric].agg(['mean','std']).reindex([40,80,160,320])
            ax.errorbar(np.arange(4),g['mean'],yerr=g['std'],marker='o',ms=3,capsize=2,color=color,label=LABELS[family])
            ax.set_xticks(np.arange(4),['40 / 1 s','80 / 2 s','160 / 4 s','320 / 8 s']);ax.set_xlabel('Window [points / seconds]');ax.set_ylabel(label)
    _legend_below(fig,axs[0,0],3)
    for i,ax in enumerate(axs.flat):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig01_window_scan')

def _future_figure(overall,run):
    _style();fig,axs=plt.subplots(1,3,figsize=(7.2,2.8));d=overall[(overall.stage=='future')&(overall.role=='test')]
    families=['lstm','gru','bigru','dual_gru','transformer','tcn','tcn_long']
    for i,f in enumerate(families):
        sub=d[d.family==f]
        for ax,metric,label in zip(axs,['score','gyro_rmse','accel_rmse'],['TEST relative error score','Gyro RMSE [rad/s]','Accel RMSE [m/s²]']):
            means=[];sds=[];pos=[]
            for j,look in enumerate(['0','6','full']):
                cell=sub[sub.lookahead.astype(str)==look]
                if len(cell):
                    m,s=_mean(cell,metric);means.append(m);sds.append(s);pos.append(j)
            ax.errorbar(pos,means,yerr=sds,marker='o',ms=3,capsize=2,color=PALETTE[i],label=LABELS[f])
            ax.set_xticks([0,1,2],['0 / causal','6 / 150 ms','Full window']);ax.tick_params(axis='x',labelsize=6)
            ax.set_xlabel('Allowed future input');ax.set_ylabel(label)
    _legend_below(fig,axs[0],4)
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig02_equal_future')

def _mechanism_figure(overall,run):
    _style();fig,axs=plt.subplots(1,3,figsize=(7.2,3.0));d=overall[(overall.stage=='mechanism')&(overall.role=='test')]
    combos=[(n,a) for n in ['bn','ln','none'] for a in ['gelu','identity']]
    for ax,metric,title in zip(axs[:2],['score','accel_rmse'],['TEST score','Accel RMSE [m/s²]']):
        arr=np.full((2,6),np.nan)
        for i,depth in enumerate([1,3]):
            for j,(norm,act) in enumerate(combos):
                cell=d[(d.depth==depth)&(d.norm==norm)&(d.activation==act)&(d.dilation==1)&(d.kernel.isin([5,13]))]
                arr[i,j]=cell[metric].mean()
        im=ax.imshow(arr,aspect='auto',cmap='viridis_r');ax.grid(False)
        ax.set_yticks([0,1],['Depth 1 / w68','Depth 3 / w64']);ax.set_xticks(np.arange(6),[n.upper()+'\n'+('GELU' if a=='gelu' else 'linear') for n,a in combos],fontsize=6)
        ax.set_title(title+'; local RF=13')
        for i in range(2):
            for j in range(6):ax.text(j,i,'%.3f'%arr[i,j],ha='center',va='center',fontsize=6,color='black' if arr[i,j]<np.nanmean(arr) else 'white')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    rf=d[(d.depth==3)&(d.kernel==5)&(d.norm=='ln')&(d.activation=='gelu')]
    g=rf.groupby('dilation').score.agg(['mean','std']).reindex([1,2,4])
    axs[2].errorbar([13,25,49],g['mean'],yerr=g['std'],marker='o',capsize=2,color=PALETTE[2]);axs[2].set_xticks([13,25,49])
    axs[2].set_xlabel('Frontend receptive field [points]');axs[2].set_ylabel('TEST score');axs[2].set_title('Identical parameter count')
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig03_frontend_mechanisms')

def _hpo_figure(overall,registry,lock,run):
    _style();fig,axs=plt.subplots(1,3,figsize=(7.2,3.4));families=sorted(lock['hpo_selected_candidates'])
    labels=[LABELS[f] for f in families]
    pilots=registry[(registry.stage=='search')&(registry.seed==registry.seed.min())]
    for i,c in enumerate(['c0','c1','c2']):
        values=[pilots[(pilots.family==f)&(pilots.candidate==c)].best_val_score.iloc[0] for f in families]
        axs[0].barh(np.arange(len(families))+(i-1)*.22,values,height=.21,color=PALETTE[i],label=c)
    axs[0].set_yticks(np.arange(len(families)),labels,fontsize=6);axs[0].set_xlabel('Pilot VALID score');_legend_below(fig,axs[0],3)
    data=overall[(overall.stage=='search')&(overall.role=='test')]
    for ax,metric,label in [(axs[1],'score','Selected TEST score'),(axs[2],'accel_rmse','Accel RMSE [m/s²]')]:
        for i,f in enumerate(families):
            cell=data[(data.family==f)&(data.candidate==lock['hpo_selected_candidates'][f])];mean,sd=_mean(cell,metric)
            ax.barh(i,mean,xerr=sd,color=PALETTE[i],height=.7,capsize=2)
        ax.set_yticks(np.arange(len(families)),labels,fontsize=6);ax.set_xlabel(label)
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig04_equal_budget_baselines')

def _generalization_figure(overall,run):
    _style();fig,axs=plt.subplots(1,3,figsize=(7.2,3.1));data=overall[(overall.stage=='generalization')&(overall.role=='test')]
    sensors=sorted(data.held_sensor.unique());families=sorted(data.family.unique())
    for ax,metric,label in zip(axs,['score','gyro_rmse','accel_rmse'],['TEST relative error score','Gyro RMSE [rad/s]','Accel RMSE [m/s²]']):
        for i,f in enumerate(families):
            means=[];sds=[]
            for s in sensors:
                m,sd=_mean(data[(data.family==f)&(data.held_sensor==s)],metric);means.append(m);sds.append(sd)
            ax.bar(np.arange(len(sensors))+(i-1)*.24,means,width=.23,yerr=sds,color=PALETTE[i],label=LABELS[f],capsize=2)
        affine=overall[(overall.stage=='baseline')&(overall.family=='affine')&(overall.role=='test')]
        reference=[affine[affine.held_sensor==sensor][metric].iloc[0] for sensor in sensors]
        ax.plot(np.arange(len(sensors)),reference,color='#1f2933',marker='x',ls='--',label='Affine (TRAIN only)')
        ax.set_xticks(np.arange(len(sensors)),sensors,fontsize=6);ax.set_xlabel('Held-out device (unseen recordings)');ax.set_ylabel(label)
    _legend_below(fig,axs[0],4)
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig05_strict_device_recording_generalization')

def _training_figure(run,jobs):
    _style();fig,axs=plt.subplots(2,3,figsize=(7.2,5.0))
    selected=[j for j in jobs.values() if j.get('diagnostics') and j['model_options']['lookahead'] in [0,'full'] and j['family'] in ['gru','bigru']]
    for i,job in enumerate(selected):
        hist=pd.read_csv(Path(job['directory'])/'history.csv');label=LABELS[job['family']]+' / future '+str(job['model_options']['lookahead'])
        for ax,metric,ylabel in zip(axs[0],['train_loss','val_score','gradient_norm'],['TRAIN Huber loss','VALID score','Gradient norm before clipping']):
            ax.plot(hist.epoch,hist[metric],color=PALETTE[i],label=label);ax.set_xlabel('Epoch');ax.set_ylabel(ylabel)
    _legend_below(fig,axs[0,1],3)
    job=next(j for j in selected if j['family']=='gru');folder=run/'diagnostics'/job['id']
    for ax,label in zip(axs[1],['epoch_001','epoch_010','best']):
        p=folder/(label+'_intermediate.npz');d=np.load(p,allow_pickle=True)
        z=d['hidden'][0].T;z=(z-z.mean(1,keepdims=True))/(z.std(1,keepdims=True)+1e-8)
        ax.imshow(z,aspect='auto',origin='lower',extent=[-4,3.975,0,z.shape[0]],cmap='RdBu_r',vmin=-2,vmax=2);ax.grid(False)
        ax.axvline(0,color='black',lw=.6);ax.set_title(label.replace('_',' ')+'; epoch '+str(int(d['epoch'])));ax.set_xlabel('Time from explained sample [s]');ax.set_ylabel('Hidden channel (z-score)')
    for i,ax in enumerate(axs.flat):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig06_training_and_hidden_states')

def _shap_figure(run,jobs):
    _style();fig,axs=plt.subplots(2,3,figsize=(7.2,5.0));selected=[]
    for family,look in [('gru',0),('bigru',0),('bigru','full')]:
        selected.append(next(j for j in jobs.values() if j.get('diagnostics') and j['family']==family and j['model_options']['lookahead']==look))
    shared_max={}
    for group in ['axis','lag']:
        values=[]
        for job in selected:
            item=np.load(run/'diagnostics'/job['id']/('best_'+group+'_shapley.npz'))
            values.append(np.mean(np.abs(item['residual_shapley'])/item['residual_std'][None,None],axis=0).max())
        shared_max[group]=max(values)
    for col,job in enumerate(selected):
        for row,group in enumerate(['axis','lag']):
            d=np.load(run/'diagnostics'/job['id']/('best_'+group+'_shapley.npz'))
            importance=np.mean(np.abs(d['residual_shapley'])/d['residual_std'][None,None],axis=0).T
            im=axs[row,col].imshow(importance,aspect='auto',cmap='magma',vmin=0,vmax=shared_max[group]);axs[row,col].grid(False)
            axs[row,col].set_yticks(np.arange(6),AXES);axs[row,col].set_ylabel('Output correction axis')
            labels=AXES if group=='axis' else ['≤−81','−80:−41','−40:−13','−12:−1','0','>0']
            axs[row,col].set_xticks(np.arange(6),labels,fontsize=5.5,rotation=35 if group=='lag' else 0)
            axs[row,col].set_xlabel('Input axis group' if group=='axis' else 'Input lag group [points]')
            axs[row,col].set_title(LABELS[job['family']]+' / future '+str(job['model_options']['lookahead']))
            fig.colorbar(im,ax=axs[row,col],fraction=.04,pad=.03)
    for i,ax in enumerate(axs.flat):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig07_exact_shapley_axis_and_lag')

def _shap_evolution(run,jobs):
    _style();fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    for ax,(family,look) in zip(axs,[('gru',0),('bigru','full')]):
        job=next(j for j in jobs.values() if j.get('diagnostics') and j['family']==family and j['model_options']['lookahead']==look)
        values=[]
        for label in ['epoch_001','epoch_010','best']:
            d=np.load(run/'diagnostics'/job['id']/(label+'_axis_shapley.npz'))
            values.append(np.mean(np.abs(d['residual_shapley'])/d['residual_std'][None,None],axis=(0,2)))
        values=np.asarray(values)
        for axis in range(6):ax.plot([0,1,2],values[:,axis],marker='o',ms=3,color=PALETTE[axis],label=AXES[axis])
        best_epoch=int(np.load(run/'diagnostics'/job['id']/'best_intermediate.npz',allow_pickle=True)['epoch'])
        ax.set_xticks([0,1,2],['Epoch 1','Epoch 10','Best VALID\n(epoch %d)'%best_epoch]);ax.set_xlabel('Selected checkpoints');ax.set_ylabel('Mean |Shapley correction| / TRAIN residual std')
        ax.set_title(LABELS[family]+' / future '+str(look))
    _legend_below(fig,axs[0],6)
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig08_shapley_during_training')


def _lag_sensitivity_figure(run,jobs):
    _style();fig,axs=plt.subplots(1,2,figsize=(7.2,3.2));sensitivity=[];attribution=[];labels=[]
    for family in ['lstm','gru','bigru','dual_gru','transformer','tcn','tcn_long']:
        job=next(j for j in jobs.values() if j['stage']=='future' and j['family']==family and j['seed']==j['config']['seeds'][0] and j['model_options']['lookahead']==0)
        folder=run/'diagnostics'/job['id'];d=np.load(folder/'best_intermediate.npz',allow_pickle=True)
        gradient=d['input_gradient'].copy();stats=json.loads((Path(job['directory'])/'stats.json').read_text(encoding='utf-8'))
        for axis in range(6):gradient[:,axis,axis,160]-=stats['feature_std'][axis]/stats['residual_std'][axis]
        curve=np.mean(np.abs(gradient),axis=(0,1,2));curve=curve/max(curve.max(),1e-12)
        sensitivity.append(np.log10(np.maximum(curve,1e-10)))
        shap=np.load(folder/'best_lag_shapley.npz');attribution.append(np.mean(np.abs(shap['residual_shapley'])/shap['residual_std'][None,None],axis=(0,2)))
        labels.append(LABELS[family])
    im=axs[0].imshow(sensitivity,aspect='auto',cmap='viridis',vmin=-8,vmax=0,extent=[-160,159,len(labels)-.5,-.5]);axs[0].grid(False);axs[0].axvline(0,color='white',lw=.7)
    axs[0].set_yticks(np.arange(len(labels)),labels,fontsize=6);axs[0].set_xlabel('Lag from predicted sample [points]');axs[0].set_title('Residual gradient sensitivity (relative log10)');fig.colorbar(im,ax=axs[0],fraction=.04,pad=.02)
    im=axs[1].imshow(attribution,aspect='auto',cmap='magma');axs[1].grid(False);axs[1].set_yticks(np.arange(len(labels)),labels,fontsize=6)
    axs[1].set_xticks(np.arange(6),['≤−81','−80:−41','−40:−13','−12:−1','0','>0'],rotation=35,fontsize=6);axs[1].set_xlabel('Input lag group [points]');axs[1].set_title('Exact grouped Shapley correction / std');fig.colorbar(im,ax=axs[1],fraction=.04,pad=.02)
    for i,ax in enumerate(axs):_panel(ax,chr(97+i))
    return _save(fig,run/'figures','fig09_causal_lag_sensitivity')

def paired_contrasts(overall,run):
    d=overall[(overall.role=='test')&(overall.stage!='baseline')];pairs=[]
    def add(name,a,b,note=''):
        merge=a.merge(b,on='seed',suffixes=('_a','_b'))
        if len(merge)!=3:raise ValueError('Incomplete paired seeds for '+name)
        for _,row in merge.iterrows():
            pairs.append(dict(contrast=name,seed=row['seed'],model_a=row['condition_a'],model_b=row['condition_b'],
                              score_delta=row['score_a']-row['score_b'],gyro_rmse_delta=row['gyro_rmse_a']-row['gyro_rmse_b'],
                              accel_rmse_delta=row['accel_rmse_a']-row['accel_rmse_b'],note=note))
    w=d[d.stage=='window']
    for f in w.family.unique():add('window320_minus40_'+f,w[(w.family==f)&(w.window==320)],w[(w.family==f)&(w.window==40)])
    f=d[d.stage=='future']
    def future(family,look):return f[(f.family==family)&(f.lookahead.astype(str)==str(look))]
    for look in [0,6]:add('BiGRU_minus_GRU_look'+str(look),future('bigru',look),future('gru',look),'Same allowed future; different parameter allocation')
    for look in [0,6]:add('BiGRU_minus_dualGRU_look'+str(look),future('bigru',look),future('dual_gru',look),'Same parameter count and both-branch initialization; dual GRU keeps both past recurrences active')
    add('BiGRU_full_minus_dualGRU_causal',future('bigru','full'),future('dual_gru',0),'Same two-GRU capacity, future branch versus second past branch')
    for look in [0,6]:add('TCN_RF181_minus43_look'+str(look),future('tcn_long',look),future('tcn',look),'Identical parameters, depth, normalization, activation and initial tensors; only dilations change')
    add('TCN_RF181_minus_GRU_causal',future('tcn_long',0),future('gru',0),'Both can access all arrived history; different temporal operators')
    for a,b in [('full',0),('full',6),(6,0)]:add('BiGRU_future%s_minus%s'%(a,b),future('bigru',a),future('bigru',b),'Same BiGRU weights initialization/parameter count, different bounded future')
    for fam in ['gru','lstm','tcn','transformer']:add(fam+'_look6_minus0',future(fam,6),future(fam,0))
    m=d[d.stage=='mechanism']
    base=m[(m.depth==3)&(m.kernel==5)&(m.dilation==1)&(m.norm=='ln')&(m.activation=='gelu')]
    for dilation in [2,4]:add('frontend_RF%d_minus13'%(1+12*dilation),m[(m.depth==3)&(m.kernel==5)&(m.dilation==dilation)],base,'Identical parameter count; GRU overall prefix unchanged')
    add('pointwise_minus_RF13',m[m.kernel==1],base,'Pointwise also changes parameter count')
    for depth in [1,3]:
        for norm in ['bn','ln','none']:
            sub=m[(m.depth==depth)&(m.norm==norm)&(m.dilation==1)&(m.kernel.isin([5,13]))]
            add('GELU_minus_linear_depth%d_%s'%(depth,norm),sub[sub.activation=='gelu'],sub[sub.activation=='identity'])
        for act in ['gelu','identity']:
            sub=m[(m.depth==depth)&(m.activation==act)&(m.dilation==1)&(m.kernel.isin([5,13]))]
            add('LN_minus_BN_depth%d_%s'%(depth,act),sub[sub.norm=='ln'],sub[sub.norm=='bn'],'BN training uses temporal statistics')
            add('none_minus_LN_depth%d_%s'%(depth,act),sub[sub.norm=='none'],sub[sub.norm=='ln'])
    for norm in ['bn','ln','none']:
        for act in ['gelu','identity']:
            sub=m[(m.norm==norm)&(m.activation==act)&(m.dilation==1)&(m.kernel.isin([5,13]))]
            add('depth3_minus1_%s_%s'%(norm,act),sub[sub.depth==3],sub[sub.depth==1],'Same frontend RF13; width64 versus68; approximately matched parameters')
    pairs=pd.DataFrame(pairs);pairs.to_csv(run/'paired_contrasts.csv',index=False)
    summary=pairs.groupby('contrast')[['score_delta','gyro_rmse_delta','accel_rmse_delta']].agg(['mean','std','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(run/'paired_contrasts_summary.csv')
    return pairs,summary

def _table(frame,columns):
    # Avoid optional tabulate dependency in pinn_imu.
    out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in frame.iterrows():
        out.append('| '+' | '.join('%.7g'%row[c] if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in columns)+' |')
    return '\n'.join(out)

def build_figures_and_report(run,cfg,jobs,lock):
    run=Path(run);(run/'figures').mkdir(exist_ok=True)
    metrics=pd.read_csv(run/'metrics_per_seed.csv',keep_default_na=False);registry=pd.read_csv(run/'fit_registry.csv',keep_default_na=False)
    for frame in [metrics,registry]:
        for col in ['window','seed','depth','kernel','dilation','width','elapsed_seconds','best_val_score','score','gyro_rmse','accel_rmse']:
            if col in frame:frame[col]=pd.to_numeric(frame[col],errors='coerce')
    overall=metrics[metrics.aggregation=='overall'].copy()
    # Non-selected HPO pilots have no TEST rows and do not enter this leaderboard.
    overall.to_csv(run/'overall_per_seed.csv',index=False)
    pairs,contrasts=paired_contrasts(overall,run)
    deploy=json.loads((run/'deployment_selection.lock.json').read_text(encoding='utf-8'))
    latency=json.loads((run/'deployment_latency_audit.json').read_text(encoding='utf-8'))
    builders=[('fig01_window_scan',lambda:_window_figure(overall,run),'Window scan with common max320 anchors and identical center40 targets. Points show means ± sample SD over three optimization seeds. VALID choices are locked before new TEST evaluation.'),
              ('fig02_equal_future',lambda:_future_figure(overall,run),'Seven backends, including a capacity-matched dual causal GRU and an exactly parameter-matched RF181 TCN, share a causal CNN/token-LN frontend and the same allowed input prefix. Zero future is verified in train/eval and by the forbidden-future Jacobian. Full future exists only for BiGRU and Transformer.'),
              ('fig03_frontend_mechanisms',lambda:_mechanism_figure(overall,run),'Frontend depth/normalization/activation factorial at local RF13, with approximately matched parameter count. Dilation varies RF with identical parameter count. GRU still has a prefix receptive field.'),
              ('fig04_equal_budget_baselines',lambda:_hpo_figure(overall,registry,lock,run),'Seven model families receive three LR/dropout packages, one pilot seed, 40-epoch caps and identical data/optimizer. The VALID-selected package is repeated with three seeds. Parameters and wall times differ.'),
              ('fig05_strict_device_recording_generalization',lambda:_generalization_figure(overall,run),'Three held-device folds also hold out TEST physical recordings. TRAIN/VALID exclude the held device; a fixed recipe avoids target-device hyperparameter selection. Error bars are optimization-seed SD.'),
              ('fig06_training_and_hidden_states',lambda:_training_figure(run,jobs),'Training loss, VALID error and gradient norm, followed by one fixed VALID sample’s GRU hidden states at epochs1/10/best. Hidden channels are z-scored over time only for display. All token states are shown; positive-time columns are not consumed by the index160 causal prediction.'),
              ('fig07_exact_shapley_axis_and_lag',lambda:_shap_figure(run,jobs),'Exact interventional Shapley values enumerate all64 six-group coalitions, using8 TRAIN backgrounds and8 VALID examples. Values show physical residual-correction attribution divided by TRAIN residual standard deviation; the raw skip contribution is removed analytically.'),
              ('fig09_causal_lag_sensitivity',lambda:_lag_sensitivity_figure(run,jobs),'For seven zero-future backends, left shows per-model-normalized residual-correction gradients; right shows exact six-lag-group Shapley correction on the same fixed VALID examples. Gradient magnitude is sensitivity, not SHAP or a causal sensor claim.'),
              ('fig08_shapley_during_training',lambda:_shap_evolution(run,jobs),'Input-axis attribution at epochs1/10/best VALID illustrates training evolution on fixed examples/backgrounds. This is a model explanation under interventional feature mixing, not a causal claim about sensor physics.')]
    builders.sort(key=lambda item:item[0])
    index=['# Figure index','']
    for stem,builder,caption in builders:
        builder();index.extend(['## '+stem,'',caption,''])
    (run/'figures/figure_index.md').write_text('\n'.join(index),encoding='utf-8')
    report=['# ALiMU 上下文、结构机制与严格泛化实验','',
            '本轮用 BMX160、LSM9DS1、MPU9250 → 实测 VN100 的六轴配对信号。表中gyro为逐轴池化RMSE [rad/s]，accel为 [m/s²]，SD为优化seed标准差。40 Hz；统一最大320点锚点和中心40点目标，TRAIN拟合归一化。分数 S 为设备等权的陀螺仪与加速度相对原始误差比的均值，越低越好；S=1对应原始LOW。','',
            '这是已看过历史TEST后的预声明补充实验，不是全新未触碰数据。TEST只有dyn08/dyn09两条独立物理采集；三seed仅描述优化变动，不能当作三个独立数据集或正式显著性证明。','',
            '## 矩阵与预算','',_table(registry.groupby(['stage','reused'],dropna=False).agg(fits=('job_id','count'),seconds=('elapsed_seconds','sum')).reset_index(),['stage','reused','fits','seconds']),'',
            '窗口组沿用80轮上限、patience15的旧配方；其他组为40轮上限、patience10。组内所有结构一致。跨组不混排名。共享模型用AdamW，与最初独立设备DDPM实验的Adam协议区分。','',
            '## 短窗口和长窗口','']
    w=overall[(overall.stage=='window')&(overall.role=='test')]
    wsum=w.groupby(['family','window']).agg(S=('score','mean'),SD=('score','std'),gyro=('gyro_rmse','mean'),accel=('accel_rmse','mean')).reset_index()
    wsum['reduction_pct']=100*(1-wsum.S)
    wsum.to_csv(run/'window_results.csv',index=False);report.append(_table(wsum,['family','window','S','SD','reduction_pct','gyro','accel']))
    report.extend(['','VALID选择的窗口：'+json.dumps(lock['validation_selected_windows'],ensure_ascii=False)+'。',''])
    for window in [40,80,160,320]:
        cell=wsum[wsum.window==window].sort_values('S');winner=cell.iloc[0]
        report.append('- %d点（%.1fs）TEST观察最优：%s，S=%.6f。'%(window,window/40,LABELS[winner.family],winner.S))
    report.extend(['','GRU/BiGRU的320−40点配对差值（负值表示长窗口改善）：',_table(contrasts.reset_index().query("contrast in ['window320_minus40_cnn_gru','window320_minus40_cnn_bigru']"),['contrast','score_delta_mean','score_delta_std']),'','## 相同未来范围与因果部署',''])
    f=overall[(overall.stage=='future')&(overall.role=='test')]
    fsum=f.groupby(['family','lookahead']).agg(S=('score','mean'),SD=('score','std'),gyro=('gyro_rmse','mean'),accel=('accel_rmse','mean')).reset_index();fsum.to_csv(run/'future_results.csv',index=False)
    report.append(_table(fsum,['family','lookahead','S','SD','gyro','accel']))
    zero=fsum[fsum.lookahead.astype(str)=='0'].sort_values('S').iloc[0]
    fv=overall[(overall.stage=='future')&(overall.role=='validation')&(overall.lookahead.astype(str)=='0')].groupby('family').score.mean().sort_values()
    report.extend(['','固定前端B组严格零前视神经网络的TEST观察最优：%s，S=%.6f；VALID选择的零前视神经网络部署候选：%s。 此处比较预声明的共同40轮配方，未对每种因果结构分别重新调参；后面的三候选等预算搜索属于离线整窗模型。因此这里的部署推荐限于已测试预算，不能据此认定某类结构在充分调参后始终更优。'%(LABELS[zero.family],zero.S,LABELS[deploy['zero_future_family']]),
                   '原对称CNN的单向RNN含未来6点，不能用于上述零前视结论。本轮逐样本前缀回放证明无需等待整窗后半段或中心40点整块收齐；RNN仍每窗重置，没有宣称无限流式状态部署。现有数据清单另记录LOW插值的25ms已知可用性延迟，网络0点/6点前视是在此基础上额外增加0/150ms，不能把网络零前视写成原始传感器零总延迟。上游外参也未独立验证。','',
                   'BiGRU0与GRU0比较表示相同输入可用范围下的结构/参数分配差异；BiGRU0/6/full内部比较保持参数量和初始化不变，能直接检查未来上下文收益。零前视BiGRU的反向分支只有当前token，不再提供长程反向循环；因此另外训练参数量和初始化相同的双单向GRU，两条分支都完整处理过去，控制有效循环容量。该补充在TEST数值读取前预声明并独立锁定。','',
                   _table(contrasts.reset_index()[contrasts.reset_index().contrast.str.startswith('BiGRU')],['contrast','score_delta_mean','score_delta_std']),'',
                   '## 局部感受野、深度、归一化和非线性',''])
    configuration_results=None
    if 'zero_future_configurations' in deploy:
        from virtual_imu.context_study.causal_candidates import causal_configuration_results
        configuration_results=causal_configuration_results(overall,deploy['zero_future_configurations'])
        configuration_results.to_csv(run/'causal_configuration_results.csv',index=False)
        chosen=configuration_results[configuration_results.VALID_selected].iloc[0];observed=configuration_results.iloc[0]
        report.extend(['','已训练的零前视神经配置（单独的配置搜索）：',_table(configuration_results,['candidate_id','validation_score','S','SD','gyro','accel','VALID_selected']),
                       '固定前端的B组用于后端结构的公平比较。另把B组与满足训练/推理均零前视的前端机制配置合并，按三seed平均VALID选择实际部署候选；GRU探索的前端组合较多，因此此表比较已探索的配置，不用于断言GRU家族拥有相同搜索预算下的固有优势。默认GRU0精确复用只计一次，BN机制行单独分析。',
                       'VALID锁定的零前视神经配置：%s（TEST S=%.6f±%.6f）；TEST描述性观察最低配置：%s（S=%.6f）。'%(chosen.candidate_id,chosen.S,chosen.SD,observed.candidate_id,observed.S),''])
    mech=overall[(overall.stage=='mechanism')&(overall.role=='test')].groupby(['condition','depth','kernel','dilation','norm','activation']).agg(S=('score','mean'),SD=('score','std')).reset_index()
    mech.to_csv(run/'mechanism_results.csv',index=False);report.append(_table(mech,['condition','S','SD']))
    factor_rows=[]
    digest=contrasts.reset_index()
    for label,prefix in [('Frontend depth3 minus1','depth3_minus1_'),('GELU minus identity','GELU_minus_linear_'),('Token LN minus BN','LN_minus_BN_'),('No norm minus token LN','none_minus_LN_'),('Frontend RF49 minus13','frontend_RF49_minus13'),('TCN total RF181 minus43 (causal)','TCN_RF181_minus43_look0')]:
        values=digest[digest.contrast.str.startswith(prefix)].score_delta_mean.to_numpy()
        factor_rows.append(dict(factor=label,conditions=len(values),mean_score_delta=float(values.mean()),min_condition_delta=float(values.min()),max_condition_delta=float(values.max())))
    factors=pd.DataFrame(factor_rows);factors.to_csv(run/'factor_effect_digest.csv',index=False)
    report.append(_table(factors,['factor','conditions','mean_score_delta','min_condition_delta','max_condition_delta']))
    report.extend(['','完整配对差值见paired_contrasts_summary.csv。另用同一TCN检查点裁剪输入，验证160点已覆盖RF43：窗口变长并未扩大该TCN推理时的实际感受野，分别训练160/320产生的差异也可能来自训练归一化与优化。见fixed_checkpoint_context_diagnostic.csv和fig09。另外TCN以相同4层、相同参数和初始张量，将dilation从[1,2,4,8]改为[3,9,27,45]，在不增加深度/归一化/非线性的条件下比较总RF43与181；前端RF13填补各级膨胀采样间隙，覆盖完整历史前缀。RF181为理论感受野；在本实验每窗重置、中心40点输出的协议中，零前视实际只有141–180个已到达样本，RF181覆盖该前缀，各级卷积的每个抽头均可接触真实输入；更远组合仍会接触左侧填充。因此不能把RF181写成每个目标实际使用181点历史。深度对照固定前端RF13，width68与64使总参数接近而非完全相同；归一化和GELU因素做完整交叉；这些开关仅作用于CNN前端，GRU门控和回归头保持不变。RF13/25/49只改dilation，参数完全相同；这改变CNN的局部聚合跨度，GRU的总体历史前缀并未改变。BN训练使用整窗统计，推理使用固定的运行统计量。B组和配置部署表要求训练与推理均限制未来信息，因此BN行用于单独的归一化机制分析。','',
                   '## 相同协议、相同搜索预算的强基线',''])
    h=overall[(overall.stage=='search')&(overall.role=='test')].groupby(['family','candidate']).agg(S=('score','mean'),SD=('score','std'),gyro=('gyro_rmse','mean'),accel=('accel_rmse','mean')).reset_index().sort_values('S')
    h['reduction_pct']=100*(1-h.S)
    h.to_csv(run/'equal_budget_results.csv',index=False);report.append(_table(h,['family','candidate','S','SD','reduction_pct','gyro','accel']))
    baseline_shared=overall[(overall.stage=='baseline')&(overall.role=='test')&(overall.held_sensor=='shared')][['family','score','gyro_rmse','accel_rmse']]
    report.extend(['','同一监督目标支持上的确定性基线（无seed方差）：',_table(baseline_shared,['family','score','gyro_rmse','accel_rmse'])])
    report.extend(['','此固定小网格下七类神经网络的TEST观察最优：%s（S=%.6f）。这不是对所有可能超参数或官方SOTA排名的断言。Informer是仓库内ProbSparse/蒸馏的IMU回归适配实现。'%(LABELS[h.iloc[0].family],h.iloc[0].S),'',
                   '## 同时留出设备与采集序列',''])
    g=overall[(overall.stage=='generalization')&(overall.role=='test')].groupby(['held_sensor','family']).agg(S=('score','mean'),SD=('score','std'),gyro=('gyro_rmse','mean'),accel=('accel_rmse','mean')).reset_index()
    baseline_rows=[]
    for _,row in overall[(overall.stage=='baseline')&(overall.role=='test')&(overall.held_sensor!='shared')].iterrows():
        baseline_rows.append(dict(held_sensor=row.held_sensor,family=row.family,S=row.score,SD=0.,gyro=row.gyro_rmse,accel=row.accel_rmse))
    g=pd.concat([g,pd.DataFrame(baseline_rows)],ignore_index=True)
    g['VALID_selected']=[deploy['strict_fold_families'][r.held_sensor]==r.family for _,r in g.iterrows()]
    g.to_csv(run/'strict_generalization_results.csv',index=False);report.append(_table(g,['held_sensor','family','S','SD','gyro','accel','VALID_selected']))
    for sensor,cell in g.groupby('held_sensor'):
        best=cell.sort_values('S').iloc[0];report.append('- 留出%s时，TEST观察最优%s，S=%.6f。'%(sensor,LABELS[best.family],best.S))
    macro=g.groupby('family').S.mean().sort_values();report.append('三种留出设备的等权平均S最低：%s（%.6f）。'%(LABELS[macro.index[0]],macro.iloc[0]))
    report.extend(['','严格泛化组沿用离线整窗模型，这里的严格指设备和采集隔离；因果部署仅依据零前视B组。三种fold共享dyn08/dyn09参考采集，不能当作六条独立采集。此处是固定配方下的比较。各fold的归一化、AdamW训练和早停只读取源设备TRAIN/VALID。固定配方避免使用目标设备的VALID标签。没有把该组与不同配方的共享HPO组相减作为泛化下降量。','',
                   '## 训练中间态与SHAP','',
                   'fig06给出每轮损失/VALID/梯度与隐藏表示，fig07给出六轴和时间段的精确Shapley热图，fig08给出归因随训练变化。diagnostics中保存原始数组、背景样本身份、检查点哈希和完整性验证。归因采用第一个优化种子20261010，不代表三seed归因一致性检验。这是固定8个VALID样本、8个TRAIN背景上的六组归因，精确性针对这六组联盟枚举，不代表每个时间点的全特征SHAP；替换相关IMU轴/时段可能产生非物理组合，归因表示模型依赖，不能当作传感器物理因果。梯度敏感度单独命名，不冒称SHAP；不需要额外shap包，直接枚举64种联盟计算六组精确Shapley值。','',
                   '训练wall time包含并发竞争与断点续跑，仅作执行记录；模型推理延迟在所有训练退出后，以GPU驻留FP32、5次预热/20次重复单独测量，不包含采集与CPU传输。实际4worker执行覆盖和缓存释放策略见execution_pool_override.json。图像为SVG/PDF/PNG300dpi。完整数值见metrics_per_seed.csv、summary.csv和paired_contrasts.csv；条件/seed/窗口/设备/采集记录均可追溯。'])
    answers={'short_window_observed_winners':{str(w):wsum[wsum.window==w].sort_values('S').iloc[0].to_dict() for w in [40,80,160,320]},
             'validation_selected_windows':lock['validation_selected_windows'],'zero_future_test_observed_best':zero.to_dict(),
             'zero_future_validation_selected':deploy['zero_future_family'],'equal_budget_observed_best':h.iloc[0].to_dict(),
             'shared_deterministic_baselines':baseline_shared.to_dict('records'),
             'strict_observed_per_device':{sensor:cell.sort_values('S').iloc[0].to_dict() for sensor,cell in g.groupby('held_sensor')},
             'strict_validation_selected':deploy['strict_fold_families'],'strict_macro_includes_train_only_affine_and_raw':True,'strict_macro_observed_best':{'family':macro.index[0],'S':float(macro.iloc[0])},
             'mechanism_effects':factor_rows,'paired_future_deltas':digest[digest.contrast.str.startswith('BiGRU')].to_dict('records'),'matched_rf_deltas':digest[digest.contrast.str.startswith('TCN_RF181')].to_dict('records')}
    if configuration_results is not None:
        answers['zero_future_configuration_validation_selected']=configuration_results[configuration_results.VALID_selected].iloc[0].to_dict()
        answers['zero_future_configuration_test_observed_best']=configuration_results.iloc[0].to_dict()
    from virtual_imu.context_study.precision_audit import summarize_precision
    precision=summarize_precision(run,Path(__file__).resolve().parents[3])
    if precision is not None:
        answers['numeric_precision_recovery']=precision
        report.extend(['','## 数值稳定性记录','','严格泛化Informer留出MPU9250、seed20261012在第7轮第194批次出现FP16注意力矩阵乘法溢出。输入、目标及参数均有限；原始失败证据保留。测试评估前冻结的修复仅在稀疏注意力的有限输入产生非有限AMP输出时，将该注意力运算以FP32重算并转回输入dtype；正常有限路径保持原计算图。模型、数据划分、seed、AdamW、早停与训练预算不变。该运行从第6轮完整状态续跑。此为明确的数值精度例外，不能声称全部条件采用无例外的纯FP16注意力。触发记录共%d次，逐实验/评估计数见numeric_precision_recovery.csv；修复配置、源码哈希、复现证据与验证结果均随结果包提供。'%precision['events']])
    (run/'answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (run/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
