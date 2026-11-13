"""Presentation-only revision of a sealed ALiMU experiment; numerical artifacts stay byte-identical."""
from pathlib import Path
import json,shutil,time,zipfile,hashlib
import numpy as np
import pandas as pd
import yaml
from virtual_imu.viz import plots_context_study as original
from virtual_imu.viz.plots_context_study import plt

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def mechanism(data):
    original._style();fig=plt.figure(figsize=(7.2,4.4));grid=fig.add_gridspec(2,2,width_ratios=[1.65,1.0],hspace=.65,wspace=.6)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[1,0]),fig.add_subplot(grid[:,1])]
    d=data[(data.stage=='mechanism')&(data.role=='test')];combos=[(n,a) for n in ['bn','ln','none'] for a in ['gelu','identity']]
    for ax,metric,title in zip(axes[:2],['score','accel_rmse'],['TEST relative error score S','Accel RMSE [m/s²]']):
        values=np.array([[d[(d.depth==depth)&(d.norm==norm)&(d.activation==act)&(d.dilation==1)&d.kernel.isin([5,13])][metric].mean() for norm,act in combos] for depth in [1,3]])
        im=ax.imshow(values,aspect='auto',cmap='viridis_r');ax.grid(False)
        ax.set_yticks([0,1],['Depth 1\nwidth 68','Depth 3\nwidth 64'],fontsize=7)
        ax.set_xticks(range(6),[(n.upper() if n!='none' else 'None')+'\n'+('GELU' if a=='gelu' else 'Linear') for n,a in combos],fontsize=6.5)
        ax.set_title(title+'; RF13',fontsize=8,pad=7)
        for i in range(2):
            for j in range(6):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=6.5,color='black' if values[i,j]<values.mean() else 'white')
        bar=fig.colorbar(im,ax=ax,fraction=.045,pad=.025);bar.ax.tick_params(labelsize=6.5)
    rf=d[(d.depth==3)&(d.kernel==5)&(d.norm=='ln')&(d.activation=='gelu')].groupby('dilation').score.agg(['mean','std']).reindex([1,2,4])
    axes[2].errorbar([13,25,49],rf['mean'],yerr=rf['std'],color=original.PALETTE[2],marker='o',capsize=2,ms=4)
    axes[2].set_xticks([13,25,49]);axes[2].set_xlabel('Frontend RF [points]');axes[2].set_ylabel('TEST relative error score S');axes[2].set_title('Matched parameters',fontsize=8,pad=7)
    for i,ax in enumerate(axes):ax.text(-.2,1.17 if i<2 else 1.045,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=10)
    return fig

def generalization(data):
    original._style();fig,axes=plt.subplots(1,3,figsize=(7.2,3.4));models=data[(data.stage=='generalization')&(data.role=='test')]
    sensors=sorted(models.held_sensor.unique());families=sorted(models.family.unique());positions=np.arange(3)
    for ax,metric,label in zip(axes,['score','gyro_rmse','accel_rmse'],['TEST relative error score S','Gyro RMSE [rad/s]','Accel RMSE [m/s²]']):
        for i,family in enumerate(families):
            cells=[models[(models.family==family)&(models.held_sensor==s)][metric] for s in sensors]
            ax.bar(positions+(i-1)*.24,[c.mean() for c in cells],width=.23,yerr=[c.std(ddof=1) for c in cells],color=original.PALETTE[i],label=original.LABELS[family],capsize=2)
        for family,style in [('affine',dict(color='#5b6875',marker='x',ls='--')),('raw',dict(color='#20252b',marker='o',ms=3,mfc='white',ls=':'))]:
            baseline=data[(data.stage=='baseline')&(data.role=='test')&(data.family==family)]
            values=[float(baseline[baseline.held_sensor==s][metric].iloc[0]) for s in sensors]
            ax.plot(positions,values,label=original.LABELS[family],**style)
        ax.set_xticks(positions,sensors,fontsize=6);ax.set_xlabel('Held-out device');ax.set_ylabel(label)
    original._legend_below(fig,axes[0],3)
    for i,ax in enumerate(axes):original._panel(ax,chr(97+i))
    return fig

def revise_saved(fig,out,stem):
    if stem.startswith('fig08'):
        for ax in fig.axes:
            ax.set_ylabel('Mean |SHAP correction|\n[TRAIN residual std]',fontsize=8);ax.set_title(ax.get_title(),fontsize=8,pad=8)
    elif stem.startswith('fig09'):
        for ax,title in zip(fig.axes[:2],['Residual lag sensitivity','Exact grouped SHAP']):
            ax.set_title(title,fontsize=8,pad=9)
            for text in ax.texts:
                if text.get_text() in ['a','b']:text.set_position((-.22,1.08))
        for ax,label in zip(fig.axes[2:],['log10(relative)','|SHAP| [residual std]']):ax.set_ylabel(label,fontsize=7)
    return _SAVE(fig,out,stem)

_SAVE=original._save

def build(root,config_path):
    root=Path(root);cfg=yaml.safe_load((root/config_path).read_text(encoding='utf-8'));source=root/'runs'/cfg['source_run'];dest=root/'runs'/cfg['experiment_id']
    assert (source/'complete.json').exists(),'Numerical source must be sealed'
    if (dest/'complete.json').exists():print('SKIP sealed figure revision '+str(dest));return dest
    dest.mkdir(exist_ok=True);(dest/'figures').mkdir(exist_ok=True)
    print('READ sealed numerical source '+str(source)+'; WRITE figure revision '+str(dest)+'; SKIP training and numerical source mutations',flush=True)
    manifest=json.loads((source/'artifact_manifest.json').read_text(encoding='utf-8'));copied={}
    for path in list((source/'figures').glob('*'))+list(source.glob('*.csv'))+[source/n for n in ['answers.json','config.yaml','data_manifest.json','environment.json','independent_audit.json','verification.json','deployment_selection.lock.json','causal_configuration_reporting.lock.json','deployment_latency_audit.json','numeric_precision_recovery.lock.json','numeric_precision_audit.json']]:
        if not path.is_file():continue
        relative=path.relative_to(source).as_posix()
        if relative in manifest['files'] and digest(path)!=manifest['files'][relative]:raise AssertionError('Source changed '+relative)
        target=dest/relative;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(path,target);copied[relative]=digest(path)
    data=pd.read_csv(source/'metrics_per_seed.csv',keep_default_na=False);data=data[data.aggregation=='overall'];jobs=json.loads((source/'executed_jobs_with_supplement.json').read_text(encoding='utf-8'))
    _SAVE(mechanism(data),dest/'figures','fig03_frontend_mechanisms');_SAVE(generalization(data),dest/'figures','fig05_strict_device_recording_generalization')
    def redirect(fig,out,stem):return revise_saved(fig,dest/'figures',stem)
    original._save=redirect
    try:original._shap_evolution(source,jobs);original._lag_sensitivity_figure(source,jobs)
    finally:original._save=_SAVE
    note='\n\nPresentation revision: fig03 uses wider heatmap cells; fig05 explicitly includes Raw LOW alongside TRAIN-only affine and neural models; fig08 shortens the normalized attribution axis label; fig09 separates panel letters, titles and colorbar units. All numerical sources are the sealed parent run.\n'
    index=dest/'figures/figure_index.md';index.write_text(index.read_text(encoding='utf-8')+note,encoding='utf-8')
    intro=f'> 图形修订包：{cfg["experiment_id"]}。所有训练、预测和数值结论来自已封存的 {cfg["source_run"]}；本目录修订排版与基线呈现，新增训练0次。\n\n'
    report=(source/'report.md').read_text(encoding='utf-8').replace('因果部署仅依据零前视B组','因果部署依据零前视B组及通过训练/推理因果复核的非BN前端机制配置')
    (dest/'report.md').write_text(intro+report,encoding='utf-8')
    for path in [root/config_path,Path(__file__),root/'scripts/revise_context_figures.py']:
        target=dest/'source_snapshot'/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    write_json(dest/'revision.json',dict(source_run=str(source.resolve()),source_manifest_sha256=digest(source/'artifact_manifest.json'),source_complete_sha256=digest(source/'complete.json'),source_metrics_sha256=digest(source/'metrics_per_seed.csv'),revised_figures=cfg['revised_figures'],new_training_fits=0,copied_source_sha256=copied,revision_sources={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in [root/config_path,Path(__file__),root/'scripts/revise_context_figures.py']}))
    print('WRITE9figures27exports; READ sealed metrics and SHAP arrays; SKIP checkpoint changes',flush=True);return dest

def seal(root,config_path):
    root=Path(root);cfg=yaml.safe_load((root/config_path).read_text(encoding='utf-8'));dest=root/'runs'/cfg['experiment_id'];source=root/'runs'/cfg['source_run']
    if (dest/'complete.json').exists():print('SKIP sealed revision');return
    qa=json.loads((dest/'visual_qa.json').read_text(encoding='utf-8'));assert qa['status']=='PASS'
    receipt=json.loads((dest/'revision.json').read_text(encoding='utf-8'));assert digest(source/'artifact_manifest.json')==receipt['source_manifest_sha256']
    for relative,expected in receipt['copied_source_sha256'].items():
        if relative.startswith('figures/'):continue
        assert digest(dest/relative)==expected,'Numerical copy changed '+relative
    for name,expected in receipt['revision_sources'].items():assert digest(root/name)==expected
    archive_path=dest/'ALiMU_context_mechanisms_results_revised.zip'
    files=[p for p in dest.rglob('*') if p.is_file() and p!=archive_path and p.name not in ['complete.json','artifact_manifest.json']]
    replacement={p.relative_to(dest).as_posix():p for p in files}
    with zipfile.ZipFile(source/'ALiMU_context_mechanisms_results.zip') as old,zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            if name not in replacement:new.writestr(name,old.read(name))
        for name,path in replacement.items():new.write(path,name)
    protected={p.relative_to(dest).as_posix():digest(p) for p in dest.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json','complete.json']}
    write_json(dest/'artifact_manifest.json',dict(files=protected,policy='Presentation-only revision; numerical parent stays sealed'))
    write_json(dest/'complete.json',dict(status='PASS',kind='figure_revision',new_training_fits=0,numerical_source=cfg['source_run'],figures=9,exports=27,artifact_manifest_sha256=digest(dest/'artifact_manifest.json'),completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('READ visual QA and unchanged numerical copies; WRITE reviewed archive and sealed revision; SKIP all numerical source mutations',flush=True)
