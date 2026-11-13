"""Build ALiMU Nature-style figures from completed, hash-bound local runs.

No training/data/checkpoint access and no mutation of the source runs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from virtual_imu.viz.plots_alimu_nature import LABELS, render_figures

LOG=logging.getLogger('alimu_nature')
KEYS=['protocol','split','variant','window','seed']
IDENTITY=['protocol','split','variant','window','aggregation','dataset_id','pair_id','original_sequence_id']
METRICS=['gyro_rmse','accel_rmse','normalized_score']
SOURCES={
 'shared':['summary.csv','metrics_per_seed.csv','paired_summary.csv','paired_deltas.csv','validation_ranking.csv',
           'compute_summary.csv','compute_cost.csv','config.yaml','data_manifest.json','selections.lock.json','provenance.json','verification.json'],
 'windows':['summary.csv','metrics_per_seed.csv','config.yaml','data_manifest.json','provenance.json','verification.json'],
 'matched':['summary.csv','metrics_per_seed.csv','matched_seed_deltas.csv','config.yaml','provenance.json','selections.lock.json','verification.json'],
}
CAPTIONS={
'fig1_shared_selection':('Shared-model selection on ALiMU',
 'a, Validation ranking of eleven fixed-recipe architectures, ordered by the three-seed mean balanced normalized score. '
 'b, Observed ALiMU test scores in exactly the same model order. CNN–GRU is selected by validation; CNN–BiGRU has the lowest observed test score. '
 'For each LOW device pair and seed, score = (gyro component RMSE / Raw gyro RMSE + acceleration component RMSE / Raw acceleration RMSE)/2; device pairs are equally weighted. '
 'Raw denominators are split-specific, so connecting validation to test as a distribution shift estimate is inappropriate. '
 'The horizontal axis is nonzero; Raw=1 lies outside its displayed range. Markers/error bars denote mean ± sample SD (ddof=1) over three optimization seeds; small hollow dots denote individual seeds. '
 'One shared ALiMU-only model maps BMX160, MPU9250 and LSM9DS1 to measured VN100. Input320 at40Hz, central40 supervision, stride40. '
 'TEST was previously inspected; a descriptive minimum is not a model-selection decision.'),
'fig2_shared_physical':('Physical accuracy of shared ALiMU models',
 'a,b, Gyroscope and accelerometer three-axis component RMSE, respectively, for the same eleven shared-model architectures and deterministic Raw LOW. '
 'Errors are pooled across all target samples and axes before the square root for each seed, then summarized across three optimization seeds as mean ± sample SD. '
 'Each seed contains88,080 device–time target rows, from two physical test recordings observed by three LOW devices. '
 'CNN–BiGRU has the lowest observed pooled test means in both physical metrics; CNN–GRU remains the validation-selected model. '
 'VN100 is a measured reference. Raw has one value and no seed SD. These are signal-mapping errors, not navigation errors or absolute sensor-grade claims.'),
'fig3_ablation_evidence':('Module, window and device evidence',
 'a,b, Same-seed percentage changes in balanced test score relative to CNN–LSTM64 when replacing its backend or CNN frontend. '
 'Negative changes favor the replacement; each percentage is calculated before seed averaging. '
 'Points and error bars show three seeds and mean ± sample SD. Seven new non-deletion replacements are within±5% of reference parameters; BiLSTM32, CNN-only and Linear–LSTM64 controls are outside that group. '
 'c, Earlier frozen CNN–BiLSTM window sweep on the same retained central40 target support:40/80/160/320 samples correspond to1/2/4/8s nominal input context. '
 'VALID favors160 samples, while TEST is lowest at320; this does not justify test-based window selection. '
 'd, Device-specific normalized test scores for the shared CNN–GRU, CNN–BiGRU and CNN–LSTM64 models; no separate per-device model selection is performed. '
 'BiGRU uses both temporal directions; the unidirectional CNN hybrids still see six future input samples via the CNN. Operator, normalization and information-access differences limit mechanistic attribution.'),
'fig4_per_device_ddpm':('Separate fixed80 per-device comparison and diffusion ablation',
 'a,b, ALiMU gyroscope/acceleration component RMSE for the earlier uniform80-sample per-device experiment: each LOW device has a separately trained model. '
 'Each method has three optimization seeds after validation-only selection over three declared candidates; Raw and train-fitted Affine are deterministic. '
 'Three-device errors are pooled per seed before taking the square root (89,949 device–time rows), then reported as mean ± sample SD. '
 'Informer has the lowest observed gyro mean; independently tuned no-diffusion regression has the lowest observed acceleration mean. '
 'c,d,100×(DDPM/strict-no-diffusion RMSE−1) calculated for each device, signal group and seed. Strict removal retains the DDPM-selected backbone, hyperparameters and initial weights; all18 differences are positive. '
 'This protocol uses80 samples, full-window prediction, Adam and independent device training. It must not share an absolute leaderboard with the320-sample, central40, AdamW shared-model experiment.'),
'fig5_scope_and_cost':('Training-scope sensitivity and compute',
 'a, ALiMU-only test scores after training one shared model on ALiMU alone (filled circles) or jointly on ALiMU and PennCOSYVIO (open squares). '
 'All rows evaluate only the same ALiMU test support. Joint selection uses equal dataset weight on VALID from both datasets and selects CNN–LSTM64; the observed ALiMU-only test minimum inside the joint group is CNN–BiLSTM32. '
 'b, Uniformly measured batch1 forward latency of the ALiMU-only models on local RTX3060Ti: five warmups and20 synchronized repeats per fit. '
 'Plotted means and sample SD summarize the three per-fit latency means. Measurements exclude input buffering and future-context waiting and are not stable hardware constants. '
 'No per-family hyperparameter search was run for module replacements. These data compare the tested implementations under the fixed recipe.'),
}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_json(path,obj):
 LOG.info('WRITE %s',path)
 path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path,frame):
 LOG.info('WRITE %s (%d rows)',path,len(frame))
 # Keep complete numeric precision in machine-readable figure data.
 frame.to_csv(path,index=False,float_format='%.17g',encoding='utf-8')

def read_sources(cfg,out):
 frames={};receipts={}; bindings=[]
 for scope,names in SOURCES.items():
  run=ROOT/cfg['sources'][scope]
  complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
  if complete.get('status')!='complete':raise ValueError('Incomplete source '+scope)
  if (run/'artifact_manifest.json').exists():
   manifest=json.loads((run/'artifact_manifest.json').read_text(encoding='utf-8'))
   if sha(run/'artifact_manifest.json')!=complete['artifact_manifest_sha256']:raise ValueError('Bad source seal')
   sealed=manifest['files_sha256']
  else:sealed=complete['artifacts_sha256']
  target=out/'inputs'/scope;target.mkdir(parents=True,exist_ok=True)
  for name in names+['complete.json']:
   p=run/name;digest=sha(p);LOG.info('READ %s SHA256=%s',p,digest)
   if name in sealed and sealed[name]!=digest:raise ValueError('Sealed input changed: '+str(p))
   if name not in sealed and name not in ('complete.json','verification.json'):
    raise ValueError('Input not included in source seal: '+str(p))
   if name=='verification.json' and json.loads(p.read_text(encoding='utf-8')).get('status')!='PASS':raise ValueError('Source verification failed')
   shutil.copy2(p,target/name)
   bindings.append(dict(scope=scope,source_run=run.name,file=name,sha256=digest,verified_against_seal=name in sealed))
   if name.endswith('.csv'):frames[(scope,name)]=pd.read_csv(p,float_precision='round_trip')
  receipts[scope]=dict(run_id=run.name,complete_sha256=sha(run/'complete.json'),data_manifest_sha256=complete.get('data_manifest_sha256'),git_commit=complete.get('git_commit'))
 return frames,receipts,bindings

def add_scores(frame):
 """Recalculate the normalized metric independently from physical pair errors."""
 d=frame.copy();d['normalized_score']=np.nan
 pairs=d[d.aggregation.eq('pair')].copy()
 keys=['protocol','split','dataset_id','pair_id']
 raw=pairs[pairs.variant.eq('raw')][keys+['gyro_rmse','accel_rmse']]
 p=pairs.reset_index().merge(raw,on=keys,suffixes=('','_raw'),validate='many_to_one')
 p['score']=.5*(p.gyro_rmse/np.maximum(p.gyro_rmse_raw,1e-12)+p.accel_rmse/np.maximum(p.accel_rmse_raw,1e-12))
 d.loc[p['index'],'normalized_score']=p.score.to_numpy()
 datasets=p.groupby(KEYS+['dataset_id']).score.mean()
 overall=datasets.groupby(level=KEYS).mean()
 for i,r in d[d.aggregation.isin(['dataset','overall'])].iterrows():
  k=tuple(r[x] for x in KEYS)
  d.at[i,'normalized_score']=datasets.loc[k+(r.dataset_id,)] if r.aggregation=='dataset' else overall.loc[k]
 return d

def assert_near(a,b,desc):
 if not np.allclose(a,b,rtol=2e-10,atol=2e-12,equal_nan=True):raise ValueError('Numeric audit failed: '+desc)

def audit_long(summary,seeds):
 count=0
 for _,r in summary[summary.metric.isin(METRICS)&summary.aggregation.isin(['pair','dataset','overall'])].iterrows():
  mask=pd.Series(True,index=seeds.index)
  for k in IDENTITY:mask &= seeds[k].eq(r[k])
  a=seeds.loc[mask,r.metric].to_numpy()
  if not len(a) or len(a)!=r.n:raise ValueError('Missing/ambiguous summary seeds')
  assert_near(a.mean(),r['mean'],'summary mean');assert_near(a.std(ddof=1) if len(a)>1 else np.nan,r.sd,'summary SD')
  count+=1
 return count

def prepare(frames):
 s=add_scores(frames['shared','metrics_per_seed.csv']);w=add_scores(frames['windows','metrics_per_seed.csv'])
 audit={"shared_summary_cells_recomputed":audit_long(frames['shared','summary.csv'],s),
        "window_summary_cells_recomputed":audit_long(frames['windows','summary.csv'],w)}
 p=frames['shared','paired_deltas.csv'];summary=frames['shared','paired_summary.csv']
 pp=p[p.metric.isin(METRICS)&p.aggregation.isin(['dataset','pair','overall'])]
 for _,r in pp.iterrows():
  keys=[k for k in IDENTITY if k!='variant']+['seed']
  mask=pd.Series(True,index=s.index)
  for k in keys:mask &= s[k].eq(r[k])
  a=s[mask&s.variant.eq(r.variant)];b=s[mask&s.variant.eq('cnn_lstm64')]
  if len(a)!=1 or len(b)!=1:raise ValueError('Bad paired source')
  av=float(a.iloc[0][r.metric]);bv=float(b.iloc[0][r.metric])
  assert_near(av,r.value,'paired value');assert_near(bv,r.reference_value,'paired reference')
  assert_near(100*(av-bv)/bv,r.relative_delta_pct,'paired percent')
 for _,r in summary[summary.metric.isin(METRICS)&summary.aggregation.isin(['dataset','pair','overall'])].iterrows():
  mask=p.metric.eq(r.metric)
  for k in IDENTITY:mask &= p[k].eq(r[k])
  a=p[mask].relative_delta_pct
  assert_near(a.mean(),r.relative_delta_pct_mean,'paired summary mean')
  assert_near(a.std(ddof=1),r.relative_delta_pct_sd,'paired summary SD')
 audit['same_seed_paired_percentages_recomputed']=len(pp)
 ranking=frames['shared','validation_ranking.csv']
 for protocol in ['alimu_only','pooled']:
  a=s[(s.protocol==protocol)&(s.split=='validation')&(s.aggregation=='overall')&(s.variant!='raw')].groupby('variant').normalized_score.mean().sort_values()
  b=ranking[ranking.protocol.eq(protocol)].sort_values('rank')
  if a.index.tolist()!=b.variant.tolist():raise ValueError('VALID rank disagreement')
  assert_near(a.values,b['mean'].values,'VALID ranking')
 m=frames['matched','metrics_per_seed.csv'];ms=frames['matched','summary.csv']
 for _,r in ms.iterrows():
  a=m[(m.pair_id==r.pair_id)&(m.method==r.method)&(m.component==r.component)].rmse.to_numpy()
  if len(a)!=r.n:raise ValueError('Bad matched n')
  assert_near(a.mean(),r.rmse_mean,'matched mean');assert_near(a.std(ddof=1) if len(a)>1 else np.nan,r.rmse_sd,'matched SD')
 strict=frames['matched','matched_seed_deltas.csv']
 for _,r in strict.iterrows():
  for method,col in [('ddpm','ddpm_rmse'),('no_diffusion_matched','no_diffusion_rmse')]:
   a=m[(m.pair_id==r.pair_id)&(m.seed==r.seed)&(m.component==r.component)&(m.method==method)].rmse
   if len(a)!=1:raise ValueError('Bad strict pair')
   assert_near(a.iloc[0],r[col],'strict pair input')
  assert_near(100*(r.ddpm_rmse/r.no_diffusion_rmse-1),r.ddpm_change_pct,'strict ratio')
 if len(strict)!=18 or not strict.ddpm_change_pct.gt(0).all():raise ValueError('Strict DDPM result changed')
 audit['matched_summary_cells_recomputed']=len(ms);audit['strict_diffusion_deltas_recomputed']=len(strict)
 c=frames['shared','compute_cost.csv'];cs=frames['shared','compute_summary.csv']
 cc=cs[cs.cost_scope.eq('uniform_new_inference')]
 for _,r in cc.iterrows():
  a=c[(c.protocol==r.protocol)&(c.variant==r.variant)][r.metric]
  assert_near(a.mean(),r['mean'],'uniform compute mean');assert_near(a.std(ddof=1),r.sd,'uniform compute SD')
 audit['uniform_compute_cells_recomputed']=len(cc)
 order=ranking[ranking.protocol.eq('alimu_only')].sort_values('rank').variant.tolist()
 return dict(shared_seed=s,window_seed=w,paired_seed=p,matched_seed=m,strict_seed=strict,compute_seed=c,order=order),audit

def markdown_table(frame):
 cols=frame.columns.tolist()
 def fmt(x):return str(x).replace('|','/')
 return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def materialize_tables(data,frames,out):
 p=out/'source_data';p.mkdir(exist_ok=True)
 s=data['shared_seed'];valid=frames['shared','validation_ranking.csv'];summ=frames['shared','summary.csv']
 # Panel files retain individual seeds and full units/identities; no plotting-only rounding.
 shared=s[(s.dataset_id=='alimu')&s.aggregation.isin(['dataset','pair'])][KEYS+['aggregation','dataset_id','pair_id','samples']+METRICS]
 files={'fig1_shared_selection':shared[shared.aggregation.eq('dataset')&shared.protocol.eq('alimu_only')],
 'fig2_shared_physical':shared[shared.aggregation.eq('dataset')&shared.protocol.eq('alimu_only')&shared.split.eq('test')],
 'fig3_paired_changes':data['paired_seed'].query("protocol == 'alimu_only' and split == 'test' and aggregation == 'dataset' and dataset_id == 'alimu'").loc[lambda x:x.metric.isin(METRICS)],
 'fig3_window_sweep':data['window_seed'].query("protocol == 'alimu_only' and variant == 'cnn_bilstm' and aggregation == 'dataset' and dataset_id == 'alimu'")[KEYS+['aggregation','dataset_id','samples']+METRICS],
 'fig3_device_scores':shared[shared.aggregation.eq('pair')&shared.protocol.eq('alimu_only')&shared.split.eq('test')],
 'fig4_per_device_metrics':data['matched_seed'].copy(),
 'fig4_strict_diffusion_pairs':data['strict_seed'].copy(),
 'fig5_training_scope':shared[shared.aggregation.eq('dataset')&shared.split.eq('test')],
 'fig5_uniform_cost':data['compute_seed'].copy()}
 for name,df in files.items():
  # Remove unrelated source-machine directory names from public panel data.
  df=df.drop(columns=[c for c in df if c.startswith('source_') or c.startswith('reference_source_')],errors='ignore')
  write_csv(p/(name+'.csv'),df)
 rows=[]
 for v in data['order']+['raw']:
  r={'variant':v,'model':LABELS[v]}
  for metric in METRICS:
   cell=summ[(summ.protocol=='alimu_only')&(summ.split=='test')&(summ.aggregation=='dataset')&(summ.dataset_id=='alimu')&(summ.variant==v)&(summ.metric==metric)].iloc[0]
   for field in ['mean','sd','n']:r[metric+'_'+field]=cell[field]
  if v!='raw':
   a=valid[(valid.protocol=='alimu_only')&(valid.variant==v)].iloc[0]
   r.update(validation_score_mean=a['mean'],validation_score_sd=a.sd,validation_rank=a['rank'])
  rows.append(r)
 table=pd.DataFrame(rows);write_csv(p/'shared_alimu_model_results.csv',table)
 write_csv(p/'matched_pooled_results.csv',frames['matched','summary.csv'].query("pair_id == '__all_pairs__' and component in ['gyro','accel']"))
 return table

def report(data,frames,table,config,out,audit):
 winners={'shared_alimu_validation_selected':'cnn_gru','shared_alimu_observed_test_minimum':'cnn_bigru','joint_validation_selected':'cnn_lstm64',
          'matched80_observed_gyro_minimum':'informer','matched80_observed_accel_minimum':'no_diffusion'}
 test=table[table.variant.ne('raw')].set_index('variant')
 if test.validation_score_mean.idxmin()!='cnn_gru' or test.normalized_score_mean.idxmin()!='cnn_bigru':raise ValueError('Winner identity disagreement')
 for m in ['gyro_rmse','accel_rmse']:
  if test[m+'_mean'].idxmin()!='cnn_bigru':raise ValueError('Physical winner disagreement')
 ms=frames['matched','summary.csv'].query("pair_id == '__all_pairs__'")
 for metric,v in [('gyro','informer'),('accel','no_diffusion')]:
  if ms[ms.component.eq(metric)].sort_values('rmse_mean').iloc[0].method!=v:raise ValueError('Matched winner disagreement')
 write_json(out/'conclusions.json',winners)
 human=[]
 for _,r in table.iterrows():
  def est(metric,scale=1,digits=4):
   a=r[metric+'_mean']*scale;b=r[metric+'_sd']*scale
   return f'{a:.{digits}f}' if pd.isna(b) else f'{a:.{digits}f} ± {b:.{digits}f}'
  human.append({'模型':r.model,'VALID分数':f'{r.validation_score_mean:.4f}' if r.variant!='raw' else '1.0000',
   'TEST角速度 (10⁻³ rad/s)':est('gyro_rmse',1000,3),'TEST加速度 (m/s²)':est('accel_rmse',1,4),'TEST综合分数':est('normalized_score',1,4)})
 raw=table[table.variant.eq('raw')].iloc[0];best=table[table.variant.eq('cnn_bigru')].iloc[0]
 gain_g=100*(1-best.gyro_rmse_mean/raw.gyro_rmse_mean);gain_a=100*(1-best.accel_rmse_mean/raw.accel_rmse_mean)
 ps=frames['shared','paired_summary.csv'];pct=ps[(ps.protocol=='alimu_only')&(ps.split=='test')&(ps.variant=='cnn_bigru')&(ps.aggregation=='dataset')&(ps.dataset_id=='alimu')].set_index('metric').relative_delta_pct_mean
 text=f'''# ALiMU 结果与 Nature 风格证据图

本图集直接整理已完成实验，未新增训练。主图结论：**ALiMU 单独训练的共享模型应按 VALID 选择 CNN-GRU；CNN-BiGRU 是这轮 TEST 均值最低的候选。** 两者含义不同。早期逐设备80点实验中，**Informer 的角速度误差最低，独立调参的无扩散 Transformer 的加速度误差最低**。

## 共享模型：11种结构的统一比较

三种 LOW IMU 共享一个模型映射到实测 VN100。40Hz，320点输入（名义8s），中心40点监督，stride40。ALiMU TRAIN dyn01/02/04/05/06/stat11，VALID dyn07，TEST dyn08/09，排除重复 dyn03/stat10。统一 AdamW、固定训练配方、早停与三个 seed；本轮没有为每种结构单独搜索超参数。

{markdown_table(pd.DataFrame(human))}

表按 VALID 综合分数排序；Raw 单列为控制。均值±样本标准差来自三个优化 seed，不是独立录制置信区间。物理指标先合并样本误差再开方；综合分数先在设备对内平均 gyro/Raw 与 accel/Raw，再等权设备对，不能用表中合并物理 RMSE 重新构造。三个设备共享两条物理 TEST 录制；每个 seed 有88,080个设备时间目标行。

CNN-BiGRU 相对 Raw 的合并角速度/加速度 RMSE 分别降低 **{gain_g:.2f}% / {gain_a:.2f}%**。相对同 seed CNN-LSTM64，其角速度、加速度、综合分数平均变化分别为 **{pct['gyro_rmse']:+.2f}% / {pct['accel_rmse']:+.2f}% / {pct['normalized_score']:+.2f}%**，三项均在三个 seed 中更低。这里与 Raw 的比较采用均值之比，与 CNN-LSTM 的比较采用同 seed 百分比均值，二者统计口径明确区分。

CNN-GRU 的 VALID 综合分数为0.3319±0.0033，CNN-BiGRU 为0.3337±0.0018；领先很小，不能据此宣称显著差异。部署选型保留 VALID 选出的 CNN-GRU，CNN-BiGRU 可作为后续新录制验证的候选。BiGRU 使用双向上下文；单向 CNN-RNN 也因CNN看到6个未来点，不能当作严格因果实时模型。

## 消融、窗口与训练范围

模块消融支持保留本轮 CNN 前端：更换为 Linear、逐点 MLP、GRU 或 Transformer 后接相同 LSTM64，TEST 综合分数均升高；这限定于被测实现及配方，并非模型家族上限。后端更换为 GRU/BiGRU 是有竞争力的选择。七个新非删除替换结构参数量在参考±5%内；BiLSTM32约少10.06%，CNN-only和Linear控制不属于容量匹配组。

早期固定中心监督的 CNN-BiLSTM 窗口扫描中，VALID 最优160点（4s），已观察 TEST 最低320点（8s）。图中同时展示两者；不能因为 TEST 最低而把8s称为验证选定窗口。

当训练范围改为 ALiMU+PennCOSYVIO 联合共享模型时，两个数据集等权 VALID 选出 CNN-LSTM64。该范围下只看 ALiMU TEST 综合分数，CNN-BiLSTM32 的均值最低；不能据 ALiMU TEST 为联合模型重新选型。图5全部测试点仍只评价 ALiMU，避免混入另一数据集的绝对误差。

## 独立逐设备80点实验

{markdown_table(pd.DataFrame([{'模型':LABELS[v],**{('角速度 (10⁻³ rad/s)' if m=='gyro' else '加速度 (m/s²)'):f"{ms[(ms.method==v)&(ms.component==m)].iloc[0].rmse_mean*(1000 if m=='gyro' else 1):.4f}" for m in ['gyro','accel']}} for v in ['raw','affine','ddpm','cnn_bilstm','informer','transformer','no_diffusion','no_diffusion_matched']]))}

该表用三设备分别训练、80点输入、整窗预测、Adam、每方法三个候选验证选择；每个seed89,949个设备时间行。它与共享320点、中心监督、AdamW实验的训练范围、目标支持和配方不同，**两张表不能合并成全局排行榜**。严格去扩散消融的18个设备×seed×信号组差值全部支持去掉扩散；目前没有证据表明 DDPM 带来收益。

## 图集与使用

- 图1：验证选择与已观察测试排名，最直接回答“哪个模型最优”。
- 图2：全部共享候选的物理误差与 Raw 基线。
- 图3：前后模块、窗口与设备层面的支持证据。
- 图4：逐设备80点对照及严格去扩散消融。
- 图5：训练范围敏感性与统一 CUDA 推理延迟。

图片统一7.2英寸宽，采用可编辑文字SVG、嵌入TrueType字体PDF、300dpi PNG与600dpi LZW TIFF。图注见 figures/figure_index.md，逐seed源数据见 source_data/。使用项目Matplotlib样式和Nature图形技能的证据层级、简洁配色、可编辑文字、点与误差线规范；未声称已满足特定期刊最终投稿检查。

所有误差线为优化随机性；未做显著性检验或添加p值。TEST此前已被查看，且只有两条物理录制。VN100为实测参考，结论限于六轴信号映射，不推出导航收益或绝对器件等级。所有图片由同一Python后端生成，未改动原始实验数值。

## 可复现与核验

源运行、配置快照、数据清单摘要和git提交保存在 provenance.json 与 data_manifest.json；每个读取的文件都记录SHA256，原实验目录保持只读。当前工作区有未提交实现，故同时保存本次绘图代码快照及其哈希。命令为：

```powershell
& 'D:/Anaconda3/envs/pinn_imu/python.exe' scripts/build_alimu_nature_figures.py --config configs/analysis/alimu_nature_figures.yaml
```

本次从逐seed数据重算共享汇总{audit['shared_summary_cells_recomputed']}格、窗口汇总{audit['window_summary_cells_recomputed']}格、同seed百分比{audit['same_seed_paired_percentages_recomputed']}项、逐设备汇总{audit['matched_summary_cells_recomputed']}格、严格扩散差值18项，以及统一推理成本{audit['uniform_compute_cells_recomputed']}格；结果与封存表一致。verification.json记录数值与导出检查。全部图在最终尺寸下检查，数值源文件保持完整精度。
'''
 LOG.info('WRITE %s',out/'report.md');(out/'report.md').write_text(text,encoding='utf-8')
 index='# ALiMU figure index and caption drafts\n\nAll figures use Python/Matplotlib; SVG text is editable. Error bars are sample SD over optimization seeds, not recording confidence intervals. No significance tests.\n\n'
 for stem,(title,caption) in CAPTIONS.items():
  index+=f'## {stem}: {title}\n\n[SVG]({stem}.svg) · [PDF]({stem}.pdf) · [PNG]({stem}.png) · [TIFF]({stem}.tiff)\n\n{caption}\n\n'
 (out/'figures/figure_index.md').write_text(index,encoding='utf-8')
 return winners

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
 logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s',stream=sys.stdout)
 logging.getLogger('fontTools').setLevel(logging.WARNING)
 LOG.info('READ %s',args.config);cfg=yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
 if cfg['backend']!='python':raise ValueError('Selected backend must be Python')
 np.random.seed(cfg['seed']);out=ROOT/cfg['output_dir']
 # Only our declared presentation output can be created; source directories stay sealed.
 if out.resolve() in [ (ROOT/x).resolve() for x in cfg['sources'].values()]:raise ValueError('Output overlaps source')
 if (out/'complete.json').exists():
  LOG.info('SKIP sealed presentation run %s; choose a new experiment_id/output_dir for revisions',out);return
 out.mkdir(parents=True,exist_ok=True)
 shutil.copy2(args.config,out/'config.yaml')
 frames,receipts,bindings=read_sources(cfg,out)
 LOG.info('READ numeric audit: independent normalization, mean/SD, validation ranks, paired deltas and inference costs')
 data,audit=prepare(frames);table=materialize_tables(data,frames,out)
 LOG.info('WRITE all five figure bundles under %s',out/'figures');exports=render_figures(data,out/'figures',cfg)
 winners=report(data,frames,table,cfg,out,audit)
 git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 sources=['scripts/build_alimu_nature_figures.py','src/virtual_imu/viz/plots_alimu_nature.py','src/virtual_imu/viz/style.py']
 source_hashes={}
 for rel in sources:
  dest=out/'plotting_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);source_hashes[rel]=sha(ROOT/rel)
 manifest={'source_runs':receipts,'inputs':bindings,'source_data':{p.name:sha(p) for p in (out/'source_data').glob('*.csv')}}
 write_json(out/'data_manifest.json',manifest)
 write_json(out/'figure_contract.json',cfg['contract'])
 write_json(out/'provenance.json',dict(run_id=cfg['experiment_id'],created_utc=datetime.now(timezone.utc).isoformat(),git_commit=git,
  source_runs=receipts,input_config_sha256=sha(args.config),plotting_sources_sha256=source_hashes,
  data_manifest_sha256=sha(out/'data_manifest.json'),backend='Python/Matplotlib only',python=sys.version,
  numerical_results='Copied from frozen completed runs; no new training',private_templates='Not used'))
 # Re-read source files to establish that every consumed artifact stayed unchanged.
 for b in bindings:
  source=ROOT/cfg['sources'][b['scope']]/b['file']
  if sha(source)!=b['sha256']:raise ValueError('Read-only source changed during rendering')
 write_json(out/'verification.json',dict(status='PASS',numeric_audit=audit,exports=exports,source_inputs_unchanged=len(bindings),
  all_panel_data_from_real_runs=True,selection_separated_from_observed_test_minimum=True,
  cross_protocol_absolute_ranking=False,visual_review='pending manual review of all five PNG exports',
  note='This is a presentation-only audit; source runs retain their independent experiment verifications.'))
 LOG.info('WRITE build receipt; visual QA is required before completion seal')
 LOG.info('SKIP training, raw datasets, checkpoints, source-run mutation and outputs/latest')

if __name__=='__main__':main()
