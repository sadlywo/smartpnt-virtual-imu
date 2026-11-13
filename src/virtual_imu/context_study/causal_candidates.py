"""VALID-only selection among the already planned zero-future configurations."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics

def eligible_causal_configuration(job):
    options=job.get('model_options',{})
    if options.get('lookahead')!=0:return False
    if job['stage']=='future':return True
    if job['stage']!='mechanism' or options.get('norm','ln')=='bn':return False
    # This exact three-seed configuration reuses future/GRU0; count it once.
    alias=all(options.get(k,v)==v for k,v in dict(depth=3,width=64,kernel=5,dilation=1,norm='ln',activation='gelu').items())
    return not alias

def select_causal_configurations(jobs,fit_scores,seeds):
    groups=defaultdict(list)
    for job in jobs.values():
        if eligible_causal_configuration(job):groups[job['stage']+'__'+job['id'].split('__')[1]].append(job)
    candidates=[]
    for identity,group in sorted(groups.items()):
        group=sorted(group,key=lambda j:j['seed'])
        if sorted(j['seed'] for j in group)!=sorted(seeds):raise ValueError('Incomplete causal candidate seeds '+identity)
        values=[float(fit_scores[j['id']]) for j in group]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite VALID score '+identity)
        first=group[0]
        candidates.append(dict(candidate_id=identity,stage=first['stage'],family=first['family'],condition=first['id'].split('__')[1],
                               job_ids=[j['id'] for j in group],validation_score=statistics.mean(values),validation_sd=statistics.stdev(values)))
    if not candidates:raise ValueError('No zero-future configurations')
    chosen=min(candidates,key=lambda c:(c['validation_score'],c['candidate_id']))
    return dict(role='validation',criterion='Three-seed mean best VALID score; no TEST fields accepted',
                eligibility='B zero-future backends plus non-BN zero-future mechanism configurations; exact GRU0 reuse counted once',
                interpretation='Configuration search; GRU frontend variants have more explored configurations than the other backends. Use B alone for the fixed-frontend family comparison.',
                selected_candidate_id=chosen['candidate_id'],candidates=candidates)

def causal_configuration_results(overall,selection):
    """Report the locked selection and descriptive TEST outcomes separately."""
    import pandas as pd
    rows=[]
    for candidate in selection['candidates']:
        cell=overall[overall.job_id.isin(candidate['job_ids'])&overall.role.eq('test')]
        if len(cell)!=len(candidate['job_ids']):raise ValueError('Missing TEST rows for locked causal candidate')
        rows.append({k:v for k,v in candidate.items() if k!='job_ids'}|dict(S=float(cell.score.mean()),SD=float(cell.score.std(ddof=1)),
                    gyro=float(cell.gyro_rmse.mean()),accel=float(cell.accel_rmse.mean()),VALID_selected=candidate['candidate_id']==selection['selected_candidate_id']))
    return pd.DataFrame(rows).sort_values(['S','candidate_id']).reset_index(drop=True)
