"""Selection/reporting tests use synthetic scores, never research measurements."""
import pytest
import pandas as pd
from virtual_imu.context_study.causal_candidates import eligible_causal_configuration,select_causal_configurations,causal_configuration_results

def make_group(stage,tag,options,score):
    jobs={};scores={}
    for seed in [1,2,3]:
        identity='__'.join([stage,tag,'w320','s'+str(seed)])
        jobs[identity]=dict(id=identity,stage=stage,family='gru',seed=seed,model_options=options.copy(),test_score=0.)
        scores[identity]=score+seed*.001
    return jobs,scores

def test_valid_choice_includes_frontend_candidates_and_excludes_bn_future_and_exact_alias():
    jobs={};scores={}
    groups=[('future','gru_look0',dict(lookahead=0),.5),('mechanism','linear',dict(lookahead=0,norm='none',activation='identity'),.3),
            ('mechanism','bn',dict(lookahead=0,norm='bn'),.01),('future','gru_look6',dict(lookahead=6),.01),
            ('mechanism','default_alias',dict(lookahead=0,norm='ln',activation='gelu'),.01)]
    for stage,tag,options,score in groups:
        group,values=make_group(stage,tag,options,score);jobs.update(group);scores.update(values)
    selection=select_causal_configurations(jobs,scores,[1,2,3])
    assert selection['selected_candidate_id']=='mechanism__linear'
    assert len(selection['candidates'])==2
    assert selection['candidates'][1]['validation_score']==pytest.approx(.302)
    assert selection['candidates'][1]['validation_sd']==pytest.approx(.001)
    assert all(not eligible_causal_configuration(j) for j in jobs.values() if j['model_options'].get('norm')=='bn')

def test_incomplete_and_nonfinite_causal_candidates_fail_instead_of_dropping_seeds():
    jobs,scores=make_group('future','gru_look0',dict(lookahead=0),.5)
    missing=dict(jobs);missing.pop(next(iter(missing)))
    with pytest.raises(ValueError,match='Incomplete'):select_causal_configurations(missing,scores,[1,2,3])
    scores[next(iter(scores))]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):select_causal_configurations(jobs,scores,[1,2,3])

def test_test_observed_ranking_keeps_the_prior_valid_choice():
    a,sa=make_group('future','gru_look0',dict(lookahead=0),.3)
    b,sb=make_group('mechanism','linear',dict(lookahead=0,norm='none',activation='identity'),.4)
    jobs={**a,**b};selection=select_causal_configurations(jobs,{**sa,**sb},[1,2,3])
    rows=[]
    for identity,job in jobs.items():
        score=(.7 if job['stage']=='future' else .2)+job['seed']*.001
        rows.append(dict(job_id=identity,role='test',score=score,gyro_rmse=score*.001,accel_rmse=score*.2))
    table=causal_configuration_results(pd.DataFrame(rows),selection)
    assert table.iloc[0].candidate_id=='mechanism__linear' and not table.iloc[0].VALID_selected
    chosen=table[table.VALID_selected].iloc[0]
    assert chosen.candidate_id=='future__gru_look0' and chosen.S==pytest.approx(.702)
    assert chosen.SD==pytest.approx(.001)
