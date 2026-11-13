"""Audit the explicit, pre-TEST mixed precision recovery amendment."""
from pathlib import Path
import json
import pandas as pd
from virtual_imu.replacements.training import sha,write_json

def validate_precision_policy(run,repository):
    run=Path(run);repository=Path(repository);path=run/'numeric_precision_recovery.lock.json'
    if not path.exists():return None
    policy=json.loads(path.read_text(encoding='utf-8'))
    if policy['test_payload_loaded'] or policy['architecture_changed'] or policy['hyperparameters_changed'] or policy['seed_or_split_changed']:
        raise AssertionError('Numeric amendment changed the scientific protocol')
    for name,digest in policy['sources'].items():
        if sha(repository/name)!=digest:raise AssertionError('Numerical recovery source changed '+name)
    if sha(run/policy['diagnostic'])!=policy['diagnostic_sha256']:raise AssertionError('Original numerical failure evidence changed')
    evidence=json.loads((run/policy['diagnostic']).read_text(encoding='utf-8'))
    if evidence['status']!='REPRODUCED' or evidence['roles']!=['train','validation'] or not evidence['parameters_finite']:
        raise AssertionError('Numerical diagnosis did not establish finite-input overflow')
    return policy

def precision_receipts(run):
    run=Path(run);rows=[]
    for path in sorted(run.glob('fits/*/numeric_precision_events.jsonl'))+list(run.glob('numeric_precision_events.jsonl')):
        events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if any(not e['output_finite'] or e['recompute_dtype']!='torch.float32' for e in events):raise AssertionError('Invalid precision recovery event')
        rows.append(dict(job_id=path.parent.name if path.parent!=run else 'evaluation',training_events=sum(e['training'] for e in events),evaluation_events=sum(not e['training'] for e in events),total_events=len(events),receipt_sha256=sha(path)))
    return rows

def summarize_precision(run,repository):
    run=Path(run);policy=validate_precision_policy(run,repository)
    if policy is None:return None
    rows=precision_receipts(run)
    pd.DataFrame(rows,columns=['job_id','training_events','evaluation_events','total_events','receipt_sha256']).to_csv(run/'numeric_precision_recovery.csv',index=False)
    result=dict(status='PASS',policy_sha256=sha(run/'numeric_precision_recovery.lock.json'),operation='ProbSparseAttention',trigger='finite-input autocast output overflow only',events=sum(r['total_events'] for r in rows),rows=rows,
                source_and_diagnostic_hashes_verified=True,pretest_amendment=True,precision_exception=True,normal_finite_path_unchanged=True)
    write_json(run/'numeric_precision_audit.json',result);return result

def audit_precision(run,repository):
    run=Path(run);policy=validate_precision_policy(run,repository)
    if policy is None:return None
    rows=precision_receipts(run);recorded=json.loads((run/'numeric_precision_audit.json').read_text(encoding='utf-8'))
    if rows!=recorded['rows'] or sum(r['total_events'] for r in rows)!=recorded['events']:raise AssertionError('Numeric precision event accounting mismatch')
    return dict(status='PASS',events=recorded['events'],precision_exception=True,pretest_policy_sha256=sha(run/'numeric_precision_recovery.lock.json'))
