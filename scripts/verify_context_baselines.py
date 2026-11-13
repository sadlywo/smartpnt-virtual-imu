"""Cold-check persisted baseline predictions and replay selected states."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from virtual_imu.ablations.data import CommonWindows, HIGH, LOW
from virtual_imu.context_study.classical_models import load_state, predict
from virtual_imu.context_study.classical_pipeline import independent_metrics, verify
from virtual_imu.context_study.data import StudySource
from virtual_imu.fir_arx_research.data import sha256_file

LOG = logging.getLogger(__name__)


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Audit output already exists; preserve it and choose another path')
    verification = verify(run)
    complete = json.loads((run/'complete.json').read_text(encoding='utf-8'))
    if sha256_file(run/'artifact_manifest.json') != complete['artifact_manifest_sha256']:
        raise ValueError('Completion manifest identity mismatch')
    lock = json.loads((run/'selection.lock.json').read_text(encoding='utf-8'))
    access = json.loads((run/'test_access.json').read_text(encoding='utf-8'))
    if lock['test_loaded'] or access['selection_lock_sha256'] != sha256_file(run/'selection.lock.json'):
        raise ValueError('Selection/TEST access lock mismatch')
    provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    source_run = Path(provenance['source_run'])
    source_cfg = yaml.safe_load((source_run/'config.yaml').read_text(encoding='utf-8'))
    trials = pd.read_csv(run/'trials.csv')
    for group, rows in trials.groupby(['group', 'method']):
        if len(rows) != 12 or rows.primary.sum() != 3 or set(rows.role) != {'validation'}:
            raise ValueError('Candidate budget or selection role mismatch '+str(group))
    for identity, chosen in lock['selections'].items():
        rows = trials[(trials.group == chosen['group']) & (trials.method == chosen['method']) & trials.valid]
        if chosen['tier'] == 'primary3':
            rows = rows[rows.primary]
        expected = rows.sort_values(['score', 'candidate_id']).iloc[0]
        if expected.candidate_id != chosen['candidate_id']:
            raise ValueError('Saved selection is not VALID-optimal '+identity)
    metrics = pd.read_csv(run/'metrics.csv')
    errors, model_replays = [], []
    for held in ['shared', 'BMX160', 'LSM9DS1', 'MPU9250']:
        if not any(r['condition']['held'] == held for r in lock['selections'].values()):
            continue
        stats = json.loads((run/'supports'/(held+'__stats.json')).read_text(encoding='utf-8'))
        for role in ['validation', 'test']:
            LOG.info('READ persisted and original supports cohort=%s role=%s', held, role)
            saved = pd.read_parquet(run/'supports'/(held+'__'+role+'.parquet'))
            original = pd.read_parquet(source_run/'supports'/(held+'__'+role+'.parquet'))
            pd.testing.assert_frame_equal(saved, original, check_exact=True)
            source = StudySource(ROOT, source_cfg['data'], held_sensor=None if held == 'shared' else held)
            data = CommonWindows(source.load(role), stats=stats)
            pd.testing.assert_frame_equal(data.target_frame, saved, check_exact=True)
            arrays = {}
            for identity, chosen in lock['selections'].items():
                condition = chosen['condition']
                if condition['held'] != held:
                    continue
                path = run/'predictions'/(identity+'__'+role+'.npz')
                with np.load(path, allow_pickle=False) as stored:
                    prediction = stored['prediction']
                    if str(stored['support_sha256']) != data.support_sha256:
                        raise ValueError('Prediction support digest mismatch '+identity)
                rows = metrics[(metrics.job_id == identity) & (metrics.role == role)]
                for _, row in rows.iterrows():
                    if row.aggregation == 'overall' or row.aggregation == 'dataset':
                        indices = np.arange(len(saved))
                    elif row.aggregation == 'physical_recording':
                        indices = np.flatnonzero(saved.physical_group_id.eq(row.physical_group_id))
                    elif row.aggregation == 'pair':
                        indices = np.flatnonzero(saved.pair_id.eq(row.pair_id))
                    elif row.aggregation == 'recording':
                        indices = np.flatnonzero(saved.original_sequence_id.eq(row.original_sequence_id))
                    else:
                        raise ValueError('Unknown aggregation '+row.aggregation)
                    frame = saved.iloc[indices]
                    values = independent_metrics(frame, prediction[indices])
                    residual = prediction[indices]-frame[HIGH].to_numpy(float)
                    for axis, name in enumerate(['gx', 'gy', 'gz', 'ax', 'ay', 'az']):
                        values[name+'_rmse'] = float(np.sqrt(np.mean(residual[:, axis]**2)))
                    for name, part in [('gyro', slice(0, 3)), ('accel', slice(3, 6))]:
                        values[name+'_vector_rmse'] = float(np.sqrt(np.mean(np.sum(residual[:, part]**2, axis=1))))
                        raw = frame[LOW].to_numpy(float)-frame[HIGH].to_numpy(float)
                        values[name+'_improvement_pct'] = 100*(1-values[name+'_rmse']/np.sqrt(np.mean(raw[:, part]**2)))
                    values['relative_error_reduction_pct'] = 100*(1-values['score'])
                    error = max(abs(float(row[k])-v) for k, v in values.items())
                    if error > 1e-10:
                        raise ValueError('Saved physical/axis metric mismatch '+identity)
                    errors.append(error)
                window = condition['window']
                if window not in arrays:
                    arrays[window] = data.arrays(window)[0].transpose(0, 2, 1)
                state_path = run/chosen['state']
                if sha256_file(state_path) != chosen['state_sha256']:
                    raise ValueError('Selected state changed '+identity)
                normal = predict(load_state(state_path), arrays[window], center=(window-40)//2,
                                 block_size=40, batch_anchors=48)
                physical = data.physical(normal.transpose(0, 2, 1))
                replay_error = float(np.max(np.abs(physical-prediction)))
                if replay_error > 1e-10:
                    raise ValueError('Full persisted prediction replay mismatch '+identity)
                model_replays.append(dict(job_id=identity, role=role, samples=len(saved),
                                          maximum_error=replay_error))
                LOG.info('PASS persisted metrics and full state replay %s %s max_error=%.3g', identity, role, replay_error)
            del data, arrays, saved, original
    result = dict(status='PASS', source_run=str(run), source_artifact_manifest_sha256=sha256_file(run/'artifact_manifest.json'),
                  verifier_sha256=sha256_file(Path(__file__)), hash_verification=verification,
                  selections_rechecked=len(lock['selections']), persisted_metric_rows=len(errors),
                  max_metric_error=max(errors), full_prediction_replays=model_replays,
                  max_prediction_replay_error=max(r['maximum_error'] for r in model_replays),
                  support_values_identical_to_original=True, candidate_selection_valid_only=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), output.parent/'verify_context_baselines.py')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    LOG.info('WRITE independent verification %s; SKIP sealed-run mutations', output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/alimu_context_classical_20261112')
    parser.add_argument('--out', default='runs/alimu_context_classical_20261112_verification/verification.json')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = audit(ROOT/args.run, ROOT/args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'full_prediction_replays'}, indent=2))
