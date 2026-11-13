"""Metadata-first device AND physical-recording separation."""
from __future__ import annotations
import copy
import logging
from virtual_imu.ablations.data import SharedSource,CommonWindows
LOG=logging.getLogger(__name__)

def strict_fold_records(records,held_sensor):
    result={role:[copy.deepcopy(r) for r in values if (r['low_sensor']==held_sensor)==(role=='test')]
            for role,values in records.items()}
    validate_split_records(result,held_sensor)
    return result

def validate_split_records(records,held_sensor=None):
    for role in ['train','validation','test']:
        if not records[role]: raise ValueError('Empty split '+role)
    groups={k:{r['physical_group_id'] for r in v} for k,v in records.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if groups[a]&groups[b]: raise ValueError('Physical recording leakage: '+a+'/'+b)
    if held_sensor is not None:
        for role in ['train','validation']:
            if any(r['low_sensor']==held_sensor for r in records[role]): raise ValueError('Held device leakage')
        if any(r['low_sensor']!=held_sensor for r in records['test']): raise ValueError('Wrong test device')
    return {'physical_groups':{k:sorted(v) for k,v in groups.items()},
            'sensors':{k:sorted({r['low_sensor'] for r in v}) for k,v in records.items()}}

class StudySource(SharedSource):
    def __init__(self,root,config,held_sensor=None):
        super().__init__(root,config,'alimu_only')
        if self.dataset.manifest_sha256!=config['manifest_sha256']: raise ValueError('Manifest changed')
        if held_sensor is not None: self.records=strict_fold_records(self.records,held_sensor)
        self.split_audit=validate_split_records(self.records,held_sensor)
        LOG.info('READ split metadata: %s; SKIP all unrequested numerical roles',self.split_audit)


def windows(source,role,stats=None):
    opts={k:source.config[k] for k in ['max_window','block_size','stride']}
    return CommonWindows(source.load(role),stats=stats,**opts)
