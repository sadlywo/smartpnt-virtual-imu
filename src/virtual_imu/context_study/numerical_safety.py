"""Numerical recovery for finite-input FP16 ProbSparse overflow, without changing finite forwards."""
from __future__ import annotations
import json,logging,time
from pathlib import Path
import torch
from virtual_imu.models.matched import ProbSparseAttention
_ORIGINAL=ProbSparseAttention.forward
_INSTALLED=False
_RUN=None

def safe_sparse_forward(self,q,k,v):
    output=_ORIGINAL(self,q,k,v)
    if bool(torch.isfinite(output).all()):return output
    if not torch.is_autocast_enabled() or not all(bool(torch.isfinite(t).all()) for t in (q,k,v)):
        raise FloatingPointError('Nonfinite sparse attention outside the finite-input autocast recovery policy')
    with torch.autocast(device_type=q.device.type,enabled=False):
        output=_ORIGINAL(self,q.float(),k.float(),v.float()).to(q.dtype)
    if not bool(torch.isfinite(output).all()):raise FloatingPointError('FP32 sparse attention recovery was nonfinite')
    event=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),operation='ProbSparseAttention',sequence_length=self.sequence_length,
               input_dtype=str(q.dtype),recompute_dtype='torch.float32',training=bool(self.training),grad_enabled=torch.is_grad_enabled(),
               q_max_abs=float(q.detach().abs().max()),k_max_abs=float(k.detach().abs().max()),v_max_abs=float(v.detach().abs().max()),output_finite=True)
    loggers=logging.getLogger().handlers
    directories=[Path(h.baseFilename).parent for h in loggers if isinstance(h,logging.FileHandler) and Path(h.baseFilename).name=='training.log']
    target=(directories[-1] if directories else _RUN)
    if target is not None:
        with (Path(target)/'numeric_precision_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
    logging.getLogger(__name__).warning('WRITE finite-input sparse attention FP32 recovery length=%d training=%s',self.sequence_length,self.training)
    return output

def install(run=None):
    global _INSTALLED,_RUN
    if run is not None:_RUN=Path(run)
    if not _INSTALLED:ProbSparseAttention.forward=safe_sparse_forward;_INSTALLED=True

def initialize_safe_worker():
    from virtual_imu.context_study.execution import initialize_worker
    initialize_worker();install()
