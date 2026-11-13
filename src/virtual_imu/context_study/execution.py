"""Recorded execution-only concurrency/cache control; frozen fit arithmetic is unchanged."""
from __future__ import annotations
import concurrent.futures as futures
import gc
import torch
_LAST_HELD=object()

def initialize_worker():
    import virtual_imu.context_study.training as training
    original=training.cached_data
    def bounded_cached_data(job,device):
        global _LAST_HELD
        held=job.get('held_sensor')
        if held!=_LAST_HELD:
            training._CACHE.clear();gc.collect();torch.cuda.empty_cache();_LAST_HELD=held
        return original(job,device)
    training.cached_data=bounded_cached_data

def install_pool_override(workers):
    original=futures.ProcessPoolExecutor
    def recorded_pool(*args,**kwargs):
        if args or kwargs.get('initializer') is not None:raise ValueError('Unexpected pool constructor signature')
        kwargs['max_workers']=workers;kwargs['initializer']=initialize_worker
        return original(**kwargs)
    futures.ProcessPoolExecutor=recorded_pool
