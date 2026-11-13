"""Read-only status and artifact verification for the ALiMU context study."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]

def alive(pid):
    if os.name!='nt':
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.GetExitCodeProcess.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return ctypes.get_last_error()==5
    code=wintypes.DWORD();kernel.GetExitCodeProcess(handle,ctypes.byref(code));kernel.CloseHandle(handle)
    return code.value==259

def status(run):
    completed=[];active=[];pending_resume=[]
    for path in (run/'fits').glob('*/fit.json'):
        try:completed.append(json.loads(path.read_text(encoding='utf-8')))
        except json.JSONDecodeError:continue
    for path in (run/'fits').glob('*/history.csv'):
        if (path.parent/'fit.json').exists():continue
        lines=path.read_text(encoding='utf-8').splitlines()
        if len(lines)>1:
            item=dict(job=path.parent.name,epoch=int(lines[-1].split(',')[0]))
            (active if time.time()-path.stat().st_mtime<120 else pending_resume).append(item)
    pid_path=run/'coordinator.pid';pid=int(pid_path.read_text()) if pid_path.exists() else None
    return dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),complete=(run/'complete.json').exists(),
                completed_slots=len(completed),completed_new=sum(not r['reused'] for r in completed),completed_reused=sum(r['reused'] for r in completed),
                total_planned_slots=227,active=active,pending_resume=pending_resume,coordinator_pid=pid,coordinator_running=alive(pid) if pid else False)

def verify(run):
    if not (run/'complete.json').exists():raise RuntimeError('Run remains incomplete; no final-result verification is claimed')
    manifest=json.loads((run/'artifact_manifest.json').read_text(encoding='utf-8'))
    for relative,expected in manifest['files'].items():
        path=(run/relative).resolve()
        if run.resolve() not in path.parents:raise RuntimeError('Artifact path escapes run')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
        if digest.hexdigest()!=expected:raise RuntimeError('Artifact hash mismatch: '+relative)
    print('READ sealed artifacts; SKIP writes; PASS',len(manifest['files']),'hashes',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='runs/alimu_context_mechanisms_20261108');p.add_argument('--verify',action='store_true');p.add_argument('--follow',action='store_true')
    args=p.parse_args();run=(ROOT/args.run).resolve()
    print('READ run',run,'; SKIP all artifact mutations',flush=True)
    if args.verify:verify(run)
    else:
        previous=None
        while True:
            state=status(run)
            signature=(state['completed_slots'],state['complete'],state['coordinator_running'],tuple(item['job'] for item in state['active']))
            if signature!=previous:print(json.dumps(state,ensure_ascii=False),flush=True)
            previous=signature
            if not args.follow or state['complete'] or not state['coordinator_running']:break
            time.sleep(30)
