"""Detached single-model manager; completion is artifact-gated, failures do not retry."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from run_b1_fewshot import BASE,OUT,TARGET,guarded,sha,atomic_json


def manager():
    child=None
    def stop(sig,frame):
        atomic_json(OUT/'SUPERVISOR_SIGNAL.json',dict(signal=sig,pid=os.getpid(),time=time.time()))
        if child is not None and child.poll() is None: child.send_signal(signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]: signal.signal(sig,stop)
    for stage in ['train','evaluate','summary']:
        guarded()
        if stage!='summary':
            active=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
            if active: raise RuntimeError(f'GPU already occupied by{active}')
        command=[sys.executable,'-u',str(Path(__file__).parent/'run_b1_fewshot.py'),stage]
        if stage=='summary': command=[sys.executable,'-u',str(Path(__file__).parent/'summarize_b1_fewshot.py')]
        with (OUT/f'{stage}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.time()
            atomic_json(OUT/f'{stage}_LAUNCH.json',dict(pid=child.pid,ppid=os.getpid(),command=command,start=start))
            while child.poll() is None:
                atomic_json(OUT/'queue_state.json',dict(status='RUNNING',phase=stage,model='B1',pid=child.pid,
                            manager_pid=os.getpid(),time=time.time(),elapsed_s=time.time()-start))
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
                with (OUT/'health.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),pid=child.pid,stage=stage,gpu=gpu.stdout.strip(),
                                           alive=child.poll() is None))+'\n')
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: pass
            rc=child.returncode
        atomic_json(OUT/f'{stage}_EXIT.json',dict(exit_code=rc,signal=-rc if rc<0 else rc-128 if rc in [129,130,143] else None,
                    end=time.time(),elapsed_s=time.time()-start))
        if rc!=0:
            atomic_json(OUT/'queue_state.json',dict(status='BLOCKED',phase=stage,exit_code=rc,automatic_retry=False))
            return
        if stage=='train':
            gate=json.loads((TARGET/'TRAIN_COMPLETE.json').read_text())
            assert gate['step']==500 and sha(gate['checkpoint'])==gate['checkpoint_sha256']
        elif stage=='evaluate':
            gate=json.loads((TARGET/'MODEL_COMPLETE.json').read_text())
            assert gate['steps']==500 and gate['query_subjects']==15 and gate['roi_rows']==120
        else:
            assert json.loads((OUT/'COMPLETE.json').read_text())['new_models']==1
    atomic_json(OUT/'queue_state.json',dict(status='COMPLETE',models=1,steps=500,query_subjects=15,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manager',action='store_true'); a=p.parse_args()
    if a.manager:
        try: manager()
        except Exception as e:
            atomic_json(OUT/'SUPERVISOR_FAILURE.json',dict(error=str(e),pid=os.getpid(),time=time.time()))
            atomic_json(OUT/'queue_state.json',dict(status='BLOCKED',error=str(e),automatic_retry=False))
            raise
    else:
        guarded()
        if list(OUT.glob('failure_*.json')) or (OUT/'SUPERVISOR_FAILURE.json').exists():
            raise RuntimeError('Unresolved failure; no automatic launch/retry')
        with (OUT/'launch.lock').open('x') as f: f.write(str(os.getpid()))
        env=os.environ.copy(); env.update(OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',PYTHONUNBUFFERED='1')
        with (OUT/'supervisor.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--manager'],cwd=BASE,env=env,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/'LAUNCH.json',dict(pid=proc.pid,launch_parent=os.getpid(),detached=True,time=time.time()))
        print(json.dumps(dict(manager_pid=proc.pid,output=str(OUT))))
