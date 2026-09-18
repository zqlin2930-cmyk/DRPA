"""Detached serial launcher; no retries, no parameter changes, no early stopping."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
def main():
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'queue_start.json').exists(),'Already launched; no retry allowed'
    assert json.loads((ROOT/'preflight.json').read_text())['status']=='PREFLIGHT_PASS'
    for model in ('DRPA','PDFT','FullFT'):
        assert json.loads((ROOT/f'model_gate_{model}.json').read_text())['status']=='PASS'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').glob('*.py')}
    write('queue_start.json',{'pid':os.getpid(),'start_unix':time.time(),'code_sha256':hashes,'order':['DRPA','PDFT','FullFT'],'automatic_retry':False})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
    for name,args in [('DRPA',['experiment.py','--model','DRPA']),('PDFT',['experiment.py','--model','PDFT']),('FullFT',['experiment.py','--model','FullFT']),('REPORT',['report.py','--out',str(ROOT)])]:
        for p,digest in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,'Code changed during queue'
        command=[sys.executable,'-u',str(ROOT/'code'/args[0]),*args[1:]]
        start=time.time()
        with (ROOT/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
            write('queue_state.json',{'status':'RUNNING','stage':name,'child_pid':proc.pid,'supervisor_pid':os.getpid(),'start_unix':start})
            while proc.poll() is None:
                elapsed=time.time()-start
                write('health.json',{'stage':name,'child_pid':proc.pid,'process_alive':True,'elapsed_sec':elapsed,'unix':time.time(),
                      'timeout_advisory':elapsed>7*86400,'policy':'7-day timeout is advisory; no automatic kill or protocol change'})
                time.sleep(30)
            write(f'{name}.exit.json',{'returncode':proc.returncode,'start_unix':start,'end_unix':time.time(),'child_pid':proc.pid})
        if proc.returncode:
            write('queue_state.json',{'status':'FAILED','stage':name,'returncode':proc.returncode,'automatic_retry':False});return proc.returncode
        if name!='REPORT':assert (ROOT/name/'complete.json').exists(),'Missing completion gate'
    write('queue_state.json',{'status':'COMPLETE','end_unix':time.time()});return 0
if __name__=='__main__':sys.exit(main())
