"""Continue the existing trajectory, then run remaining models with verified validation v2."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV=ROOT/'validation_v2'
def write(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
def main():
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready=json.loads((REV/'TRANSITION_READY.json').read_text());step=ready['step']
    assert json.loads((REV/'EQUIVALENCE_GATE.json').read_text())['status']=='PASS'
    assert json.loads((REV/'RESUME_EQUIVALENCE_GATE.json').read_text())['status']=='PASS'
    assert not (ROOT/'queue_revision_v2.json').exists(),'No automatic restart'
    paths=list((ROOT/'code').glob('*.py'))+[REV/x for x in ['experiment_v2.py','fast_validation.py','state_tools.py','report_v2.py','supervise_v2.py']]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write('queue_revision_v2.json',{'pid':os.getpid(),'start_unix':time.time(),'code_sha256':hashes,'resume_step':step,
          'order':['DRPA','PDFT','FullFT'],'automatic_retry':False,'change':'Equivalent validation computation and exact-state continuation'})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
    stages=[('DRPA',['experiment_v2.py','--model','DRPA','--resume-step',str(step)]),('PDFT',['experiment_v2.py','--model','PDFT']),
            ('FullFT',['experiment_v2.py','--model','FullFT']),('REPORT',['report_v2.py','--out',str(ROOT)])]
    for name,args in stages:
        for p,digest in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,'Code changed during v2 queue'
        command=[sys.executable,'-u',str(REV/args[0]),*args[1:]];start=time.time()
        with (ROOT/f'{name}.log').open('a' if name=='DRPA' else 'x') as log:
            if name=='DRPA':log.write(f'\nEXACT_STATE_CONTINUATION step={step}; optimized validation v2\n');log.flush()
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
            write('queue_state.json',{'status':'RUNNING','stage':name,'child_pid':proc.pid,'supervisor_pid':os.getpid(),'start_unix':start,'validation_revision':2})
            while proc.poll() is None:
                elapsed=time.time()-start
                write('health.json',{'stage':name,'child_pid':proc.pid,'process_alive':True,'elapsed_sec':elapsed,'unix':time.time(),
                      'timeout_advisory':elapsed>7*86400,'policy':'Advisory only; no kill/retry/protocol change','validation_revision':2})
                time.sleep(15)
            exit_record={'returncode':proc.returncode,'start_unix':start,'end_unix':time.time(),'child_pid':proc.pid,'validation_revision':2}
            write(f'{name}.exit.json',exit_record)
            if name=='DRPA':write('validation_v2/DRPA.segment2.exit.json',exit_record)
        if proc.returncode:
            write('queue_state.json',{'status':'FAILED','stage':name,'returncode':proc.returncode,'automatic_retry':False,'validation_revision':2});return proc.returncode
        if name!='REPORT':assert (ROOT/name/'complete.json').exists(),'Missing completion gate'
    write('queue_state.json',{'status':'COMPLETE','end_unix':time.time(),'validation_revision':2});return 0
if __name__=='__main__':sys.exit(main())
