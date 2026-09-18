"""Adopt the live DRPA process without restarting it; manage validation-based stops."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from policy import POLICY,atomic_json,sha
from controller import identity,monitor,alive

ROOT=Path(__file__).resolve().parents[1];REV=ROOT/'plateau_v3'

def main():
    lock=(ROOT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'queue_revision_v3.json').exists(),'No automatic restart'
    adoption=json.loads((REV/'ADOPTION_REQUEST.json').read_text());pid=adoption['child_pid']
    assert alive(pid,adoption['child_process_start_token'])
    paths=[*list((ROOT/'code').glob('*.py')),*list((ROOT/'validation_v2').glob('*.py')),*list(REV.glob('*.py'))]
    hashes={str(p):sha(p) for p in paths}
    atomic_json(ROOT/'queue_revision_v3.json',{'pid':os.getpid(),'start_unix':time.time(),'code_sha256':hashes,
         'policy':POLICY,'order':['DRPA','PDFT','FullFT'],'automatic_retry':False,'adopted_child_pid':pid})
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(ROOT/'matplotlib_cache'))
    for model in ['DRPA','PDFT','FullFT']:
        assert all(sha(p)==h for p,h in hashes.items()),'Frozen code changed'
        out=ROOT/model;out.mkdir(exist_ok=True)
        atomic_json(out/'effective_protocol_v3.json',{'policy':POLICY,'original_training_config':'config.json',
             'overrides':['required_steps','scientific_scope: validation-based practical-plateau stopping'],
             'mandatory_endpoints':POLICY['required_endpoints'],'maximum_updates':12000,
             'validation_implementation':'raw_shared_edt_threads4_v2','optimizer_changes':False,
             'protocol_amendment':'../plateau_v3/PROTOCOL_AMENDMENT.md','unix':time.time()})
        start=time.time();proc=None;log=None
        if model!='DRPA':
            log=(ROOT/f'{model}.log').open('x')
            proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'validation_v2/experiment_v2.py'),'--model',model],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT));pid=proc.pid
        atomic_json(ROOT/'queue_state.json',{'status':'RUNNING','stage':model,'child_pid':pid,'supervisor_pid':os.getpid(),
             'start_unix':start,'validation_revision':2,'stopping_revision':3,'adopted_existing_process':model=='DRPA'})
        if model=='DRPA':
            atomic_json(REV/'ADOPTED_ACTIVE.json',{'status':'ACTIVE','child_pid':pid,'child_process_start_token':identity(pid)[1],
                 'new_supervisor_pid':os.getpid(),'training_process_restarted':False,'optimizer_resets':0,'unix':time.time(),
                 'progress_at_adoption':json.loads((out/'progress.json').read_text()),'policy':POLICY})
        done=monitor(ROOT,model,pid,os.getpid(),proc)
        if log is not None:log.close()
        atomic_json(ROOT/f'{model}.exit.json',{'status':'COMPLETE','runner_returncode':None if proc is None else proc.returncode,
            'authorized_policy_stop':done['final_step']<12000,'start_unix':start,'end_unix':time.time(),
            'final_step':done['final_step'],'stop_reason':done['stop_reason'],'stopping_revision':3})
    assert all(sha(p)==h for p,h in hashes.items()),'Frozen code changed before report'
    atomic_json(ROOT/'queue_state.json',{'status':'RUNNING','stage':'REPORT','stopping_revision':3,'supervisor_pid':os.getpid()})
    with (ROOT/'REPORT.log').open('x') as log:
        result=subprocess.run([sys.executable,'-u',str(REV/'report_v3.py'),'--out',str(ROOT)],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
    assert result.returncode==0,'Final report failed; no automatic retry'
    atomic_json(ROOT/'queue_state.json',{'status':'COMPLETE','end_unix':time.time(),'stopping_revision':3})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(REV/'SUPERVISOR_FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc(),'unix':time.time()})
        atomic_json(ROOT/'queue_state.json',{'status':'FAILED','stopping_revision':3,'error':repr(exc),'automatic_retry':False})
        raise
