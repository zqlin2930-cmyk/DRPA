"""Replace only the queue supervisor; retain the live training process and PID."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json,os,signal,subprocess,sys,time
from pathlib import Path
from policy import POLICY,atomic_json,sha
from controller import identity,alive,suspend,terminate_suspended
ROOT=Path(__file__).resolve().parents[1];REV=ROOT/'plateau_v3'

def main():
    assert not (REV/'ADOPTION_REQUEST.json').exists()
    assert json.loads((REV/'TEST_RESULTS.json').read_text())['status']=='PASS'
    gate=json.loads((REV/'TEST_RESULTS.json').read_text())
    assert all(sha(REV/p)==h for p,h in gate['tested_source_hashes'].items())
    q=json.loads((ROOT/'queue_state.json').read_text());assert q['status']=='RUNNING' and q['stage']=='DRPA'
    pid=q['child_pid'];sup=q['supervisor_pid'];ct=identity(pid)[1];st=identity(sup)[1]
    assert b'validation_v2/experiment_v2.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'validation_v2/supervise_v2.py' in Path(f'/proc/{sup}/cmdline').read_bytes()
    progress=json.loads((ROOT/'DRPA/progress.json').read_text());assert progress['step']<8500
    old=json.loads((ROOT/'queue_revision_v2.json').read_text())
    assert all(sha(p)==h for p,h in old['code_sha256'].items())
    archive=REV/'prior_supervision';archive.mkdir()
    for p in ['queue_state.json','health.json','queue_revision_v2.json']:(archive/p).write_bytes((ROOT/p).read_bytes())
    atomic_json(REV/'ADOPTION_REQUEST.json',{'child_pid':pid,'child_process_start_token':ct,'old_supervisor_pid':sup,
         'old_supervisor_start_token':st,'progress_before':progress,'policy':POLICY,'unix':time.time(),
         'amendment_before_first_decision_endpoint':True})
    suspend(sup,st)
    assert alive(pid,ct),'Training stopped unexpectedly before supervisor change'
    terminate_suspended(sup,st)
    with (REV/'supervisor_v3.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(REV/'supervise_v3.py')],stdin=subprocess.DEVNULL,stdout=log,
              stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT),env=os.environ.copy())
    for _ in range(100):
        if (REV/'ADOPTED_ACTIVE.json').exists():
            proof=json.loads((REV/'ADOPTED_ACTIVE.json').read_text())
            assert proof['child_pid']==pid and proof['child_process_start_token']==ct and alive(pid,ct)
            atomic_json(REV/'DEPLOYMENT_AUDIT.json',{'status':'ACTIVE_VERIFIED','new_supervisor_pid':p.pid,
              'child_pid_unchanged':pid,'training_process_restarted':False,'optimizer_resets':0,
              'progress_before':progress,'progress_after':json.loads((ROOT/'DRPA/progress.json').read_text()),
              'policy':POLICY,'unix':time.time(),'tested_sources':gate['tested_source_hashes']})
            print((REV/'DEPLOYMENT_AUDIT.json').read_text());return
        if p.poll() is not None:raise RuntimeError('New supervisor failed; live child retained, inspect supervisor_v3.log')
        time.sleep(.1)
    raise TimeoutError('Adoption confirmation unavailable; inspect state without relaunching')
if __name__=='__main__':main()
