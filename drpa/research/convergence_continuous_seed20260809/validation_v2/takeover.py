"""Gated process transition at a fully committed endpoint; preserve every source artifact."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('__DRPA_WORKSPACE__/convergence_continuous_seed20260809');REV=ROOT/'validation_v2'
sys.path.insert(0,str(ROOT/'code'))
import experiment as e
import state_tools as s
def stopped(pid):return '\nState:\tT' in Path(f'/proc/{pid}/status').read_text()
def terminate_stopped(pid):
    assert stopped(pid)
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(120):
        path=Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2]=='Z':return
        time.sleep(.25)
    raise RuntimeError(f'Original process did not terminate: {pid}')
def main():
    start=time.time()
    assert json.loads((REV/'EQUIVALENCE_GATE.json').read_text())['status']=='PASS'
    gate=json.loads((REV/'RESUME_EQUIVALENCE_GATE.json').read_text());assert gate['status']=='PASS'
    import fast_validation as fast
    assert e.sha(Path(fast.__file__))==json.loads((REV/'EQUIVALENCE_GATE.json').read_text())['optimized_source_sha256']
    build=json.loads((REV/'BUILD_PROVENANCE.json').read_text())
    assert e.sha(ROOT/'code/experiment.py')==build['original_experiment_sha256']
    assert e.sha(ROOT/'code/report.py')==build['original_report_sha256']
    assert e.sha(REV/'experiment_v2.py')==build['versioned_experiment_sha256']
    assert e.sha(REV/'report_v2.py')==build['versioned_report_sha256']
    while not (REV/'TRANSITION_READY.json').exists():
        if time.time()-start>7200:raise TimeoutError('No completed boundary; original queue left unchanged')
        time.sleep(2)
    ready=json.loads((REV/'TRANSITION_READY.json').read_text());step=ready['step'];pid=ready['pid'];sup=ready['supervisor_pid']
    assert step==gate['source_step'],'Exact recovery test belongs to another boundary; old processes remain paused'
    assert stopped(pid) and stopped(sup)
    assert b'/code/experiment.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert b'code/supervise.py' in Path(f'/proc/{sup}/cmdline').read_bytes()
    state=s.verify_completed_endpoint(e,ROOT/'DRPA',step)
    assert e.sha(ROOT/'DRPA/checkpoints'/f'step_{step:05d}.pt')==gate['checkpoint_sha256']
    archive=REV/'transition_state';archive.mkdir(exist_ok=False)
    for path in ['queue_start.json','queue_state.json','health.json','DRPA/progress.json','DRPA/config.json','DRPA/continuity.csv',
                 'DRPA/train_curve_raw.csv','DRPA/val_curve.csv','DRPA/val_roi_records.csv','DRPA/val_ptid_records.csv','DRPA/checkpoint_manifest.csv']:
        src=ROOT/path;dst=archive/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    runtime=archive/f'optimizer_rng_step{step:05d}.pt';shutil.copyfile(ROOT/'DRPA/latest_optimizer_rng.pt',runtime)
    assert e.sha(runtime)==e.sha(ROOT/'DRPA/latest_optimizer_rng.pt')
    audit={'status':'VERIFIED_BOUNDARY','step':step,'old_child_pid':pid,'old_supervisor_pid':sup,
           'original_files_preserved':True,'optimizer_resets':0,'logged_updates_replayed':0,
           'runtime_archive':str(runtime),'runtime_sha256':e.sha(runtime),'equivalence_gate_sha256':e.sha(REV/'EQUIVALENCE_GATE.json'),
           'recovery_gate_sha256':e.sha(REV/'RESUME_EQUIVALENCE_GATE.json'),'unix':time.time()}
    e.atomic_json(REV/'DEPLOYMENT_AUDIT.json',audit)
    terminate_stopped(sup);terminate_stopped(pid)
    subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],check=True)
    with (REV/'supervisor_v2.log').open('x') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(REV/'supervise_v2.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(ROOT))
    audit.update(status='LAUNCHED',new_supervisor_pid=proc.pid,launch_unix=time.time());e.atomic_json(REV/'DEPLOYMENT_AUDIT.json',audit)
    for _ in range(300):
        proof=ROOT/'DRPA/first_resumed_update_verification.json'
        if proof.exists():
            first=json.loads(proof.read_text());assert first['status']=='PASS'
            audit.update(status='ACTIVE_VERIFIED',first_resumed_update=first,new_state=json.loads((ROOT/'queue_state.json').read_text()))
            e.atomic_json(REV/'DEPLOYMENT_AUDIT.json',audit);print(json.dumps(audit,indent=2),flush=True);return
        q=json.loads((ROOT/'queue_state.json').read_text())
        if q.get('status')=='FAILED':raise RuntimeError(f'Versioned runner failed: {q}')
        time.sleep(2)
    raise TimeoutError('New process launched; first-update proof not yet available; inspect logs without retry')
if __name__=='__main__':main()
