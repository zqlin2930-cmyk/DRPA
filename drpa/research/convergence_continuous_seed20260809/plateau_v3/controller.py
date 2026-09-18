"""Apply the same stopping policy to an existing or newly launched runner."""
import json, math, os, signal, time
from pathlib import Path
from policy import POLICY, atomic_json, decision, record_decision, rows, sha

def identity(pid):
    p=Path(f'/proc/{pid}/stat')
    if not p.exists(): return None
    fields=p.read_text().rsplit(')',1)[1].split()
    return fields[0],fields[19]

def alive(pid, token):
    state=identity(pid)
    return state is not None and state[0]!='Z' and state[1]==token

def last_training_step(out):
    path=Path(out)/'train_curve_raw.csv'
    if not path.exists(): return 0
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-4096));text=f.read().decode()
    if not text.endswith('\n'): return None
    line=text.strip().splitlines()[-1]
    return int(line.split(',')[2]) if not line.startswith('model,') else 0

def suspend(pid,token):
    assert alive(pid,token);os.kill(pid,signal.SIGSTOP)
    for _ in range(200):
        s=identity(pid)
        if s is not None and s[0]=='T' and s[1]==token:return
        time.sleep(.005)
    raise RuntimeError('Cannot confirm suspension')

def terminate_suspended(pid,token):
    assert identity(pid)==('T',token)
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    for _ in range(300):
        if not alive(pid,token):return
        time.sleep(.1)
    raise RuntimeError('Runner did not terminate after policy stop')

def verify_endpoint(out,step):
    out=Path(out); expected=list(range(0,step+1,1500))
    train=rows(out/'train_curve_raw.csv'); val=rows(out/'val_curve.csv')
    cp=rows(out/'checkpoint_manifest.csv');cont=rows(out/'continuity.csv')
    assert [int(r['global_step']) for r in train]==list(range(1,step+1))
    assert [int(r['step']) for r in val]==expected
    assert [int(r['global_step']) for r in cp]==expected
    assert [int(r['global_step']) for r in cont]==expected
    assert all(int(r['samples_seen'])==int(r['global_step']) and int(r['nan_inf'])==0 for r in train)
    assert all(math.isfinite(float(r[k])) for r in train for k in ['total_loss','dice_loss','bce_loss','gradient_norm'])
    raw=rows(out/'val_roi_records.csv');ptid=rows(out/'val_ptid_records.csv')
    assert len(raw)==len(expected)*1976 and len(ptid)==len(expected)*85
    assert len({(r['step'],r['case_id'],r['prompt']) for r in raw})==len(raw)
    assert len({(r['step'],r['ptid']) for r in ptid})==len(ptid)
    final=cont[-1]
    for r in cont:
        assert int(r['global_step'])==int(r['optimizer_step_min'])==int(r['optimizer_step_max'])
        assert r['optimizer_state_finite']=='True' and r['rng_restored_after_validation']=='True'
        assert int(r['optimizer_resets'])==int(r['scheduler_resets'])==0
    assert sha(cp[-1]['path'])==cp[-1]['sha256']
    import torch
    state=torch.load(out/'latest_optimizer_rng.pt',map_location='cpu',weights_only=False)
    assert state['global_step']==step and state['scheduler'] is None and state['model_checkpoint']==cp[-1]['path']
    counters=[int(s['step'].item()) for s in state['optimizer']['state'].values() if 'step' in s]
    assert counters and min(counters)==max(counters)==step
    return {**{k:int(final[k]) for k in ['global_step','optimizer_state_entries','optimizer_step_min','optimizer_step_max']},
            'optimizer_state_finite':True,'checkpoint_path':cp[-1]['path'],'checkpoint_sha256':cp[-1]['sha256']}

def stop_at_endpoint(out,pid,token,result):
    step=result['step'];suspend(pid,token)
    try:
        actual=last_training_step(out)
        assert actual==step,f'Stop boundary missed: committed {actual}, expected {step}; no truncation allowed'
        final=verify_endpoint(out,step)
        progress=json.loads((out/'progress.json').read_text())
        elapsed=progress['wall_clock_time_sec']+max(0,time.time()-progress['updated_unix'])
        record={'status':'VERIFIED_BEFORE_POLICY_STOP','pid':pid,'process_start_token':token,'step':step,
                'last_committed_step':actual,'decision':result,'unix':time.time(),'optimizer_resets':0,
                'in_flight_uncommitted_work':'May be discarded; no committed update is deleted or replayed',**final}
        atomic_json(out/'policy_stop_audit.json',record)
    except BaseException:
        if alive(pid,token):os.kill(pid,signal.SIGCONT)
        raise
    terminate_suspended(pid,token)
    resumed=(out/'exact_resume_bridge.json').exists()
    complete={'status':'COMPLETE','final_step':step,'stop_reason':result['reason'],'practical_plateau':result['practical_plateau'],
              'policy_version':POLICY['version'],'optimizer_continuous':True,'scheduler_continuous':'not applicable: absent',
              'optimizer_reset_count':0,'nan_inf_training':0,'wall_clock_time_sec':elapsed,
              'exact_resume_count':int(resumed),'runtime_segments':1+int(resumed),
              'termination':'authorized policy stop at verified complete endpoint','runner_signal':'SIGTERM',**final}
    assert not (out/'complete.json').exists()
    atomic_json(out/'complete.json',complete)
    atomic_json(out/'progress.json',{'status':'COMPLETE','step':step,'pid':pid,'updated_unix':time.time(),
                                   'wall_clock_time_sec':elapsed,'stop_reason':result['reason'],'policy_version':POLICY['version']})
    atomic_json(out/'policy_stop_audit.json',{**record,'status':'POLICY_STOP_COMPLETE','end_unix':time.time()})

def monitor(root,model,pid,supervisor_pid,proc=None):
    root=Path(root);out=root/model;token=identity(pid)[1];processed=set();last_health=0
    while alive(pid,token):
        if proc is not None:proc.poll()
        now=time.time()
        if now-last_health>=15:
            atomic_json(root/'health.json',{'stage':model,'child_pid':pid,'supervisor_pid':supervisor_pid,'process_alive':True,
                'unix':now,'policy_version':POLICY['version'],'timeout_advisory':now-json.loads((root/'queue_revision_v3.json').read_text())['start_unix']>7*86400,
                'policy':'No crash retry; no timeout kill; only prespecified endpoint stops'})
            last_health=now
        validation=rows(out/'val_curve.csv');by_step={int(r['step']):r for r in validation}
        for step in (9000,10500,12000):
            if step in by_step and step not in processed:
                cont=rows(out/'continuity.csv')
                if not cont or int(cont[-1]['global_step'])<step:continue
                result=record_decision(out,by_step[step-1500],by_step[step]);processed.add(step)
                if result['stop'] and step<12000:
                    stop_at_endpoint(out,pid,token,result)
                    if proc is not None:proc.wait(timeout=30)
                    return json.loads((out/'complete.json').read_text())
        step=last_training_step(out)
        time.sleep(.025 if step is not None and step>=8950 else 1)
    if proc is not None:proc.wait(timeout=30)
    assert (out/'complete.json').exists(),f'{model} exited without successful completion; inspect failure_context and log'
    done=json.loads((out/'complete.json').read_text());assert done['final_step']==12000
    val={int(r['step']):r for r in rows(out/'val_curve.csv')}
    for step in (9000,10500,12000):
        result=record_decision(out,val[step-1500],val[step])
        assert step==12000 or not result['stop'],'A required policy stop was missed'
    atomic_json(out/'complete_before_policy_annotation.json',done)
    done.update(policy_version=POLICY['version'],stop_reason=result['reason'],practical_plateau=result['practical_plateau'])
    atomic_json(out/'complete.json',done)
    return done
