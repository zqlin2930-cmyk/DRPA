"""Audited exact checkpoint continuation; never reset optimizer moments or RNG."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b

def checkpoint_paths(out,step):return out/'checkpoints'/f'step_{step:05d}.pt',out/'latest_optimizer_rng.pt'

def verify_completed_endpoint(e,out,step):
    train=pd.read_csv(out/'train_curve_raw.csv');val=pd.read_csv(out/'val_curve.csv');cp=pd.read_csv(out/'checkpoint_manifest.csv')
    assert train.global_step.tolist()==list(range(1,step+1)),'Cannot rewind or skip any logged update'
    assert val.step.tolist()==[s for s in e.STEPS if s<=step]
    assert cp.global_step.tolist()==[s for s in e.STEPS if s<=step]
    raw=pd.read_csv(out/'val_roi_records.csv');ptid=pd.read_csv(out/'val_ptid_records.csv')
    assert len(raw[raw.step==step])==1976 and len(ptid[ptid.step==step])==85
    assert not raw.duplicated(['step','case_id','prompt']).any()
    assert not ptid.duplicated(['step','ptid']).any()
    model,runtime=checkpoint_paths(out,step)
    assert e.sha(model)==cp.loc[cp.global_step==step,'sha256'].iloc[0]
    state=torch.load(runtime,map_location='cpu',weights_only=False)
    assert state['global_step']==step and Path(state['model_checkpoint'])==model
    assert state['scheduler'] is None
    return state

def restore_state(e,wrapper,optimizer,model_path,state,step):
    payload=torch.load(model_path,map_location='cpu',weights_only=False)
    if 'adapter_state_dict' in payload:incoming=payload['adapter_state_dict']
    elif 'trainable_state_dict' in payload:incoming=payload['trainable_state_dict']
    else:
        wrapper.model.load_state_dict(payload['model_state'],strict=True);incoming=None
    if incoming is not None:
        expected={n:p for n,p in wrapper.model.named_parameters() if p.requires_grad}
        assert expected.keys()==incoming.keys(),'Trainable parameter identity mismatch'
        with torch.no_grad():
            for n,p in expected.items():p.copy_(incoming[n].to(device=p.device,dtype=p.dtype))
        assert all(torch.equal(p.detach().cpu(),incoming[n]) for n,p in expected.items())
        # Canonical adapter checkpoints omit buffers. Refuse if any module could update running statistics.
        assert not any(getattr(mod,'track_running_stats',False) for mod in wrapper.model.modules()),'Mutable normalization buffers require full state'
    saved=state['optimizer']
    assert len(saved['param_groups'])==len(optimizer.param_groups)
    for old,new in zip(saved['param_groups'],optimizer.param_groups):
        for key in ['name','lr','weight_decay','betas','eps']:
            assert old[key]==new[key],f'Optimizer protocol mismatch: {key}'
        assert len(old['params'])==len(new['params'])
    optimizer.load_state_dict(saved)
    assert tree_equal(optimizer.state_dict(),saved),'Restored optimizer is not identical'
    assert e.optimizer_audit(optimizer,step)['optimizer_step_min']==step
    e.restore_rng(state['rng']);assert tree_equal(e.rng_state(),state['rng'])
    return {'model_parameters_equal':True,'optimizer_state_equal':True,'rng_equal':True,
            'global_step':step,'optimizer_resets':0,'scheduler_resets':0,
            'buffer_names':[n for n,_ in wrapper.model.named_buffers()]}

def initialize_resume(e,wrapper,optimizer,out,step,config):
    gate=json.loads((e.OUT/'validation_v2/RESUME_EQUIVALENCE_GATE.json').read_text())
    assert gate['status']=='PASS' and gate['exact_next_update_equal']
    ready=json.loads((e.OUT/'validation_v2/TRANSITION_READY.json').read_text())
    assert ready['step']==step
    state=verify_completed_endpoint(e,out,step)
    previous=json.loads((out/'config.json').read_text())
    assert config==previous,'Resume training configuration changed'
    model,runtime=checkpoint_paths(out,step)
    assert gate['source_step']==step and gate['checkpoint_sha256']==e.sha(model),'Recovery gate must match the exact source checkpoint'
    restored=restore_state(e,wrapper,optimizer,model,state,step)
    prior=json.loads((out/'progress.json').read_text())
    origin=prior['updated_unix']-prior['wall_clock_time_sec']
    elapsed=max(float(pd.read_csv(out/'train_curve_raw.csv').wall_clock_time_sec.iloc[-1]),time.time()-origin)
    continuity=pd.read_csv(out/'continuity.csv')
    previous_id=int(continuity.iloc[-1].optimizer_id);previous_model_id=int(continuity.iloc[-1].model_id)
    if step not in continuity.global_step.to_list():
        e.append_csv(out/'continuity.csv',[{**e.optimizer_audit(optimizer,step),'optimizer_id':previous_id,'model_id':previous_model_id,
                     'optimizer_resets':0,'scheduler_resets':0,'rng_restored_after_validation':True}])
    bridge={**restored,'status':'EXACT_STATE_RESTORED','model':config['model'],'resume_step':step,
            'old_pid':ready['pid'],'new_pid':os.getpid(),'old_optimizer_id':previous_id,'new_optimizer_id':id(optimizer),
            'old_model_id':previous_model_id,'new_model_id':id(wrapper.model),'checkpoint_path':str(model),
            'checkpoint_sha256':e.sha(model),'optimizer_rng_sha256':e.sha(runtime),
            'next_committed_update':step+1,'no_logged_steps_replayed':True,'wall_clock_origin_unix':origin,
            'restore_unix':time.time(),'equivalence_gate_sha256':e.sha(e.OUT/'validation_v2/RESUME_EQUIVALENCE_GATE.json')}
    assert not (out/'exact_resume_bridge.json').exists(),'Duplicate process transition refused'
    e.atomic_json(out/'exact_resume_bridge.json',bridge)
    return state,elapsed

def verify_first_update(e,wrapper,optimizer,losses,norm,case,out,step):
    dest=e.OUT/'validation_v2'
    gate=json.loads((dest/'RESUME_EQUIVALENCE_GATE.json').read_text())
    assert gate['next_step']==step and gate['next_case_id']==case
    expected=torch.load(dest/'resume_gate_direct_restoration.pt',map_location='cpu',weights_only=False)
    actual={'losses':losses,'gradient_norm':norm,'model':wrapper.adapter_state_dict(),
            'optimizer':optimizer.state_dict(),'rng':e.rng_state()}
    assert tree_equal(actual,expected),'First resumed optimizer update diverged from exact-recovery reference'
    e.atomic_json(out/'first_resumed_update_verification.json',{'status':'PASS','step':step,'case_id':case,
        'all_losses_weights_optimizer_rng_exact':True,'reference_sha256':e.sha(dest/'resume_gate_direct_restoration.pt')})
