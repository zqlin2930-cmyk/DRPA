"""Next-update equivalence after exact model/AdamW/RNG round-trip, disposable only."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import gc,json,sys,time
from pathlib import Path
import torch
ROOT=Path('__DRPA_WORKSPACE__/convergence_continuous_seed20260809')
sys.path.insert(0,str(ROOT/'code'))
import experiment as e
import state_tools as s

def update(wrapper,optimizer,params,image,target,cache,rng):
    wrapper.set_training_mode();e.restore_rng(rng);optimizer.zero_grad(set_to_none=True);losses=[]
    for pi,prompt in enumerate(e.shared.PROMPTS):
        logits=wrapper(image,cache.get([prompt],torch.device('cuda')))
        loss,dice,bce=e.shared.loss_fp32(logits,target[:,[pi]])
        (loss/8).backward();losses.append([float(loss.detach()),float(dice.detach()),float(bce.detach())]);del logits,loss,dice,bce
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step()
    return {'losses':losses,'gradient_norm':norm,'model':wrapper.adapter_state_dict(),
            'optimizer':{k:v for k,v in optimizer.state_dict().items()},'rng':e.rng_state()}

def main():
    dest=ROOT/'validation_v2';e.shared.seed_all();step=3000
    runtime=ROOT/'DRPA/latest_optimizer_rng.pt'
    state=torch.load(runtime,map_location='cpu',weights_only=False)
    assert state['global_step']==step
    model=ROOT/'DRPA/checkpoints'/f'step_{step:05d}.pt'
    torch.save(state,dest/'resume_gate_initial_state.pt')
    train=e.shared.records(e.manifests('DRPA')[0]);frame=e.pd.read_csv(e.manifests('DRPA')[0],dtype=str)
    nextrow=next(x for x in e.order_rows(frame) if x[0]==step+1)
    ds=e.shared.BilateralGroupedPatchDataset([train[nextrow[3]]],e.shared.load_crop_spec(e.shared.PILOT/'crop_spec.json'),cache_cases=False)
    item=ds[0];image=item['image'].unsqueeze(0).cuda();target=item['mask'].unsqueeze(0).cuda()
    cache=e.shared.TextEmbeddingCache(e.shared.BANK,e.shared.MODEL,e.shared.CACHE)
    results=[]
    for branch in ['direct_restoration','save_reload_restoration']:
        wrapper,groups,params=e.make_model('DRPA');optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
        source=state if branch=='direct_restoration' else torch.load(dest/'resume_gate_initial_state.pt',map_location='cpu',weights_only=False)
        check=s.restore_state(e,wrapper,optimizer,model,source,step)
        result=update(wrapper,optimizer,params,image,target,cache,source['rng'])
        assert e.optimizer_audit(optimizer,step+1)['optimizer_step_min']==step+1
        torch.save(result,dest/f'resume_gate_{branch}.pt');results.append({'branch':branch,'losses':result['losses'],'gradient_norm':result['gradient_norm'],'restore':check})
        del result,wrapper,optimizer,groups,params,source;gc.collect();torch.cuda.empty_cache()
    a=torch.load(dest/'resume_gate_direct_restoration.pt',map_location='cpu',weights_only=False)
    b=torch.load(dest/'resume_gate_save_reload_restoration.pt',map_location='cpu',weights_only=False)
    assert s.tree_equal(a,b),'Next update differs after state round-trip'
    report={'status':'PASS','source_step':step,'next_step':step+1,'next_case_id':nextrow[4],
            'exact_next_update_equal':True,'compared':['all8prompt losses','gradient norm','all trainable weights','all AdamW states','all RNG streams'],
            'checkpoint_sha256':e.sha(model),'runtime_snapshot_sha256':e.sha(dest/'resume_gate_initial_state.pt'),
            'branches':results,'scope':'Disposable engineering test; no formal optimizer update or output CSV was modified'}
    e.atomic_json(dest/'RESUME_EQUIVALENCE_GATE.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
