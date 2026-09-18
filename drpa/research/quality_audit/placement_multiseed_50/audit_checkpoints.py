import csv,gc,hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path('__DRPA_WORKSPACE__/quality_audit/placement_multiseed_50')
DEST=ROOT/'final_statistics'
DEST.mkdir(exist_ok=True)

def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from tensors(v)
    elif isinstance(x,(tuple,list)):
        for v in x:yield from tensors(v)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

results=[];provenance=[]
for seed in [20260809,3407,2026]:
 for condition in ['b1','b1_projection','b1_decoder','drpa']:
    p=ROOT/f'seed{seed}_{condition}';ck=p/'checkpoints/step_06000.pt'
    x=torch.load(ck,map_location='cpu',weights_only=False)
    disk_config=json.loads((p/'config.json').read_text())
    assert x['global_step']==6000 and x['condition']==condition
    assert x['config']==disk_config and disk_config['seed']==seed
    model_params=sum(t.numel() for t in x['trainable_model_state'].values())
    assert model_params==disk_config['trainable_parameters']
    ts=list(tensors(x));nonfinite=sum(int(not torch.isfinite(t).all()) for t in ts if t.is_floating_point() or t.is_complex())
    scalar={k:v for k,v in x.items() if isinstance(v,(int,str,float,bool,type(None)))}
    row=dict(run=p.name,checkpoint=str(ck),bytes=ck.stat().st_size,sha256=sha(ck),readable=True,
        tensor_count=len(ts),nonfinite_tensors=nonfinite,model_state_parameters=model_params,checkpoint_config_match=True,metadata=json.dumps(scalar),keys=json.dumps(list(x)),
        active_failure_artifacts=json.dumps([str(f) for f in p.glob('*failure*.json')]),
        post_completion_shutdown_warning=(p/'post_completion_dataloader_shutdown_warning.json').exists())
    assert nonfinite==0
    results.append(row)
    for name in ['validation_rows_step_06000.csv','validation_summary_step_06000.csv','run_summary.json','config.json','training_dynamics.csv','parameter_summary.csv','initialization.json','post_completion_dataloader_shutdown_warning.json']:
        f=p/name
        if f.exists():provenance.append(dict(run=p.name,file=name,source_path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
    print(json.dumps(row),flush=True);del x,ts;gc.collect()
for name,rows in [('CHECKPOINT_INTEGRITY.csv',results),('SOURCE_ARTIFACT_MANIFEST.csv',provenance)]:
 with (DEST/name).open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
