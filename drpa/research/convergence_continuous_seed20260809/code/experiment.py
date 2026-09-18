"""Fresh, isolated, continuous convergence trajectories; never imports old scores."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse, csv, hashlib, importlib.util, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

BASE = Path('__DRPA_WORKSPACE__')
OUT = BASE / 'convergence_continuous_seed20260809'
SEED = 20260809
STEPS = list(range(0,12001,1500))
COUNTS = {'DRPA':10969696,'PDFT':81930624,'FullFT':440029541}
DISPLAY = {'DRPA':'DRPA-8','PDFT':'PD-FT','FullFT':'VoxTell-FullFT'}
sys.path[:0] = [str(BASE/'VoxTell'), str(BASE), *[str(p) for p in (BASE/'quality_audit').iterdir() if p.is_dir()]]
import drpa8_full_data_train as shared
from drpa8_wrapper import DRPA8Wrapper
from b3_full_data_wrapper import B3FullDataWrapper

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

formal=module('original_fullft',BASE/'scripts/fullft/train_fullft_10pct_formal.py')
gate=formal.load_gate(BASE)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    path=Path(path); tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str)+'\n'); os.replace(tmp,path)

def append_csv(path,rows):
    if not rows: return
    path=Path(path); exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        if not exists:w.writeheader()
        w.writerows(rows); f.flush(); os.fsync(f.fileno())

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])

def manifests(model):
    root=BASE/'quality_audit'
    if model=='FullFT':return root/'drpa_data_capacity_scaling/manifests/train_100pct.csv',root/'drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv'
    return root/'voxtell_mtl_drpa8_full_data/full_data_train_cases.csv',root/'voxtell_mtl_drpa8_full_data/full_data_val_cases.csv'

def order_rows(frame,steps=12000):
    for step in range(1,steps+1):
        epoch=(step-1)//len(frame)+1; offset=(step-1)%len(frame)
        if offset==0:order=np.random.default_rng(SEED+epoch).permutation(len(frame))
        index=int(order[offset]); yield step,epoch,offset,index,str(frame.iloc[index].case_id)

def audit_split(train,val):
    assert len(train)==971 and train.ptid.nunique()==337
    assert len(val)==247 and val.ptid.nunique()==85
    assert not set(train.ptid)&set(val.ptid)
    for f in (train,val):
        assert not f.case_id.duplicated().any()
        assert (f.case_id.str.split('_').str[0]==f.ptid).all()
        for col in ('image_path','label_path'):
            missing=[p for p in f[col] if not Path(p).is_file()]
            if missing:raise FileNotFoundError(f'{len(missing)} missing {col}: {missing[:2]}')

def preflight():
    OUT.mkdir(exist_ok=True)
    for d in ('source_snapshot','summary','figures'): (OUT/d).mkdir(exist_ok=True)
    frames={}
    for m in COUNTS:
        t,v=manifests(m); a,b=pd.read_csv(t,dtype=str),pd.read_csv(v,dtype=str);audit_split(a,b)
        frames[m]=(a,b)
    cols=['case_id','ptid','image_path','label_path']
    for m in COUNTS:
        assert frames[m][0][cols].equals(frames['DRPA'][0][cols]),'Training order/input mismatch'
        assert frames[m][1][cols].equals(frames['DRPA'][1][cols]),'Validation order/input mismatch'
    source_paths={Path(shared.__file__),Path(shared.evaluator.__file__),Path(formal.__file__),Path(gate.__file__)}
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and str(p).startswith(str(BASE)) and not Path(p).is_relative_to(OUT) and Path(p).suffix=='.py':source_paths.add(Path(p))
    source_paths.update([BASE/'quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_fp32_formal_runner.py',
                         BASE/'quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json',
                         BASE/'VoxTell_weights/voxtell_v1.1/plans.json'])
    hashes={}
    for p in sorted(source_paths):
        hashes[str(p)]=sha(p)
        dest=OUT/'source_snapshot'/p.relative_to(BASE); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    checkpoint=BASE/'VoxTell_weights/voxtell_v1.1/fold_0/checkpoint_final.pth'
    hashes[str(checkpoint)]=sha(checkpoint)
    bank=Path(shared.BANK); hashes[str(bank)]=sha(bank)
    cache=shared.TextEmbeddingCache(shared.BANK,shared.MODEL,shared.CACHE)
    official_cache=shared.TextEmbeddingCache(shared.BANK,shared.MODEL,None)
    for p in shared.PROMPTS:
        assert cache.contains(p) and official_cache.contains(p),'Missing original prompt embedding'
        assert torch.equal(cache.get([p],torch.device('cpu')),official_cache.get([p],torch.device('cpu'))),'Embedding override differs across models'
    if Path(shared.CACHE).exists():hashes[shared.CACHE]=sha(shared.CACHE)
    for m in COUNTS:
        for p in manifests(m):hashes[str(p)]=sha(p)
    train,val=frames['DRPA']; records=shared.records(manifests('DRPA')[0])
    spec=shared.load_crop_spec(shared.PILOT/'crop_spec.json')
    ds=shared.BilateralGroupedPatchDataset(records[:1],spec,cache_cases=True)
    first=ds[0]; second=ds[0]
    assert torch.equal(first['image'],second['image']) and torch.equal(first['mask'],second['mask'])
    cached=sum(x.nbytes for x in ds._cache[records[0].case_id][:3])
    projected=cached*(971+247)
    from drpa.runtime import memory_limit_bytes
    limit=memory_limit_bytes()
    assert projected+16*1024**3<limit,'Original cache estimate exceeds cgroup; do not start'
    assert shutil.disk_usage(OUT).free>30*1024**3,'Insufficient disk for all endpoints'
    data={'status':'PREFLIGHT_PASS','seed':SEED,'training_ptids':337,'training_visits':971,'validation_ptids':85,
          'validation_visits':247,'ptid_overlap':0,'initialization':str(checkpoint),'initialization_sha256':hashes[str(checkpoint)],
          'source_hashes':hashes,'prompts':list(shared.PROMPTS),'evaluator_groups':shared.evaluator.GROUPS,
          'cache_projected_bytes':projected,'cgroup_memory_limit_bytes':limit,'cache_hit_exact_equivalence':True,
          'torch':torch.__version__,'cuda':torch.version.cuda,'required_steps':STEPS,
          'checkpoint_policy':'All endpoints retain model weights; one latest optimizer/RNG sidecar per model; no runtime resume in this runner',
          'timestamp':time.time()}
    atomic_json(OUT/'preflight.json',data)
    pd.DataFrame(order_rows(train),columns=['global_step','epoch','epoch_offset','dataset_index','case_id']).to_csv(OUT/'training_order.csv',index=False)
    print(json.dumps({k:v for k,v in data.items() if k!='source_hashes'},indent=2),flush=True)

def make_model(m):
    if m=='FullFT':
        model=gate.initialise_official_model(BASE).to('cuda'); wrapper=formal.FullFTWrapper(model)
        params=[p for _,p in gate.unique_named_parameters(model)]
        groups=[{'params':params,'lr':1e-5,'name':'segmentation_network'}]
    else:
        cls=DRPA8Wrapper if m=='DRPA' else B3FullDataWrapper
        wrapper=cls(shared.MODEL,shared.BANK,device=torch.device('cuda'));model=wrapper.model
        grouped={k:[] for k in ('cross_attention_lora','projection_adapter','decoder_stages')}
        for _,p,g in wrapper.trainable_parameter_groups():grouped[g].append(p)
        expected=[294912,442624 if m=='DRPA' else 71403552,10232160]
        assert [sum(p.numel() for p in v) for v in grouped.values()]==expected
        rates=[1e-4,1e-4 if m=='DRPA' else 1e-5,1e-5]
        groups=[{'params':ps,'lr':lr,'name':k} for (k,ps),lr in zip(grouped.items(),rates)]
        params=[p for g in groups for p in g['params']]
    assert len({id(p) for p in params})==len(params)
    assert sum(p.numel() for p in params)==COUNTS[m]
    assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
    return wrapper,groups,params

def optimizer_audit(optimizer,step):
    counters=[int(s['step'].item()) for s in optimizer.state.values() if 'step'in s]
    assert (not counters and step==0) or (counters and min(counters)==max(counters)==step),f'Optimizer discontinuity {step}'
    finite=all(bool(torch.isfinite(t).all().item()) for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t))
    if not finite:raise FloatingPointError('Nonfinite optimizer state')
    return {'global_step':step,'optimizer_state_entries':len(counters),'optimizer_step_min':min(counters,default=0),
            'optimizer_step_max':max(counters,default=0),'optimizer_state_finite':finite}

def save_endpoint(m,wrapper,optimizer,step,config,out):
    assert shutil.disk_usage(out).free>8*1024**3,'Disk safety threshold reached'
    path=out/'checkpoints'/f'step_{step:05d}.pt'
    assert not path.exists(),'Endpoint overwrite refused'
    metadata={'model':DISPLAY[m],'global_step':step,'seed':SEED,'config':config,'optimizer_audit':optimizer_audit(optimizer,step)}
    tmp=path.with_suffix('.tmp')
    if m=='FullFT':torch.save({'model_state':wrapper.model.state_dict(),**metadata},tmp)
    else:wrapper.save_checkpoint(str(tmp),metadata)
    os.replace(tmp,path)
    runtime=out/'latest_optimizer_rng.pt'; tmpr=runtime.with_suffix('.tmp')
    torch.save({'optimizer':optimizer.state_dict(),'rng':rng_state(),'scheduler':None,'global_step':step,
                'model_checkpoint':str(path),'config':config},tmpr);os.replace(tmpr,runtime)
    row={**metadata['optimizer_audit'],'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    append_csv(out/'checkpoint_manifest.csv',[row]);return row

def aggregate_validation(raw,m,step):
    assert len(raw)==247*8 and raw.case_id.nunique()==247 and raw.ptid.nunique()==85
    assert not raw.duplicated(['case_id','prompt']).any()
    result={'model':DISPLAY[m],'seed':SEED,'step':step,'samples_seen':step,
            'mean_dice':raw.dice.mean(),'hd95_mm':raw.hd95_mm.mean(),
            'surface_dice_2mm':raw.surface_dice_2mm.mean(),
            'fp_ml':raw.false_positive_volume_ml.mean(),'fn_ml':raw.false_negative_volume_ml.mean(),
            'hd95_undefined_roi_count':int(raw.hd95_mm.isna().sum())}
    for short,structure in [('hipp','hippocampus'),('ec','entorhinal cortex'),('phg','parahippocampal gyrus'),('amy','amygdala')]:
        result[short+'_dice']=raw.loc[raw.structure==structure,'dice'].mean()
    metrics=['dice','hd95_mm','surface_dice_2mm','false_positive_volume_ml','false_negative_volume_ml']
    ptid=raw.groupby('ptid')[metrics].mean().reset_index()
    ptid['model']=DISPLAY[m];ptid['seed']=SEED;ptid['step']=step
    ptid['n_visits']=ptid.ptid.map(raw.groupby('ptid').case_id.nunique())
    ptid['hd95_undefined_roi_count']=ptid.ptid.map(raw.groupby('ptid').hd95_mm.apply(lambda x:int(x.isna().sum())))
    for short,structure in [('hipp','hippocampus'),('ec','entorhinal cortex'),('phg','parahippocampal gyrus'),('amy','amygdala')]:
        ptid[short+'_dice']=ptid.ptid.map(raw[raw.structure==structure].groupby('ptid').dice.mean())
    return result,ptid

def train(m,precision):
    out=OUT/m;out.mkdir(exist_ok=True);(out/'checkpoints').mkdir(exist_ok=True)
    assert not (out/'train_curve_raw.csv').exists() and not list((out/'checkpoints').iterdir()),'Fresh trajectory only; refusing restart'
    pre=json.loads((OUT/'preflight.json').read_text())
    for p,h in pre['source_hashes'].items():assert sha(p)==h,f'Frozen source changed: {p}'
    formal.refuse_busy_gpu();shared.seed_all()
    assert precision=='fp32','This protocol implements confirmed strict FP32 only'
    config={'model':DISPLAY[m],'seed':SEED,'batch_size':1,'max_updates':12000,'required_steps':STEPS,
            'train_ptids':337,'train_visits':971,'val_ptids':85,'val_visits':247,'overlap':0,
            'precision':'FP32_no_training_autocast_no_gradscaler','activation_checkpointing':m!='FullFT',
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'backend_precision_policy':'Preserve original formal-runner backend defaults; no TF32 override was present in those training sources',
            'optimizer':'AdamW','weight_decay':1e-5,'betas':[.9,.999],'eps':1e-8,'gradient_clip':1.0,'scheduler':None,
            'initialization':pre['initialization'],'initialization_sha256':pre['initialization_sha256'],
            'trainable_parameters':COUNTS[m],'data_cache_cases':m!='FullFT','num_workers':0,
            'data_order':'np.random.default_rng(20260809 + epoch).permutation(971), epoch starts at1',
            'evaluator':'original grouped FP16 inference; FP32 sigmoid; >=0.5; LCC0; physical spacing',
            'aggregation':'canonical visit-by-ROI macro mean; undefined HD95 retained and counted; pandas skip-NaN matches original',
            'scientific_scope':'single-seed convergence sensitivity, no selection/early stop'}
    wrapper,groups,params=make_model(m)
    config['learning_rates']={g['name']:g['lr'] for g in groups}
    atomic_json(out/'config.json',config)
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
    optimizer_id=id(optimizer); model_id=id(wrapper.model)
    cache=shared.TextEmbeddingCache(shared.BANK,shared.MODEL,shared.CACHE if m!='FullFT' else None)
    original_infer=shared.evaluator.infer
    def checked_infer(*args,**kwargs):
        probabilities=original_infer(*args,**kwargs)
        if not all(np.isfinite(p).all() for p in probabilities.values()):
            raise FloatingPointError('Nonfinite validation probabilities')
        return probabilities
    shared.evaluator.infer=checked_infer
    t,v=manifests(m);frame=pd.read_csv(t,dtype=str);train_records=shared.records(t);val_records=shared.records(v)
    spec=shared.load_crop_spec(shared.PILOT/'crop_spec.json')
    train_ds=shared.BilateralGroupedPatchDataset(train_records,spec,cache_cases=m!='FullFT')
    val_ds=shared.BilateralGroupedPatchDataset(val_records,spec,cache_cases=m!='FullFT')
    start=time.perf_counter();step=0
    def progress(status,**more):
        atomic_json(out/'progress.json',{'status':status,'step':step,'pid':os.getpid(),'updated_unix':time.time(),
                    'wall_clock_time_sec':time.perf_counter()-start,**more})
    def training_mode():
        if m=='FullFT':wrapper.model.train()
        else:wrapper.set_training_mode()
    def endpoint():
        assert id(optimizer)==optimizer_id and id(wrapper.model)==model_id
        rng=rng_state();modes=[(m,m.training) for m in wrapper.model.modules()]
        checkpoint=save_endpoint(m,wrapper,optimizer,step,config,out)
        progress('VALIDATING',checkpoint=checkpoint)
        rows=formal.evaluate_fullft(f'{DISPLAY[m]}@{step:05d}',wrapper,val_records,val_ds,cache,shared.evaluator)
        rows.to_csv(out/f'validation_all_branches_step_{step:05d}.csv',index=False)
        raw=rows.loc[rows.lcc==0].copy()
        assert set(raw.case_id)=={r.case_id for r in val_records}
        result,ptid=aggregate_validation(raw,m,step)
        raw['model']=DISPLAY[m];raw['seed']=SEED;raw['step']=step
        append_csv(out/'val_roi_records.csv',raw.to_dict('records'))
        append_csv(out/'val_ptid_records.csv',ptid.to_dict('records'))
        append_csv(out/'val_curve.csv',[result])
        restore_rng(rng)
        for mod,mode in modes:mod.training=mode
        assert torch.equal(torch.get_rng_state(),rng['torch'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda']))
        append_csv(out/'continuity.csv',[{**optimizer_audit(optimizer,step),'optimizer_id':optimizer_id,'model_id':model_id,
                    'optimizer_resets':0,'scheduler_resets':0,'rng_restored_after_validation':True}])
        progress('ENDPOINT_COMPLETE',metrics=result);print(json.dumps(result),flush=True)
    try:
        training_mode();endpoint()
        for expected_step,epoch,offset,index,case in order_rows(frame):
            training_mode();tick=time.perf_counter()
            item=train_ds[index];assert item['case_id']==case
            image=item['image'].unsqueeze(0).to('cuda',dtype=torch.float32)
            target=item['mask'].unsqueeze(0).to('cuda',dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True);losses=[]
            for pi,prompt in enumerate(shared.PROMPTS):
                embedding=cache.get([prompt],torch.device('cuda'))
                logits=wrapper(image,embedding)
                total,dice,bce=shared.loss_fp32(logits,target[:,[pi]])
                if not bool(torch.isfinite(total)):raise FloatingPointError(f'Nonfinite loss before update{expected_step}')
                (total/len(shared.PROMPTS)).backward()
                losses.append([float(total.detach()),float(dice.detach()),float(bce.detach())])
                del logits,total,dice,bce,embedding
            if m!='FullFT' and any(p.grad is None for p in params):raise RuntimeError('Unexpected inactive trainable parameter')
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True))
            optimizer.step();step=expected_step
            if not all(bool(torch.isfinite(p).all()) for p in params):raise FloatingPointError(f'Nonfinite parameters after{step}')
            torch.cuda.synchronize();loss=np.mean(losses,axis=0)
            row={'model':DISPLAY[m],'seed':SEED,'global_step':step,'samples_seen':step,'epoch_equivalent':step/971,
                 'data_pass_index':epoch,'case_id':case,'total_loss':loss[0],'dice_loss':loss[1],'bce_loss':loss[2],
                 'learning_rate':json.dumps(config['learning_rates'],sort_keys=True),'gradient_norm':norm,
                 'wall_clock_time_sec':time.perf_counter()-start,'update_wall_time_sec':time.perf_counter()-tick,
                 'optimizer_state_entries':len(optimizer.state),'nan_inf':0}
            append_csv(out/'train_curve_raw.csv',[row]);progress('TRAINING',last_loss=loss[0],epoch=epoch)
            del image,target,item
            if step%25==0:print(json.dumps({'model':m,'step':step,'loss':loss[0],'seconds':time.perf_counter()-start}),flush=True)
            if step in STEPS:endpoint()
        final=optimizer_audit(optimizer,step)
        assert step==12000
        atomic_json(out/'complete.json',{'status':'COMPLETE','final_step':step,'optimizer_continuous':True,
            'scheduler_continuous':'not applicable: absent','optimizer_reset_count':0,'nan_inf_training':0,
            'wall_clock_time_sec':time.perf_counter()-start,**final})
        progress('COMPLETE')
    except BaseException as e:
        atomic_json(out/'failure_context.json',{'step':step,'error':repr(e),'traceback':traceback.format_exc(),'unix':time.time()})
        progress('FAILED');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--model',choices=list(COUNTS));p.add_argument('--precision',default='fp32')
    a=p.parse_args()
    if a.preflight:preflight()
    elif a.model:train(a.model,a.precision)
    else:p.error('Choose --preflight or --model')
