#!/usr/bin/env python3
"""One authorized FullFT500→query15 task; frozen source imports, isolated outputs."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fullft_addendum_contract import FULLFT_SHA256, prepare as parent_contract
from dependency_inventory import sha, expected_paths

BASE=Path('__DRPA_WORKSPACE__')
OUT=BASE/'quality_audit/oasis_fullft_fewshot_addendum_20260912'
PREP=OUT/'preparation'
PARENT=PREP/'parent_completed'
SOURCE=BASE/'scripts/evaluate/run_oasis_external_frozen_test.py'
GATE=BASE/'scripts/fullft/fullft_runtime_gate.py'
CHECKPOINT=BASE/'quality_audit/voxtell_fullft_baseline/formal_100pct_retry_20260905/checkpoints/voxtell_fullft_100pct_step06000.pt'
TRAINER=BASE/'quality_audit/voxtell_mtl_drpa8_full_data/drpa8_full_data_train.py'
CROP=BASE/'quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json'
TARGET=OUT/'FullFT'
COUNT=440029541
LOSS_PATH=339291552
SEED=20260809


def atomic_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_csv(path,frame):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    frame.to_csv(tmp,index=False); os.replace(tmp,path)


def safe_disk():
    free=shutil.disk_usage(OUT).free/2**30
    if free<20: raise RuntimeError(f'DISK_BELOW20GiB:{free:.3f}')
    return free


def verify(hashes):
    for p,h in hashes.items():
        if not Path(p).is_file() or sha(p)!=h: raise RuntimeError(f'SOURCE_HASH_MISMATCH:{p}')


def imported_source():
    spec=importlib.util.spec_from_file_location('fullft_frozen_source',SOURCE)
    old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    # Same immutable bytes, alternate canonical storage location on authorized host.
    old.CHECKPOINTS['FullFT']=CHECKPOINT
    return old,old.imports()


def loss_function():
    text=TRAINER.read_text(); tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='loss_fp32')
    context={'torch':torch,'F':F}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(TRAINER),'exec'),context)
    return context['loss_fp32'],ast.get_source_segment(text,node)


def cache_and_dataset(old,mod,manifest):
    cache=mod.TextEmbeddingCache(str(old.BANK),str(old.MODEL_DIR),str(old.TEXT_CACHE))
    assert all(cache.contains(p) for p in mod.evaluator.PROMPTS),'Missing cached embedding; no Qwen generation'
    frame=pd.read_csv(manifest)
    records=[mod.CaseRecord(r.case_id,r.image_path,r.label_path) for r in frame.itertuples()]
    return cache,records,mod.Dataset(records,mod.load_crop_spec(CROP),cache_cases=False)


def model_setup(device):
    old,mod=imported_source()
    wrapper=old.load_fullft_model(device)
    model=wrapper.model
    named=dict(model.named_parameters(remove_duplicate=True))
    assert len({id(p) for p in named.values()})==len(named)
    assert sum(p.numel() for p in named.values())==COUNT
    assert all(p.requires_grad and p.dtype==torch.float32 for p in named.values())
    assert not any('qwen' in k.lower() or 'lora' in k.lower() for k in named)
    assert all(p.requires_grad for p in model.encoder.parameters())
    return old,mod,wrapper,named


def prepare():
    if (OUT/'PREFLIGHT_PASS.json').exists(): raise RuntimeError('Already frozen; no refreeze')
    safe_disk()
    contract=parent_contract(PARENT)
    hashes=expected_paths(json.loads((PREP/'parent_preflight.json').read_text()))
    verify(hashes)
    extras=json.loads((PREP/'extra_source_hashes.json').read_text())
    verify(extras); hashes.update(extras)
    if sha(CHECKPOINT)!=FULLFT_SHA256: raise RuntimeError('FullFT canonical checkpoint identity mismatch')
    hashes[str(CHECKPOINT)]=FULLFT_SHA256
    for name in ['source_tensor_identity.json','target_tensor_identity.json']:
        assert (PREP/name).is_file()
    source_tensor=json.loads((PREP/'source_tensor_identity.json').read_text())
    target_tensor=json.loads((PREP/'target_tensor_identity.json').read_text())
    assert source_tensor==target_tensor,'Cross-host MRI/masks/prompts not bitwise equal'
    old,mod,wrapper,named=model_setup(torch.device('cpu'))
    assert all(torch.isfinite(p).all() for p in named.values())
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    step=payload.get('global_step',payload.get('metadata',{}).get('optimizer_step'))
    assert int(step)==6000
    assert int(payload.get('configured_trainable',COUNT))==COUNT
    assert all(torch.isfinite(t).all() for t in payload['model_state'].values())
    del payload,wrapper,named
    cache,records,ds=cache_and_dataset(old,mod,PARENT/'support_manifest.csv')
    assert len(records)==5
    for i in range(5):
        item=ds[i]
        assert tuple(item['image'].shape)==(1,192,192,192)
        assert tuple(item['mask'].shape)==(8,192,192,192)
        assert torch.isfinite(item['image']).all() and torch.isfinite(item['mask']).all()
        assert list(item['prompts'])==list(contract['protocol']['prompts'])
        del item
    _,loss_source=loss_function()
    assert loss_source==contract['protocol']['loss_source'],'Canonical loss changed'
    before=pd.read_csv(PREP/'FROZEN_ALL20_ORIGINAL_ROWS.csv')
    before=before[before.model.eq('FullFT') & before.case_id.isin(contract['query']) & before.lcc.eq(0)].copy()
    assert len(before)==120 and before.case_id.nunique()==15
    assert not before.duplicated(['case_id','prompt']).any() and np.isfinite(before.dice).all()
    before['phase']='before'; atomic_csv(OUT/'QUERY_BEFORE_FROM_FROZEN.csv',before)
    conf=contract['protocol']
    atomic_json(OUT/'FROZEN_CONFIG.json',conf)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    assert 'RTX 6000D' in hardware,'Unexpected authorized hardware'
    for p in list(PARENT.iterdir())+list(PREP.glob('*.json'))+list(Path(__file__).parent.glob('*.py'))+[OUT/'FROZEN_CONFIG.json',OUT/'QUERY_BEFORE_FROM_FROZEN.csv',PREP/'FROZEN_ALL20_ORIGINAL_ROWS.csv']:
        if p.is_file(): hashes[str(p)]=sha(p)
    verify(hashes)
    assert not torch.cuda.is_initialized(),'CPU preflight unexpectedly created CUDA context'
    atomic_json(OUT/'PREFLIGHT_PASS.json',dict(status='FULLFT_FEWSHOT_READY',guarded_hashes=hashes,
                configured_unique_trainable=COUNT,checkpoint_step=6000,checkpoint_sha256=FULLFT_SHA256,
                hardware=hardware,torch=torch.__version__,finite_cpu_state=True,
                tensor_embedding_cross_host_bitwise_equal=True,free_GiB=safe_disk(),support=5,query=15,
                optional_cache_absent=not old.TEXT_CACHE.exists(),text_cache=str(old.TEXT_CACHE),
                retrospective_comparator=True,no_query_selection=True,time=time.time()))
    print(json.dumps({'status':'FULLFT_FEWSHOT_READY','parameters':COUNT,'support':5,'query':15,'hardware':hardware}),flush=True)


def guarded():
    gate=json.loads((OUT/'PREFLIGHT_PASS.json').read_text())
    verify(gate['guarded_hashes']); safe_disk()
    if gate['optional_cache_absent'] and Path(gate['text_cache']).exists():
        raise RuntimeError('Previously absent optional prompt cache appeared')
    return gate


def save_state(model,opt,step,final=False):
    path=TARGET/('step_00500.pt' if final else 'latest_resume.pt')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    data=dict(format='oasis_fullft_fewshot_v1',global_step=step,configured_trainable=COUNT,model_state=state,
              optimizer=None if final else opt.state_dict(),scheduler=None,scaler=None,
              python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
              cuda_rng=torch.cuda.get_rng_state_all(),next_order_index=step,
              sample_order_sha256=sha(PARENT/'sample_order_500.csv'),config_sha256=sha(OUT/'FROZEN_CONFIG.json'),
              parent_adni_step=6000,parent_checkpoint_sha256=FULLFT_SHA256)
    tmp=path.with_suffix('.tmp')
    torch.save(data,tmp); os.replace(tmp,path)


def train():
    guarded(); TARGET.mkdir(exist_ok=True)
    if (TARGET/'training.csv').exists(): raise RuntimeError('Training already exists; no retry/resume')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    old,mod,wrapper,named=model_setup(torch.device('cuda')); wrapper.model.train()
    cache,records,ds=cache_and_dataset(old,mod,PARENT/'support_manifest.csv')
    params=list(named.values()); opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5,betas=(.9,.999),eps=1e-8)
    assert sum(p.numel() for g in opt.param_groups for p in g['params'])==COUNT
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(params)
    order=pd.read_csv(PARENT/'sample_order_500.csv'); loss_fn,_=loss_function()
    atomic_csv(TARGET/'TRAINABLE_SCOPE.csv',pd.DataFrame([dict(name=k,numel=p.numel(),dtype=str(p.dtype)) for k,p in named.items()]))
    atomic_json(TARGET/'RUNTIME_CONFIG.json',dict(model='FullFT',configured_trainable=COUNT,optimizer_registered=COUNT,
                seed=SEED,precision='strictFP32',autocast=False,GradScaler=False,TF32=False,
                activation_checkpointing=False,base_sha256=FULLFT_SHA256,base_checkpoint=str(CHECKPOINT),
                sample_order_sha256=sha(PARENT/'sample_order_500.csv'),torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),pid=os.getpid(),ppid=os.getppid(),num_workers=0))
    fields=['step','epoch','case_id','total_loss','dice_loss','bce_loss','grad_norm_pre_clip',
            'grad_present_numel','grad_none_numel','sec','peak_allocated_GiB','peak_reserved_GiB','finite']
    torch.cuda.reset_peak_memory_stats()
    with (TARGET/'training.csv').open('x',buffering=1) as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for o in order.itertuples():
            safe_disk(); start=time.perf_counter()
            atomic_json(TARGET/'progress.json',dict(phase='TRAIN',step_completed=int(o.step)-1,current_case=o.case_id,pid=os.getpid(),time=time.time()))
            item=ds[int(o.support_index)]; assert item['case_id']==o.case_id
            image=item['image'].unsqueeze(0).cuda(); truth=item['mask'].unsqueeze(0).cuda().float()
            assert image.dtype==torch.float32 and torch.isfinite(image).all()
            opt.zero_grad(set_to_none=True); vals=[]
            assert not torch.is_autocast_enabled()
            for j,prompt in enumerate(mod.evaluator.PROMPTS):
                logits=wrapper(image,cache.get([prompt],torch.device('cuda')))
                assert logits.dtype==torch.float32 and torch.isfinite(logits).all()
                loss,dl,bl=loss_fn(logits,truth[:,[j]])
                if not torch.isfinite(loss): raise RuntimeError(f'NONFINITE_LOSS:{o.step}:{o.case_id}:{prompt}')
                (loss/8).backward(); vals.append([loss.item(),dl.item(),bl.item()]); del logits,loss,dl,bl
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            present=sum(p.numel() for p in params if p.grad is not None)
            if present!=LOSS_PATH: raise RuntimeError(f'UNEXPECTED_LOSS_PATH:{present}')
            if o.step==1:
                ledger=[]
                for k,p in named.items():
                    status='grad_none' if p.grad is None else 'nonzero' if torch.count_nonzero(p.grad) else 'zero'
                    ledger.append(dict(name=k,numel=p.numel(),status=status))
                atomic_csv(TARGET/'FIRST_STEP_GRADIENT_LEDGER.csv',pd.DataFrame(ledger))
            opt.step()
            assert all(torch.isfinite(p).all() for p in params),'Nonfinite updated model'
            assert all(torch.isfinite(v).all() for st in opt.state.values() for v in st.values() if torch.is_tensor(v)),'Nonfinite optimizer'
            torch.cuda.synchronize(); mean=np.mean(vals,axis=0)
            row=dict(step=int(o.step),epoch=int(o.epoch),case_id=o.case_id,total_loss=float(mean[0]),dice_loss=float(mean[1]),
                     bce_loss=float(mean[2]),grad_norm_pre_clip=float(norm),grad_present_numel=present,grad_none_numel=COUNT-present,
                     sec=time.perf_counter()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finite=True)
            writer.writerow(row); f.flush(); print(json.dumps(row),flush=True)
            atomic_json(TARGET/'progress.json',dict(phase='TRAIN',step_completed=int(o.step),pid=os.getpid(),time=time.time(),loss=row['total_loss']))
            del item,image,truth
            if o.step%100==0: save_state(wrapper.model,opt,int(o.step))
    save_state(wrapper.model,opt,500,final=True)
    check=torch.load(TARGET/'step_00500.pt',map_location='cpu',weights_only=False)
    assert check['global_step']==500 and check['configured_trainable']==COUNT
    assert all(torch.equal(v.detach().cpu(),check['model_state'][k]) for k,v in wrapper.model.state_dict().items())
    del check; guarded()
    atomic_json(TARGET/'TRAIN_COMPLETE.json',dict(status='TRAIN_COMPLETE',step=500,finite=True,
                checkpoint=str(TARGET/'step_00500.pt'),checkpoint_sha256=sha(TARGET/'step_00500.pt'),
                latest_resume_sha256=sha(TARGET/'latest_resume.pt'),training_sha256=sha(TARGET/'training.csv')))


def evaluate():
    guarded(); gate=json.loads((TARGET/'TRAIN_COMPLETE.json').read_text())
    assert gate['step']==500 and sha(gate['checkpoint'])==gate['checkpoint_sha256']
    old,mod,wrapper,named=model_setup(torch.device('cuda'))
    payload=torch.load(gate['checkpoint'],map_location='cpu',weights_only=False)
    wrapper.model.load_state_dict(payload['model_state'],strict=True); del payload
    wrapper.model.eval()
    cache,records,ds=cache_and_dataset(old,mod,PARENT/'query_manifest.csv')
    old.OUT=TARGET/'query'; old.OUT.mkdir(exist_ok=True)
    original_infer=mod.evaluator.infer
    def finite_infer(*args):
        p=original_infer(*args)
        if not all(np.isfinite(v).all() for v in p.values()): raise RuntimeError('NONFINITE_QUERY_PROBABILITY')
        return p
    mod.evaluator.infer=finite_infer
    frames=[]
    for i,rec in enumerate(records):
        safe_disk(); dest=old.OUT/f'{rec.case_id}_rows.csv'
        if dest.exists(): raise RuntimeError('Query already exists; no rerun')
        frame=old.evaluate_one('FullFT',wrapper,[rec],[ds[i]],cache,mod.evaluator,torch.device('cuda'))
        assert len(frame)==8 and frame.prompt.nunique()==8 and frame.lcc.eq(0).all()
        assert np.isfinite(frame.dice).all()
        frame['phase']='after'; atomic_csv(dest,frame); frames.append(frame)
        masks=old.OUT/'raw_masks_lcc0/FullFT'/f'{rec.case_id}.npz'
        with np.load(masks) as m: assert m['masks'].shape[0]==8 and np.isfinite(m['masks']).all()
        atomic_json(old.OUT/f'{rec.case_id}.done.json',dict(rows_sha256=sha(dest),mask_sha256=sha(masks)))
        atomic_json(TARGET/'progress.json',dict(phase='QUERY',cases_completed=i+1,total=15,pid=os.getpid(),time=time.time()))
        print(f'query progress {i+1}/15',flush=True)
    rows=pd.concat(frames,ignore_index=True)
    assert len(rows)==120 and rows.case_id.nunique()==15 and not rows.duplicated(['case_id','prompt']).any()
    assert set(rows.case_id)==set(pd.read_csv(PARENT/'query_manifest.csv').case_id)
    atomic_csv(TARGET/'QUERY_METRICS.csv',rows); guarded()
    atomic_json(TARGET/'MODEL_COMPLETE.json',dict(status='COMPLETE',model='FullFT',steps=500,
                query_subjects=15,roi_rows=120,missing=0,duplicate=0,checkpoint_sha256=gate['checkpoint_sha256'],
                metrics_sha256=sha(TARGET/'QUERY_METRICS.csv')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','evaluate']); args=parser.parse_args()
    def on_signal(sig,frame):
        atomic_json(OUT/f'SIGNAL_{args.stage}.json',dict(signal=sig,pid=os.getpid(),time=time.time()))
        raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]: signal.signal(sig,on_signal)
    try: {'prepare':prepare,'train':train,'evaluate':evaluate}[args.stage]()
    except Exception as e:
        atomic_json(OUT/f'failure_{args.stage}.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise
