#!/usr/bin/env python3
"""Isolated approved OASIS5/15 adaptation. Never invokes historical main()."""
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
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fewshot_contract import SEED, MODELS, sha256, verify_hashes, split_rows, sample_order, atomic_json, config

BASE = Path('__DRPA_WORKSPACE__')
OLD = BASE/'quality_audit/oasis_trt20_external_frozen_test'
OUT = BASE/'quality_audit/oasis_fewshot_adaptation_20260912'
SOURCE = BASE/'scripts/evaluate/run_oasis_external_frozen_test.py'
TRAINER = BASE/'quality_audit/voxtell_mtl_drpa8_full_data/drpa8_full_data_train.py'
CROP = BASE/'quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json'
EXPECTED = {
 'FrozenVoxTell': 'f45e61c34c56af7b71711a6de54e8414dbc4cb52441003894f3370ed68d8feaa',
 'DRPA8': 'b6f0e8b5be8a3167a8bb893da27bfd15506a009d117ba3e31df592f40d94c9fe',
 'B3Canonical': '65f364167a2df39a6def7c501c0a491d4522078a50254cbbf402dd89e12a66da'}


def imported_source():
    spec = importlib.util.spec_from_file_location('frozen_oasis_source', SOURCE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def canonical_loss():
    tree = ast.parse(TRAINER.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_fp32')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAINER), 'exec'), namespace)
    return namespace['loss_fp32'], ast.get_source_segment(TRAINER.read_text(), node)


def safe_disk():
    free = shutil.disk_usage(OUT.parent).free / 2**30
    if free < 20: raise RuntimeError(f'DISK_BELOW20GiB: {free:.3f}')
    return free


def atomic_csv(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_csv(tmp, index=False); os.replace(tmp, path)


def guarded():
    receipt = json.loads((OUT/'PREFLIGHT_PASS.json').read_text())
    for p in receipt.get('required_absent', []):
        if Path(p).exists(): raise RuntimeError(f'Previously absent optional input appeared: {p}')
    verify_hashes(receipt['guarded_hashes']); safe_disk()
    return receipt


def prepare():
    if OUT.exists(): raise RuntimeError('Output already exists; inspect, never overwrite/re-freeze')
    safe_disk()
    if json.loads((BASE/'quality_audit/adni_locked_frozen_test_20260911/queue_state.json').read_text())['status'] != 'ALL_FIVE_MODELS_COMPLETE':
        raise RuntimeError('ADNI prerequisite incomplete')
    old = imported_source(); mod = old.imports()
    source_manifest = OLD/'preflight/external_subject_manifest_mapped.csv'
    df = pd.read_csv(source_manifest)
    support, query = split_rows(df.to_dict('records'))
    original = pd.read_csv('__DRPA_OASIS_ROOT__/02_metadata/subject_manifest.csv')
    assert set(original.subject_id) == set(df.subject_id)
    for row in df.itertuples():
        ref = original[original.subject_id == row.subject_id].iloc[0]
        assert row.image_path == ref.t1_mri_path and row.source_label_path == ref.dkt31_cma_label_path
    assert df.image_sha256.nunique() == 20
    hashes = {str(source_manifest): sha256(source_manifest)}
    for row in df.to_dict('records'):
        for p, h in [('image_path','image_sha256'),('source_label_path','source_label_sha256'),('label_path','derived_label_sha256')]:
            hashes[row[p]] = row[h]
    verify_hashes(hashes)
    for name, h in EXPECTED.items():
        path = old.CHECKPOINTS[name]; assert sha256(path) == h, (name, 'checkpoint hash mismatch')
        hashes[str(path)] = h
    payload_audits = {}
    for name in MODELS:
        p = torch.load(old.CHECKPOINTS[name], map_location='cpu', weights_only=False)
        assert p['format'] == old.EXPECTED[name][0]
        meta = p['metadata']; assert int(meta.get('optimizer_step', meta.get('step', -1))) == 6000
        assert int(meta['trainable_parameters']) == MODELS[name]
        weights = p.get('adapter_state_dict', p.get('trainable_state_dict'))
        assert weights is not None and all(torch.isfinite(x).all() for x in weights.values())
        assert sum(x.numel() for x in weights.values()) == MODELS[name]
        payload_audits[name] = {'checkpoint': str(old.CHECKPOINTS[name]), 'sha256': EXPECTED[name], 'metadata':meta, 'state_finite':True}
        del p, weights
    # Freeze source modules as well as original results; no method-level edits allowed after launch.
    files = [SOURCE, TRAINER, CROP, old.BANK,
             BASE/'quality_audit/voxtell_mtl_drpa8_pilot/drpa8_wrapper.py',
             BASE/'quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_full_data_wrapper.py',
             BASE/'quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound/voxtell_decoder_capacity_wrapper.py',
             BASE/'quality_audit/voxtell_mtl_peft/voxtell_peft_wrapper.py',
             BASE/'quality_audit/voxtell_mtl_peft/text_embedding_cache.py',
             BASE/'quality_audit/voxtell_mtl_peft/mtl_grounding_dataset.py',
             BASE/'quality_audit/voxtell_mtl_peft_pilot/pilot_preprocessing.py',
             BASE/'quality_audit/voxtell_mtl_b1_bilateral_crop/b1_preprocessing.py',
             BASE/'quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py']
    files += list(OLD.glob('*.csv')) + list(OLD.glob('*.md')) + list(OLD.glob('*.json'))
    files += list((BASE/'VoxTell/voxtell').rglob('*.py')) + list(Path(__file__).parent.glob('*.py'))
    if old.TEXT_CACHE.is_file(): files.append(old.TEXT_CACHE)
    for p in files: hashes[str(p)] = sha256(p)
    cache = mod.TextEmbeddingCache(str(old.BANK), str(old.MODEL_DIR), str(old.TEXT_CACHE))
    assert all(cache.contains(p) for p in mod.evaluator.PROMPTS), 'Missing prompt cannot invoke Qwen or write sharedcache'
    assert len(mod.evaluator.PROMPTS) == 8
    assert config()['prompt_reduction'].startswith('8 separate')
    loss_fn, loss_source = canonical_loss()
    z = torch.zeros(1,1,2,2,2); y = torch.ones_like(z)
    loss, d, b = loss_fn(z, y); assert torch.isfinite(loss) and torch.equal(loss, d+b)
    OUT.mkdir()
    # Split frozen BEFORE any access to the old score table.
    atomic_csv(OUT/'support_manifest.csv', pd.DataFrame(support))
    atomic_csv(OUT/'query_manifest.csv', pd.DataFrame(query))
    atomic_csv(OUT/'sample_order_500.csv', pd.DataFrame(sample_order(support)))
    conf = config(); conf['prompts'] = mod.evaluator.PROMPTS; conf['loss_source'] = loss_source
    atomic_json(OUT/'FROZEN_CONFIG.json', conf)
    atomic_json(OUT/'SPLIT_FREEZE.json', dict(seed=SEED, support=[r['case_id'] for r in support],
                 query=[r['case_id'] for r in query], outcome_blind=True, time=time.time(),
                 files={p.name:sha256(p) for p in OUT.glob('*') if p.is_file()}))
    before = pd.read_csv(OLD/'OASIS_EXTERNAL_CASE_LEVEL_METRICS.csv')
    before = before[before.model.isin(MODELS) & before.case_id.isin([r['case_id'] for r in query]) & before.lcc.eq(0)].copy()
    assert len(before) == 240 and not before.duplicated(['model','case_id','prompt']).any()
    assert before.groupby('model').size().eq(120).all()
    before['phase'] = 'before'; atomic_csv(OUT/'QUERY_BEFORE_FROM_FROZEN.csv', before)
    # Five support items checked on CPU only, with the unchanged native reader.
    recs = [mod.CaseRecord(r['case_id'],r['image_path'],r['label_path']) for r in support]
    ds = mod.Dataset(recs, mod.load_crop_spec(CROP), cache_cases=False)
    tensor_audits = []
    for i, rec in enumerate(recs):
        item = ds[i]; assert tuple(item['image'].shape) == (1,192,192,192)
        assert tuple(item['mask'].shape) == (8,192,192,192)
        assert torch.isfinite(item['image']).all() and torch.isfinite(item['mask']).all()
        assert list(item['prompts']) == list(mod.evaluator.PROMPTS)
        tensor_audits.append(dict(case_id=rec.case_id, image_shape=list(item['image'].shape),mask_shape=list(item['mask'].shape), finite=True))
        del item
    for p in OUT.glob('*'):
        if p.is_file(): hashes[str(p)] = sha256(p)
    verify_hashes(hashes)
    assert not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA'
    atomic_json(OUT/'PREFLIGHT_PASS.json',dict(status='OASIS_FEWSHOT_READY',guarded_hashes=hashes,
                required_absent=[] if old.TEXT_CACHE.is_file() else [str(old.TEXT_CACHE)],
                checkpoints=payload_audits, support_tensor_checks=tensor_audits, free_GiB=safe_disk(),
                torch_version=torch.__version__, source_evaluation_precision='original grouped FP16, FP32 sigmoid',
                support_subjects=[r['case_id'] for r in support], query_subjects=[r['case_id'] for r in query]))
    print(json.dumps({'status':'OASIS_FEWSHOT_READY','support':[r['case_id'] for r in support],'query':15}), flush=True)


def model_setup(name, device):
    old = imported_source(); mod = old.imports()
    wrapper = old.load_model(name, mod, device)
    named = dict(wrapper.trainable_named_parameters())
    assert sum(p.numel() for p in named.values()) == MODELS[name]
    assert len({id(p) for p in named.values()}) == len(named)
    assert set(id(p) for p in wrapper.model.parameters() if p.requires_grad) == set(id(p) for p in named.values())
    assert all(p.dtype == torch.float32 for p in wrapper.model.parameters())
    assert all(not p.requires_grad for p in wrapper.model.encoder.parameters())
    assert all('qwen' not in k.lower() for k in named)
    cache = mod.TextEmbeddingCache(str(old.BANK), str(old.MODEL_DIR), str(old.TEXT_CACHE))
    assert all(cache.contains(p) for p in mod.evaluator.PROMPTS)
    return old, mod, wrapper, named, cache


def save_training_state(path, wrapper, optimizer, named, step, order_hash):
    state = dict(format='oasis_fewshot_training_state_v1', step=step,
                 trainable_state={k:p.detach().cpu().clone() for k,p in named.items()},
                 optimizer=optimizer.state_dict(), scheduler=None, scaler=None,
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(), sample_order_sha256=order_hash,
                 next_order_index=step, config_sha256=sha256(OUT/'FROZEN_CONFIG.json'))
    tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)


def train(name):
    guarded(); target = OUT/name; target.mkdir(exist_ok=True)
    if (target/'training.csv').exists(): raise RuntimeError('Training exists; no automatic retry/resume')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    old, mod, wrapper, named, cache = model_setup(name, torch.device('cuda'))
    wrapper.set_training_mode(); params=list(named.values())
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5,betas=(.9,.999),eps=1e-8)
    assert sum(p.numel() for g in optimizer.param_groups for p in g['params']) == MODELS[name]
    support=pd.read_csv(OUT/'support_manifest.csv')
    recs=[mod.CaseRecord(r.case_id,r.image_path,r.label_path) for r in support.itertuples()]
    ds=mod.Dataset(recs,mod.load_crop_spec(CROP),cache_cases=False)
    order=pd.read_csv(OUT/'sample_order_500.csv'); loss_fn,_=canonical_loss()
    prompts=list(mod.evaluator.PROMPTS)
    atomic_csv(target/'TRAINABLE_SCOPE.csv',pd.DataFrame([dict(name=k,numel=p.numel(),dtype=str(p.dtype)) for k,p in named.items()]))
    atomic_json(target/'RUNTIME_CONFIG.json',dict(model=name,trainable=MODELS[name],optimizer_registered=MODELS[name],
                seed=SEED,precision='strictFP32',GradScaler=False,autocast=False,TF32=False,
                base_checkpoint=str(old.CHECKPOINTS[name]),base_sha256=EXPECTED[name],
                sample_order_sha256=sha256(OUT/'sample_order_500.csv'),torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),pid=os.getpid(),ppid=os.getppid(),num_workers=0,
                canonical_wrapper_decoder_checkpointing='unchanged preserve_rng_state=True'))
    torch.cuda.reset_peak_memory_stats()
    fields=['step','epoch','case_id','total_loss','dice_loss','bce_loss','grad_norm_pre_clip','sec','peak_allocated_GiB','peak_reserved_GiB','finite']
    with (target/'training.csv').open('x',buffering=1) as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for o in order.itertuples():
            safe_disk(); start=time.perf_counter()
            atomic_json(target/'progress.json',dict(phase='TRAIN',step_completed=int(o.step)-1,current_case=o.case_id,pid=os.getpid(),time=time.time()))
            item=ds[int(o.support_index)]; assert item['case_id']==o.case_id
            image=item['image'].unsqueeze(0).cuda(); truth=item['mask'].unsqueeze(0).cuda().float()
            assert torch.isfinite(image).all() and image.dtype==torch.float32
            optimizer.zero_grad(set_to_none=True); vals=[]
            assert not torch.is_autocast_enabled()
            for j,prompt in enumerate(prompts):
                logits=wrapper(image,cache.get([prompt],torch.device('cuda')))
                assert logits.dtype==torch.float32
                loss,dl,bl=loss_fn(logits,truth[:,[j]])
                if not torch.isfinite(loss): raise RuntimeError(f'NONFINITE_LOSS step{o.step} {o.case_id} {prompt}')
                (loss/8).backward(); vals.append([loss.item(),dl.item(),bl.item()]); del logits,loss,dl,bl
            norm=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            assert all(p.grad is not None for p in params), 'Trainable disconnected from loss'
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in params), 'Nonfinite updated model'
            assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)), 'Nonfinite optimizer'
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start; mean=np.mean(vals,axis=0)
            row=dict(step=int(o.step),epoch=int(o.epoch),case_id=o.case_id,total_loss=float(mean[0]),dice_loss=float(mean[1]),bce_loss=float(mean[2]),
                     grad_norm_pre_clip=float(norm),sec=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,finite=True)
            writer.writerow(row); stream.flush()
            atomic_json(target/'progress.json',dict(phase='TRAIN',step_completed=int(o.step),pid=os.getpid(),time=time.time(),**{'loss':row['total_loss']}))
            print(json.dumps(row),flush=True)
            del item,image,truth
            if o.step%100==0:
                save_training_state(target/'latest_resume.pt',wrapper,optimizer,named,int(o.step),sha256(OUT/'sample_order_500.csv'))
    path=target/'step_00500.pt'; tmp=path.with_suffix('.tmp')
    wrapper.save_checkpoint(str(tmp),dict(experiment='OASIS_FEWSHOT_ADAPTATION',optimizer_step=500,trainable_parameters=MODELS[name],base_checkpoint_sha256=EXPECTED[name]))
    os.replace(tmp,path)
    check=torch.load(path,map_location='cpu',weights_only=False)
    state=check.get('adapter_state_dict',check.get('trainable_state_dict'))
    assert set(state)==set(named) and all(torch.isfinite(x).all() for x in state.values())
    assert all(torch.equal(state[k],p.detach().cpu()) for k,p in named.items()); del check,state
    guarded()
    atomic_json(target/'TRAIN_COMPLETE.json',dict(step=500,status='TRAIN_COMPLETE',checkpoint_sha256=sha256(path),
                checkpoint=str(path),training_sha256=sha256(target/'training.csv'),latest_resume_sha256=sha256(target/'latest_resume.pt'),
                sample_order_sha256=sha256(OUT/'sample_order_500.csv'),finite=True))


def evaluate(name):
    guarded(); target=OUT/name; gate=json.loads((target/'TRAIN_COMPLETE.json').read_text())
    assert gate['step']==500 and sha256(gate['checkpoint'])==gate['checkpoint_sha256']
    old,mod,wrapper,named,cache=model_setup(name,torch.device('cuda'))
    payload=torch.load(gate['checkpoint'],map_location='cpu',weights_only=False)
    state=payload.get('adapter_state_dict',payload.get('trainable_state_dict'))
    assert set(state)==set(named)
    with torch.no_grad():
        for k,p in named.items(): p.copy_(state[k].to(p.device))
    del payload,state; wrapper.eval()
    query=pd.read_csv(OUT/'query_manifest.csv')
    recs=[mod.CaseRecord(r.case_id,r.image_path,r.label_path) for r in query.itertuples()]
    ds=mod.Dataset(recs,mod.load_crop_spec(CROP),cache_cases=False)
    # Only redirect output destination. Reuse original infer/restore/metrics unchanged.
    old.OUT=target/'query'; old.OUT.mkdir(exist_ok=True)
    cache_get=cache.get
    def safe_get(prompts,device):
        assert all(cache.contains(p) for p in prompts)
        return cache_get(prompts,device)
    cache.get=safe_get
    original_infer=mod.evaluator.infer
    def finite_infer(*args):
        probabilities=original_infer(*args)
        if not all(np.isfinite(p).all() for p in probabilities.values()): raise RuntimeError('NONFINITE_QUERY_PROBABILITY')
        return probabilities
    mod.evaluator.infer=finite_infer
    frames=[]
    for i,rec in enumerate(recs):
        safe_disk(); case_path=old.OUT/f'{rec.case_id}_rows.csv'
        if case_path.exists(): raise RuntimeError('Query case already exists: no automatic rerun')
        frame=old.evaluate_one(name,wrapper,[rec],[ds[i]],cache,mod.evaluator,torch.device('cuda'))
        assert len(frame)==8 and frame.prompt.nunique()==8
        assert np.isfinite(frame.dice).all() and frame.lcc.eq(0).all()
        frame['phase']='after'; atomic_csv(case_path,frame); frames.append(frame)
        masks=old.OUT/'raw_masks_lcc0'/name/f'{rec.case_id}.npz'
        with np.load(masks) as m: assert m['masks'].shape[0]==8 and np.isfinite(m['masks']).all()
        atomic_json(old.OUT/f'{rec.case_id}.done.json',dict(rows_sha256=sha256(case_path),masks_sha256=sha256(masks)))
        atomic_json(target/'progress.json',dict(phase='QUERY',cases_completed=i+1,total=15,pid=os.getpid(),time=time.time()))
        print(f'query progress {name} {i+1}/15',flush=True)
    rows=pd.concat(frames,ignore_index=True)
    assert len(rows)==120 and rows.case_id.nunique()==15 and not rows.duplicated(['case_id','prompt']).any()
    assert set(rows.case_id)==set(query.case_id)
    atomic_csv(target/'QUERY_METRICS.csv',rows); guarded()
    atomic_json(target/'MODEL_COMPLETE.json',dict(model=name,status='COMPLETE',steps=500,query_subjects=15,roi_rows=120,
                checkpoint_sha256=gate['checkpoint_sha256'],metrics_sha256=sha256(target/'QUERY_METRICS.csv'),missing=0,duplicate=0))


def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['prepare','train','evaluate']); p.add_argument('--model',choices=list(MODELS)); a=p.parse_args()
    if a.stage!='prepare' and a.model is None: p.error('model required')
    target=OUT/a.model if a.model else OUT
    def on_signal(sig,frame):
        atomic_json(target/f'SIGNAL_{a.stage}.json',dict(signal=sig,pid=os.getpid(),time=time.time()))
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,on_signal)
    try:
        if a.stage=='prepare': prepare()
        elif a.stage=='train': train(a.model)
        else: evaluate(a.model)
    except Exception as e:
        if target.exists(): atomic_json(target/f'failure_{a.stage}.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise


if __name__=='__main__': main()
