#!/usr/bin/env python3
"""One B1 comparator; reuse the original few-shot loops without changing forward."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import types
import numpy as np
import pandas as pd
import torch
import reused_fewshot_runner as core
from fewshot_contract import SEED, sha256 as sha, verify_hashes, atomic_json
from dependency_inventory import expected_paths
from fullft_addendum_contract import prepare as parent_contract, HASHES

BASE=core.BASE
OUT=BASE/'quality_audit/oasis_b1_fewshot_addendum_20260912'
TARGET=OUT/'B1'
PREP=OUT/'preparation'
FULLFT=BASE/'quality_audit/oasis_fullft_fewshot_addendum_20260912'
PARENT=FULLFT/'preparation/parent_completed'
CHECKPOINT=BASE/'quality_audit/canonical_data_capacity_completion/task3_b1_100pct/checkpoints/step_06000.pt'
CHECKPOINT_SHA='751fd50ed3c571ba76838b506a5bdfc05ba3c1095041f6d36e3668e927493f06'
COUNT=294912
atomic_csv=core.atomic_csv


def canonical_b1_training_mode(self):
    # Exact canonical_b1_100_data_capacity_runner.py:35-38 behavior.
    self.model.eval()
    for layer in self.model.transformer_decoder.layers:
        layer.multihead_attn.parametrizations.in_proj_weight[0].train()
        layer.multihead_attn.out_proj.parametrizations.weight[0].train()


def model_setup(name,device):
    assert name=='B1'
    old=core.imported_source(); mod=old.imports()
    assert old.CHECKPOINTS['B1']==CHECKPOINT
    wrapper=old.load_model('B1',mod,device)
    named=dict(wrapper.trainable_named_parameters())
    assert sum(p.numel() for p in named.values())==COUNT
    assert len({id(p) for p in named.values()})==len(named)
    assert set(map(id,named.values()))=={id(p) for p in wrapper.model.parameters() if p.requires_grad}
    assert all('multihead_attn' in k and k.endswith(('lora_A','lora_B')) for k in named)
    assert all(p.dtype==torch.float32 for p in wrapper.model.parameters())
    assert all(not p.requires_grad for p in wrapper.model.encoder.parameters())
    assert all(not p.requires_grad for p in wrapper.model.decoder.parameters())
    assert all(not p.requires_grad for p in wrapper.model.project_to_decoder_channels.parameters())
    assert not any('qwen' in k.lower() for k in named)
    wrapper.set_training_mode=types.MethodType(canonical_b1_training_mode,wrapper)
    wrapper.save_checkpoint=wrapper.save_adapter_checkpoint
    cache=mod.TextEmbeddingCache(str(old.BANK),str(old.MODEL_DIR),str(old.TEXT_CACHE))
    assert all(cache.contains(p) for p in mod.evaluator.PROMPTS),'Missing cached prompt; no Qwen generation'
    return old,mod,wrapper,named,cache


core.OUT=OUT
core.MODELS={'B1':COUNT}
core.EXPECTED={'B1':CHECKPOINT_SHA}
core.model_setup=model_setup
guarded=core.guarded


def prepare():
    if (OUT/'PREFLIGHT_PASS.json').exists(): raise RuntimeError('Already frozen; no refreeze')
    contract=parent_contract(PARENT)
    hashes=expected_paths(json.loads((FULLFT/'preparation/parent_preflight.json').read_text()))
    extras=json.loads((FULLFT/'preparation/extra_source_hashes.json').read_text())
    hashes.update(extras)
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    hashes[str(CHECKPOINT)]=CHECKPOINT_SHA
    verify_hashes(hashes)
    provenance=pd.read_csv(PREP/'MODEL_PROVENANCE_AUDIT.csv')
    b1=provenance[provenance.model.eq('B1')]
    assert len(b1)==1 and CHECKPOINT_SHA in b1.iloc[0].astype(str).tolist()
    source_config=json.loads((PREP/'config.json').read_text())
    payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    assert payload['model']=='b1' and payload['subset']=='100pct' and payload['global_step']==6000
    assert payload['config']==source_config
    assert source_config['seed']==SEED and source_config['train_ptids']==337 and source_config['train_visits']==971
    assert source_config['parameter_groups']=={'cross_attention_lora':COUNT}
    assert not source_config['amp_forward'] and not source_config['grad_scaler']
    state=payload['trainable_model_state']
    assert sum(v.numel() for v in state.values())==COUNT and all(torch.isfinite(v).all() for v in state.values())
    old,mod,wrapper,named,cache=model_setup('B1',torch.device('cpu'))
    assert set(state)==set(named)
    assert all(torch.equal(state[k],p) for k,p in named.items())
    wrapper.set_training_mode()
    assert not wrapper.model.training and not wrapper.model.encoder.training and not wrapper.model.decoder.training
    mode_rows=[]
    for n,m in wrapper.model.named_modules():
        if hasattr(m,'lora_A'):
            assert m.training and m.lora_A.shape[0]==4 and m.scaling==2 and m.dropout==.05
            mode_rows.append(dict(name=n,rank=4,alpha=8,dropout=m.dropout,training=m.training))
    assert len(mode_rows)==12
    assert all(torch.isfinite(p).all() for p in wrapper.model.parameters())
    # Exercise the compatibility save/load interface on CPU; no optimizer or forward.
    save_test=PREP/'B1_CPU_ADAPTER_INTERFACE_TEST.pt'
    if save_test.exists(): raise RuntimeError('CPU interface test already exists; inspect before retry')
    wrapper.save_checkpoint(str(save_test),{'purpose':'CPU_INTERFACE_TEST_NOT_FORMAL','optimizer_steps':0})
    reread=torch.load(save_test,map_location='cpu',weights_only=False)
    assert set(reread['adapter_state_dict'])==set(named)
    assert all(torch.equal(reread['adapter_state_dict'][k],p) for k,p in named.items())
    wrapper.eval()
    assert all(not m.training for _,m in wrapper.model.named_modules())
    del wrapper,named,payload,state,reread
    for name in ['support_manifest.csv','query_manifest.csv','sample_order_500.csv']:
        dest=OUT/name
        if dest.exists(): raise RuntimeError('Frozen destination already exists')
        shutil.copyfile(PARENT/name,dest)
        assert sha(dest)==HASHES[name]
    _,loss_source=core.canonical_loss()
    assert loss_source==contract['protocol']['loss_source']
    conf=dict(contract['protocol']); conf['models']={'B1':COUNT}
    conf.update(model_training_mode='canonical model.eval; LoRA parametrizations.train',
                canonical_checkpoint_sha256=CHECKPOINT_SHA,
                source_model_train_precision='strictFP32 according to embedded ADNI100 checkpoint config',
                addendum_after_existing_results=True)
    atomic_json(OUT/'FROZEN_CONFIG.json',conf)
    support=pd.read_csv(OUT/'support_manifest.csv')
    recs=[mod.CaseRecord(r.case_id,r.image_path,r.label_path) for r in support.itertuples()]
    ds=mod.Dataset(recs,mod.load_crop_spec(core.CROP),cache_cases=False)
    tensor_rows=[]
    for i,rec in enumerate(recs):
        item=ds[i]
        assert tuple(item['image'].shape)==(1,192,192,192) and tuple(item['mask'].shape)==(8,192,192,192)
        assert torch.isfinite(item['image']).all() and torch.isfinite(item['mask']).all()
        assert list(item['prompts'])==conf['prompts']
        tensor_rows.append(dict(case_id=rec.case_id,image_shape=list(item['image'].shape),mask_shape=list(item['mask'].shape),finite=True))
        del item
    source_tensor=json.loads((FULLFT/'preparation/source_tensor_identity.json').read_text())
    target_tensor=json.loads((FULLFT/'preparation/target_tensor_identity.json').read_text())
    assert source_tensor==target_tensor
    before_path=FULLFT/'preparation/FROZEN_ALL20_ORIGINAL_ROWS.csv'
    assert sha(before_path)=='b8c863347062f2c314f4deb1f535026d2f3a23eb518bdc8b212ab0fbb6044121'
    before=pd.read_csv(before_path)
    before=before[before.model.eq('B1') & before.case_id.isin(contract['query']) & before.lcc.eq(0)].copy()
    assert len(before)==120 and before.case_id.nunique()==15 and not before.duplicated(['case_id','prompt']).any()
    assert np.isfinite(before.dice).all()
    before['phase']='before'; atomic_csv(OUT/'QUERY_BEFORE_FROM_FROZEN.csv',before)
    full_gate=json.loads((FULLFT/'COMPLETE.json').read_text())
    assert full_gate['status']=='OASIS_FULLFT_FEWSHOT_ADAPTATION_COMPLETE' and full_gate['combined_rows']==720
    for name,h in full_gate['outputs'].items():
        assert sha(FULLFT/name)==h
        hashes[str(FULLFT/name)]=h
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    assert 'RTX 6000D' in hardware
    files=list(Path(__file__).parent.glob('*.py'))+list(PREP.iterdir())+list(OUT.glob('*.json'))+list(OUT.glob('*.csv'))
    files += [before_path,FULLFT/'COMPLETE.json',FULLFT/'preparation/source_tensor_identity.json',FULLFT/'preparation/target_tensor_identity.json']
    for p in files:
        if p.is_file(): hashes[str(p)]=sha(p)
    verify_hashes(hashes)
    assert not torch.cuda.is_initialized()
    atomic_json(OUT/'PREFLIGHT_PASS.json',dict(status='B1_FEWSHOT_READY',guarded_hashes=hashes,
                required_absent=[] if old.TEXT_CACHE.exists() else [str(old.TEXT_CACHE)],
                checkpoint_sha256=CHECKPOINT_SHA,checkpoint_step=6000,configured_unique_trainable=COUNT,
                source_config=source_config,model_mode_checks=mode_rows,support_tensor_checks=tensor_rows,
                source_target_tensor_embedding_bitwise_equal=True,hardware=hardware,torch=torch.__version__,
                finite_cpu_state=True,support=5,query=15,free_GiB=core.safe_disk(),time=time.time()))
    print(json.dumps({'status':'B1_FEWSHOT_READY','parameters':COUNT,'support':5,'query':15}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','evaluate']); a=parser.parse_args()
    def stop(sig,frame):
        atomic_json(OUT/f'SIGNAL_{a.stage}.json',dict(signal=sig,pid=os.getpid(),time=time.time()))
        raise SystemExit(128+sig)
    for s in [signal.SIGINT,signal.SIGHUP,signal.SIGTERM]: signal.signal(s,stop)
    try:
        if a.stage=='prepare': prepare()
        elif a.stage=='train': core.train('B1')
        else: core.evaluate('B1')
    except Exception as e:
        atomic_json(OUT/f'failure_{a.stage}.json',dict(error=str(e),traceback=traceback.format_exc(),pid=os.getpid(),time=time.time()))
        raise


if __name__=='__main__': main()
