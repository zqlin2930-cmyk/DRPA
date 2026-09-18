#!/usr/bin/env python3
"""Canonical B1 100% fixed-step runner with explicit FP32 forward/backward."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse, gc, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

BASE=Path(os.environ.get('MTL_MODEL_ROOT', '__DRPA_WORKSPACE__')); AUDIT=BASE/'quality_audit/drpa_data_capacity_scaling'; FULL=BASE/'quality_audit/voxtell_mtl_drpa8_full_data'; PILOT=BASE/'quality_audit/voxtell_mtl_drpa8_pilot'; PEFT=BASE/'quality_audit/voxtell_mtl_peft'; PREP=BASE/'quality_audit/voxtell_mtl_b1_bilateral_crop'; B3=BASE/'quality_audit/voxtell_mtl_stable_decoder_capacity_baseline'
sys.path[:0]=[str(PILOT),str(PEFT),str(PREP),str(B3)]
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
from text_embedding_cache import TextEmbeddingCache
from voxtell_peft_wrapper import VoxTellPEFTWrapper
from drpa8_wrapper import DRPA8Wrapper
from voxtell_decoder_capacity_wrapper import VoxTellDecoderCapacityWrapper
import evaluate_drpa8 as evaluator

SEED=20260809; DEVICE=torch.device('cuda'); MODEL=str(BASE/'VoxTell_weights/voxtell_v1.1'); BANK=str(BASE/'VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz'); CACHE=str(PEFT/'text_embedding_cache.npz'); EXPECTED={'b1':294912,'drpa8':10969696,'b3':83873893}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def refuse_busy_gpu():
    s=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
    if s: raise RuntimeError('GPU already occupied: '+s)
def records(path):
    d=pd.read_csv(path,dtype=str); return [CaseRecord(str(r.case_id),str(r.image_path),str(r.label_path)) for r in d.itertuples()]
def loss_fp32(z,y):
    z=z.float(); y=y.float(); b=F.binary_cross_entropy_with_logits(z,y); p=torch.sigmoid(z); dims=tuple(range(2,p.ndim)); inter=(p*y).sum(dims); den=p.sum(dims)+y.sum(dims); dl=1-((2*inter+1e-5)/(den+1e-5)).mean(); return dl+b
def make_wrapper(kind):
    if kind=='b1':
        w=VoxTellPEFTWrapper(MODEL,BANK,rank=4,alpha=8,dropout=.05,device=DEVICE); w.model.eval()
        for l in w.model.transformer_decoder.layers:
            l.multihead_attn.parametrizations.in_proj_weight[0].train(); l.multihead_attn.out_proj.parametrizations.weight[0].train()
        groups={'cross_attention_lora':list(w.trainable_parameters())}; save=w.save_adapter_checkpoint
    elif kind=='drpa8':
        w=DRPA8Wrapper(MODEL,BANK,device=DEVICE); w.set_training_mode(); groups={'cross_attention_lora':[],'projection_adapter':[],'decoder_stages':[]}
        for _,p,g in w.trainable_parameter_groups(): groups[g].append(p)
        save=w.save_checkpoint
    else:
        w=VoxTellDecoderCapacityWrapper(MODEL,BANK,rank=4,alpha=8,dropout=.05,device=DEVICE); w.model.eval()
        for n in w.decoder_owned_children: getattr(w.model.decoder,n).train()
        for p in w.model.project_to_decoder_channels: p.train()
        for l in w.model.transformer_decoder.layers:
            l.multihead_attn.parametrizations.in_proj_weight[0].train(); l.multihead_attn.out_proj.parametrizations.weight[0].train()
        groups={'cross_attention_lora':[],'decoder_projection':[]}
        for _,p,g in w.trainable_parameter_groups(): groups['cross_attention_lora' if g=='cross_attention_lora' else 'decoder_projection'].append(p)
        save=w.save_adapter_checkpoint
    count=sum(p.numel() for ps in groups.values() for p in ps)
    if count!=EXPECTED[kind]: raise RuntimeError(f'{kind} trainable count {count} != {EXPECTED[kind]}')
    return w,groups,save
def grad_stats(groups):
    out={}; params=[]
    for g,ps in groups.items():
        finite=nonzero=none=0; sq=0.
        for p in ps:
            params.append(p)
            if p.grad is None: none+=1; continue
            finite+=int(torch.isfinite(p.grad).all()); nonzero+=int(torch.count_nonzero(p.grad).item()>0); sq+=float(torch.sum(p.grad.float()**2).cpu())
        out[g]={'param_tensors':len(ps),'finite_grad_tensors':finite,'nonzero_grad_tensors':nonzero,'none_grad_tensors':none,'grad_norm':sq**.5}
    return out,params
def save_state(w,opt,path,step,kind,subset,config):
    # The frozen VoxTell backbone is identical for every screen arm. Persist
    # only trainable tensors plus optimizer/RNG/config state; this preserves
    # audit reproducibility without writing the multi-GB frozen backbone nine
    # times. The official initialization and manifest hashes are recorded in
    # initialization.json/config.json for each independent run.
    trainable_state={n:p.detach().cpu() for n,p in w.model.named_parameters() if p.requires_grad}
    tmp=path.with_suffix('.tmp'); torch.save({'trainable_model_state':trainable_state,'optimizer_state':opt.state_dict(),'rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'global_step':step,'model':kind,'subset':subset,'config':config},tmp); os.replace(tmp,path)
def validate(w,val,val_ds,cache,step,out,kind,train_len):
    rows=evaluator.evaluate_condition(f'DATA_CAPACITY_{kind}@step{step:05d}',w,val,val_ds,cache); df=pd.DataFrame(rows); sm=evaluator.summarize(df); sm.insert(1,'step',step); df.to_csv(out/f'validation_rows_step_{step:05d}.csv',index=False); sm.to_csv(out/f'validation_summary_step_{step:05d}.csv',index=False); raw=sm[sm.lcc==0]; o=raw[raw.scope=='overall'].iloc[0]; r={'step':step,'equivalent_epoch':step/train_len,'mean_dice':float(o.dice),'hd95_mm':float(o.hd95_mm),'surface_dice_2mm':float(o.surface_dice_2mm),'components':float(o.connected_components),'fp_volume_ml':float(o.false_positive_volume_ml),'max_fp_distance_mm':float(o.max_false_positive_distance_mm),'empty_mask_rate':float(o.empty_mask_rate)}
    for s,k in [('hippocampus','hipp'),('entorhinal cortex','ec'),('parahippocampal gyrus','phg'),('amygdala','amy')]: r[k+'_dice']=float(raw[raw.scope==s].iloc[0].dice)
    return r
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',choices=['b1','drpa8','b3'],required=True); ap.add_argument('--subset',choices=['10pct','25pct','50pct','100pct'],required=True); ap.add_argument('--max-steps',type=int,default=6000); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True)
    refuse_busy_gpu(); seed_all(); train_path=AUDIT/'manifests'/f'train_{a.subset}.csv'; val_path=AUDIT/'manifests/val_100pct_frozen.csv'; train_frame=pd.read_csv(train_path,dtype=str); val_frame=pd.read_csv(val_path,dtype=str); train=records(train_path); val=records(val_path); spec=load_crop_spec(PILOT/'crop_spec.json'); train_ds=BilateralGroupedPatchDataset(train,spec,cache_cases=False); val_ds=BilateralGroupedPatchDataset(val,spec,cache_cases=False); cache=TextEmbeddingCache(BANK,MODEL,CACHE); w,groups,save=make_wrapper(a.model); counts={g:sum(p.numel() for p in ps) for g,ps in groups.items()}; config={'model':a.model,'subset':a.subset,'seed':SEED,'max_optimizer_steps':a.max_steps,'train_visits':len(train),'train_ptids':int(train_frame.ptid.nunique()),'val_visits':len(val),'val_ptids':int(val_frame.ptid.nunique()),'batch_size':1,'gradient_accumulation':1,'preprocessing':'canonical RAS + official reader-space + fixed bilateral 192^3 crop','prompts':'8 fixed canonical prompts','loss':'FP32 Dice+BCE','amp_forward':False,'grad_scaler':False,'clip_norm':1.0,'weight_decay':1e-5,'parameter_groups':counts,'train_manifest_sha256':sha256(train_path),'val_manifest_sha256':sha256(val_path)}; (out/'config.json').write_text(json.dumps(config,indent=2)); pd.DataFrame([{'module_group':g,'parameter_count':n,'requires_grad':True} for g,n in counts.items()]).to_csv(out/'parameter_summary.csv',index=False)
    opt_groups=[{'params':groups['cross_attention_lora'],'lr':1e-4}]
    if a.model=='drpa8': opt_groups += [{'params':groups['projection_adapter'],'lr':1e-4},{'params':groups['decoder_stages'],'lr':1e-5}]
    elif a.model=='b3': opt_groups += [{'params':groups['decoder_projection'],'lr':1e-5}]
    opt=torch.optim.AdamW(opt_groups,weight_decay=1e-5); scaler=None
    # Keep the fresh-initialization audit trail without duplicating the multi-GB
    # base model. Formal screen checkpoints are only written at validation steps.
    (out/'initialization.json').write_text(json.dumps({
        'status':'FRESH_OFFICIAL_INITIALIZATION',
        'model':a.model, 'subset':a.subset, 'seed':SEED,
        'train_manifest_sha256':config['train_manifest_sha256'],
        'val_manifest_sha256':config['val_manifest_sha256'],
        'trainable_parameters':sum(counts.values()),
    }, indent=2))
    steps=[]; epochs=[]; vals=[]; best=None; epoch=0; t0=time.time()
    while len(steps)<a.max_steps:
        epoch+=1; es=time.time(); losses=[]; order=np.random.default_rng(SEED+epoch).permutation(len(train_ds))
        for idx in order:
            if len(steps)>=a.max_steps: break
            item=train_ds[int(idx)]; image=item['image'].unsqueeze(0).to(DEVICE); target=item['mask'].unsqueeze(0).to(DEVICE).float(); opt.zero_grad(set_to_none=True); lv=[]
            for j,prompt in enumerate(PROMPTS):
                emb=cache.get([prompt],DEVICE)
                logits=w(image,emb)
                loss=loss_fp32(logits,target[:,[j]])
                if not torch.isfinite(loss): raise RuntimeError(f'nonfinite loss step={len(steps)+1} case={item["case_id"]}')
                (loss/len(PROMPTS)).backward(); lv.append(float(loss.detach().cpu())); del emb,logits,loss
            ga,params=grad_stats(groups)
            # Some full decoder/segmentation parameters are structurally not on
            # the selected final-logit path (for example unused deep-supervision
            # heads). Record those inactive tensors, but fail only on an
            # existing non-finite gradient. This preserves the B3 contract
            # without silently treating legitimate unused parameters as a
            # numerical failure.
            if any(v['finite_grad_tensors'] < v['param_tensors'] - v['none_grad_tensors'] for v in ga.values()): raise RuntimeError(f'nonfinite gradient step={len(steps)+1}: {ga}')
            pre={g:v['grad_norm'] for g,v in ga.items()}; torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True); post,_=grad_stats(groups); before=1.0; opt.step(); after=1.0; step=len(steps)+1; steps.append({'step':step,'epoch':epoch,'case_id':item['case_id'],'loss':float(np.mean(lv)),'pre_clip_norm_json':json.dumps(pre),'post_clip_norm_json':json.dumps({g:v['grad_norm'] for g,v in post.items()}),'scaler_before':before,'scaler_after':after,'clip_triggered':any(post[g]['grad_norm']<pre[g]-1e-12 for g in post),'nan_inf':False,'skipped_step':False,'step_seconds':time.time()-es}); losses.extend(lv); del image,target; gc.collect(); torch.cuda.empty_cache()
            # Data-capacity follow-up contract: only the terminal 6000-step
            # validation is a formal performance node.  The already-running
            # 25pct_DRPA process has loaded the previous code and therefore
            # remains unchanged; this source affects only subsequently
            # launched runs.
            if step == a.max_steps and a.max_steps == 6000:
                vr=validate(w,val,val_ds,cache,step,out,a.model,len(train_ds)); vals.append(vr); save_state(w,opt,out/'checkpoints'/f'step_{step:05d}.pt',step,a.model,a.subset,config)
                if best is None or vr['mean_dice']>best['mean_dice']: best=vr
                pd.DataFrame(vals).to_csv(out/'validation_metrics.csv',index=False); print(json.dumps({'model':a.model,'subset':a.subset,'step':step,'validation':vr}),flush=True)
        epochs.append({'epoch':epoch,'steps_end':len(steps),'train_loss':float(np.mean(losses)) if losses else float('nan'),'elapsed_sec':time.time()-es,'equivalent_epoch':len(steps)/len(train_ds)}); pd.DataFrame(steps).to_csv(out/'training_dynamics.csv',index=False); pd.DataFrame(epochs).to_csv(out/'training_curve.csv',index=False)
    summary={'status':'COMPLETE','model':a.model,'subset':a.subset,'steps':len(steps),'train_visits':len(train),'train_ptids':int(train_frame.ptid.nunique()),'val_visits':len(val),'val_ptids':int(val_frame.ptid.nunique()),'best_validation':best,'total_wall_sec':time.time()-t0,'parameter_groups':counts}; (out/'run_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
