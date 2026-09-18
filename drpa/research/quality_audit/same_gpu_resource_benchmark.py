#!/usr/bin/env python3
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse, gc, json, os, sys, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE=Path(os.environ.get("MTL_MODEL_ROOT","__DRPA_WORKSPACE__"))
DEVICE=torch.device("cuda")
MODEL=str(BASE/"VoxTell_weights/voxtell_v1.1")
BANK=str(BASE/"VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz")
CACHE=str(BASE/"quality_audit/voxtell_mtl_peft/text_embedding_cache.npz")
TRAIN_MANIFEST=BASE/"quality_audit/voxtell_mtl_drpa8_full_data/full_data_train_cases.csv"
CROP=BASE/"quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
OUT=BASE/"quality_audit/same_gpu_resource_benchmark"
SEED=20260809

sys.path[:0]=[
 str(BASE/"VoxTell"),
 str(BASE/"quality_audit/voxtell_mtl_peft"),
 str(BASE/"quality_audit/voxtell_mtl_b1_bilateral_crop"),
 str(BASE/"quality_audit/voxtell_mtl_drpa8_pilot"),
 str(BASE/"quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound"),
 str(BASE/"quality_audit/voxtell_full_data_b3_vs_drpa_efficiency"),
 str(BASE/"scripts/fullft"),
]
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
from text_embedding_cache import TextEmbeddingCache
from voxtell_peft_wrapper import VoxTellPEFTWrapper
from drpa8_wrapper import DRPA8Wrapper
from b3_full_data_wrapper import B3FullDataWrapper
import fullft_runtime_gate as gate

def loss_fn(logits, target):
    z=logits.float(); y=target.float()
    bce=F.binary_cross_entropy_with_logits(z,y)
    p=torch.sigmoid(z); dims=tuple(range(2,p.ndim))
    inter=(p*y).sum(dims); den=p.sum(dims)+y.sum(dims)
    dice=1-((2*inter+1e-5)/(den+1e-5)).mean()
    return dice+bce

def make_model(kind):
    if kind=="B1":
        w=VoxTellPEFTWrapper(MODEL,BANK,rank=4,alpha=8,dropout=.05,device=DEVICE)
        w.model.eval()
        for layer in w.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()
        params=list(w.trainable_parameters()); forward=w
    elif kind=="DRPA-8":
        w=DRPA8Wrapper(MODEL,BANK,device=DEVICE); w.set_training_mode()
        params=[p for _,p,_ in w.trainable_parameter_groups()]; forward=w
    elif kind=="B3-Canonical":
        w=B3FullDataWrapper(MODEL,BANK,rank=4,alpha=8,dropout=.05,device=DEVICE)
        w.set_training_mode()
        params=[p for _,p,_ in w.trainable_parameter_groups()]; forward=w
    elif kind=="FullFT":
        model=gate.initialise_official_model(BASE).to(DEVICE)
        for p in model.parameters(): p.requires_grad_(True)
        model.train()
        class Wrap:
            def __call__(self,image,emb): return model(image,emb)
        w=Wrap(); params=[p for _,p in gate.unique_named_parameters(model) if p.requires_grad]; forward=w
    else: raise ValueError(kind)
    return w,forward,params

def nvidia_snapshot():
    q=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout.strip()
    vals=[x.strip() for x in q.split(",")]
    return {"gpu_util_pct":float(vals[0]),"gpu_used_mib":float(vals[1]),"gpu_total_mib":float(vals[2])}

def run_one(kind,item,cache,warmup,steps):
    wrapper,forward,params=make_model(kind)
    expected={"B1":294912,"DRPA-8":10969696,"B3-Canonical":81930624,"FullFT":440029541}[kind]
    actual=sum(p.numel() for p in params)
    if actual!=expected: raise RuntimeError(f"{kind} parameter mismatch: {actual} != {expected}")
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5)
    image=item["image"].unsqueeze(0).to(DEVICE,dtype=torch.float32)
    target=item["mask"].unsqueeze(0).to(DEVICE,dtype=torch.float32)
    for wi in range(warmup):
        opt.zero_grad(set_to_none=True)
        for j,prompt in enumerate(PROMPTS):
            emb=cache.get([prompt],DEVICE); logits=forward(image,emb); loss=loss_fn(logits,target[:,[j]])
            (loss/len(PROMPTS)).backward()
            del emb,logits,loss
        opt.step()
    torch.cuda.synchronize(DEVICE); torch.cuda.reset_peak_memory_stats(DEVICE)
    rows=[]
    for si in range(steps):
        opt.zero_grad(set_to_none=True)
        ft=bt=ot=0.0
        for j,prompt in enumerate(PROMPTS):
            emb=cache.get([prompt],DEVICE)
            torch.cuda.synchronize(DEVICE); t=time.perf_counter()
            logits=forward(image,emb)
            torch.cuda.synchronize(DEVICE); ft+=time.perf_counter()-t
            t=time.perf_counter(); loss=loss_fn(logits,target[:,[j]])
            finite=bool(torch.isfinite(loss).item())
            if not finite:
                raise RuntimeError("nonfinite benchmark loss")
            (loss/len(PROMPTS)).backward()
            torch.cuda.synchronize(DEVICE); bt+=time.perf_counter()-t
            del emb,logits,loss
        torch.cuda.synchronize(DEVICE); t=time.perf_counter(); opt.step(); torch.cuda.synchronize(DEVICE); ot=time.perf_counter()-t
        snap=nvidia_snapshot()
        rows.append({"model":kind,"step":si+1,"forward_sec":ft,"backward_sec":bt,"optimizer_sec":ot,"sec_per_step":ft+bt+ot,"gpu_util_pct":snap["gpu_util_pct"],"gpu_used_mib":snap["gpu_used_mib"],"peak_allocated_bytes":int(torch.cuda.max_memory_allocated(DEVICE)),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved(DEVICE)),"loss_finite":finite})
    df=pd.DataFrame(rows)
    result={"model":kind,"configured_trainable_params":actual,"warmup_steps":warmup,"measured_steps":steps,"peak_allocated_gib":float(df.peak_allocated_bytes.max()/2**30),"peak_reserved_gib":float(df.peak_reserved_bytes.max()/2**30),"mean_sec_per_step":float(df.sec_per_step.mean()),"median_sec_per_step":float(df.sec_per_step.median()),"samples_per_sec":float(1/df.sec_per_step.mean()),"mean_forward_sec":float(df.forward_sec.mean()),"mean_backward_sec":float(df.backward_sec.mean()),"mean_optimizer_sec":float(df.optimizer_sec.mean()),"mean_gpu_util_pct":float(df.gpu_util_pct.mean()),"all_loss_finite":bool(df.loss_finite.all())}
    df.to_csv(OUT/f"{kind.replace('-','_').lower()}_step_metrics.csv",index=False)
    del opt,image,target,wrapper,forward,params
    gc.collect(); torch.cuda.empty_cache()
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--warmup",type=int,default=5); ap.add_argument("--steps",type=int,default=100)
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    frame=pd.read_csv(TRAIN_MANIFEST,dtype=str).iloc[0]
    record=CaseRecord(str(frame.case_id),str(frame.image_path),str(frame.label_path))
    ds=BilateralGroupedPatchDataset([record],load_crop_spec(CROP),cache_cases=False)
    item=ds[0]; cache=TextEmbeddingCache(BANK,MODEL,CACHE)
    results=[]; order=["B1","DRPA-8","B3-Canonical","FullFT"]
    for kind in order:
        print(json.dumps({"event":"start","model":kind}),flush=True)
        results.append(run_one(kind,item,cache,args.warmup,args.steps))
        print(json.dumps(results[-1]),flush=True)
    pd.DataFrame(results).to_csv(OUT/"SAME_GPU_RESOURCE_BENCHMARK.csv",index=False)
    report=["# Same-GPU Resource Benchmark","","GPU: RTX PRO 6000; input: 192^3; batch=1; FP32; same case and prompt loop; no validation; warmup/measurement protocol recorded in config.json.","","| Model | Params | Peak alloc GiB | Peak reserved GiB | mean sec/step | median sec/step | samples/s | forward s | backward s | optimizer s | GPU util |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        report.append(f"| {r['model']} | {r['configured_trainable_params']:,} | {r['peak_allocated_gib']:.3f} | {r['peak_reserved_gib']:.3f} | {r['mean_sec_per_step']:.4f} | {r['median_sec_per_step']:.4f} | {r['samples_per_sec']:.6f} | {r['mean_forward_sec']:.4f} | {r['mean_backward_sec']:.4f} | {r['mean_optimizer_sec']:.4f} | {r['mean_gpu_util_pct']:.1f}% |")
    report += ["","This is a same-GPU runtime benchmark, not a validation-performance comparison. FullFT remains the high-capacity reference; no cross-GPU efficiency claim is made."]
    (OUT/"SAME_GPU_RESOURCE_BENCHMARK.md").write_text("\n".join(report)+"\n")
    (OUT/"config.json").write_text(json.dumps({"seed":SEED,"gpu":"RTX PRO 6000","input":"192x192x192","batch_size":1,"precision":"FP32","warmup_steps":args.warmup,"measured_steps":args.steps,"same_case":str(frame.case_id),"model_order":order,"no_validation":True},indent=2))
    print(json.dumps({"status":"COMPLETE","models":order},indent=2),flush=True)
if __name__=="__main__": main()
