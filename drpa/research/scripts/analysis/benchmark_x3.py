#!/usr/bin/env python3
"""Isolated, benchmark-only optimizer updates. Never saves model weights."""

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import traceback

MODELS = ['B1', 'DRPA-8', 'B3-Canonical', 'FullFT']
PARAMS = dict(zip(MODELS, [294912, 10969696, 81930624, 440029541]))
SEED = 20260809
WARMUP, MEASURED, REPEATS = 20, 100, 3

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def emit(event, **kw):
    print(json.dumps(dict(event=event, timestamp=time.time(), **kw)), flush=True)

def old_module(base):
    spec = importlib.util.spec_from_file_location('frozen_resource_factory', base/'quality_audit/same_gpu_resource_benchmark.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def gpu_processes():
    return subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()

def preflight(base, out):
    import numpy as np
    m = old_module(base)
    bank = m.TextEmbeddingCache(m.BANK, m.MODEL, m.CACHE)
    assert all(bank.contains(p) for p in m.PROMPTS), 'Missing prompt cache; Qwen fallback forbidden'
    assert len(m.PROMPTS) == 8
    files = [Path(m.MODEL)/'fold_0/checkpoint_final.pth', Path(m.MODEL)/'plans.json', Path(m.BANK), m.TRAIN_MANIFEST, m.CROP,
      base/'quality_audit/same_gpu_resource_benchmark.py', base/'scripts/fullft/fullft_runtime_gate.py',
      base/'quality_audit/voxtell_mtl_peft/voxtell_peft_wrapper.py',
      base/'quality_audit/voxtell_mtl_b1_bilateral_crop/b1_preprocessing.py',
      base/'quality_audit/voxtell_mtl_drpa8_pilot/drpa8_wrapper.py',
      base/'quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_full_data_wrapper.py',
      base/'quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound/voxtell_decoder_capacity_wrapper.py',
      base/'VoxTell/voxtell/model/voxtell_model.py', Path(__file__).resolve()]
    if Path(m.CACHE).is_file(): files.append(Path(m.CACHE))
    frame = m.pd.read_csv(m.TRAIN_MANIFEST, dtype=str).iloc[0]
    files.extend([Path(frame.image_path), Path(frame.label_path)])
    for f in files: assert f.is_file(), str(f)
    ds = m.BilateralGroupedPatchDataset([m.CaseRecord(str(frame.case_id), str(frame.image_path), str(frame.label_path))], m.load_crop_spec(m.CROP), cache_cases=False)
    item = ds[0]
    assert list(item['image'].shape)==[1,192,192,192]
    assert list(item['mask'].shape)==[8,192,192,192]
    assert item['image'].dtype == m.torch.float32 and item['mask'].dtype == m.torch.float32
    assert m.torch.isfinite(item['image']).all() and m.torch.isfinite(item['mask']).all()
    tensors = {k:hashlib.sha256(item[k].contiguous().numpy().tobytes()).hexdigest() for k in ('image','mask')}
    frozen = dict(status='PREFLIGHT_PASS', seed=SEED, models=MODELS, expected_parameters=PARAMS,
      warmup=WARMUP, measured=MEASURED, repeats=REPEATS, batch=1, input_shape=[1,1,192,192,192],
      precision='strict_FP32', autocast=False, grad_scaler=False, tf32=False,
      optimizer={'name':'AdamW','lr':1e-5,'weight_decay':1e-5,'betas':[.9,.999],'eps':1e-8,'foreach':None,'fused':None},
      seed_semantics='fresh process and canonical initialization, same seed each repeat; repeats measure runtime not scientific seed variation',
      step_definition='one visit, eight prompt-wise forward/loss/backward passes; average gradient then one AdamW step',
      data_loader={'num_workers':0,'batch_size':1,'pin_memory':True,'shuffle':False,'drop_last':False},
      data_source='REPEATED_FIXED_PREPROCESSED_CPU_TENSOR', preprocessing_timed=False,
      h2d='one float32 image and eight masks per update; non_blocking=True',
      checkpointing={'B1':'unchanged official forward','DRPA-8':'existing decoder checkpoint','B3-Canonical':'existing decoder checkpoint','FullFT':'no checkpointing'},
      qwen='not loaded; frozen precomputed canonical text embeddings; GPU text embeddings preloaded before warmup',
      optional_text_cache_exists=Path(m.CACHE).is_file(),all_prompts_available_in_existing_bank=True,
      train_manifest=str(m.TRAIN_MANIFEST), case_id=str(frame.case_id), ptid=str(frame.ptid), prompts=list(m.PROMPTS), tensor_sha256=tensors,
      torch=m.torch.__version__, numpy=np.__version__, cuda=m.torch.version.cuda,
      source_sha256={str(f):sha(f) for f in files})
    atomic(out/'PROTOCOL.json', frozen)
    emit('PREFLIGHT_PASS', case_id=str(frame.case_id), source_files=len(files))
    return frozen

def assert_sources(protocol):
    for f,h in protocol['source_sha256'].items():
        assert sha(f)==h, 'Source changed: '+f

def monitor(stop, rows):
    while not stop.is_set():
        try:
            r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
            v=r.stdout.strip().split(',')
            rows.append(dict(timestamp=time.time(),gpu_util_pct=float(v[0]),gpu_memory_mib=float(v[1]),temperature_c=float(v[2]),power_w=float(v[3])))
        except Exception as e: rows.append(dict(timestamp=time.time(),error=repr(e)))
        stop.wait(2)

def worker(base, out, model, repeat):
    import numpy as np
    import torch
    m=old_module(base)
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    run=out/f'repeat{repeat}_{model.replace("-","_")}'
    run.mkdir(exist_ok=False)
    atomic(run/'process.json',dict(pid=os.getpid(),ppid=os.getppid(),start_timestamp=time.time(),model=model,repeat=repeat))
    def on_signal(signum, frame):
        atomic(run/'signal.json',dict(signal=signum,timestamp=time.time(),pid=os.getpid()))
        raise SystemExit(128+signum)
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(s,on_signal)
    if gpu_processes(): raise RuntimeError('GPU occupied before model initialization')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert not torch.is_autocast_enabled()
    frame=m.pd.read_csv(m.TRAIN_MANIFEST,dtype=str).iloc[0]
    ds=m.BilateralGroupedPatchDataset([m.CaseRecord(str(frame.case_id),str(frame.image_path),str(frame.label_path))],m.load_crop_spec(m.CROP),cache_cases=False)
    item=ds[0]
    for k in ('image','mask'):
        assert hashlib.sha256(item[k].contiguous().numpy().tobytes()).hexdigest()==protocol['tensor_sha256'][k],k
    class FixedCase(torch.utils.data.Dataset):
        def __len__(self): return WARMUP+MEASURED
        def __getitem__(self,index): return item['image'],item['mask']
    loader=torch.utils.data.DataLoader(FixedCase(),batch_size=1,num_workers=0,pin_memory=True,shuffle=False)
    bank=m.TextEmbeddingCache(m.BANK,m.MODEL,m.CACHE)
    assert all(bank.contains(p) for p in m.PROMPTS)
    embeddings=[bank.get([p],m.DEVICE) for p in m.PROMPTS]
    wrapper,forward,params=m.make_model(model)
    assert len({id(p) for p in params})==len(params)
    assert sum(p.numel() for p in params)==PARAMS[model]
    assert all(p.requires_grad and p.dtype==torch.float32 for p in params)
    opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=1e-5)
    assert len({id(p) for g in opt.param_groups for p in g['params']})==len(params)
    atomic(run/'parameter_gate.json',dict(status='PASS',configured_trainable=PARAMS[model],unique_optimizer_parameter_tensors=len(params),wrapper_checkpointing=protocol['checkpointing'][model]))
    stop=threading.Event();telemetry=[]
    thread=threading.Thread(target=monitor,args=(stop,telemetry),daemon=True);thread.start()
    it=iter(loader); rows=[]; measured_start=None
    emit('MODEL_INITIALIZED',model=model,repeat=repeat,params=PARAMS[model])
    torch.cuda.synchronize()
    fh=(run/'step_metrics.csv').open('x',newline='');writer=None
    try:
      for step in range(1,WARMUP+MEASURED+1):
        if step==WARMUP+1:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();measured_start=time.time()
        torch.cuda.synchronize(); start=time.perf_counter();cpu0=time.process_time()
        t=start;im,gt=next(it);data_s=time.perf_counter()-t
        t=time.perf_counter();image=im.to(m.DEVICE,non_blocking=True);target=gt.to(m.DEVICE,non_blocking=True)
        torch.cuda.synchronize();h2d_s=time.perf_counter()-t
        opt.zero_grad(set_to_none=True);ft=lt=bt=0.;losses=[]
        for j,emb in enumerate(embeddings):
            torch.cuda.synchronize();t=time.perf_counter();logits=forward(image,emb);torch.cuda.synchronize();ft+=time.perf_counter()-t
            assert logits.dtype==torch.float32 and list(logits.shape)==[1,1,192,192,192]
            t=time.perf_counter();loss=m.gate.canonical_loss(logits,target[:,[j]]);torch.cuda.synchronize();lt+=time.perf_counter()-t
            lv=float(loss.detach());assert np.isfinite(lv),'nonfinite loss';losses.append(lv)
            t=time.perf_counter();(loss/8).backward();torch.cuda.synchronize();bt+=time.perf_counter()-t
            del logits,loss
        t=time.perf_counter()
        norms=torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in params if p.grad is not None])
        grad_norm=float(torch.linalg.vector_norm(norms));assert np.isfinite(grad_norm),'nonfinite gradient norm'
        check_s=time.perf_counter()-t
        torch.cuda.synchronize();t=time.perf_counter();opt.step();torch.cuda.synchronize();ot=time.perf_counter()-t
        end=time.perf_counter();wall=end-start
        row=dict(model=model,repeat=repeat,step=step,phase='warmup' if step<=WARMUP else 'measured',loss=float(np.mean(losses)),grad_norm=grad_norm,
          data_sec=data_s,h2d_sec=h2d_s,forward_sec=ft,loss_sec=lt,backward_sec=bt,optimizer_sec=ot,finite_check_sec=check_s,
          compute_sec=ft+lt+bt+ot,end_to_end_sec=wall,cpu_util_pct=100*(time.process_time()-cpu0)/wall,
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
          timestamp=time.time(),source=protocol['data_source'])
        if writer is None:writer=csv.DictWriter(fh,fieldnames=list(row));writer.writeheader()
        writer.writerow(row);fh.flush();rows.append(row)
        if step==1 or step%10==0:
            atomic(run/'progress.json',dict(status='RUNNING',model=model,repeat=repeat,step=step,total_steps=120,warmup_completed=min(step,20),measured_completed=max(step-20,0),last=row))
            emit('PROGRESS',model=model,repeat=repeat,step=step,measured_completed=max(step-20,0),sec_per_step=wall)
        del image,target,im,gt,norms
      measured_end=time.time()
      for p in params: assert torch.isfinite(p).all().item(),'nonfinite updated parameter'
      for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v) and v.is_floating_point():assert torch.isfinite(v).all().item(),'nonfinite optimizer state'
      df=m.pd.DataFrame(rows);d=df[df.phase=='measured'];assert len(d)==100
      stop.set();thread.join(timeout=12)
      m.pd.DataFrame(telemetry).to_csv(run/'telemetry.csv',index=False)
      util=[r['gpu_util_pct'] for r in telemetry if measured_start<=r['timestamp']<=measured_end and 'gpu_util_pct' in r]
      assert util,'No valid measured-window GPU telemetry'
      result=dict(status='PASS',model=model,repeat=repeat,configured_trainable_params=PARAMS[model],warmup_steps=20,measured_steps=100,
        peak_allocated_gib=float(d.peak_allocated_gib.max()),peak_reserved_gib=float(d.peak_reserved_gib.max()),
        mean_sec_per_step=float(d.compute_sec.mean()),median_sec_per_step=float(d.compute_sec.median()),samples_per_sec=1/float(d.compute_sec.mean()),
        end_to_end_sec_per_step=float(d.end_to_end_sec.mean()),end_to_end_samples_per_sec=1/float(d.end_to_end_sec.mean()),
        gpu_util_pct=float(np.mean(util)),median_gpu_util_pct=float(np.median(util)),cpu_util_pct=float(d.cpu_util_pct.mean()),
        forward_sec=float(d.forward_sec.mean()),loss_sec=float(d.loss_sec.mean()),backward_sec=float(d.backward_sec.mean()),optimizer_sec=float(d.optimizer_sec.mean()),
        data_sec=float(d.data_sec.mean()),h2d_sec=float(d.h2d_sec.mean()),finite_checks_pass=True,checkpoint_saved=False,
        worker_processes=0,measured_start=measured_start,measured_end=measured_end)
      atomic(run/'result.json',result);emit('RUN_PASS',**result)
    except BaseException as e:
      atomic(run/'failure_context.json',dict(error=repr(e),traceback=traceback.format_exc(),timestamp=time.time(),model=model,repeat=repeat,completed_steps=len(rows)))
      raise
    finally:fh.close();stop.set();thread.join(timeout=12)

def summarize(out):
    import numpy as np
    import pandas as pd
    results=[];allsteps=[]
    for rep in range(1,4):
      for model in MODELS:
        run=out/f'repeat{rep}_{model.replace("-","_")}'
        r=json.loads((run/'result.json').read_text());assert r['status']=='PASS'
        assert not (run/'failure_context.json').exists()
        exitinfo=json.loads((out/f'{run.name}.exit.json').read_text());assert exitinfo['exit_code']==0
        df=pd.read_csv(run/'step_metrics.csv');assert len(df)==120 and list(df.step)==list(range(1,121))
        d=df[df.phase=='measured'];assert len(d)==100
        assert np.isfinite(d.select_dtypes('number')).all().all()
        results.append(r);allsteps.append(df)
    df=pd.DataFrame(results);df.to_csv(out/'RUN_SUMMARY.csv',index=False)
    pd.concat(allsteps,ignore_index=True).to_csv(out/'BENCHMARK_RAW.csv',index=False)
    fields=['configured_trainable_params','peak_allocated_gib','peak_reserved_gib','mean_sec_per_step','median_sec_per_step','samples_per_sec',
      'end_to_end_sec_per_step','end_to_end_samples_per_sec','forward_sec','loss_sec','backward_sec','optimizer_sec','data_sec','h2d_sec','gpu_util_pct','median_gpu_util_pct','cpu_util_pct']
    rows=[]
    for model in MODELS:
      sub=df[df.model==model];assert len(sub)==3
      for k in fields:rows.append(dict(model=model,metric=k,mean=float(sub[k].mean()),sd=float(sub[k].std(ddof=1)),n_runs=3))
    summary=pd.DataFrame(rows);summary.to_csv(out/'BENCHMARK_X3_SUMMARY.csv',index=False)
    def ms(model,k):
        r=summary[(summary.model==model)&(summary.metric==k)].iloc[0]
        return f'{r["mean"]:.4f} ± {r["sd"]:.4f}'
    report=['# Same-GPU Resource Benchmark ×3','', 'Status: `SAME_GPU_RESOURCE_BENCHMARK_X3_COMPLETE`','',
      '## Frozen measurement contract','',
      '- RTX PRO 6000; batch=1;192³; strict FP32; TF32/autocast/GradScaler disabled; canonical model wrappers unchanged.',
      '- Three fresh processes per model, each initialized from canonical pretrained VoxTell with seed20260809,20 warm-up +100 measured optimizer updates.',
      '- One sample means one visit with eight sequential canonical prompt forwards/backwards and one averaged-gradient AdamW update. Not eight independent visits.',
      '- AdamW lr=1e-5,weight_decay=1e-5 (common resource-benchmark optimizer, not replacement of the models’ formal training LR groups).',
      '- Qwen is not loaded; frozen text embeddings are precomputed and resident. No validation or checkpoint saving.',
      '- DRPA and B3 retain their existing decoder checkpointing; B1/FullFT preserve their original forward paths. This measures actual implementations, not parameter count in isolation.',
      '- Same preprocessed real training case repeated through DataLoader(num_workers=0,pin_memory=True); no raw-MRI preprocessing or disk I/O is included. Case and tensor/source hashes are in PROTOCOL.json.',
      '- Compute sec/step=forward+loss+backward+optimizer, synchronized by CUDA. End-to-end warm-input sec/step also includes CPU collation,H2D,zero-grad and finite-check instrumentation. Both reported separately.',
      '- All CPU math-library thread limits and Torch threads are fixed across runs. Each process is isolated to prevent cross-model allocator carryover. Peak memory resets after warm-up, with initialized optimizer state retained.',
      '- GPU telemetry sampled every2s during measured window; CPU% is process CPU time/wall time (may exceed100%). SD is across3 run summaries, not300 independent steps.','',
      '| Model | Trainable params | Peak allocated GiB | Peak reserved GiB | Compute sec/step | Compute visits/s | Warm-input end-to-end sec/step |','|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:report.append(f'| {model} | {PARAMS[model]:,} | '+ ' | '.join(ms(model,k) for k in ['peak_allocated_gib','peak_reserved_gib','mean_sec_per_step','samples_per_sec','end_to_end_sec_per_step'])+' |')
    report+=['','## Interpretation limits','','`PARAMETER_EFFICIENCY != COMPUTE_EFFICIENCY`. Do not infer speed or memory ratios from trainable parameter ratios. Use measured same-GPU results; these are warm-input model training costs, not whole-cohort wall-clock costs.','',
      'Old single-run5-warmup benchmark remains historical and is not overwritten. Benchmark repetitions do not establish training-seed performance robustness.',
      '', 'Artifacts: PROTOCOL.json, RUN_SUMMARY.csv, BENCHMARK_X3_SUMMARY.csv, BENCHMARK_RAW.csv, per-run timing/telemetry/process/exit records. All12 gates passed; no formal artifact modified.','']
    (out/'SAME_GPU_RESOURCE_BENCHMARK.md').write_text('\n'.join(report))

def supervisor(base,out):
    out.mkdir(exist_ok=False,parents=True)
    atomic(out/'supervisor_process.json',dict(pid=os.getpid(),ppid=os.getppid(),start_timestamp=time.time()))
    child=None
    def on_signal(signum,frame):
        atomic(out/'supervisor_signal.json',dict(signal=signum,timestamp=time.time()))
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(s,on_signal)
    try:
      assert not gpu_processes(),'GPU occupied'
      protocol=preflight(base,out)
      completed=[]
      for rep in range(1,4):
        for model in MODELS:
          run=f'repeat{rep}_{model.replace("-","_")}'
          assert not gpu_processes(),'Unexpected concurrent GPU process'
          atomic(out/'STATE.json',dict(status='RUNNING',current=run,completed=completed,total_runs=12))
          env=os.environ.copy()
          env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',PYTHONUNBUFFERED='1')
          with (out/f'{run}.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','worker','--base',str(base),'--out',str(out),'--model',model,'--repeat',str(rep)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic(out/f'{run}.launch.json',dict(parent_pid=os.getpid(),child_pid=child.pid,start_timestamp=time.time()))
            emit('LAUNCH',model=model,repeat=rep,pid=child.pid)
            code=child.wait()
            atomic(out/f'{run}.exit.json',dict(exit_code=code,signal=(-code if code<0 else None),end_timestamp=time.time(),child_pid=child.pid))
          assert code==0,f'{run} exit={code}; no auto-retry'
          result=json.loads((out/run/'result.json').read_text());assert result['status']=='PASS'
          assert not (out/run/'failure_context.json').exists()
          completed.append(run);emit('ARTIFACT_GATE_PASS',run=run,completed=len(completed))
      assert_sources(protocol);summarize(out)
      atomic(out/'STATE.json',dict(status='SAME_GPU_RESOURCE_BENCHMARK_X3_COMPLETE',completed=completed,total_runs=12,source_hashes_unchanged=True))
    except BaseException as e:
      atomic(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),timestamp=time.time()))
      atomic(out/'STATE.json',dict(status='BLOCKED',error=repr(e)))
      raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['run','worker','preflight','summarize'],required=True)
    ap.add_argument('--base',type=Path,default=Path('__DRPA_WORKSPACE__'));ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--model',choices=MODELS);ap.add_argument('--repeat',type=int,choices=[1,2,3]);a=ap.parse_args()
    if a.mode=='run':supervisor(a.base,a.out)
    elif a.mode=='worker':worker(a.base,a.out,a.model,a.repeat)
    elif a.mode=='preflight':a.out.mkdir(exist_ok=False,parents=True);preflight(a.base,a.out)
    else:summarize(a.out)

if __name__=='__main__':main()
