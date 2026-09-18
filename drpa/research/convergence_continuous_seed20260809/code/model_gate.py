"""Disposable initialization/backward check, no optimizer update or model save."""
import argparse,gc,json,time
import torch
import experiment as e
p=argparse.ArgumentParser();p.add_argument('--model',choices=list(e.COUNTS),required=True);a=p.parse_args()
e.shared.seed_all();start=time.time()
w,groups,params=e.make_model(a.model)
if a.model=='FullFT':w.model.train()
else:w.set_training_mode()
ds=e.shared.BilateralGroupedPatchDataset(e.shared.records(e.manifests(a.model)[0])[:1],e.shared.load_crop_spec(e.shared.PILOT/'crop_spec.json'),cache_cases=False)
item=ds[0];image=item['image'].unsqueeze(0).cuda();target=item['mask'].unsqueeze(0).cuda()
cache=e.shared.TextEmbeddingCache(e.shared.BANK,e.shared.MODEL,e.shared.CACHE if a.model!='FullFT' else None)
losses=[]
for pi,prompt in enumerate(e.shared.PROMPTS):
    logits=w(image,cache.get([prompt],torch.device('cuda')))
    loss,_,_=e.shared.loss_fp32(logits,target[:,[pi]])
    assert logits.dtype==torch.float32 and torch.isfinite(loss)
    (loss/8).backward();losses.append(float(loss.detach()));del logits,loss
norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
assert sum(p.numel() for p in params)==e.COUNTS[a.model]
missing=[n for n,p in w.model.named_parameters() if p.requires_grad and p.grad is None]
if a.model!='FullFT':assert not missing
record={'model':e.DISPLAY[a.model],'status':'PASS','optimizer_updates':0,'checkpoint_written':False,
        'configured_trainable':sum(p.numel() for p in params),'loss':sum(losses)/8,'gradient_norm':float(norm),
        'no_gradient_parameters':missing,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'elapsed_sec':time.time()-start,'learning_rates':{g['name']:g['lr'] for g in groups},
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
e.atomic_json(e.OUT/f'model_gate_{a.model}.json',record);print(json.dumps(record),flush=True)
