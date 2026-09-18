#!/usr/bin/env python3
"""Step-controlled DRPA-8 full-data formal training."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import gc
import json
import math
import random
import subprocess
import sys
import time
import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = Path(os.environ.get('MTL_MODEL_ROOT', '__DRPA_WORKSPACE__'))
OUT = BASE / 'quality_audit/voxtell_mtl_drpa8_full_data'
FORMAL = OUT / 'formal_training'
PILOT = BASE / 'quality_audit/voxtell_mtl_drpa8_pilot'
PEFT = BASE / 'quality_audit/voxtell_mtl_peft'
PREP = BASE / 'quality_audit/voxtell_mtl_b1_bilateral_crop'
sys.path[:0] = [str(PILOT), str(PEFT), str(PREP)]

from b1_preprocessing import (  # noqa: E402
    BilateralGroupedPatchDataset,
    CaseRecord,
    PROMPTS,
    load_crop_spec,
)
from text_embedding_cache import TextEmbeddingCache  # noqa: E402
from drpa8_wrapper import DRPA8Wrapper  # noqa: E402
import evaluate_drpa8 as evaluator  # noqa: E402

SEED = 20260809
MAX_STEPS = 6000
# Formal validation is intentionally sparse: one full validation at step 3000
# and one final validation at step 6000. This is a protocol choice, not a
# change to the optimizer or model contract.
VALIDATION_STEPS = (3000, 6000)
VALIDATION_INTERVAL = 3000
DECODER_LR = 1e-5
PROJECTION_LR = 1e-4
LORA_LR = 1e-4
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
DEVICE = torch.device('cuda')
MODEL = str(BASE / 'VoxTell_weights/voxtell_v1.1')
BANK = str(BASE / 'VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz')
CACHE = str(PEFT / 'text_embedding_cache.npz')
EXPECTED_GROUPS = {
    'cross_attention_lora': 294_912,
    'projection_adapter': 442_624,
    'decoder_stages': 10_232_160,
}
EXPECTED_TOTAL = sum(EXPECTED_GROUPS.values())
RUN_MAX_STEPS = MAX_STEPS
RUN_CACHE_CASES = True
RUN_OUTPUT = FORMAL
PROFILE_ROWS = []
PROFILE_START = time.perf_counter()
PROFILE_LAST_READ = 0


def process_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    return float(line.split()[1]) / 1024.0
        except Exception:
            pass
        return 0.0


def read_bytes() -> int:
    try:
        for line in Path('/proc/self/io').read_text().splitlines():
            if line.startswith('read_bytes:'):
                return int(line.split()[1])
    except Exception:
        pass
    return 0


def gpu_vram_mb() -> float:
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=False,
        ).stdout.strip().splitlines()
        return float(out[0]) if out else 0.0
    except Exception:
        return 0.0


def shm_usage_mb() -> float:
    try:
        stat = os.statvfs('/dev/shm')
        used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
        return used / 2**20
    except Exception:
        return 0.0


def cache_lifecycle(event, record, **extra):
    PROFILE_ROWS.append({
        'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
        'event': event, 'case_id': record.case_id,
        'elapsed_sec': time.perf_counter() - PROFILE_START,
        'rss_mb': process_rss_mb(), 'read_bytes': read_bytes(), **extra,
    })


def write_profile():
    if PROFILE_ROWS:
        pd.DataFrame(PROFILE_ROWS).to_csv(RUN_OUTPUT / 'full_data_cache_profile.csv', index=False)


def write_step_rows(rows):
    pd.DataFrame(rows).to_csv(RUN_OUTPUT / 'drpa8_full_gradient_summary.csv', index=False)


def write_runtime_curve(rows):
    pd.DataFrame(rows).to_csv(RUN_OUTPUT / 'drpa8_full_runtime_curve.csv', index=False)


def phase(event: str, step: int, case_id: str = ''):
    print(json.dumps({'event': event, 'step': step, 'case_id': case_id,
                      'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
                      'rss_mb': process_rss_mb(), 'read_bytes': read_bytes()},
                     sort_keys=True), flush=True)


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_if_gpu_busy() -> None:
    active = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
         check=False,
    ).stdout.strip()
    if active:
        raise RuntimeError(f'GPU has active compute processes; refusing full-data training: {active}')


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path)
    return [CaseRecord(str(row.case_id), str(row.image_path), str(row.label_path))
            for row in frame.itertuples()]


def loss_fp32(logits: torch.Tensor, target: torch.Tensor):
    z = logits.float()
    y = target.float()
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z)
    dims = tuple(range(2, p.ndim))
    inter = (p * y).sum(dims)
    den = p.sum(dims) + y.sum(dims)
    dice_loss = 1.0 - ((2 * inter + 1e-5) / (den + 1e-5)).mean()
    return dice_loss + bce, dice_loss, bce


def grouped_parameters(wrapper: DRPA8Wrapper):
    groups = {key: [] for key in EXPECTED_GROUPS}
    audit = []
    seen: set[int] = set()
    for name, param, group in wrapper.trainable_parameter_groups():
        if group not in groups:
            raise RuntimeError(f'unexpected trainable group {group}: {name}')
        if id(param) in seen:
            raise RuntimeError(f'duplicate optimizer parameter: {name}')
        if not param.requires_grad:
            raise RuntimeError(f'trainable parameter lacks requires_grad: {name}')
        seen.add(id(param))
        groups[group].append(param)
        audit.append({
            'module': name,
            'group': group,
            'parameter_count': int(param.numel()),
            'requires_grad': bool(param.requires_grad),
        })
    counts = {key: sum(p.numel() for p in params) for key, params in groups.items()}
    if counts != EXPECTED_GROUPS or sum(counts.values()) != EXPECTED_TOTAL:
        raise RuntimeError(f'formal DRPA-8 trainable parameter mismatch: {counts}')
    trainable_ids = {id(p) for params in groups.values() for p in params}
    unexpected = [name for name, p in wrapper.model.named_parameters()
                  if p.requires_grad and id(p) not in trainable_ids]
    if unexpected:
        raise RuntimeError(f'unexpected trainable/frozen-contract violation: {unexpected[:5]}')
    return groups, audit, counts


def gradient_stats(groups):
    output = {}
    for group, params in groups.items():
        finite = nonzero = none = 0
        square_sum = 0.0
        for param in params:
            if param.grad is None:
                none += 1
                continue
            if torch.isfinite(param.grad).all():
                finite += 1
            if torch.count_nonzero(param.grad).item() > 0:
                nonzero += 1
            square_sum += float(torch.sum(param.grad.float() ** 2).cpu())
        output[group] = {
            'grad_tensors': len(params),
            'finite_grad_tensors': finite,
            'nonzero_grad_tensors': nonzero,
            'none_grad_tensors': none,
            'grad_norm': math.sqrt(square_sum),
        }
    return output


def effective_rank(values: torch.Tensor) -> float:
    values = values.float()
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def projection_rank_rows(wrapper: DRPA8Wrapper, step: int, equivalent_epoch: float):
    rows = []
    with torch.no_grad():
        for pidx, projection in enumerate(wrapper.model.project_to_decoder_channels):
            for child_name, child in projection.named_modules():
                if not isinstance(child, torch.nn.Linear):
                    continue
                adapter = child.parametrizations.weight[0]
                delta = adapter.delta_matrix().detach().float()
                singular = torch.linalg.svdvals(delta)
                base = child.parametrizations.weight.original.detach().float()
                rows.append({
                    'step': step,
                    'equivalent_epoch': equivalent_epoch,
                    'projection': f'project_to_decoder_channels.{pidx}.{child_name}',
                    'configured_rank': int(adapter.rank),
                    'effective_rank': effective_rank(singular),
                    'stable_rank': float((singular.square().sum() /
                                          singular.square().max().clamp_min(1e-12)).cpu()),
                    'delta_l2': float(delta.norm().cpu()),
                    'relative_update': float((delta.norm() /
                                               base.norm().clamp_min(1e-12)).cpu()),
                    'A_norm': float(adapter.lora_A.detach().float().norm().cpu()),
                    'B_norm': float(adapter.lora_B.detach().float().norm().cpu()),
                })
    return rows


def validation_metrics(wrapper, val, val_ds, cache, step: int, steps_per_epoch: int):
    label = f'DRPA-8-full@step{step:05d}'
    rows = evaluator.evaluate_condition(label, wrapper, val, val_ds, cache)
    frame = pd.DataFrame(rows)
    summary = evaluator.summarize(frame)
    summary.insert(1, 'step', step)
    summary.insert(2, 'equivalent_epoch', step / steps_per_epoch)
    raw = summary[summary.lcc == 0]
    overall = raw[raw.scope == 'overall'].iloc[0]
    get = lambda scope, field: float(raw[raw.scope == scope].iloc[0][field])
    event = {
        'step': step,
        'equivalent_epoch': step / steps_per_epoch,
        'val_mean_dice': float(overall.dice),
        'val_hd95_mm': float(overall.hd95_mm),
        'val_surface_dice': float(overall.surface_dice_2mm),
        'val_components': float(overall.connected_components),
        'val_fp_volume_ml': float(overall.false_positive_volume_ml),
        'val_max_fp_distance_mm': float(overall.max_false_positive_distance_mm),
        'val_empty_mask_rate': float(overall.empty_mask_rate),
        'val_hippocampus_dice': get('hippocampus', 'dice'),
        'val_entorhinal_cortex_dice': get('entorhinal cortex', 'dice'),
        'val_parahippocampal_gyrus_dice': get('parahippocampal gyrus', 'dice'),
        'val_amygdala_dice': get('amygdala', 'dice'),
        'val_hippocampus_hd95_mm': get('hippocampus', 'hd95_mm'),
        'val_entorhinal_cortex_hd95_mm': get('entorhinal cortex', 'hd95_mm'),
        'val_parahippocampal_gyrus_hd95_mm': get('parahippocampal gyrus', 'hd95_mm'),
        'val_amygdala_hd95_mm': get('amygdala', 'hd95_mm'),
    }
    return event, frame, summary


def better_checkpoint(current, best) -> bool:
    if best is None:
        return True
    delta = current['val_mean_dice'] - best['val_mean_dice']
    if abs(delta) >= 0.001:
        return delta > 0
    for key, direction in [
        ('val_entorhinal_cortex_dice', 1),
        ('val_parahippocampal_gyrus_dice', 1),
        ('val_hd95_mm', -1),
        ('val_fp_volume_ml', -1),
    ]:
        if abs(current[key] - best[key]) > 1e-12:
            return direction * (current[key] - best[key]) > 0
    return False


def plateau_detected(events) -> bool:
    if len(events) < 4 or events[-1]['step'] < 3000:
        return False
    last = events[-4:]
    dice_deltas = [last[i + 1]['val_mean_dice'] - last[i]['val_mean_dice'] for i in range(3)]
    ec_deltas = [last[i + 1]['val_entorhinal_cortex_dice'] - last[i]['val_entorhinal_cortex_dice'] for i in range(3)]
    phg_deltas = [last[i + 1]['val_parahippocampal_gyrus_dice'] - last[i]['val_parahippocampal_gyrus_dice'] for i in range(3)]
    hd_deltas = [last[i + 1]['val_hd95_mm'] - last[i]['val_hd95_mm'] for i in range(3)]
    fp_deltas = [last[i + 1]['val_fp_volume_ml'] - last[i]['val_fp_volume_ml'] for i in range(3)]
    return (max(dice_deltas) < 0.001 and max(ec_deltas) < 0.001 and
            max(phg_deltas) < 0.001 and min(hd_deltas) > -0.001 and
            min(fp_deltas) > -0.001)


def write_report(events, counts, rank_frame, grad_frame, checkpoint_frame, plateau):
    best = max(events, key=lambda row: row['val_mean_dice']) if events else None
    final = events[-1] if events else None
    if best is None:
        (RUN_OUTPUT / 'DRPA8_FULL_DATA_TRAINING_REPORT.md').write_text(
            '# DRPA-8 Full-data Cache Audit Run Report\n\n'
            f'- Completed: `{len(grad_frame)}/{RUN_MAX_STEPS}` optimizer steps\n'
            '- Validation points: `0` (this run used the configured sparse validation schedule)\n'
            f'- cache_cases: `{RUN_CACHE_CASES}`\n'
            '- This short run has no validation performance result.\n'
        )
        return
    pilot_best = 0.7803389563180912
    recommendation = 'B3-stable full-data under the same 6000-step contract' if best['val_mean_dice'] >= pilot_best else 'STOP; do not start another experiment until the full-data result is reviewed'
    # The formal run records per-group norms and a step-level nan_inf flag;
    # it does not emit finite-gradient-tensor count columns.  Do not infer
    # tensor counts from absent columns: use the fields that are actually in
    # the gradient audit CSV.
    finite_grad_steps = int((~grad_frame['nan_inf'].astype(bool)).sum()) if not grad_frame.empty else 0
    skipped = int(grad_frame['skipped_optimizer_step'].sum()) if not grad_frame.empty else -1
    report = f'''# DRPA-8 Full-data Formal Training Report

## Contract

- Train: `971 visits / 337 PTIDs`; validation: `247 visits / 85 PTIDs`; PTID overlap `0`.
- Batch size `1`, gradient accumulation `1`, `971` optimizer steps per complete epoch.
- Fresh official VoxTell v1.1 initialization; no pilot/B2/B3 checkpoint or B3 SVD initialization.
- DRPA rank `8`; cross-attention LoRA rank `4`; LoRA/projection LR `1e-4`; decoder LR `1e-5`.
- AdamW weight decay `1e-5`; FP32 Dice+BCE; AMP; unscale then clip `1.0`; fixed LR; seed `20260809`.

## Completion

- Completed: **{final['step']}/6000 optimizer steps**
- Best step: **{best['step']}**
- Best equivalent epoch: **{best['equivalent_epoch']:.6f}**
- Best Raw Dice: **{best['val_mean_dice']:.6f}**
- Final Raw Dice: **{final['val_mean_dice']:.6f}**
- Best-vs-50/12 DRPA-8 pilot Dice change: **{best['val_mean_dice'] - pilot_best:+.6f}**

## Best/final metrics

| checkpoint | Dice | HD95 (mm) | Surface Dice | Hipp | EC | PHG | Amy | components | FP volume (ml) | max FP distance (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best step {best['step']} | {best['val_mean_dice']:.6f} | {best['val_hd95_mm']:.6f} | {best['val_surface_dice']:.6f} | {best['val_hippocampus_dice']:.6f} | {best['val_entorhinal_cortex_dice']:.6f} | {best['val_parahippocampal_gyrus_dice']:.6f} | {best['val_amygdala_dice']:.6f} | {best['val_components']:.4f} | {best['val_fp_volume_ml']:.6f} | {best['val_max_fp_distance_mm']:.6f} |
| final step {final['step']} | {final['val_mean_dice']:.6f} | {final['val_hd95_mm']:.6f} | {final['val_surface_dice']:.6f} | {final['val_hippocampus_dice']:.6f} | {final['val_entorhinal_cortex_dice']:.6f} | {final['val_parahippocampal_gyrus_dice']:.6f} | {final['val_amygdala_dice']:.6f} | {final['val_components']:.4f} | {final['val_fp_volume_ml']:.6f} | {final['val_max_fp_distance_mm']:.6f} |

## Stability and effective rank

- Gradient/loss finite audit steps: `{finite_grad_steps}/{len(grad_frame) if not grad_frame.empty else 0}`.
- Skipped optimizer steps: `{skipped}`.
- NaN/Inf events: `{int(grad_frame.nan_inf.sum()) if not grad_frame.empty else -1}`.
- Projection effective-rank curve: `drpa8_full_effective_rank_curve.csv`.
- Final projection effective rank: mean `{rank_frame[rank_frame.step == rank_frame.step.max()].effective_rank.mean():.4f}`, range `{rank_frame[rank_frame.step == rank_frame.step.max()].effective_rank.min():.4f}–{rank_frame[rank_frame.step == rank_frame.step.max()].effective_rank.max():.4f}`.
- Plateau status: **{'PLATEAU_DETECTED' if plateau else 'NOT_DETECTED'}**. Training was not early-stopped.

## Next experiment recommendation

Only one candidate is recorded: **{recommendation}**. It was not started automatically.
'''
    (RUN_OUTPUT / 'DRPA8_FULL_DATA_TRAINING_REPORT.md').write_text(report)


def main() -> None:
    global RUN_MAX_STEPS, RUN_CACHE_CASES, RUN_OUTPUT, PROFILE_START
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--cache-cases', choices=['true', 'false'], default='true')
    parser.add_argument('--output-dir', default=str(FORMAL))
    args = parser.parse_args()
    RUN_MAX_STEPS = int(args.max_steps)
    RUN_CACHE_CASES = args.cache_cases == 'true'
    RUN_OUTPUT = Path(args.output_dir)
    RUN_OUTPUT.mkdir(parents=True, exist_ok=True)
    PROFILE_START = time.perf_counter()
    refuse_if_gpu_busy()
    seed_all()
    FORMAL.mkdir(parents=True, exist_ok=True)
    (RUN_OUTPUT / 'checkpoints').mkdir(parents=True, exist_ok=True)
    train_path = OUT / 'full_data_train_cases.csv'
    val_path = OUT / 'full_data_val_cases.csv'
    train = records(train_path)
    val = records(val_path)
    if len(train) != 971 or len(val) != 247:
        raise RuntimeError(f'formal manifest count mismatch train={len(train)} val={len(val)}')
    train_ds = BilateralGroupedPatchDataset(train, load_crop_spec(PILOT / 'crop_spec.json'), cache_cases=RUN_CACHE_CASES, lifecycle_hook=cache_lifecycle)
    val_ds = BilateralGroupedPatchDataset(val, load_crop_spec(PILOT / 'crop_spec.json'), cache_cases=RUN_CACHE_CASES)
    cache = TextEmbeddingCache(BANK, MODEL, CACHE)
    steps_per_epoch = len(train_ds)
    if steps_per_epoch != 971:
        raise RuntimeError(f'dataloader length mismatch: {steps_per_epoch}')

    wrapper = DRPA8Wrapper(MODEL, BANK, device=DEVICE)
    wrapper.set_training_mode()
    groups, audit, counts = grouped_parameters(wrapper)
    print(json.dumps({'trainable_parameter_groups': counts,
                      'total_trainable_parameters': sum(counts.values()),
                      'expected_total': EXPECTED_TOTAL}), flush=True)
    pd.DataFrame(audit).to_csv(RUN_OUTPUT / 'full_trainable_parameter_audit.csv', index=False)
    (RUN_OUTPUT / 'initialization_audit.json').write_text(json.dumps({
        'base_initialization': 'official VoxTell v1.1 fresh; no pilot/B2/B3/SVD resume',
        'train_visits': len(train), 'train_ptids': 337,
        'val_visits': len(val), 'val_ptids': 85, 'ptid_overlap': 0,
        'batch_size_visits': 1, 'gradient_accumulation_steps': 1,
        'steps_per_epoch': steps_per_epoch,
        'max_optimizer_steps': RUN_MAX_STEPS,
        'cache_cases': RUN_CACHE_CASES,
        'validation_steps': list(VALIDATION_STEPS),
        'parameter_groups': counts,
        'total_trainable_parameters': sum(counts.values()),
        'scheduler': 'none_constant_group_learning_rates',
    }, indent=2))
    wrapper.save_checkpoint(str(RUN_OUTPUT / 'checkpoints' / 'initial_fresh.pt'), {
        'experiment': 'VOXTELL_MTL_DRPA8_FULL_DATA',
        'initialization': 'official v1.1 fresh; no resume',
    })

    optimizer = torch.optim.AdamW([
        {'params': groups['cross_attention_lora'], 'lr': LORA_LR},
        {'params': groups['projection_adapter'], 'lr': PROJECTION_LR},
        {'params': groups['decoder_stages'], 'lr': DECODER_LR},
    ], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    global_step = 0
    epoch = 0
    events = []
    curve_rows = []
    rank_rows = []
    grad_rows = []
    runtime_rows = []
    validation_rows = []
    roi_rows = []
    checkpoint_rows = []
    interval_losses, interval_dice_losses, interval_bces = [], [], []
    plateau = False
    best = None
    # Persist an explicit process-start marker so a hard external termination
    # cannot be mistaken for a completed or resumable training state.
    (RUN_OUTPUT / 'run_status.json').write_text(json.dumps({
        'status': 'RUNNING', 'optimizer_steps_completed': 0,
        'max_optimizer_steps': RUN_MAX_STEPS, 'resume_allowed': False,
        'cache_cases': RUN_CACHE_CASES,
    }, indent=2))

    while global_step < RUN_MAX_STEPS:
        epoch += 1
        wrapper.set_training_mode()
        order = np.random.default_rng(SEED + epoch).permutation(len(train_ds))
        for index in order:
            if global_step >= RUN_MAX_STEPS:
                break
            data_start = time.perf_counter()
            phase('data_fetch_start', global_step + 1)
            item = train_ds[int(index)]
            data_elapsed = time.perf_counter() - data_start
            phase('data_fetch_complete', global_step + 1, item['case_id'])
            image = item['image'].unsqueeze(0).to(DEVICE)
            target = item['mask'].unsqueeze(0).to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)
            forward_elapsed = 0.0
            backward_elapsed = 0.0
            losses, dice_losses, bces = [], [], []
            for prompt_idx, prompt in enumerate(PROMPTS):
                embedding = cache.get([prompt], DEVICE)
                forward_start = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = wrapper(image, embedding)
                forward_elapsed += time.perf_counter() - forward_start
                with torch.autocast('cuda', enabled=False):
                    loss, dice_loss, bce = loss_fp32(logits, target[:, [prompt_idx]])
                finite_loss = bool(torch.isfinite(loss).item())
                if not finite_loss:
                    raise RuntimeError(f'nonfinite loss at step={global_step + 1} case={item["case_id"]}')
                backward_start = time.perf_counter()
                scaler.scale(loss / len(PROMPTS)).backward()
                backward_elapsed += time.perf_counter() - backward_start
                losses.append(float(loss.detach().cpu()))
                dice_losses.append(float(dice_loss.detach().cpu()))
                bces.append(float(bce.detach().cpu()))
                del embedding, logits, loss
            scaler.unscale_(optimizer)
            pre = gradient_stats(groups)
            grad_finite = all(v['finite_grad_tensors'] + v['none_grad_tensors'] == v['grad_tensors'] for v in pre.values())
            grad_nonzero = all(v['nonzero_grad_tensors'] > 0 for v in pre.values())
            if not grad_finite or not grad_nonzero:
                raise RuntimeError(f'gradient gate failed at step={global_step + 1}: {pre}')
            clip_value = torch.nn.utils.clip_grad_norm_(
                [p for params in groups.values() for p in params], CLIP_NORM,
                error_if_nonfinite=True,
            )
            post = gradient_stats(groups)
            scale_before = float(scaler.get_scale())
            optimizer_start = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer_elapsed = time.perf_counter() - optimizer_start
            phase('optimizer_complete_before_step_increment', global_step + 1, item['case_id'])
            scale_after = float(scaler.get_scale())
            scaler_backoff = scale_after < scale_before
            if scaler_backoff:
                raise RuntimeError(f'GradScaler backoff at step={global_step + 1}')
            global_step += 1
            if global_step == 1:
                (RUN_OUTPUT / 'run_status.json').write_text(json.dumps({
                    'status': 'RUNNING', 'optimizer_steps_completed': 1,
                    'max_optimizer_steps': RUN_MAX_STEPS, 'resume_allowed': False,
                    'cache_cases': RUN_CACHE_CASES,
                }, indent=2))
            clip_triggered = bool(float(clip_value) > CLIP_NORM)
            grad_rows.append({
                'step': global_step,
                'epoch': epoch,
                'equivalent_epoch': global_step / steps_per_epoch,
                'case_id': item['case_id'],
                'loss': float(np.mean(losses)),
                'dice_loss': float(np.mean(dice_losses)),
                'bce_loss': float(np.mean(bces)),
                'lora_pre_clip_norm': pre['cross_attention_lora']['grad_norm'],
                'projection_pre_clip_norm': pre['projection_adapter']['grad_norm'],
                'decoder_pre_clip_norm': pre['decoder_stages']['grad_norm'],
                'lora_post_clip_norm': post['cross_attention_lora']['grad_norm'],
                'projection_post_clip_norm': post['projection_adapter']['grad_norm'],
                'decoder_post_clip_norm': post['decoder_stages']['grad_norm'],
                'pre_clip_total_norm': float(clip_value),
                'post_clip_total_norm': float(math.sqrt(sum(v['grad_norm'] ** 2 for v in post.values()))),
                'clip_triggered': clip_triggered,
                'scaler_before': scale_before,
                'scaler_after': scale_after,
                'scaler_backoff': scaler_backoff,
                'nan_inf': False,
                'skipped_case': False,
                'skipped_optimizer_step': False,
                'data_fetch_sec': data_elapsed,
                'forward_sec': forward_elapsed,
                'backward_sec': backward_elapsed,
                'optimizer_sec': optimizer_elapsed,
                'total_step_sec': data_elapsed + forward_elapsed + backward_elapsed + optimizer_elapsed,
                'rss_mb': process_rss_mb(),
                'read_bytes': read_bytes(),
                'cache_cases': RUN_CACHE_CASES,
            })
            write_step_rows(grad_rows)
            phase('step_record_flushed', global_step, item['case_id'])
            interval_losses.extend(losses)
            interval_dice_losses.extend(dice_losses)
            interval_bces.extend(bces)
            if global_step in VALIDATION_STEPS:
                wrapper.model.eval()
                event, case_frame, summary = validation_metrics(wrapper, val, val_ds, cache, global_step, steps_per_epoch)
                event.update({
                    'train_loss': float(np.mean(interval_losses)),
                    'train_dice_loss': float(np.mean(interval_dice_losses)),
                    'train_bce_loss': float(np.mean(interval_bces)),
                    'clip_trigger_rate': float(np.mean([r['clip_triggered'] for r in grad_rows if r['step'] > global_step - VALIDATION_INTERVAL and r['step'] <= global_step])),
                    'grad_scaler_min': float(min(r['scaler_after'] for r in grad_rows if r['step'] > global_step - VALIDATION_INTERVAL and r['step'] <= global_step)),
                    'grad_scaler_max': float(max(r['scaler_after'] for r in grad_rows if r['step'] > global_step - VALIDATION_INTERVAL and r['step'] <= global_step)),
                    'plateau_status': 'PLATEAU_DETECTED' if plateau_detected(events + [event]) else 'NOT_DETECTED',
                })
                events.append(event)
                plateau = plateau or event['plateau_status'] == 'PLATEAU_DETECTED'
                event['plateau_status'] = 'PLATEAU_DETECTED' if plateau else 'NOT_DETECTED'
                curve_rows.append(event)
                validation_rows.extend(summary.to_dict('records'))
                roi_rows.extend(summary[summary.scope != 'overall'].to_dict('records'))
                rank_rows.extend(projection_rank_rows(wrapper, global_step, event['equivalent_epoch']))
                pd.DataFrame(curve_rows).to_csv(RUN_OUTPUT / 'drpa8_full_training_curve.csv', index=False)
                pd.DataFrame(validation_rows).to_csv(RUN_OUTPUT / 'drpa8_full_validation_metrics.csv', index=False)
                pd.DataFrame(roi_rows).to_csv(RUN_OUTPUT / 'drpa8_full_roi_metrics.csv', index=False)
                pd.DataFrame(rank_rows).to_csv(RUN_OUTPUT / 'drpa8_full_effective_rank_curve.csv', index=False)
                pd.DataFrame(grad_rows).to_csv(RUN_OUTPUT / 'drpa8_full_gradient_summary.csv', index=False)
                interval = pd.DataFrame([r for r in grad_rows if r['step'] > global_step - VALIDATION_INTERVAL and r['step'] <= global_step])
                runtime_event = {
                    **event,
                    'train_data_fetch_median_sec': float(interval.data_fetch_sec.median()),
                    'train_data_fetch_p90_sec': float(interval.data_fetch_sec.quantile(.9)),
                    'train_total_step_median_sec': float(interval.total_step_sec.median()),
                    'train_total_step_p90_sec': float(interval.total_step_sec.quantile(.9)),
                    'train_rss_median_mb': float(interval.rss_mb.median()),
                    'train_rss_p90_mb': float(interval.rss_mb.quantile(.9)),
                    'gpu_vram_mb': gpu_vram_mb(),
                    'shm_used_mb': shm_usage_mb(),
                }
                runtime_rows.append(runtime_event)
                write_runtime_curve(runtime_rows)
                checkpoint_name = f'step_{global_step:05d}.pt'
                metadata = {**event, 'experiment': 'VOXTELL_MTL_DRPA8_FULL_DATA', 'optimizer_step': global_step,
                            'equivalent_epoch': event['equivalent_epoch'], 'selection': 'scheduled validation checkpoint',
                            'trainable_parameters': EXPECTED_TOTAL, 'scheduler': 'none'}
                wrapper.save_checkpoint(str(RUN_OUTPUT / 'checkpoints' / checkpoint_name), metadata)
                checkpoint_rows.append({**event, 'checkpoint': str(RUN_OUTPUT / 'checkpoints' / checkpoint_name), 'selection': 'scheduled_validation'})
                if better_checkpoint(event, best):
                    best = event.copy()
                    wrapper.save_checkpoint(str(RUN_OUTPUT / 'checkpoints' / 'validation_best.pt'), {**metadata, 'selection': 'validation_best_raw_mean_dice'})
                    checkpoint_rows.append({**event, 'checkpoint': str(RUN_OUTPUT / 'checkpoints' / 'validation_best.pt'), 'selection': 'validation_best_update'})
                pd.DataFrame(checkpoint_rows).to_csv(RUN_OUTPUT / 'drpa8_full_checkpoint_manifest.csv', index=False)
                interval_losses, interval_dice_losses, interval_bces = [], [], []
                print(json.dumps(event), flush=True)
                wrapper.set_training_mode()
            del image, target
            phase('cleanup_start', global_step, item['case_id'])
            gc.collect()
            torch.cuda.empty_cache()
            phase('cleanup_complete', global_step, item['case_id'])
        if global_step >= RUN_MAX_STEPS:
            break

    pd.DataFrame(curve_rows).to_csv(RUN_OUTPUT / 'drpa8_full_training_curve.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(RUN_OUTPUT / 'drpa8_full_validation_metrics.csv', index=False)
    pd.DataFrame(roi_rows).to_csv(RUN_OUTPUT / 'drpa8_full_roi_metrics.csv', index=False)
    pd.DataFrame(rank_rows).to_csv(RUN_OUTPUT / 'drpa8_full_effective_rank_curve.csv', index=False)
    pd.DataFrame(grad_rows).to_csv(RUN_OUTPUT / 'drpa8_full_gradient_summary.csv', index=False)
    pd.DataFrame(checkpoint_rows).to_csv(RUN_OUTPUT / 'drpa8_full_checkpoint_manifest.csv', index=False)
    write_runtime_curve(runtime_rows)
    write_profile()
    torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(), 'torch_cuda_all': torch.cuda.get_rng_state_all(),
    }}, RUN_OUTPUT / 'final_optimizer_scaler_rng.pt')
    write_report(events, counts, pd.DataFrame(rank_rows), pd.DataFrame(grad_rows), pd.DataFrame(checkpoint_rows), plateau)
    aliases = {
        'DRPA8_FULL_DATA_TRAINING_REPORT.md': 'DRPA8_FULL_DATA_FINAL_REPORT.md',
        'drpa8_full_training_curve.csv': 'drpa8_full_learning_curve.csv',
    }
    for source, target in aliases.items():
        src = RUN_OUTPUT / source
        if src.exists():
            shutil.copyfile(src, RUN_OUTPUT / target)
    (RUN_OUTPUT / 'run_status.json').write_text(json.dumps({
        'status': 'COMPLETED', 'optimizer_steps_completed': global_step,
        'max_optimizer_steps': RUN_MAX_STEPS, 'resume_allowed': False,
        'cache_cases': RUN_CACHE_CASES,
    }, indent=2))
    print(json.dumps({'completed_steps': global_step, 'max_steps': RUN_MAX_STEPS,
                      'best_step': best['step'] if best else None,
                      'plateau': plateau}, indent=2), flush=True)


if __name__ == '__main__':
    main()
