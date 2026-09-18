#!/usr/bin/env python3
"""Independent B3-Full FP32 formal runner.

This is a project-side runner.  It reuses the audited data/evaluation loop and
B3 wrapper, but disables AMP and GradScaler at runtime.  It never resumes an
old B3 or DRPA checkpoint.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(os.environ.get('MTL_MODEL_ROOT', '__DRPA_WORKSPACE__'))
OUT = BASE / 'quality_audit/voxtell_full_data_b3_vs_drpa_efficiency'
RUN = OUT / 'b3_full_fp32_formal_3000_6000_rerun'
DRPA_LOOP = BASE / 'drpa8_full_data_train.py'
PILOT = BASE / 'quality_audit/voxtell_mtl_drpa8_pilot'
PEFT = BASE / 'quality_audit/voxtell_mtl_peft'
PREP = BASE / 'quality_audit/voxtell_mtl_b1_bilateral_crop'
MODEL = BASE / 'VoxTell_weights/voxtell_v1.1'
DRPA_DATA = BASE / 'quality_audit/voxtell_mtl_drpa8_full_data'
TRAIN_MANIFEST = DRPA_DATA / 'full_data_train_cases.csv'
VAL_MANIFEST = DRPA_DATA / 'full_data_val_cases.csv'
OFFICIAL_CHECKPOINT = MODEL / 'fold_0/checkpoint_final.pth'
sys.path[:0] = [str(DRPA_LOOP.parent), str(PILOT), str(PEFT), str(PREP), str(OUT)]

import drpa8_full_data_train as shared  # noqa: E402
from b3_full_data_wrapper import B3FullDataWrapper  # noqa: E402

EXPECTED_GROUPS = {
    'cross_attention_lora': 294_912,
    'projection_adapter': 71_403_552,
    'decoder_stages': 10_232_160,
}
EXPECTED_TOTAL = sum(EXPECTED_GROUPS.values())
LORA_LR = 1e-4
PROJECTION_LR = 1e-5
DECODER_LR = 1e-5
WEIGHT_DECAY = 1e-5
SEED = 20260809
MAX_STEPS = 6000

_latest = {'wrapper': None, 'optimizer': None, 'phase': {}, 'prompt_idx': 0, 'failure_written': False, 'contract': None}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ptid(case_id: str) -> str:
    return case_id.split('_', 1)[0]


def audit_protocol() -> dict:
    train = pd.read_csv(TRAIN_MANIFEST)
    val = pd.read_csv(VAL_MANIFEST)
    train_ptids = {ptid(str(x)) for x in train.case_id}
    val_ptids = {ptid(str(x)) for x in val.case_id}
    overlap = sorted(train_ptids & val_ptids)
    if len(train) != 971 or len(train_ptids) != 337:
        raise RuntimeError(f'train manifest mismatch: visits={len(train)}, PTIDs={len(train_ptids)}')
    if len(val) != 247 or len(val_ptids) != 85:
        raise RuntimeError(f'val manifest mismatch: visits={len(val)}, PTIDs={len(val_ptids)}')
    if overlap:
        raise RuntimeError(f'PTID overlap: {overlap[:5]}')

    # This exactly mirrors the existing formal loop: a new deterministic
    # permutation at each epoch, with seed + epoch.
    order_tokens = []
    step = 0
    epoch = 0
    case_ids = train.case_id.astype(str).tolist()
    while step < MAX_STEPS:
        epoch += 1
        order = np.random.default_rng(SEED + epoch).permutation(len(case_ids))
        for idx in order:
            if step >= MAX_STEPS:
                break
            order_tokens.append(f'{step + 1}:{case_ids[int(idx)]}')
            step += 1
    order_hash = sha256_text('\n'.join(order_tokens))
    crop_spec = PILOT / 'crop_spec.json'
    prep_source = PREP / 'b1_preprocessing.py'
    evaluator_path = Path(shared.evaluator.__file__).resolve()
    contract = {
        'status': 'PASS',
        'train_visits': int(len(train)), 'train_ptids': int(len(train_ptids)),
        'val_visits': int(len(val)), 'val_ptids': int(len(val_ptids)),
        'ptid_overlap': overlap,
        'train_manifest_sha256': sha256_file(TRAIN_MANIFEST),
        'val_manifest_sha256': sha256_file(VAL_MANIFEST),
        'sample_order_sha256_6000_steps': order_hash,
        'sample_order_definition': 'np.random.default_rng(SEED + epoch).permutation(971), exact shared-loop order',
        'seed': SEED,
        'prompts': list(shared.PROMPTS),
        'preprocessing': 'canonical RAS + official reader-space preprocessing + bilateral MTL crop',
        'crop_spec_sha256': sha256_file(crop_spec),
        'preprocessing_source_sha256': sha256_file(prep_source),
        'evaluator_path': str(evaluator_path),
        'evaluator_sha256': sha256_file(evaluator_path),
        'initialization': 'official VoxTell v1.1 fresh initialization only',
        'official_checkpoint_sha256': sha256_file(OFFICIAL_CHECKPOINT),
        'model_dir': str(MODEL),
        'optimizer': 'AdamW', 'weight_decay': WEIGHT_DECAY,
        'learning_rates': {'cross_attention_lora': LORA_LR, 'projection_full': PROJECTION_LR, 'decoder_stages': DECODER_LR},
        'loss': 'FP32 Dice+BCE', 'amp': False, 'grad_scaler': False,
        'gradient_clip_norm': 1.0,
        'max_optimizer_steps': MAX_STEPS,
        'validation_steps': [3000, 6000],
        'batch_size': 1, 'gradient_accumulation': 1,
        'trainable_parameter_groups': EXPECTED_GROUPS,
        'trainable_parameters': EXPECTED_TOTAL,
        'forbidden_resume_sources': ['original B3 checkpoints', 'DRPA checkpoints', 'B3 SVD factors', 'optimizer/scaler states'],
    }
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'B3_FP32_PROTOCOL_AUDIT.json').write_text(json.dumps(contract, indent=2))
    (RUN / 'B3_FP32_PROTOCOL_AUDIT.md').write_text(
        '# B3-Full FP32 6000-step protocol audit\n\n'
        f'- Status: **{contract["status"]}**\n'
        f'- Train/val: `{len(train)}/{len(val)} visits`, `{len(train_ptids)}/{len(val_ptids)} PTIDs`; overlap `{len(overlap)}`.\n'
        f'- Sample order hash: `{order_hash}`.\n'
        f'- Official checkpoint hash: `{contract["official_checkpoint_sha256"]}`.\n'
        f'- Evaluator: `{evaluator_path}`\n'
        f'- Evaluator hash: `{contract["evaluator_sha256"]}`\n'
        f'- Preprocessing: `{contract["preprocessing"]}`; crop hash `{contract["crop_spec_sha256"]}`.\n'
        f'- Prompts: `{", ".join(shared.PROMPTS)}`.\n'
        f'- Fresh init only; no old checkpoint/optimizer/scaler/SVD resume.\n'
        f'- B3 groups: `{json.dumps(EXPECTED_GROUPS)}`; total `{EXPECTED_TOTAL}`.\n'
        '- Optimizer: AdamW, weight decay `1e-5`; LR LoRA `1e-4`, full projection `1e-5`, decoder stages `1e-5`.\n'
        '- Loss: FP32 Dice+BCE; AMP disabled; GradScaler disabled; unscale step is not applicable; gradient clip `1.0`.\n'
        '- Validation: complete validation only at optimizer steps 3000 and 6000; raw Mean Dice is the primary selection metric.\n'
    )
    return contract


class NoAutocast(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


class NoGradScaler:
    """Compatibility shim: no scaling, no AMP state, direct optimizer.step."""
    def __init__(self, *args, **kwargs):
        pass
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        return None
    def step(self, optimizer):
        return optimizer.step()
    def update(self):
        return None
    def get_scale(self):
        return 1.0
    def state_dict(self):
        return {'enabled': False}


def _rng_state():
    return {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'torch_cpu': torch.random.get_rng_state(),
        'torch_cuda_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _failure_payload(reason: str, extra: dict | None = None):
    payload = {
        'status': 'B3_FULL_FP32_NUMERICAL_FAILURE',
        'reason': reason,
        'phase': _latest['phase'],
        'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
        'optimizer_present': _latest['optimizer'] is not None,
        'prompt_index': _latest['prompt_idx'],
        'prompt': shared.PROMPTS[_latest['prompt_idx']] if _latest['prompt_idx'] < len(shared.PROMPTS) else None,
        'recent_valid_checkpoints': sorted(str(p) for p in (RUN / 'checkpoints').glob('step_*.pt'))[-3:],
        'rng_saved': True,
    }
    if extra:
        payload.update(extra)
    (RUN / 'failure_diagnostic.json').write_text(json.dumps(payload, indent=2, default=str))
    torch.save({'rng': _rng_state(), 'optimizer': _latest['optimizer'].state_dict() if _latest['optimizer'] else None}, RUN / 'failure_optimizer_rng.pt')
    wrapper = _latest.get('wrapper')
    if wrapper is not None:
        try:
            wrapper.save_checkpoint(str(RUN / 'failure_model_state.pt'), {'failure': reason, 'phase': _latest['phase']})
        except Exception as exc:
            payload['model_state_save_error'] = repr(exc)
    _latest['failure_written'] = True


def _safe_b3_projection_rank_rows(wrapper, step: int, equivalent_epoch: float):
    """Non-blocking audit for mixed parametrized and full-FT projections.

    LoRA/parametrized projections expose a delta matrix.  B3 full-FT
    projections are ordinary Linear modules, so they are recorded as full-rank
    without accessing ``module.parametrizations``.  This audit must never be a
    prerequisite for optimizer progress.
    """
    rows = []
    with torch.no_grad():
        for pidx, projection in enumerate(wrapper.model.project_to_decoder_channels):
            for child_name, child in projection.named_modules():
                if not isinstance(child, torch.nn.Linear):
                    continue
                prefix = f'project_to_decoder_channels.{pidx}.{child_name}'
                parametrized = hasattr(child, 'parametrizations') and hasattr(child.parametrizations, 'weight')
                if parametrized:
                    adapter = child.parametrizations.weight[0]
                    delta = adapter.delta_matrix().detach().float()
                    singular = torch.linalg.svdvals(delta)
                    base = child.parametrizations.weight.original.detach().float()
                    rows.append({
                        'step': step, 'equivalent_epoch': equivalent_epoch,
                        'projection': prefix, 'configured_rank': int(adapter.rank),
                        'effective_rank': shared.effective_rank(singular),
                        'stable_rank': float((singular.square().sum() / singular.square().max().clamp_min(1e-12)).cpu()),
                        'delta_l2': float(delta.norm().cpu()),
                        'relative_update': float((delta.norm() / base.norm().clamp_min(1e-12)).cpu()),
                        'A_norm': float(adapter.lora_A.detach().float().norm().cpu()),
                        'B_norm': float(adapter.lora_B.detach().float().norm().cpu()),
                        'audit_mode': 'parametrized_delta',
                    })
                else:
                    weight = child.weight.detach().float()
                    rows.append({
                        'step': step, 'equivalent_epoch': equivalent_epoch,
                        'projection': prefix, 'configured_rank': 'full',
                        'effective_rank': 'full', 'stable_rank': 'full',
                        'delta_l2': float('nan'), 'relative_update': float('nan'),
                        'A_norm': float('nan'), 'B_norm': float('nan'),
                        'weight_l2': float(weight.norm().cpu()),
                        'audit_mode': 'ordinary_full_finetune_linear',
                    })
    return rows


def safe_b3_projection_rank_rows(wrapper, step: int, equivalent_epoch: float):
    """Never let an optional projection audit stop the training loop."""
    try:
        return _safe_b3_projection_rank_rows(wrapper, step, equivalent_epoch)
    except Exception as exc:
        return [{
            'step': step, 'equivalent_epoch': equivalent_epoch,
            'projection': 'AUDIT_ERROR', 'configured_rank': 'unknown',
            'effective_rank': 'unknown', 'stable_rank': 'unknown',
            'audit_mode': 'non_blocking_audit_error', 'audit_error': repr(exc),
        }]


def install_runtime_patches():
    # Replace only runtime behavior; the official model and shared loop files
    # remain unchanged on disk.
    shared.DRPA8Wrapper = B3FullDataWrapper
    shared.EXPECTED_GROUPS = EXPECTED_GROUPS
    shared.EXPECTED_TOTAL = EXPECTED_TOTAL
    shared.PROJECTION_LR = PROJECTION_LR
    shared.LORA_LR = LORA_LR
    shared.DECODER_LR = DECODER_LR
    shared.VALIDATION_STEPS = (3000, 6000)
    shared.VALIDATION_INTERVAL = 3000
    # Defer the diagnostic until after metrics and checkpoint persistence.  The
    # shared loop calls this hook before writing the validation artifacts; an
    # empty row list keeps that path non-blocking. The save hook below performs
    # the actual rank audit after the scheduled checkpoint is fully written.
    shared.projection_rank_rows = lambda *args, **kwargs: []
    shared.write_report = lambda *args, **kwargs: None
    torch.autocast = NoAutocast
    torch.amp.GradScaler = NoGradScaler

    original_phase = shared.phase
    def phase(event, step, case_id=''):
        _latest['phase'] = {'event': event, 'step': int(step), 'case_id': str(case_id)}
        if event == 'data_fetch_complete':
            _latest['prompt_idx'] = 0
        return original_phase(event, step, case_id)
    shared.phase = phase

    original_init = B3FullDataWrapper.__init__
    def wrapper_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _latest['wrapper'] = self
    B3FullDataWrapper.__init__ = wrapper_init

    original_adamw = torch.optim.AdamW
    class CaptureAdamW(original_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _latest['optimizer'] = self
            torch.save({'optimizer': self.state_dict(), 'rng': _rng_state()}, RUN / 'initial_optimizer_rng.pt')
    torch.optim.AdamW = CaptureAdamW

    original_loss = shared.loss_fp32
    def loss_fp32(logits, target):
        out = original_loss(logits, target)
        prompt_idx = _latest['prompt_idx']
        _latest['prompt_idx'] += 1
        if not bool(torch.isfinite(out[0]).item()):
            vals = logits.detach().float().cpu()
            _failure_payload('nonfinite_loss', {
                'prompt_index': prompt_idx,
                'prompt': shared.PROMPTS[prompt_idx] if prompt_idx < len(shared.PROMPTS) else None,
                'logits_dtype': str(logits.dtype),
                'logits_min': float(vals.nan_to_num().min()),
                'logits_max': float(vals.nan_to_num().max()),
                'loss_finite': False,
            })
        return out
    shared.loss_fp32 = loss_fp32

    original_grad_stats = shared.gradient_stats
    def gradient_stats(groups):
        out = original_grad_stats(groups)
        params_finite = all(torch.isfinite(param).all().item() for values in groups.values() for param in values)
        grad_norms_finite = all(math.isfinite(float(v['grad_norm'])) for v in out.values())
        bad = (not params_finite or not grad_norms_finite or
               any(v['finite_grad_tensors'] < v['grad_tensors'] - v['none_grad_tensors'] or v['none_grad_tensors'] > 0 or v['nonzero_grad_tensors'] == 0 for v in out.values()))
        if bad:
            _failure_payload('gradient_or_parameter_gate_failed', {'gradient_stats': out, 'parameters_finite': params_finite, 'gradient_norms_finite': grad_norms_finite})
        return out
    shared.gradient_stats = gradient_stats

    # Persist GPU/RSS audit fields at every optimizer step, not only at
    # validation nodes.
    original_write_step_rows = shared.write_step_rows
    def write_step_rows(rows):
        if rows:
            rows[-1]['gpu_vram_mb'] = shared.gpu_vram_mb()
            rows[-1]['shm_used_mb'] = shared.shm_usage_mb()
        return original_write_step_rows(rows)
    shared.write_step_rows = write_step_rows

    # The shared loop writes model-only trainable checkpoints.  For the two
    # formal validation nodes, replace those files with auditable payloads
    # containing optimizer and RNG/CUDA-RNG state, while also preserving the
    # loop's ordinary checkpoint files.
    original_save_checkpoint = B3FullDataWrapper.save_checkpoint
    def save_checkpoint(self, path, metadata):
        original_save_checkpoint(self, path, metadata)
        name = Path(path).name
        if name not in {'step_03000.pt', 'step_06000.pt', 'validation_best.pt'}:
            return
        payload = torch.load(path, map_location='cpu')
        payload.update({
            'optimizer_state_dict': _latest['optimizer'].state_dict() if _latest['optimizer'] is not None else None,
            'rng_state': _rng_state(),
            'experiment_config': {
                'experiment': 'B3_FULL_FP32_6000_FORMAL_EXPERIMENT',
                'validation_steps': [3000, 6000], 'max_optimizer_steps': MAX_STEPS,
                'amp': False, 'grad_scaler': False, 'loss': 'FP32 Dice+BCE',
                'learning_rates': {'cross_attention_lora': LORA_LR, 'projection_full': PROJECTION_LR, 'decoder_stages': DECODER_LR},
                'weight_decay': WEIGHT_DECAY, 'seed': SEED,
            },
            'global_step': metadata.get('optimizer_step', metadata.get('step')),
            'manifest_hashes': {
                'train': (_latest['contract'] or {}).get('train_manifest_sha256'),
                'val': (_latest['contract'] or {}).get('val_manifest_sha256'),
            },
        })
        torch.save(payload, path)
        if name.startswith('step_'):
            step = int(name[5:10])
            alias = RUN / f'b3_full_fp32_step{step}.pt'
            shutil.copyfile(path, alias)
            hashes_path = RUN / 'checkpoint_hashes.json'
            hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
            hashes[alias.name] = sha256_file(alias)
            hashes_path.write_text(json.dumps(hashes, indent=2))
            # Rank analysis is deliberately after checkpoint persistence and
            # is diagnostic-only. Its own failure is recorded, never raised.
            try:
                audit_rows = safe_b3_projection_rank_rows(self, int(step), float(metadata.get('equivalent_epoch', 0.0)))
                audit_path = RUN / 'b3_full_fp32_rank_audit.csv'
                audit_frame = pd.DataFrame(audit_rows)
                if audit_path.exists():
                    prior = pd.read_csv(audit_path)
                    audit_frame = pd.concat([prior, audit_frame], ignore_index=True)
                audit_frame.to_csv(audit_path, index=False)
            except Exception as exc:
                with (RUN / 'b3_full_fp32_rank_audit_errors.log').open('a') as f:
                    f.write(f'step={step} error={exc!r}\n')
    B3FullDataWrapper.save_checkpoint = save_checkpoint


def copy_aliases(run: Path):
    aliases = {
        'drpa8_full_training_curve.csv': 'b3_full_fp32_learning_curve.csv',
        'drpa8_full_validation_metrics.csv': 'b3_full_fp32_validation_metrics.csv',
        'drpa8_full_roi_metrics.csv': 'b3_full_fp32_roi_metrics.csv',
        'drpa8_full_runtime_curve.csv': 'b3_full_fp32_runtime_curve.csv',
        'drpa8_full_gradient_summary.csv': 'b3_full_fp32_gradient_summary.csv',
        'drpa8_full_effective_rank_curve.csv': 'b3_full_fp32_effective_rank_curve.csv',
        'drpa8_full_checkpoint_manifest.csv': 'b3_full_fp32_checkpoint_manifest.csv',
    }
    for src, dst in aliases.items():
        if (run / src).exists():
            shutil.copyfile(run / src, run / dst)


def write_final_report(contract: dict, run: Path, error: Exception | None = None):
    copy_aliases(run)
    curve = run / 'drpa8_full_training_curve.csv'
    grad = run / 'drpa8_full_gradient_summary.csv'
    status = 'FAILED' if error else 'COMPLETED'
    lines = [
        '# B3-Full FP32 6000-step formal experiment', '',
        f'- Status: **{status}**',
        '- This arm is independent and starts from fresh official VoxTell v1.1 initialization.',
        '- The prior AMP B3 run and canonical DRPA artifacts were not resumed or overwritten.',
        '- Forward and loss are full FP32; AMP and GradScaler are disabled.',
        f'- Train/val: {contract["train_visits"]}/{contract["val_visits"]} visits, {contract["train_ptids"]}/{contract["val_ptids"]} PTIDs; overlap {len(contract["ptid_overlap"])}.',
        f'- Trainable groups: `{json.dumps(EXPECTED_GROUPS)}`; total `{EXPECTED_TOTAL}`.',
        f'- LR: LoRA `{LORA_LR}`, projection `{PROJECTION_LR}`, decoder `{DECODER_LR}`; AdamW weight decay `{WEIGHT_DECAY}`.',
        '- Validation nodes: 3000 and 6000; raw Mean Dice is the primary selection metric.',
    ]
    if curve.exists() and curve.stat().st_size:
        frame = pd.read_csv(curve)
        if not frame.empty:
            best = frame.loc[frame.val_mean_dice.idxmax()]
            final = frame.iloc[-1]
            lines += ['', '## Validation summary', '', f'- Completed validation step: `{int(final.step)}`', f'- Best step: `{int(best.step)}`', f'- Best Mean Dice: `{best.val_mean_dice:.6f}`', f'- Final Mean Dice: `{final.val_mean_dice:.6f}`', f'- Best HD95: `{best.val_hd95_mm:.6f} mm`', f'- Best Surface Dice: `{best.val_surface_dice:.6f}`']
            for name, col in [('Hipp', 'val_hippocampus_dice'), ('EC', 'val_entorhinal_cortex_dice'), ('PHG', 'val_parahippocampal_gyrus_dice'), ('Amy', 'val_amygdala_dice')]:
                lines.append(f'- Best {name} Dice: `{best[col]:.6f}`')
    if grad.exists() and grad.stat().st_size:
        gf = pd.read_csv(grad)
        lines += ['', '## Stability', f'- Logged optimizer steps: `{len(gf)}`', f'- Non-finite rows: `{int(gf.nan_inf.sum()) if "nan_inf" in gf else "not_recorded"}`', f'- Clip-triggered rows: `{int(gf.clip_triggered.sum()) if "clip_triggered" in gf else "not_recorded"}`']
    if error:
        lines += ['', '## Failure', f'- Exception: `{repr(error)}`', '- No rescue, LR change, checkpoint resume, or protocol modification was performed.', '- See `failure_diagnostic.json` and `failure_optimizer_rng.pt`.']
    (run / 'B3_FULL_FP32_TRAINING_REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if args.max_steps != MAX_STEPS:
        raise RuntimeError('This formal runner is fixed at 6000 optimizer steps.')
    contract = audit_protocol()
    _latest['contract'] = contract
    if args.audit_only:
        print(json.dumps({'protocol_audit': 'PASS', 'output': str(RUN)}, indent=2))
        return
    install_runtime_patches()
    try:
        sys.argv = [sys.argv[0], '--max-steps', str(MAX_STEPS), '--cache-cases', 'false', '--output-dir', str(RUN)]
        shared.main()
    except Exception as exc:
        _failure_payload('training_exception', {'exception': repr(exc)}) if not _latest['failure_written'] else None
        (RUN / 'run_status.json').write_text(json.dumps({'status': 'FAILED', 'optimizer_steps_completed': _latest['phase'].get('step', 0), 'resume_allowed': False, 'error': repr(exc)}, indent=2))
        write_final_report(contract, RUN, exc)
        raise
    else:
        write_final_report(contract, RUN)


if __name__ == '__main__':
    main()
