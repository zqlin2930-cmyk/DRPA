"""Deterministic, outcome-blind participant contract; no model imports."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

SEED = 20260809
MODELS = {'DRPA8': 10969696, 'B3Canonical': 81930624}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha256(path) != expected: raise ValueError(f'Hash mismatch: {path}')


def split_rows(rows, seed=SEED):
    rows = sorted(rows, key=lambda r: int(r['subject_id']))
    if len(rows) != 20 or len({int(r['subject_id']) for r in rows}) != 20:
        raise ValueError('Require 20 unique participants, not repeated scans')
    if len({r['case_id'] for r in rows}) != 20: raise ValueError('Duplicate case')
    for r in rows:
        if r['case_id'] != f"oasis_trt20_{int(r['subject_id']):02d}": raise ValueError('Identity mismatch')
    idx = set(int(i) for i in np.random.default_rng(seed).permutation(20)[:5])
    return [r for i, r in enumerate(rows) if i in idx], [r for i, r in enumerate(rows) if i not in idx]


def sample_order(support, seed=SEED):
    if len(support) != 5: raise ValueError('Require exactly5 support subjects')
    order = []
    for epoch in range(1, 101):
        for position, index in enumerate(np.random.default_rng(seed + epoch).permutation(5)):
            row = support[int(index)]
            order.append(dict(step=len(order)+1, epoch=epoch, position=position,
                              support_index=int(index), case_id=row['case_id'], subject_id=int(row['subject_id'])))
    return order


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def config():
    return dict(seed=SEED, support=5, query=15, steps=500, batch_size=1,
                train_precision='strict_FP32', autocast=False, GradScaler=False,
                train_TF32=False, lr=1e-5, weight_decay=1e-5, betas=[.9,.999], eps=1e-8,
                scheduler=None, clip_norm=1.0, initialization='canonical_ADNI100_step6000',
                optimizer_state='fresh_target_domain_adaptation', models=MODELS,
                prompt_reduction='8 separate prompt Dice+BCE losses; each /8 backward; one optimizer update',
                augmentation='none; unchanged canonical deterministic dataset',
                preprocessing='native_NIfTI_original_VoxTell_reader_reference_assisted_crop_and_normalization',
                use_nnunet_v1_stage0=False, cache_cases=False, num_workers=0,
                evaluation='original_grouped_FP16_forward_FP32_sigmoid', threshold='>=0.5', lcc=0,
                query_selection=False, endpoint=500, bootstrap_draws=10000, bootstrap_seed=SEED,
                query_status='adaptation-held-out with prior frozen-evaluation exposure',
                minimum_free_GiB=20)
