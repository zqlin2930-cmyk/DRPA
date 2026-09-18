#!/usr/bin/env python3
"""Unified 12-PTID Raw/LCC evaluation for DRPA-8@200 and B3-stable@200."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

BASE = Path('__DRPA_WORKSPACE__')
OUT = BASE / 'quality_audit/voxtell_mtl_drpa8_pilot'
PEFT = BASE / 'quality_audit/voxtell_mtl_peft'
PREP = BASE / 'quality_audit/voxtell_mtl_b1_bilateral_crop'
B3OUT = BASE / 'quality_audit/voxtell_mtl_stable_decoder_capacity_baseline'
B3OLD = BASE / 'quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound'
sys.path[:0] = [str(OUT), str(PEFT), str(PREP), str(B3OUT), str(B3OLD)]
from b1_preprocessing import (  # noqa: E402
    BilateralGroupedPatchDataset, CaseRecord, PROMPTS, ROI_LABEL_IDS,
    load_crop_spec, load_official_reader_case, restore_crop_to_canonical,
)
from text_embedding_cache import TextEmbeddingCache  # noqa: E402
from drpa8_wrapper import DRPA8Wrapper  # noqa: E402
from voxtell_decoder_capacity_wrapper import VoxTellDecoderCapacityWrapper  # noqa: E402

DEVICE = torch.device('cuda')
MODEL = str(BASE / 'VoxTell_weights/voxtell_v1.1')
BANK = str(BASE / 'VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz')
CACHE = str(PEFT / 'text_embedding_cache.npz')
GROUPS = [PROMPTS[i:i + 2] for i in range(0, len(PROMPTS), 2)]


def surface(mask):
    return mask ^ ndimage.binary_erosion(
        mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0
    ) if mask.any() else mask


def crop_pair(a, b, margin=4):
    union = a | b
    if not union.any():
        return a, b
    coords = np.argwhere(union)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, np.asarray(a.shape))
    sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
    return a[sl], b[sl]


def dice(a, b):
    den = int(a.sum() + b.sum())
    return float(2 * (a & b).sum() / den) if den else 1.0


def hd95(a, b, spacing):
    if not a.any() or not b.any():
        return np.nan
    a, b = crop_pair(a, b)
    sa, sb = surface(a), surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.r_[da, db], 95))


def surface_dice(a, b, spacing, tolerance=2.0):
    if not a.any() or not b.any():
        return 0.0
    a, b = crop_pair(a, b)
    sa, sb = surface(a), surface(b)
    da = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    db = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    return float((np.count_nonzero(da <= tolerance) + np.count_nonzero(db <= tolerance)) / (len(da) + len(db)))


def component_stats(mask):
    labels, n = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([], dtype=int)
    lcc = (labels == int(np.argmax(sizes) + 1)) if n else mask
    return int(n), lcc


def max_fp(mask, gt, spacing):
    fp = mask & ~gt
    if not fp.any() or not gt.any():
        return 0.0 if not fp.any() else np.nan
    return float(ndimage.distance_transform_edt(~gt, sampling=spacing)[fp].max())


def restore(patch, item, source_shape, source_affine):
    meta = dict(item['preprocess_meta'])
    start = np.rint(np.linalg.solve(source_affine, item['patch_affine'])[:3, 3]).astype(int)
    meta.update({'crop_start_voxel_canonical': tuple(int(x) for x in start), 'reader_space': False})
    return restore_crop_to_canonical(patch, source_shape, meta)


def records(path):
    df = pd.read_csv(path)
    return [CaseRecord(str(r.case_id), str(r.image_path), str(r.label_path)) for r in df.itertuples()]


def infer(wrapper, item, cache):
    result = {}
    image = item['image'].unsqueeze(0).to(DEVICE)
    wrapper.model.eval()
    with torch.inference_mode():
        for group in GROUPS:
            embedding = cache.get(group, DEVICE)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = wrapper(image, embedding)
            probs = torch.sigmoid(logits.float()).cpu().numpy()[0]
            for j, prompt in enumerate(group):
                result[prompt] = probs[j]
    return result


def evaluate_condition(name, wrapper, val, dataset, cache):
    rows = []
    for i, rec in enumerate(val):
        item = dataset[i]
        _, label, affine, _ = load_official_reader_case(rec)
        probs = infer(wrapper, item, cache)
        spacing = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
        voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
        for prompt, patch_prob in probs.items():
            prob = restore(patch_prob, item, label.shape, affine)
            raw_mask = prob >= 0.5
            n_components, lcc_mask = component_stats(raw_mask)
            gt = label == ROI_LABEL_IDS[prompt]
            for lcc, mask in ((0, raw_mask), (1, lcc_mask)):
                pred_volume = float(mask.sum() * voxel_ml)
                gt_volume = float(gt.sum() * voxel_ml)
                fp_volume = float((mask & ~gt).sum() * voxel_ml)
                rows.append({
                    'condition': name, 'lcc': lcc, 'case_id': rec.case_id,
                    'ptid': rec.case_id.split('_')[0], 'prompt': prompt,
                    'structure': prompt.split(' ', 1)[1],
                    'side': 'left' if prompt.startswith('left') else 'right',
                    'dice': dice(mask, gt), 'hd95_mm': hd95(mask, gt, spacing),
                    'surface_dice_2mm': surface_dice(mask, gt, spacing),
                    'connected_components_raw': n_components,
                    'connected_components': int(component_stats(mask)[0]),
                    'false_positive_volume_ml': fp_volume,
                    'max_false_positive_distance_mm': max_fp(mask, gt, spacing),
                    'pred_volume_ml': pred_volume, 'gt_volume_ml': gt_volume,
                    'empty_mask': int(not mask.any()),
                })
        print('evaluated', name, rec.case_id, flush=True)
    return rows


def summarize(df):
    rows = []
    scopes = ['overall', 'hippocampus', 'entorhinal cortex', 'parahippocampal gyrus', 'amygdala']
    for condition in sorted(df.condition.unique()):
        for lcc in [0, 1]:
            q = df[(df.condition == condition) & (df.lcc == lcc)]
            for scope in scopes:
                sub = q if scope == 'overall' else q[q.structure == scope]
                rows.append({
                    'condition': condition, 'lcc': lcc, 'scope': scope,
                    'n_rows': len(sub), 'n_cases': sub.case_id.nunique(),
                    'dice': float(sub.dice.mean()), 'hd95_mm': float(sub.hd95_mm.mean()),
                    'surface_dice_2mm': float(sub.surface_dice_2mm.mean()),
                    'connected_components': float(sub.connected_components.mean()),
                    'false_positive_volume_ml': float(sub.false_positive_volume_ml.mean()),
                    'max_false_positive_distance_mm': float(sub.max_false_positive_distance_mm.mean()),
                    'empty_mask_rate': float(sub.empty_mask.mean()),
                })
    return pd.DataFrame(rows)


def main():
    active = __import__('subprocess').run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True).stdout.strip()
    if active:
        raise RuntimeError(f'GPU process active; refusing evaluation: {active}')
    val = records(OUT / 'pilot_val_cases.csv')
    spec = load_crop_spec(OUT / 'crop_spec.json')
    dataset = BilateralGroupedPatchDataset(val, spec)
    cache = TextEmbeddingCache(BANK, MODEL, CACHE)
    drpa = DRPA8Wrapper(MODEL, BANK, device=DEVICE)
    drpa.load_checkpoint(str(OUT / 'drpa8_step_200.pt'))
    b3 = VoxTellDecoderCapacityWrapper(MODEL, BANK, rank=4, alpha=8, dropout=0.05, device=DEVICE)
    b3.load_adapter_checkpoint(str(B3OLD / 'diagnostics/stable_200_steps/adapter_step_200.pt'))
    rows = []
    rows.extend(evaluate_condition('DRPA-8@200', drpa, val, dataset, cache))
    rows.extend(evaluate_condition('B3-stable@200', b3, val, dataset, cache))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'drpa8_case_level_metrics.csv', index=False)
    summary = summarize(df)
    b2 = pd.read_csv(B3OUT / 'b3_stable_metric_summary.csv')
    b2 = b2[b2.condition == 'B2-highres-maskproj-LoRA'].copy()
    b2['condition'] = 'B2'
    b2 = b2.rename(columns={'scope': 'scope', 'dice': 'dice', 'hd95_mm': 'hd95_mm',
                            'surface_dice_2mm': 'surface_dice_2mm',
                            'connected_components': 'connected_components',
                            'false_positive_volume_ml': 'false_positive_volume_ml',
                            'max_false_positive_distance_mm': 'max_false_positive_distance_mm'})
    b2 = b2[['condition', 'lcc', 'scope', 'n_rows', 'n_cases', 'dice', 'hd95_mm',
             'surface_dice_2mm', 'connected_components', 'false_positive_volume_ml',
             'max_false_positive_distance_mm']]
    summary.to_csv(OUT / 'drpa8_metric_summary.csv', index=False)
    summary.to_csv(OUT / 'drpa8_metric_summary_internal.csv', index=False)
    pd.concat([summary, b2], ignore_index=True, sort=False).to_csv(OUT / 'drpa8_vs_b2_b3_metric_summary.csv', index=False)
    print(pd.concat([summary, b2], ignore_index=True, sort=False).to_string(index=False))


if __name__ == '__main__':
    main()
