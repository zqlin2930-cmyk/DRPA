#!/usr/bin/env python3
"""Recover canonical 100 percent DRPA/B3 case-level metrics without training."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

BASE = Path("__DRPA_WORKSPACE__")
OUT = BASE / "quality_audit/canonical_data_capacity_completion/task1_b3_case_level_recovery"
DRPA_RUN = BASE / "quality_audit/voxtell_mtl_drpa8_full_data/formal_training/formal_6000_sparse_val_cache_false_20260812"
B3_RUN = BASE / "quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_full_fp32_formal_3000_6000_rerun"
DRPA_CKPT = DRPA_RUN / "checkpoints/validation_best.pt"
B3_CKPT = B3_RUN / "checkpoints/validation_best.pt"
VAL_MANIFEST = BASE / "quality_audit/voxtell_mtl_drpa8_full_data/full_data_val_cases.csv"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PEFT = BASE / "quality_audit/voxtell_mtl_peft"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
B3SRC = BASE / "quality_audit/voxtell_full_data_b3_vs_drpa_efficiency"
B3BASE = BASE / "quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound"
VOXTELL = BASE / "VoxTell"
MODEL = BASE / "VoxTell_weights/voxtell_v1.1"
BANK = BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
CACHE = PEFT / "text_embedding_cache.npz"
DEVICE = torch.device("cuda")
SEED = 20260903
BOOTSTRAPS = 10000

sys.path[:0] = [str(PILOT), str(PEFT), str(PREP), str(B3BASE), str(B3SRC), str(VOXTELL)]
import evaluate_drpa8 as evaluator
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, ROI_LABEL_IDS, load_crop_spec, load_official_reader_case
from text_embedding_cache import TextEmbeddingCache
from drpa8_wrapper import DRPA8Wrapper
from b3_full_data_wrapper import B3FullDataWrapper


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def ptid(case_id: str) -> str:
    return str(case_id).split("_", 1)[0]


def assert_contract(val: pd.DataFrame) -> None:
    if len(val) != 247 or val.case_id.nunique() != 247 or val.case_id.map(ptid).nunique() != 85:
        raise RuntimeError("canonical validation manifest is not 247 visits / 85 PTIDs")
    if not DRPA_CKPT.exists() or not B3_CKPT.exists():
        raise RuntimeError("missing canonical validation-best checkpoint")
    active = __import__("subprocess").run(
        ["/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if active:
        raise RuntimeError("GPU busy; recovery refuses concurrent inference: " + active)


def load_b3() -> B3FullDataWrapper:
    wrapper = B3FullDataWrapper(str(MODEL), str(BANK), device=DEVICE)
    payload = torch.load(B3_CKPT, map_location="cpu", weights_only=False)
    incoming = payload["trainable_state_dict"]
    expected = {name: p for name, p, _ in wrapper.trainable_parameter_groups()}
    if set(expected) != set(incoming):
        raise RuntimeError("B3 checkpoint trainable-state keys do not match B3-Canonical wrapper")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(incoming[name].to(parameter.device, dtype=parameter.dtype))
    wrapper.model.eval()
    return wrapper


def load_drpa() -> DRPA8Wrapper:
    wrapper = DRPA8Wrapper(str(MODEL), str(BANK), device=DEVICE)
    wrapper.load_checkpoint(str(DRPA_CKPT))
    wrapper.model.eval()
    return wrapper


def case_rows(condition: str, wrapper, rec: CaseRecord, item: dict, cache) -> list[dict]:
    _, label, affine, _ = load_official_reader_case(rec)
    probs = evaluator.infer(wrapper, item, cache)
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
    rows = []
    for prompt, patch_prob in probs.items():
        prob = evaluator.restore(patch_prob, item, label.shape, affine)
        raw_mask = prob >= 0.5
        n_components, lcc_mask = evaluator.component_stats(raw_mask)
        gt = label == ROI_LABEL_IDS[prompt]
        for lcc, mask in ((0, raw_mask), (1, lcc_mask)):
            pred_volume = float(mask.sum() * voxel_ml)
            gt_volume = float(gt.sum() * voxel_ml)
            fp_volume = float((mask & ~gt).sum() * voxel_ml)
            fn_volume = float((gt & ~mask).sum() * voxel_ml)
            rows.append({
                "condition": condition, "lcc": lcc, "case_id": rec.case_id,
                "ptid": ptid(rec.case_id), "prompt": prompt,
                "structure": prompt.split(" ", 1)[1],
                "side": "left" if prompt.startswith("left") else "right",
                "dice": evaluator.dice(mask, gt),
                "hd95_mm": evaluator.hd95(mask, gt, spacing),
                "surface_dice_2mm": evaluator.surface_dice(mask, gt, spacing),
                "connected_components_raw": n_components,
                "connected_components": int(evaluator.component_stats(mask)[0]),
                "false_positive_volume_ml": fp_volume,
                "false_negative_volume_ml": fn_volume,
                "max_false_positive_distance_mm": evaluator.max_fp(mask, gt, spacing),
                "pred_volume_ml": pred_volume, "gt_volume_ml": gt_volume,
                "empty_mask": int(not mask.any()),
                "threshold_contract": "sigmoid(logit) >= 0.5",
            })
    return rows


def recover_condition(condition: str, factory, val: list[CaseRecord], dataset, cache) -> pd.DataFrame:
    out_path = OUT / ("case_metrics_" + condition + ".csv")
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    completed = set(existing.case_id.astype(str).unique()) if not existing.empty else set()
    wrapper = factory()
    rows = existing.to_dict("records") if not existing.empty else []
    for index, rec in enumerate(val):
        if rec.case_id in completed:
            continue
        rows.extend(case_rows(condition, wrapper, rec, dataset[index], cache))
        atomic_csv(pd.DataFrame(rows), out_path)
        atomic_json({
            "stage": "case_level_recovery",
            "condition": condition,
            "completed_visits": int(len({r["case_id"] for r in rows})),
            "total_visits": len(val),
            "current_case": rec.case_id,
            "checkpoint": str(DRPA_CKPT if condition == "drpa8" else B3_CKPT),
        }, OUT / "progress.json")
        print("RECOVERED", condition, index + 1, len(val), rec.case_id, flush=True)
    del wrapper
    torch.cuda.empty_cache()
    gc.collect()
    frame = pd.read_csv(out_path)
    if frame.case_id.nunique() != 247 or len(frame) != 247 * 8 * 2:
        raise RuntimeError("incomplete case-level recovery for " + condition)
    return frame


def summary(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    raw = frame[frame.lcc == 0].copy()
    scopes = ["overall", "hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"]
    rows = []
    for scope in scopes:
        q = raw if scope == "overall" else raw[raw.structure == scope]
        rows.append({
            "condition": condition, "scope": scope, "visits": int(q.case_id.nunique()),
            "ptids": int(q.ptid.nunique()), "rows": int(len(q)),
            "mean_dice": float(q.dice.mean()), "mean_hd95_mm": float(q.hd95_mm.mean()),
            "mean_surface_dice_2mm": float(q.surface_dice_2mm.mean()),
            "mean_fp_volume_ml": float(q.false_positive_volume_ml.mean()),
            "mean_fn_volume_ml": float(q.false_negative_volume_ml.mean()),
            "mean_components": float(q.connected_components.mean()),
        })
    return pd.DataFrame(rows)


def paired_statistics(drpa: pd.DataFrame, b3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["dice", "hd95_mm", "surface_dice_2mm", "false_positive_volume_ml", "false_negative_volume_ml", "connected_components"]
    key = ["ptid", "case_id", "prompt", "structure", "side"]
    a = drpa[drpa.lcc == 0][key + metrics].copy()
    b = b3[b3.lcc == 0][key + metrics].copy()
    merged = a.merge(b, on=key, suffixes=("_drpa", "_b3"), validate="one_to_one")
    if len(merged) != 247 * 8:
        raise RuntimeError("paired raw rows do not cover 247 x 8")
    pt = merged.groupby(["ptid", "structure"], as_index=False)[[m + "_drpa" for m in metrics] + [m + "_b3" for m in metrics]].mean()
    rng = np.random.default_rng(SEED)
    out = []
    boots = []
    for scope in ["overall", "hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"]:
        q = pt if scope == "overall" else pt[pt.structure == scope]
        ptids = np.array(sorted(q.ptid.unique()))
        for metric in metrics:
            delta_by_ptid = q.groupby("ptid").apply(lambda x: (x[metric + "_b3"] - x[metric + "_drpa"]).mean()).reindex(ptids).to_numpy()
            draws = np.array([rng.choice(delta_by_ptid, size=len(delta_by_ptid), replace=True).mean() for _ in range(BOOTSTRAPS)])
            dice_delta = q.groupby("ptid").apply(lambda x: (x["dice_b3"] - x["dice_drpa"]).mean()).reindex(ptids).to_numpy()
            out.append({
                "scope": scope, "metric": metric, "ptids": len(ptids),
                "b3_minus_drpa_mean": float(delta_by_ptid.mean()),
                "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                "dice_b3_win_rate": float((dice_delta > 0).mean()),
                "dice_drpa_win_rate": float((dice_delta < 0).mean()),
                "dice_tie_rate": float((dice_delta == 0).mean()),
            })
            boots.append(pd.DataFrame({"scope": scope, "metric": metric, "bootstrap_delta": draws}))
    return pd.DataFrame(out), pd.concat(boots, ignore_index=True)


def write_report(drpa_sum: pd.DataFrame, b3_sum: pd.DataFrame, paired: pd.DataFrame) -> None:
    merged = drpa_sum.merge(b3_sum, on="scope", suffixes=("_drpa", "_b3"))
    lines = [
        "# Canonical 100 Percent B3 Case-Level Recovery",
        "",
        "- Status: completed read-only inference recovery.",
        "- Validation: 247 visits / 85 PTIDs; raw canonical threshold sigmoid(logit) >= 0.5.",
        "- Checkpoints: DRPA validation_best and B3-Canonical validation_best; no training or checkpoint modification.",
        "- B3 configured trainable parameters: 81,930,624.",
        "",
        "## Raw-mask metric summary",
        "",
        "| scope | DRPA Dice | B3 Dice | B3-DRPA Dice | DRPA HD95 | B3 HD95 | DRPA Surface | B3 Surface | DRPA FP ml | B3 FP ml | DRPA FN ml | B3 FN ml | DRPA components | B3 components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples():
        lines.append("| %s | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |" % (
            row.scope, row.mean_dice_drpa, row.mean_dice_b3, row.mean_dice_b3 - row.mean_dice_drpa,
            row.mean_hd95_mm_drpa, row.mean_hd95_mm_b3, row.mean_surface_dice_2mm_drpa, row.mean_surface_dice_2mm_b3,
            row.mean_fp_volume_ml_drpa, row.mean_fp_volume_ml_b3, row.mean_fn_volume_ml_drpa, row.mean_fn_volume_ml_b3,
            row.mean_components_drpa, row.mean_components_b3))
    lines += ["", "## PTID-level paired bootstrap", ""]
    for row in paired.itertuples():
        lines.append("- %s / %s: B3-DRPA %.6f, 95%% CI [%.6f, %.6f], Dice PTID B3/DRPA/tie %.3f/%.3f/%.3f." % (
            row.scope, row.metric, row.b3_minus_drpa_mean, row.bootstrap_ci_low, row.bootstrap_ci_high,
            row.dice_b3_win_rate, row.dice_drpa_win_rate, row.dice_tie_rate))
    (OUT / "TASK1_B3_CASE_LEVEL_RECOVERY_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    val_frame = pd.read_csv(VAL_MANIFEST)
    assert_contract(val_frame)
    val = [CaseRecord(str(r.case_id), str(r.image_path), str(r.label_path)) for r in val_frame.itertuples()]
    dataset = BilateralGroupedPatchDataset(val, load_crop_spec(PILOT / "crop_spec.json"), cache_cases=False)
    cache = TextEmbeddingCache(str(BANK), str(MODEL), str(CACHE))
    atomic_json({
        "task": "100pct_b3_case_level_recovery",
        "validation_manifest": str(VAL_MANIFEST),
        "validation_manifest_sha256": sha256(VAL_MANIFEST),
        "drpa_checkpoint": str(DRPA_CKPT), "drpa_checkpoint_sha256": sha256(DRPA_CKPT),
        "b3_checkpoint": str(B3_CKPT), "b3_checkpoint_sha256": sha256(B3_CKPT),
        "evaluator_source": str(Path(evaluator.__file__).resolve()),
        "evaluator_sha256": sha256(Path(evaluator.__file__).resolve()),
        "threshold": "sigmoid(logit) >= 0.5", "lcc_primary": 0,
        "optimizer_step_executed": False, "training": False,
    }, OUT / "contract.json")
    drpa = recover_condition("drpa8", load_drpa, val, dataset, cache)
    b3 = recover_condition("b3_canonical", load_b3, val, dataset, cache)
    drpa_sum = summary(drpa, "drpa8")
    b3_sum = summary(b3, "b3_canonical")
    atomic_csv(pd.concat([drpa_sum, b3_sum], ignore_index=True), OUT / "case_level_summary.csv")
    paired, draws = paired_statistics(drpa, b3)
    atomic_csv(paired, OUT / "b3_vs_drpa_ptid_paired_bootstrap.csv")
    atomic_csv(draws, OUT / "b3_vs_drpa_ptid_bootstrap_draws.csv")
    write_report(drpa_sum, b3_sum, paired)
    atomic_json({"status": "TASK1_COMPLETE", "drpa_rows": len(drpa), "b3_rows": len(b3), "ptids": 85}, OUT / "progress.json")
    print("TASK1_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

