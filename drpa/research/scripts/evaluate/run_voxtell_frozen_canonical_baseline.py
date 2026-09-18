#!/usr/bin/env python3
"""Read-only 247-visit frozen original VoxTell baseline evaluation.

This entrypoint never creates an optimizer, calls backward, or writes model
weights.  It reuses the audited FullFT evaluation helpers so the preprocessing,
prompt bank, restoration, threshold, and metric definitions remain canonical.
"""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


EXPECTED_VISITS = 247
EXPECTED_PTIDS = 85
EXPECTED_TRAINABLE = 0
BASE_CHECKPOINT = Path("VoxTell_weights/voxtell_v1.1/fold_0/checkpoint_final.pth")
VALIDATION_MANIFEST = Path(
    "quality_audit/drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv"
)
EVALUATOR_SOURCE = Path("quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py")
EVALUATOR_ENTRY = Path("scripts/evaluate/evaluate_canonical.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ptid_summary(rows: pd.DataFrame) -> pd.DataFrame:
    raw = rows[rows.lcc == 0].copy()
    metric_cols = [
        "dice", "hd95_mm", "surface_dice_2mm",
        "false_positive_volume_ml", "false_negative_volume_ml",
        "connected_components", "pred_volume_ml", "gt_volume_ml",
    ]
    by_structure = raw.groupby(["ptid", "structure"], as_index=False)[metric_cols].mean()
    overall = raw.groupby(["ptid"], as_index=False)[metric_cols].mean()
    overall.insert(1, "structure", "overall")
    result = pd.concat([overall, by_structure], ignore_index=True)
    result.insert(0, "condition", raw.condition.iloc[0])
    result.insert(1, "lcc", 0)
    return result.sort_values(["ptid", "structure"]).reset_index(drop=True)


def report_text(base: Path, out: Path, config: dict, summary: pd.DataFrame,
                ptids: pd.DataFrame, elapsed: float, peak_alloc: int,
                peak_reserved: int) -> str:
    raw = summary[(summary.lcc == 0)]
    scopes = [
        ("Overall", "overall"),
        ("Hipp", "hippocampus"),
        ("EC", "entorhinal cortex"),
        ("PHG", "parahippocampal gyrus"),
        ("Amy", "amygdala"),
    ]
    lines = [
        "# VoxTell Frozen Canonical Baseline",
        "",
        "Status: `VOXTELL_FROZEN_CANONICAL_BASELINE_COMPLETE`.",
        "",
        "This is the original frozen VoxTell model evaluated without training, LoRA, backward, optimizer, or parameter updates. It is not the 12-case pilot B0.",
        "",
        "## Frozen protocol",
        "",
        f"- Validation: {config['validation_visits']} visits / {config['validation_ptids']} PTIDs; canonical internal evaluation cohort.",
        "- Model: original VoxTell checkpoint; Qwen3-Embedding-4B frozen.",
        "- Trainable parameters: **0** (all unique model parameters have `requires_grad=False`).",
        "- Input: canonical preprocessing, 192^3 reader-space crop, canonical prompts and prompt embedding cache.",
        "- Readout: original VoxTell forward path; probability threshold `>= 0.5`; raw mask is primary and LCC is diagnostic only.",
        f"- Evaluator: `{base / EVALUATOR_ENTRY}`; metric source `{base / EVALUATOR_SOURCE}`.",
        "",
        "## Validation results (raw mask, no LCC post-processing)",
        "",
        "| Scope | N | Dice | HD95 (mm) | Surface Dice@2mm | FP (mL) | FN (mL) | Components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, scope in scopes:
        row = raw[raw.scope == scope].iloc[0]
        lines.append(
            f"| {label} | {int(row.n_rows)} | {row.dice:.6f} | {row.hd95_mm:.6f} | "
            f"{row.surface_dice_2mm:.6f} | {row.false_positive_volume_ml:.6f} | "
            f"{row.false_negative_volume_ml:.6f} | {row.connected_components:.6f} |"
        )
    lines += [
        "",
        "## PTID-level aggregation",
        "",
        f"PTID metrics are the mean of visit/ROI rows within each PTID; {ptids.ptid.nunique()} PTIDs are present. The machine-readable file is `ptid_level_metrics.csv`.",
        "",
        "## Resource and integrity record",
        "",
        f"- Wall-clock inference time: `{elapsed:.3f}` s.",
        f"- Peak CUDA allocated: `{peak_alloc / 2**30:.3f}` GiB; reserved: `{peak_reserved / 2**30:.3f}` GiB.",
        f"- Checkpoint SHA256: `{config['checkpoint_sha256']}`.",
        f"- Validation manifest SHA256: `{config['validation_manifest_sha256']}`.",
        "- No checkpoint, GT, manifest, or model weight was modified.",
        "",
        "## Scientific interpretation",
        "",
        "Parameter efficiency is not compute efficiency. DRPA/B3 reduce trainable parameter count, but their same-GPU VRAM and sec/step must be compared separately; this frozen baseline supplies the zero-trainable-parameter reference and does not replace the registered B1/DRPA/B3/FullFT benchmark.",
        "",
        "Output files: `validation_rows.csv`, `validation_summary.csv`, `ptid_level_metrics.csv`, and `run_summary.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__")))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.resolve()
    out = (args.output_dir or base / "quality_audit/voxtell_frozen_canonical_baseline").resolve()
    out.mkdir(parents=True, exist_ok=True)
    final_report = out / "VOXTELL_FROZEN_CANONICAL_BASELINE_REPORT.md"
    if final_report.exists():
        raise RuntimeError(f"Refusing to overwrite completed baseline: {final_report}")

    manifest = base / VALIDATION_MANIFEST
    checkpoint = base / BASE_CHECKPOINT
    if not manifest.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"Missing manifest/checkpoint: {manifest} / {checkpoint}")
    frame = pd.read_csv(manifest, dtype=str)
    if len(frame) != EXPECTED_VISITS or frame.ptid.nunique() != EXPECTED_PTIDS:
        raise RuntimeError(f"Validation manifest mismatch: {len(frame)} visits / {frame.ptid.nunique()} PTIDs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the canonical VoxTell inference baseline")
    active = os.popen("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits").read().strip()
    if active:
        raise RuntimeError(f"GPU already occupied; refusing baseline start: {active}")

    formal = load_module(base / "scripts/fullft/train_fullft_10pct_formal.py", "fullft_eval_helpers")
    (
        BilateralGroupedPatchDataset, CaseRecord, _prompts, load_crop_spec,
        TextEmbeddingCache, evaluator,
    ) = formal.load_modules(base)
    gate = load_module(base / "scripts/fullft/fullft_runtime_gate.py", "fullft_gate_helpers")
    crop_spec_path = base / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
    model_dir = base / "VoxTell_weights/voxtell_v1.1"
    embedding_bank = base / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
    cache_path = base / "quality_audit/voxtell_mtl_peft/text_embedding_cache.npz"
    records = formal.records(frame, CaseRecord)
    dataset = BilateralGroupedPatchDataset(records, load_crop_spec(crop_spec_path), cache_cases=False)
    cache = TextEmbeddingCache(str(embedding_bank), str(model_dir), str(cache_path))
    device = torch.device("cuda")
    model = gate.initialise_official_model(base).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    configured = sum(parameter.numel() for _, parameter in gate.unique_named_parameters(model) if parameter.requires_grad)
    if configured != EXPECTED_TRAINABLE:
        raise RuntimeError(f"Frozen baseline trainable parameter mismatch: {configured}")
    model.eval()
    wrapper = formal.FullFTWrapper(model)
    config = {
        "experiment": "VOXTELL_FROZEN_CANONICAL_BASELINE",
        "model": "VoxTell-Frozen-Canonical",
        "trainable_params": configured,
        "validation_manifest": str(manifest),
        "validation_manifest_sha256": sha256(manifest),
        "validation_visits": len(frame),
        "validation_ptids": int(frame.ptid.nunique()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "threshold": ">= 0.5",
        "inference_mode": "torch.inference_mode + canonical CUDA autocast FP16",
        "canonical_evaluator_source": str(base / EVALUATOR_SOURCE),
        "no_training_or_parameter_update": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rows = formal.evaluate_fullft(
        "VoxTell-Frozen-Canonical", wrapper, records, dataset, cache, evaluator
    )
    elapsed = time.perf_counter() - started
    row_df = rows
    summary = formal.summarize_fullft(row_df, evaluator)
    ptids = ptid_summary(row_df)
    row_df.to_csv(out / "validation_rows.csv", index=False)
    summary.to_csv(out / "validation_summary.csv", index=False)
    ptids.to_csv(out / "ptid_level_metrics.csv", index=False)
    config.update({
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "rows": int(len(row_df)),
        "summary_rows": int(len(summary)),
        "ptid_summary_rows": int(len(ptids)),
    })
    (out / "run_summary.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    final_report.write_text(report_text(
        base, out, config, summary, ptids, elapsed,
        config["peak_cuda_allocated_bytes"], config["peak_cuda_reserved_bytes"],
    ))
    print(json.dumps(config, indent=2, sort_keys=True))
    print(summary[summary.lcc == 0].to_string(index=False))


if __name__ == "__main__":
    main()
