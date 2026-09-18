#!/usr/bin/env python3
"""Continue the frozen FullFT 100% run from step 6000 to step 9000.

This is an isolated convergence audit.  It restores model, AdamW and RNG
state from the formal step-6000 checkpoint and never writes to the formal
6000-step output directory.
"""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import gc
import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch


SEED = 20260809
START_STEP = 6000
TARGET_STEP = 9000
LR = 1e-5
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
EXPECTED_CONFIGURED = 440_029_541


def load_formal_module(base: Path):
    path = base / "scripts/fullft/train_fullft_10pct_formal.py"
    spec = importlib.util.spec_from_file_location("fullft_formal_for_extension", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load formal FullFT helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def convergence_decision(baseline: pd.Series, final: dict[str, Any]) -> str:
    delta = float(final["mean_dice"]) - float(baseline["mean_dice"])
    if float(final["mean_dice"]) < float(baseline["mean_dice"]):
        return "FULLFT_OVERFITTING_SIGNAL"
    if delta < 0.002:
        return "FULLFT_CONVERGENCE_PLATEAU"
    if delta <= 0.005:
        return "FULLFT_RESIDUAL_IMPROVEMENT"
    return "FULLFT_STILL_IMPROVING_STRONGLY"


def write_extension_report(
    out: Path,
    baseline: pd.Series,
    final: dict[str, Any],
    config: dict[str, Any],
    decision: str,
) -> None:
    fields = [
        ("Mean Dice", "mean_dice"),
        ("Hipp Dice", "hipp_dice"),
        ("EC Dice", "ec_dice"),
        ("PHG Dice", "phg_dice"),
        ("Amy Dice", "amy_dice"),
        ("HD95 (mm)", "hd95_mm"),
        ("Surface Dice@2mm", "surface_dice_2mm"),
        ("FP volume (ml)", "fp_volume_ml"),
        ("FN volume (ml)", "fn_volume_ml"),
        ("Components", "components"),
    ]
    lines = [
        f"# FullFT 100% Convergence Extension ({config['start_step']} → {config['target_step']})",
        "",
        f"Status: `{decision}`.",
        "",
        "This extension is a convergence-controlled upper-bound audit. The "
        "6000-step result remains the matched-update primary result and is not overwritten.",
        "",
        f"Source checkpoint: `{config['source_checkpoint']}`",
        f"Source checkpoint SHA256: `{config['source_checkpoint_sha256']}`",
        f"Optimizer state restored: `{config['optimizer_state_restored']}`",
        f"Scheduler state present/restored: `{config['scheduler_state_present']}/{config['scheduler_state_restored']}`",
        f"Global step restored: `{config['restored_global_step']}`",
        "",
        f"| Metric | Step {config['start_step']} | Step {config['target_step']} | {config['target_step']} - {config['start_step']} |",
        "|---|---:|---:|---:|",
    ]
    for label, key in fields:
        before = float(baseline[key])
        after = float(final[key])
        lines.append(f"| {label} | {before:.6f} | {after:.6f} | {after - before:+.6f} |")
    lines += [
        "",
        f"Mean Dice change: `{float(final['mean_dice']) - float(baseline['mean_dice']):+.6f}`.",
        "Decision thresholds are pre-registered for this audit only: <0.002 plateau; "
        "0.002–0.005 residual improvement; >0.005 strong improvement; validation decline is overfitting signal.",
        "No further automatic continuation is performed by this worker; an external one-shot supervisor may launch the pre-authorized 9000 → 12000 extension after successful validation.",
        "",
        "Frozen protocol: 337 PTIDs/971 visits, FP32, 192^3, batch 1, LR 1e-5, "
        "Qwen frozen, canonical Dice+BCE, canonical preprocessing/prompt/evaluator.",
    ]
    (out / "FULLFT_CONVERGENCE_EXTENSION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=6000)
    parser.add_argument("--target-step", type=int, default=9000)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.resolve()
    checkpoint_path = args.input_checkpoint.resolve()
    out = args.output_dir.resolve()
    start_step = int(args.start_step)
    target_step = int(args.target_step)
    if (start_step, target_step) not in {(6000, 9000), (9000, 12000)}:
        raise RuntimeError("Allowed extension ranges are 6000 -> 9000 or 9000 -> 12000")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if any(out.iterdir()):
        nonempty = [p.name for p in out.iterdir() if p.name != "checkpoints" or any(checkpoints.iterdir())]
        if nonempty:
            raise RuntimeError(f"Refusing to use non-empty extension output: {out}")

    formal = load_formal_module(base)
    config: dict[str, Any] = {
        "experiment": f"FULLFT_100P_CONVERGENCE_EXTENSION_{start_step}_TO_{target_step}",
        "model": "VoxTell-FullFT-100pct",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": formal.sha256(checkpoint_path),
        "start_step": start_step,
        "target_step": target_step,
        "restored_global_step": None,
        "optimizer_state_restored": False,
        "scheduler_state_present": False,
        "scheduler_state_restored": False,
        "configured_trainable": EXPECTED_CONFIGURED,
        "qwen": "Qwen3-Embedding-4B frozen",
        "train_ptids": 337,
        "train_visits": 971,
        "validation_ptids": 85,
        "validation_visits": 247,
        "input": "192x192x192",
        "batch_size": 1,
        "precision": "FP32",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "FP32 Dice+BCE",
        "seed": SEED,
        "preprocessing": "canonical",
        "prompt_contract": "8 fixed canonical prompts, equal loss contribution",
        "evaluator": str(base / "scripts/evaluate/evaluate_canonical.py"),
    }
    formal.atomic_json(out / "config.json", config)
    progress_path = out / "progress.json"
    failure_path = out / "failure_context.json"
    try:
        formal.refuse_busy_gpu()
        seed_all()
        (
            BilateralGroupedPatchDataset,
            CaseRecord,
            PROMPTS,
            load_crop_spec,
            TextEmbeddingCache,
            evaluator,
        ) = formal.load_modules(base)
        gate = formal.load_gate(base)
        train_manifest = base / "quality_audit/drpa_data_capacity_scaling/manifests/train_100pct.csv"
        validation_manifest = base / "quality_audit/drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv"
        crop_spec = base / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
        model_dir = base / "VoxTell_weights/voxtell_v1.1"
        embedding_bank = base / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
        train_frame = pd.read_csv(train_manifest, dtype=str)
        validation_frame = pd.read_csv(validation_manifest, dtype=str)
        if len(train_frame) != 971 or train_frame.ptid.nunique() != 337:
            raise RuntimeError("Unexpected frozen 100% train manifest")
        if len(validation_frame) != 247 or validation_frame.ptid.nunique() != 85:
            raise RuntimeError("Unexpected frozen validation manifest")
        if set(train_frame.ptid) & set(validation_frame.ptid):
            raise RuntimeError("PTID overlap between train and validation")
        train_records = formal.records(train_frame, CaseRecord)
        validation_records = formal.records(validation_frame, CaseRecord)
        train_dataset = BilateralGroupedPatchDataset(train_records, load_crop_spec(crop_spec), cache_cases=False)
        validation_dataset = BilateralGroupedPatchDataset(validation_records, load_crop_spec(crop_spec), cache_cases=False)
        device = torch.device("cuda")
        cache = TextEmbeddingCache(str(embedding_bank), str(model_dir), None)
        model = gate.initialise_official_model(base).to(device)
        ledger = gate.parameter_ledger(model)
        if ledger["configured_trainable_numel"] != EXPECTED_CONFIGURED:
            raise RuntimeError(f"FullFT parameter mismatch: {ledger['configured_trainable_numel']}")
        payload = torch.load(checkpoint_path, map_location="cpu")
        restored_step = int(payload.get("global_step", -1))
        if restored_step != start_step:
            raise RuntimeError(f"Expected checkpoint global_step=6000, got {restored_step}")
        model.load_state_dict(payload["model_state"], strict=True)
        unique_parameters = [parameter for _, parameter in gate.unique_named_parameters(model)]
        optimizer = torch.optim.AdamW(unique_parameters, lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer.load_state_dict(payload["optimizer_state"])
        config["restored_global_step"] = restored_step
        config["optimizer_state_restored"] = True
        if "scheduler_state" in payload:
            config["scheduler_state_present"] = True
            config["scheduler_state_restored"] = False
            raise RuntimeError("Checkpoint contains an unsupported scheduler_state")
        formal.atomic_json(out / "config.json", config)
        if "rng_state" in payload:
            torch.set_rng_state(payload["rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
        model.train()
        wrapper = formal.FullFTWrapper(model)
        atomic_json(progress_path, {
            "status": "RUNNING",
            "step": start_step,
            "event": "checkpoint_restored",
            "source_checkpoint": str(checkpoint_path),
            "updated_unix": time.time(),
        })
        start_time = time.time()
        step_rows: list[dict[str, Any]] = []
        epoch_rows: list[dict[str, Any]] = []
        validation_records_summary: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        step = start_step
        initial_epoch = start_step // len(train_dataset) + 1
        epoch = initial_epoch
        offset = start_step % len(train_dataset)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        while step < target_step:
            epoch_start = time.time()
            epoch_losses: list[float] = []
            permutation = np.random.default_rng(SEED + epoch).permutation(len(train_dataset))
            indices = permutation[offset:] if epoch == initial_epoch else permutation
            for index in indices:
                if step >= target_step:
                    break
                item = train_dataset[int(index)]
                image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
                target_all = item["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                individual_losses: list[float] = []
                step_start = time.perf_counter()
                for prompt_index, prompt in enumerate(PROMPTS):
                    embedding = cache.get([prompt], device)
                    logits = model(image, embedding)
                    loss = gate.canonical_loss(logits, target_all[:, [prompt_index]])
                    if not bool(torch.isfinite(loss).item()):
                        raise FloatingPointError(f"nonfinite loss at step {step + 1}, case {item['case_id']}, prompt {prompt}")
                    (loss / len(PROMPTS)).backward()
                    individual_losses.append(float(loss.detach().cpu()))
                    del embedding, logits, loss
                gradients = gate.gradient_ledger(model)
                if gradients["nonfinite_gradient_parameter_names"]:
                    raise FloatingPointError(f"nonfinite gradient at step {step + 1}: {gradients['nonfinite_gradient_parameter_names'][:3]}")
                pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(unique_parameters, CLIP_NORM, error_if_nonfinite=True).item())
                optimizer.step()
                if not formal.parameter_finite(model, gate):
                    raise FloatingPointError(f"nonfinite parameter after step {step + 1}")
                torch.cuda.synchronize(device)
                step += 1
                mean_loss = float(np.mean(individual_losses))
                step_rows.append({
                    "step": step,
                    "epoch": epoch,
                    "case_id": item["case_id"],
                    "loss": mean_loss,
                    "pre_clip_grad_norm": pre_clip_norm,
                    "runtime_gradient_active_numel": gradients["runtime_gradient_active_numel"],
                    "step_seconds": time.perf_counter() - step_start,
                    "peak_allocated_bytes_so_far": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes_so_far": int(torch.cuda.max_memory_reserved(device)),
                    "nan_inf": False,
                    "optimizer_state_entries": len(optimizer.state),
                })
                epoch_losses.append(mean_loss)
                del image, target_all
                if step % 25 == 0 or step == target_step:
                    pd.DataFrame(step_rows).to_csv(out / "training_dynamics.csv", index=False)
                    atomic_json(progress_path, {
                        "status": "RUNNING",
                        "step": step,
                        "epoch": epoch,
                        "last_case_id": item["case_id"],
                        "last_loss": mean_loss,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "updated_unix": time.time(),
                    })
                if step == target_step:
                    wrapper.model.eval()
                    validation_start = time.time()
                    rows = formal.evaluate_fullft(
                        f"VoxTell-FullFT-100pct@step{target_step:05d}",
                        wrapper,
                        validation_records,
                        validation_dataset,
                        cache,
                        evaluator,
                    )
                    summary = formal.summarize_fullft(rows, evaluator)
                    rows.to_csv(out / f"validation_rows_step_{target_step:05d}.csv", index=False)
                    summary.to_csv(out / f"validation_summary_step_{target_step:05d}.csv", index=False)
                    metric = formal.validation_record(summary, target_step)
                    metric["validation_runtime_sec"] = time.time() - validation_start
                    validation_records_summary.append(metric)
                    checkpoint_out = checkpoints / f"voxtell_fullft_100pct_step{target_step:05d}.pt"
                    checkpoint_hash = formal.save_checkpoint(model, optimizer, gate, target_step, config, checkpoint_out)
                    checkpoint_rows.append({
                        "step": target_step,
                        "path": str(checkpoint_out),
                        "sha256": checkpoint_hash,
                        "size_bytes": checkpoint_out.stat().st_size,
                    })
                    pd.DataFrame(validation_records_summary).to_csv(out / "validation_metrics.csv", index=False)
                    pd.DataFrame(checkpoint_rows).to_csv(out / "checkpoint_manifest.csv", index=False)
                    wrapper.model.train()
                    atomic_json(progress_path, {
                        "status": "VALIDATION_COMPLETE",
                        "step": target_step,
                        "event": "validation_and_checkpoint_complete",
                        "updated_unix": time.time(),
                    })
            epoch_rows.append({
                "epoch": epoch,
                "steps_start": max(start_step, (epoch - 1) * len(train_dataset)),
                "steps_end": step,
                "mean_train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "epoch_seconds": time.time() - epoch_start,
                "equivalent_epoch": step / len(train_dataset),
            })
            pd.DataFrame(epoch_rows).to_csv(out / "training_curve.csv", index=False)
            offset = 0
            epoch += 1

        baseline_path = args.baseline_metrics.resolve() if args.baseline_metrics else base / "quality_audit/voxtell_fullft_baseline/formal_100pct_retry_20260905/validation_metrics.csv"
        baseline_frame = pd.read_csv(baseline_path)
        baseline = baseline_frame[baseline_frame.step == start_step].iloc[0]
        final = validation_records_summary[-1]
        decision = convergence_decision(baseline, final)
        config["status"] = decision
        config["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        config["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        config["total_wall_seconds"] = time.time() - start_time
        config["validation"] = validation_records_summary
        config["checkpoint_manifest"] = checkpoint_rows
        formal.atomic_json(out / "run_summary.json", config)
        atomic_json(progress_path, {"status": decision, "step": target_step, "updated_unix": time.time()})
        write_extension_report(out, baseline, final, config, decision)
        print(json.dumps(config, indent=2))
    except Exception as error:
        status = "FULLFT_CONVERGENCE_EXTENSION_NUMERICAL_FAILURE" if isinstance(error, FloatingPointError) else "FULLFT_CONVERGENCE_EXTENSION_RUNTIME_FAILURE"
        failure = {"status": status, "error": repr(error), "traceback": traceback.format_exc(), "unix": time.time()}
        atomic_json(failure_path, failure)
        atomic_json(progress_path, {"status": status, "failure_path": str(failure_path), "updated_unix": time.time()})
        print(json.dumps(failure, indent=2))
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
