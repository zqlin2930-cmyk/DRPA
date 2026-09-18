#!/usr/bin/env python3
"""Frozen VoxTell-FullFT 10% formal training protocol.

One model, one fixed 10% train manifest, FP32, batch one, and exactly 6000
optimizer updates.  The only formal validation/checkpoint nodes are steps
3000 and 6000.  This runner does not resume, tune, or launch any follow-up
experiment; its checkpoint schema is nevertheless sufficient for an audited
manual resume if an external failure requires one.
"""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import gc
import hashlib
import importlib.util
import json
import os
import random
import subprocess
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
EXPECTED_CONFIGURED = 440_029_541
LR = 1e-5
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
MAX_STEPS = 6000
VALIDATION_STEPS = (3000, 6000)


class FullFTWrapper:
    """Evaluator-compatible wrapper around the official VoxTell model."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def __call__(self, image: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.model(image, text_embedding)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_busy_gpu() -> None:
    command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    active = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
    if active:
        raise RuntimeError(f"GPU already occupied; refusing formal start: {active}")


def load_gate(base: Path):
    path = base / "scripts/fullft/fullft_runtime_gate.py"
    spec = importlib.util.spec_from_file_location("fullft_runtime_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen FullFT gate helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_modules(base: Path):
    sys.path[:0] = [
        str(base / "VoxTell"),
        str(base / "quality_audit/voxtell_mtl_peft"),
        str(base / "quality_audit/voxtell_mtl_peft_pilot"),
        str(base / "quality_audit/voxtell_mtl_b1_bilateral_crop"),
        str(base / "quality_audit/voxtell_mtl_drpa8_pilot"),
    ]
    from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
    from text_embedding_cache import TextEmbeddingCache
    import evaluate_drpa8 as evaluator
    return BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec, TextEmbeddingCache, evaluator


def records(frame: pd.DataFrame, case_record) -> list[Any]:
    return [case_record(str(row.case_id), str(row.image_path), str(row.label_path)) for row in frame.itertuples()]


def parameter_finite(model: torch.nn.Module, gate) -> bool:
    return all(bool(torch.isfinite(parameter).all().item()) for _, parameter in gate.unique_named_parameters(model))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gate,
    step: int,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> str:
    """Atomically save model, optimizer and deterministic state at a formal node."""
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "global_step": step,
        "config": config,
        "configured_trainable": gate.parameter_ledger(model)["configured_trainable_numel"],
    }
    temporary = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    return sha256(checkpoint_path)


def evaluate_fullft(
    name: str,
    wrapper: FullFTWrapper,
    validation_records: list[Any],
    validation_dataset: Any,
    cache: Any,
    evaluator: Any,
) -> pd.DataFrame:
    """Frozen evaluator logic with FN volume added to new formal case rows."""
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(validation_records):
        item = validation_dataset[index]
        _, label, affine, _ = evaluator.load_official_reader_case(record)
        probabilities = evaluator.infer(wrapper, item, cache)
        spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))
        voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
        for prompt, patch_probability in probabilities.items():
            probability = evaluator.restore(patch_probability, item, label.shape, affine)
            raw_mask = probability >= 0.5
            raw_components, lcc_mask = evaluator.component_stats(raw_mask)
            reference = label == evaluator.ROI_LABEL_IDS[prompt]
            for lcc, mask in ((0, raw_mask), (1, lcc_mask)):
                false_positive_volume = float((mask & ~reference).sum() * voxel_ml)
                false_negative_volume = float((~mask & reference).sum() * voxel_ml)
                rows.append({
                    "condition": name,
                    "lcc": lcc,
                    "case_id": record.case_id,
                    "ptid": record.case_id.split("_")[0],
                    "prompt": prompt,
                    "structure": prompt.split(" ", 1)[1],
                    "side": "left" if prompt.startswith("left") else "right",
                    "dice": evaluator.dice(mask, reference),
                    "hd95_mm": evaluator.hd95(mask, reference, spacing),
                    "surface_dice_2mm": evaluator.surface_dice(mask, reference, spacing),
                    "connected_components_raw": raw_components,
                    "connected_components": int(evaluator.component_stats(mask)[0]),
                    "false_positive_volume_ml": false_positive_volume,
                    "false_negative_volume_ml": false_negative_volume,
                    "max_false_positive_distance_mm": evaluator.max_fp(mask, reference, spacing),
                    "pred_volume_ml": float(mask.sum() * voxel_ml),
                    "gt_volume_ml": float(reference.sum() * voxel_ml),
                    "empty_mask": int(not mask.any()),
                })
        print(f"validated {name} {record.case_id}", flush=True)
    return pd.DataFrame(rows)


def summarize_fullft(rows: pd.DataFrame, evaluator: Any) -> pd.DataFrame:
    summary = evaluator.summarize(rows)
    fn = (
        rows.groupby(["condition", "lcc", "structure"], as_index=False)["false_negative_volume_ml"]
        .mean()
    )
    overall = (
        rows.groupby(["condition", "lcc"], as_index=False)["false_negative_volume_ml"]
        .mean()
        .assign(structure="overall")
    )
    fn = pd.concat([fn, overall], ignore_index=True).rename(columns={"structure": "scope"})
    return summary.merge(fn, on=["condition", "lcc", "scope"], how="left", validate="one_to_one")


def validation_record(summary: pd.DataFrame, step: int) -> dict[str, Any]:
    raw = summary[summary.lcc == 0]
    row = raw[raw.scope == "overall"].iloc[0]
    result: dict[str, Any] = {
        "step": step,
        "mean_dice": float(row.dice),
        "hd95_mm": float(row.hd95_mm),
        "surface_dice_2mm": float(row.surface_dice_2mm),
        "fp_volume_ml": float(row.false_positive_volume_ml),
        "fn_volume_ml": float(row.false_negative_volume_ml),
        "components": float(row.connected_components),
    }
    lookup = {
        "hipp": "hippocampus",
        "ec": "entorhinal cortex",
        "phg": "parahippocampal gyrus",
        "amy": "amygdala",
    }
    for key, scope in lookup.items():
        result[f"{key}_dice"] = float(raw[raw.scope == scope].iloc[0].dice)
    return result


def write_report(out: Path, config: dict[str, Any], validation_rows: list[dict[str, Any]], status: str) -> None:
    subset_label = config["subset_label"]
    lines = [
        f"# VoxTell-FullFT {subset_label} Formal Experiment",
        "",
        f"Status: `{status}`.",
        "",
        f"Frozen protocol: FullFT configured trainable 440,029,541; Qwen frozen; {config['train_ptids']} PTIDs/{config['train_visits']} visits; FP32; batch 1; AdamW LR 1e-5 and weight decay 1e-5; 6000 optimizer steps. Canonical validation is held-out internal evaluation (85 PTIDs/247 visits) at steps 3000 and 6000 only.",
        "",
        "| Step | Mean Dice | Hipp | EC | PHG | Amy | HD95 mm | Surface Dice@2mm | FP mL | FN mL | Components |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation_rows:
        lines.append("| {step} | {mean_dice:.6f} | {hipp_dice:.6f} | {ec_dice:.6f} | {phg_dice:.6f} | {amy_dice:.6f} | {hd95_mm:.6f} | {surface_dice_2mm:.6f} | {fp_volume_ml:.6f} | {fn_volume_ml:.6f} | {components:.6f} |".format(**row))
    lines += [
        "",
        "Parameter comparison uses FullFT/DRPA = 440,029,541 / 10,969,696 = 40.113x. GPU speed and memory are recorded only for FullFT here; cross-GPU efficiency claims are not made.",
        "",
        "B1/DRPA/B3 comparison fields are intentionally deferred to the final paper-level paired analysis; no historical screen result is substituted here.",
        "",
        "Config provenance: `config.json`; case-level canonical metrics: `validation_rows_step_*.csv`; checkpoint SHA256 values: `checkpoint_manifest.json`.",
    ]
    suffix = "10P" if subset_label == "10%" else "100P"
    (out / f"VOXTELL_FULLFT_{suffix}_FORMAL_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__")))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--subset", choices=("10pct", "100pct"), default="10pct")
    args = parser.parse_args()
    base = args.base.resolve()
    out = args.output_dir or base / f"quality_audit/voxtell_fullft_baseline/formal_{args.subset}"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    train_manifest = base / f"quality_audit/drpa_data_capacity_scaling/manifests/train_{args.subset}.csv"
    validation_manifest = base / "quality_audit/drpa_data_capacity_scaling/manifests/val_100pct_frozen.csv"
    crop_spec = base / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
    model_dir = base / "VoxTell_weights/voxtell_v1.1"
    embedding_bank = base / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
    config = {
        "experiment": f"VOXTELL_FULLFT_{args.subset.upper()}_FORMAL",
        "model": f"VoxTell-FullFT-{args.subset}",
        "subset": args.subset,
        "seed": SEED,
        "configured_trainable": EXPECTED_CONFIGURED,
        "qwen": "Qwen3-Embedding-4B frozen",
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "train_manifest_sha256": sha256(train_manifest),
        "validation_manifest_sha256": sha256(validation_manifest),
        "train_ptids": 34 if args.subset == "10pct" else 337,
        "train_visits": 99 if args.subset == "10pct" else 971,
        "subset_label": "10%" if args.subset == "10pct" else "100%",
        "validation_ptids": 85,
        "validation_visits": 247,
        "input": "192x192x192",
        "batch_size": 1,
        "precision": "FP32",
        "gradient_checkpointing": False,
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "loss": "FP32 Dice+BCE",
        "prompt_contract": "8 fixed canonical prompts, equal loss contribution",
        "max_optimizer_steps": MAX_STEPS,
        "validation_steps": list(VALIDATION_STEPS),
        "evaluator": str(base / "scripts/evaluate/evaluate_canonical.py"),
        "evaluator_source": str(base / "quality_audit/voxtell_mtl_drpa8_pilot/evaluate_drpa8.py"),
    }
    atomic_json(out / "config.json", config)
    progress_path = out / "progress.json"
    failure_path = out / "failure_context.json"
    if (out / "run_summary.json").exists() or any(checkpoints.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing formal output: {out}")

    try:
        refuse_busy_gpu()
        seed_all()
        (
            BilateralGroupedPatchDataset,
            CaseRecord,
            PROMPTS,
            load_crop_spec,
            TextEmbeddingCache,
            evaluator,
        ) = load_modules(base)
        gate = load_gate(base)
        train_frame = pd.read_csv(train_manifest, dtype=str)
        validation_frame = pd.read_csv(validation_manifest, dtype=str)
        expected_visits = 99 if args.subset == "10pct" else 971
        expected_ptids = 34 if args.subset == "10pct" else 337
        if len(train_frame) != expected_visits or train_frame.ptid.nunique() != expected_ptids:
            raise RuntimeError(f"Frozen {args.subset} manifest has unexpected PTID/visit count")
        if len(validation_frame) != 247 or validation_frame.ptid.nunique() != 85:
            raise RuntimeError("Frozen validation manifest does not contain 85 PTIDs / 247 visits")
        if set(train_frame.ptid) & set(validation_frame.ptid):
            raise RuntimeError("PTID overlap between formal FullFT train and validation manifests")
        train_records = records(train_frame, CaseRecord)
        validation_records = records(validation_frame, CaseRecord)
        train_dataset = BilateralGroupedPatchDataset(train_records, load_crop_spec(crop_spec), cache_cases=False)
        validation_dataset = BilateralGroupedPatchDataset(validation_records, load_crop_spec(crop_spec), cache_cases=False)
        device = torch.device("cuda")
        cache = TextEmbeddingCache(str(embedding_bank), str(model_dir), None)
        model = gate.initialise_official_model(base).to(device)
        ledger = gate.parameter_ledger(model)
        if ledger["configured_trainable_numel"] != EXPECTED_CONFIGURED:
            raise RuntimeError(f"FullFT parameter mismatch: {ledger['configured_trainable_numel']}")
        model.train()
        wrapper = FullFTWrapper(model)
        unique_parameters = [parameter for _, parameter in gate.unique_named_parameters(model)]
        optimizer = torch.optim.AdamW(unique_parameters, lr=LR, weight_decay=WEIGHT_DECAY)
        atomic_json(progress_path, {"status": "RUNNING", "step": 0, "started_unix": time.time(), "config": config})
        start_time = time.time()
        step_rows: list[dict[str, Any]] = []
        epoch_rows: list[dict[str, Any]] = []
        validation_records_summary: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        step = 0
        epoch = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        while step < MAX_STEPS:
            epoch += 1
            epoch_start = time.time()
            epoch_losses: list[float] = []
            for index in np.random.default_rng(SEED + epoch).permutation(len(train_dataset)):
                if step >= MAX_STEPS:
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
                        raise FloatingPointError(f"nonfinite loss at pending step {step + 1}, case {item['case_id']}, prompt {prompt}")
                    (loss / len(PROMPTS)).backward()
                    individual_losses.append(float(loss.detach().cpu()))
                    del embedding, logits, loss
                gradients = gate.gradient_ledger(model)
                if gradients["nonfinite_gradient_parameter_names"]:
                    raise FloatingPointError(f"nonfinite gradient at pending step {step + 1}: {gradients['nonfinite_gradient_parameter_names'][:3]}")
                pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(unique_parameters, CLIP_NORM, error_if_nonfinite=True).item())
                optimizer.step()
                if not parameter_finite(model, gate):
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
                if step % 25 == 0 or step in VALIDATION_STEPS:
                    pd.DataFrame(step_rows).to_csv(out / "training_dynamics.csv", index=False)
                    atomic_json(progress_path, {
                        "status": "RUNNING", "step": step, "epoch": epoch,
                        "last_case_id": item["case_id"], "last_loss": mean_loss,
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "updated_unix": time.time(),
                    })
                if step in VALIDATION_STEPS:
                    wrapper.model.eval()
                    validation_start = time.time()
                    rows = evaluate_fullft(f"VoxTell-FullFT-{args.subset}@step{step:05d}", wrapper, validation_records, validation_dataset, cache, evaluator)
                    summary = summarize_fullft(rows, evaluator)
                    rows.to_csv(out / f"validation_rows_step_{step:05d}.csv", index=False)
                    summary.to_csv(out / f"validation_summary_step_{step:05d}.csv", index=False)
                    metric = validation_record(summary, step)
                    metric["validation_runtime_sec"] = time.time() - validation_start
                    validation_records_summary.append(metric)
                    checkpoint_path = checkpoints / f"voxtell_fullft_{args.subset}_step{step:05d}.pt"
                    checkpoint_hash = save_checkpoint(model, optimizer, gate, step, config, checkpoint_path)
                    checkpoint_rows.append({"step": step, "path": str(checkpoint_path), "sha256": checkpoint_hash, "size_bytes": checkpoint_path.stat().st_size})
                    pd.DataFrame(validation_records_summary).to_csv(out / "validation_metrics.csv", index=False)
                    pd.DataFrame(checkpoint_rows).to_csv(out / "checkpoint_manifest.csv", index=False)
                    wrapper.model.train()
                    atomic_json(progress_path, {"status": "RUNNING", "step": step, "event": "validation_and_checkpoint_complete", "updated_unix": time.time()})
                    print(json.dumps({"step": step, "validation": metric, "checkpoint": str(checkpoint_path)}, indent=2), flush=True)
            epoch_rows.append({"epoch": epoch, "steps_end": step, "mean_train_loss": float(np.mean(epoch_losses)), "epoch_seconds": time.time() - epoch_start, "equivalent_epoch": step / len(train_dataset)})
            pd.DataFrame(epoch_rows).to_csv(out / "training_curve.csv", index=False)

        status = f"FULLFT_{args.subset.upper()}_COMPLETE"
        summary = {
            "status": status,
            "steps": step,
            "train_visits": len(train_records),
            "train_ptids": int(train_frame.ptid.nunique()),
            "validation_visits": len(validation_records),
            "validation_ptids": int(validation_frame.ptid.nunique()),
            "configured_trainable": ledger["configured_trainable_numel"],
            "runtime_gradient_active": step_rows[-1]["runtime_gradient_active_numel"],
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "total_wall_seconds": time.time() - start_time,
            "validation": validation_records_summary,
            "checkpoint_manifest": checkpoint_rows,
        }
        atomic_json(out / "run_summary.json", summary)
        atomic_json(progress_path, {"status": status, "step": step, "updated_unix": time.time()})
        write_report(out, config, validation_records_summary, status)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        suffix = args.subset.upper()
        status = f"FULLFT_{suffix}_NUMERICAL_FAILURE" if isinstance(error, FloatingPointError) else f"FULLFT_{suffix}_RUNTIME_FAILURE"
        failure = {"status": status, "error": repr(error), "traceback": traceback.format_exc(), "unix": time.time()}
        atomic_json(failure_path, failure)
        atomic_json(progress_path, {"status": status, "updated_unix": time.time(), "failure_path": str(failure_path)})
        write_report(out, config, [], status)
        print(json.dumps(failure, indent=2), flush=True)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
