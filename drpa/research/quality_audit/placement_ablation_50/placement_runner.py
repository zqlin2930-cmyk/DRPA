#!/usr/bin/env python3
"""AMP-forward placement ablation runner for P1/P2 at 50% data."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PEFT = BASE / "quality_audit/voxtell_mtl_peft"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
PLACEMENT = BASE / "quality_audit/placement_ablation_50"
MODEL = BASE / "VoxTell_weights/voxtell_v1.1"
BANK = BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
CACHE = PEFT / "text_embedding_cache.npz"
DEVICE = torch.device("cuda")
SEED = 20260809
EXPECTED = {"b1_projection": 737536, "b1_decoder": 10527072}

sys.path[:0] = [str(PILOT), str(PEFT), str(PREP), str(PLACEMENT)]
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
from text_embedding_cache import TextEmbeddingCache
import evaluate_drpa8 as evaluator
from placement_wrappers import build_placement_wrapper


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_busy_gpu() -> None:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if out:
        raise RuntimeError("GPU already occupied: " + out)


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path, dtype=str)
    return [CaseRecord(str(row.case_id), str(row.image_path), str(row.label_path))
            for row in frame.itertuples()]


def loss_fp32(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z = logits.float()
    y = target.float()
    bce = F.binary_cross_entropy_with_logits(z, y)
    p = torch.sigmoid(z)
    dims = tuple(range(2, p.ndim))
    inter = (p * y).sum(dims)
    den = p.sum(dims) + y.sum(dims)
    dice_loss = 1.0 - ((2.0 * inter + 1e-5) / (den + 1e-5)).mean()
    return dice_loss + bce


def group_params(wrapper: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for _, param, group in wrapper.trainable_parameter_groups():
        groups.setdefault(group, []).append(param)
    return groups


def grad_stats(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for group, params in groups.items():
        none = zero = nonzero = 0
        sq = 0.0
        for param in params:
            if param.grad is None:
                none += 1
                continue
            if torch.count_nonzero(param.grad).item() == 0:
                zero += 1
            else:
                nonzero += 1
            sq += float(torch.sum(param.grad.float() ** 2).cpu())
        result[group] = {
            "parameter_tensors": len(params),
            "grad_none_tensors": none,
            "zero_grad_tensors": zero,
            "nonzero_grad_tensors": nonzero,
            "grad_norm": sq ** 0.5,
        }
    return result


def save_state(wrapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
               path: Path, step: int, kind: str, config: dict) -> None:
    state = {name: param.detach().cpu() for name, param in wrapper.model.named_parameters()
             if param.requires_grad}
    payload = {
        "format": "placement_ablation_amp_v1",
        "trainable_model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "global_step": step,
        "model": kind,
        "config": config,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def validate(wrapper: torch.nn.Module, val: list[CaseRecord], val_ds,
             cache: TextEmbeddingCache, step: int, out: Path, kind: str,
             train_len: int) -> dict:
    rows = evaluator.evaluate_condition(f"PLACEMENT_{kind}@step{step:05d}", wrapper,
                                        val, val_ds, cache)
    frame = pd.DataFrame(rows)
    summary = evaluator.summarize(frame)
    summary.insert(1, "step", step)
    frame.to_csv(out / f"validation_rows_step_{step:05d}.csv", index=False)
    summary.to_csv(out / f"validation_summary_step_{step:05d}.csv", index=False)
    raw = summary[summary.lcc == 0]
    overall = raw[raw.scope == "overall"].iloc[0]
    result = {
        "step": step,
        "equivalent_epoch": step / train_len,
        "mean_dice": float(overall.dice),
        "hd95_mm": float(overall.hd95_mm),
        "surface_dice_2mm": float(overall.surface_dice_2mm),
        "components": float(overall.connected_components),
        "fp_volume_ml": float(overall.false_positive_volume_ml),
        "max_fp_distance_mm": float(overall.max_fp_distance_mm),
        "empty_mask_rate": float(overall.empty_mask_rate),
    }
    for scope, key in [("hippocampus", "hipp"), ("entorhinal cortex", "ec"),
                       ("parahippocampal gyrus", "phg"), ("amygdala", "amy")]:
        result[key + "_dice"] = float(raw[raw.scope == scope].iloc[0].dice)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--subset", default="50pct")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    refuse_busy_gpu()
    seed_all()

    train_path = AUDIT / "manifests" / f"train_{args.subset}.csv"
    val_path = AUDIT / "manifests/val_100pct_frozen.csv"
    train_frame = pd.read_csv(train_path, dtype=str)
    val_frame = pd.read_csv(val_path, dtype=str)
    train = records(train_path)
    val = records(val_path)
    spec = load_crop_spec(PILOT / "crop_spec.json")
    train_ds = BilateralGroupedPatchDataset(train, spec, cache_cases=False)
    val_ds = BilateralGroupedPatchDataset(val, spec, cache_cases=False)
    text_cache = TextEmbeddingCache(str(BANK), str(MODEL), str(CACHE))
    wrapper = build_placement_wrapper(args.kind, str(MODEL), str(BANK), DEVICE)
    groups = group_params(wrapper)
    counts = {group: sum(p.numel() for p in params) for group, params in groups.items()}
    actual = sum(counts.values())
    if actual != EXPECTED[args.kind]:
        raise RuntimeError(f"{args.kind} count {actual} != {EXPECTED[args.kind]}")
    config = {
        "experiment": "PLACEMENT_ABLATION_50_AMP_MATCHED",
        "model": args.kind,
        "subset": args.subset,
        "seed": SEED,
        "max_optimizer_steps": args.max_steps,
        "train_visits": len(train),
        "train_ptids": int(train_frame.ptid.nunique()),
        "val_visits": len(val),
        "val_ptids": int(val_frame.ptid.nunique()),
        "batch_size": 1,
        "gradient_accumulation": 1,
        "preprocessing": "canonical RAS + official reader-space + fixed bilateral 192^3 crop",
        "prompts": "8 fixed canonical prompts",
        "loss": "FP32 Dice+BCE",
        "amp_forward": True,
        "autocast_dtype": "float16",
        "grad_scaler": True,
        "clip_norm": 1.0,
        "weight_decay": 1e-5,
        "parameter_groups": counts,
        "trainable_parameters": actual,
        "train_manifest_sha256": sha256(train_path),
        "val_manifest_sha256": sha256(val_path),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    pd.DataFrame([{"module_group": g, "parameter_count": n, "requires_grad": True}
                  for g, n in counts.items()]).to_csv(out / "parameter_summary.csv", index=False)
    (out / "initialization.json").write_text(json.dumps({
        "status": "FRESH_OFFICIAL_INITIALIZATION",
        "model": args.kind,
        "subset": args.subset,
        "seed": SEED,
        "train_manifest_sha256": config["train_manifest_sha256"],
        "val_manifest_sha256": config["val_manifest_sha256"],
        "trainable_parameters": actual,
    }, indent=2))

    optimizer_groups = [{"params": groups["cross_attention_lora"], "lr": 1e-4}]
    if args.kind == "b1_projection":
        optimizer_groups.append({"params": groups["projection_adapter"], "lr": 1e-4})
    else:
        optimizer_groups.append({"params": groups["decoder_stages"], "lr": 1e-5})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    steps = []
    epochs = []
    validations = []
    epoch = 0
    start = time.time()
    while len(steps) < args.max_steps:
        epoch += 1
        epoch_start = time.time()
        losses = []
        order = np.random.default_rng(SEED + epoch).permutation(len(train_ds))
        for index in order:
            if len(steps) >= args.max_steps:
                break
            item = train_ds[int(index)]
            image = item["image"].unsqueeze(0).to(DEVICE)
            target = item["mask"].unsqueeze(0).to(DEVICE).float()
            optimizer.zero_grad(set_to_none=True)
            prompt_losses = []
            for prompt_index, prompt in enumerate(PROMPTS):
                embedding = text_cache.get([prompt], DEVICE)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = wrapper(image, embedding)
                with torch.autocast("cuda", enabled=False):
                    loss = loss_fp32(logits, target[:, [prompt_index]])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss step={len(steps)+1} case={item['case_id']}")
                scaler.scale(loss / len(PROMPTS)).backward()
                prompt_losses.append(float(loss.detach().cpu()))
                del embedding, logits, loss
            scaler.unscale_(optimizer)
            before = grad_stats(groups)
            params = [p for ps in groups.values() for p in ps]
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            after = grad_stats(groups)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            step = len(steps) + 1
            steps.append({
                "step": step,
                "epoch": epoch,
                "case_id": item["case_id"],
                "loss": float(np.mean(prompt_losses)),
                "pre_clip_grad_json": json.dumps(before),
                "post_clip_grad_json": json.dumps(after),
                "scaler_before": scale_before,
                "scaler_after": float(scaler.get_scale()),
                "step_seconds": time.time() - epoch_start,
                "nan_inf": False,
                "skipped_step": False,
            })
            losses.extend(prompt_losses)
            del image, target
            gc.collect()
            torch.cuda.empty_cache()
            if step == args.max_steps and args.max_steps == 6000:
                result = validate(wrapper, val, val_ds, text_cache, step, out,
                                  args.kind, len(train_ds))
                validations.append(result)
                save_state(wrapper, optimizer, out / "checkpoints" / f"step_{step:05d}.pt",
                           step, args.kind, config)
                pd.DataFrame(validations).to_csv(out / "validation_metrics.csv", index=False)
                print(json.dumps({"step": step, "validation": result}), flush=True)
        epochs.append({
            "epoch": epoch,
            "steps_end": len(steps),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "elapsed_sec": time.time() - epoch_start,
            "equivalent_epoch": len(steps) / len(train_ds),
        })
        pd.DataFrame(steps).to_csv(out / "training_dynamics.csv", index=False)
        pd.DataFrame(epochs).to_csv(out / "training_curve.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "model": args.kind,
        "subset": args.subset,
        "steps": len(steps),
        "train_visits": len(train),
        "train_ptids": int(train_frame.ptid.nunique()),
        "val_visits": len(val),
        "val_ptids": int(val_frame.ptid.nunique()),
        "validation": validations[-1] if validations else None,
        "total_wall_sec": time.time() - start,
        "parameter_groups": counts,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
