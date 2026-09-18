#!/usr/bin/env python3
"""One-time Rank-4 epoch-boundary save/reload equivalence gate.

This is an engineering smoke only. It does not create a formal model result.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
OUT = BASE / "quality_audit/rank_ablation_50/rank_resume_equivalence_smoke"

import sys
sys.path[:0] = [str(BASE / "scripts/train"), str(BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop")]
from b1_preprocessing import load_crop_spec  # noqa: E402
from rank_ablation_50_runner import (  # noqa: E402
    DEVICE, PROMPTS, SEED, EpochOrderSampler, RankAblationDRPAWrapper,
    TextEmbeddingCache, TimedReaderDataset, capture_rng_state, close_loader,
    collate, grad_stats, load_runtime_resume, loss_fp32, parameter_groups,
    records, restore_rng_state, save_runtime_resume, seed_all, worker_init,
    worker_snapshot,
)


def make_wrapper_optimizer_scaler():
    wrapper = RankAblationDRPAWrapper(
        str(BASE / "VoxTell_weights/voxtell_v1.1"),
        str(BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"),
        projection_rank=4, device=DEVICE,
    )
    wrapper.set_training_mode()
    groups = parameter_groups(wrapper)
    optimizer = torch.optim.AdamW([
        {"params": groups["cross_attention_lora"], "lr": 1e-4},
        {"params": groups["projection_adapter"], "lr": 1e-4},
        {"params": groups["decoder_stages"], "lr": 1e-5},
    ], weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return wrapper, groups, optimizer, scaler


def make_loader(dataset, indices):
    sampler = EpochOrderSampler()
    sampler.set_indices(indices)
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=32,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=collate, worker_init_fn=worker_init,
        multiprocessing_context="spawn",
    )
    return loader


def train_one(item, wrapper, groups, optimizer, scaler, text_cache):
    image = item["image"].to(DEVICE, non_blocking=True)
    target = item["mask"].to(DEVICE, non_blocking=True).float()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for prompt_index, prompt in enumerate(PROMPTS):
        embedding = text_cache.get([prompt], DEVICE)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = wrapper(image, embedding)
        with torch.autocast("cuda", enabled=False):
            loss = loss_fp32(logits, target[:, [prompt_index]])
        scaler.scale(loss / len(PROMPTS)).backward()
        losses.append(float(loss.detach().cpu()))
        del embedding, logits, loss
    scaler.unscale_(optimizer)
    grad_stats(groups)
    torch.nn.utils.clip_grad_norm_([p for ps in groups.values() for p in ps], 1.0,
                                   error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    return losses


def predict_losses(item, wrapper, text_cache):
    image = item["image"].to(DEVICE, non_blocking=True)
    target = item["mask"].to(DEVICE, non_blocking=True).float()
    losses, outputs = [], []
    with torch.no_grad():
        for prompt_index, prompt in enumerate(PROMPTS):
            embedding = text_cache.get([prompt], DEVICE)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = wrapper(image, embedding)
            with torch.autocast("cuda", enabled=False):
                loss = loss_fp32(logits, target[:, [prompt_index]])
            losses.append(float(loss.detach().cpu()))
            outputs.append(logits.detach().cpu())
    return losses, outputs


def config_contract(train_path: Path, val_path: Path) -> dict:
    import hashlib
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "experiment": "DRPA_RANK_ABLATION_50_AMP_MATCHED",
        "projection_rank": 4,
        "seed": SEED,
        "max_optimizer_steps": 6000,
        "train_manifest_sha256": digest(train_path),
        "val_manifest_sha256": digest(val_path),
        "batch_size": 1,
        "amp_forward": True,
        "autocast_dtype": "float16",
        "data_source": "DISK_CACHE",
        "num_workers": 32,
        "prefetch_factor": 2,
        "multiprocessing_context": "spawn",
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"equivalence output already exists: {OUT}")
    OUT.mkdir(parents=True)
    checkpoint = OUT / "runtime_resume_roundtrip.pt"
    train_path = AUDIT / "manifests/train_50pct.csv"
    val_path = AUDIT / "manifests/val_100pct_frozen.csv"
    train = records(train_path)
    dataset = TimedReaderDataset(train, load_crop_spec(PILOT / "crop_spec.json"), "disk")
    text_cache = TextEmbeddingCache(
        str(BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"),
        str(BASE / "VoxTell_weights/voxtell_v1.1"),
        str(BASE / "quality_audit/voxtell_mtl_peft/text_embedding_cache.npz"),
    )
    config = config_contract(train_path, val_path)
    epoch1_order = np.random.default_rng(SEED + 1).permutation(len(dataset)).tolist()
    epoch2_order = np.random.default_rng(SEED + 2).permutation(len(dataset)).tolist()
    started = time.time()

    seed_all()
    wrapper, groups, optimizer, scaler = make_wrapper_optimizer_scaler()
    first_loader = make_loader(dataset, [epoch1_order[0]])
    first_item = next(iter(first_loader))
    first_loss = train_one(first_item, wrapper, groups, optimizer, scaler, text_cache)
    first_workers = worker_snapshot(first_loader)
    close_loader(first_loader)
    del first_loader

    save_runtime_resume(
        checkpoint, wrapper, optimizer, scaler, config,
        global_step=1, completed_epoch=1, epoch_order=[epoch1_order[0]],
        steps=[{"step": 1, "case_id": first_item["case_id"][0], "loss": float(np.mean(first_loss))}],
        epochs=[{"epoch": 1, "steps_end": 1, "engineering_truncated_epoch": True}],
        validations=[], sampler_offset=1,
    )
    checkpoint_size = checkpoint.stat().st_size

    reference_loader = make_loader(dataset, [epoch2_order[0]])
    reference_item = next(iter(reference_loader))
    reference_forward_rng = capture_rng_state()
    reference_losses, reference_outputs = predict_losses(reference_item, wrapper, text_cache)
    reference_workers = worker_snapshot(reference_loader)
    reference_image = reference_item["image"].clone()
    reference_mask = reference_item["mask"].clone()
    reference_case = reference_item["case_id"][0]
    close_loader(reference_loader)
    del reference_loader, reference_item, wrapper, groups, optimizer, scaler
    gc.collect(); torch.cuda.empty_cache()

    seed_all()
    replay_wrapper, replay_groups, replay_optimizer, replay_scaler = make_wrapper_optimizer_scaler()
    loaded = load_runtime_resume(checkpoint, replay_wrapper, replay_optimizer, replay_scaler, config)
    replay_loader = make_loader(dataset, [epoch2_order[0]])
    replay_item = next(iter(replay_loader))
    restore_rng_state(reference_forward_rng)
    replay_losses, replay_outputs = predict_losses(replay_item, replay_wrapper, text_cache)
    replay_workers = worker_snapshot(replay_loader)

    image_equal = bool(torch.equal(reference_image, replay_item["image"]))
    mask_equal = bool(torch.equal(reference_mask, replay_item["mask"]))
    case_equal = reference_case == replay_item["case_id"][0]
    loss_abs = [abs(a - b) for a, b in zip(reference_losses, replay_losses)]
    logits_abs = [float((a - b).abs().max()) for a, b in zip(reference_outputs, replay_outputs)]
    sampler_state = loaded[-1]
    checks = {
        "loaded_global_step": loaded[0] == 1,
        "loaded_completed_epoch": loaded[1] == 1,
        "loaded_history": len(loaded[2]) == 1 and len(loaded[3]) == 1,
        "sampler_offset": sampler_state.get("completed_epoch_offset") == 1,
        "case_identity": case_equal,
        "image_exact": image_equal,
        "mask_exact": mask_equal,
        "loss_exact": max(loss_abs) == 0.0,
        "logits_exact": max(logits_abs) == 0.0,
        "workers_stable": bool(first_workers["all_workers_alive"] and
                               reference_workers["all_workers_alive"] and
                               replay_workers["all_workers_alive"]),
        "optimizer_state_present": len(replay_optimizer.state) > 0,
        "grad_scaler_state_equal": replay_scaler.state_dict() == torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )["grad_scaler_state"],
    }
    status = "PIPELINE_RUNTIME_RESUME_EQUIVALENCE_PASS" if all(checks.values()) else "PIPELINE_RUNTIME_RESUME_EQUIVALENCE_FAIL"
    result = {
        "status": status,
        "engineering_only": True,
        "counts_as_rank_result": False,
        "formal_checkpoint": False,
        "checkpoint_bytes": checkpoint_size,
        "first_case": first_item["case_id"][0],
        "next_case": reference_case,
        "max_loss_abs_diff": max(loss_abs),
        "max_logits_abs_diff": max(logits_abs),
        "checks": checks,
        "pipeline": {"workers": 32, "spawn": True, "persistent_workers": True,
                     "prefetch_factor": 2, "batch_size": 1,
                     "amp_forward": True, "loss": "FP32 Dice+BCE"},
        "wall_sec": time.time() - started,
    }
    (OUT / "RANK_RUNTIME_RESUME_EQUIVALENCE.json").write_text(json.dumps(result, indent=2) + "\n")
    report = [
        "# Rank Runtime Resume Equivalence", "", f"Status: `{status}`", "",
        "This one-step/truncated-epoch run is an engineering-only save/reload gate and is not a Rank result.", "",
        f"- First training case: `{result['first_case']}`",
        f"- Next-batch case: `{result['next_case']}`",
        f"- MRI exact equality: `{image_equal}`",
        f"- GT/mask exact equality: `{mask_equal}`",
        f"- Maximum next-batch loss difference: `{max(loss_abs):.12g}`",
        f"- Maximum next-batch logits difference: `{max(logits_abs):.12g}`",
        f"- Runtime checkpoint size: `{checkpoint_size}` bytes", "",
        "The test uses the frozen 32-worker spawn/persistent/prefetch=2, batch-1, AMP-forward + FP32-loss pipeline.",
    ]
    (OUT / "RANK_RUNTIME_RESUME_EQUIVALENCE.md").write_text("\n".join(report) + "\n")
    close_loader(replay_loader)
    print(json.dumps(result, indent=2), flush=True)
    if status.endswith("FAIL"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
