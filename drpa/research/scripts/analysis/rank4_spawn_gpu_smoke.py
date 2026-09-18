#!/usr/bin/env python3
"""GPU integration smoke for the repaired Rank-4 data path; never writes a checkpoint."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
OUT = Path(os.environ.get("RANK4_SMOKE_OUTPUT", str(BASE / "quality_audit/rank_ablation_50/rank4_spawn_gpu_smoke")))
STEPS = int(os.environ.get("RANK4_SMOKE_STEPS", "20"))

sys.path[:0] = [str(BASE / "scripts/train"), str(PREP)]
from b1_preprocessing import load_crop_spec  # noqa: E402
from rank_ablation_50_runner import (  # noqa: E402
    DEVICE,
    PROMPTS,
    SEED,
    EpochOrderSampler,
    TextEmbeddingCache,
    TimedReaderDataset,
    collate,
    grad_stats,
    loss_fp32,
    parameter_groups,
    records,
    seed_all,
    worker_init,
    worker_snapshot,
    RankAblationDRPAWrapper,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if OUT.exists():
        raise RuntimeError(f"smoke output already exists: {OUT}")
    OUT.mkdir(parents=True)
    train = records(AUDIT / "manifests/train_50pct.csv")
    if len(train) != 461:
        raise RuntimeError(f"expected 461 training visits, got {len(train)}")
    seed_all()
    dataset = TimedReaderDataset(train, load_crop_spec(PILOT / "crop_spec.json"), "disk")
    sampler = EpochOrderSampler()
    sampler.set_indices(np.random.default_rng(SEED + 1).permutation(len(dataset)))
    loader = DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=32,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=collate, worker_init_fn=worker_init,
        multiprocessing_context="spawn",
    )
    wrapper = RankAblationDRPAWrapper(
        str(BASE / "VoxTell_weights/voxtell_v1.1"),
        str(BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"),
        projection_rank=4, device=DEVICE,
    )
    wrapper.set_training_mode()
    groups = parameter_groups(wrapper)
    count = sum(p.numel() for ps in groups.values() for p in ps)
    if count != 10_748_384:
        raise RuntimeError(f"rank4 parameter mismatch: {count}")
    optimizer = torch.optim.AdamW([
        {"params": groups["cross_attention_lora"], "lr": 1e-4},
        {"params": groups["projection_adapter"], "lr": 1e-4},
        {"params": groups["decoder_stages"], "lr": 1e-5},
    ], weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    cache = TextEmbeddingCache(
        str(BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"),
        str(BASE / "VoxTell_weights/voxtell_v1.1"),
        str(BASE / "quality_audit/voxtell_mtl_peft/text_embedding_cache.npz"),
    )
    iterator = iter(loader)
    rows = []
    started = time.time()
    try:
        for step in range(1, STEPS + 1):
            item = next(iterator)
            image = item["image"].to(DEVICE, non_blocking=True)
            target = item["mask"].to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for prompt_index, prompt in enumerate(PROMPTS):
                embedding = cache.get([prompt], DEVICE)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = wrapper(image, embedding)
                with torch.autocast("cuda", enabled=False):
                    loss = loss_fp32(logits, target[:, [prompt_index]])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss step={step} case={item['case_id'][0]}")
                scaler.scale(loss / len(PROMPTS)).backward()
                losses.append(float(loss.detach().cpu()))
                del embedding, logits, loss
            scaler.unscale_(optimizer)
            stats = grad_stats(groups)
            all_params = [p for ps in groups.values() for p in ps]
            torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            snapshot = worker_snapshot(loader)
            if not snapshot["all_workers_alive"]:
                raise RuntimeError(f"worker health failed at step={step}: {snapshot}")
            rows.append({
                "step": step, "case_id": item["case_id"][0],
                "loss": float(np.mean(losses)), "grad_scale": float(scaler.get_scale()),
                "worker_count": snapshot["worker_count"],
                "workers_alive": snapshot["all_workers_alive"],
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "grad_stats": json.dumps(stats),
            })
            del image, target
        result = {
            "status": "PASS", "steps": STEPS, "exit_code": 0,
            "trainable_parameters": count, "batch_size": 1,
            "num_workers": 32, "multiprocessing_context": "spawn",
            "persistent_workers": True, "prefetch_factor": 2,
            "amp_forward": True, "loss": "FP32 Dice+BCE",
            "optimizer_step": True, "grad_scaler": True,
            "nan_inf": False, "gpu_oom": False,
            "workers_stable": all(row["workers_alive"] for row in rows),
            "peak_allocated_gib": max(row["peak_allocated_gib"] for row in rows),
            "peak_reserved_gib": max(row["peak_reserved_gib"] for row in rows),
            "wall_sec": time.time() - started,
            "formal_checkpoint_written": False,
            "counts_as_rank_result": False,
        }
        (OUT / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
        pd.DataFrame(rows).to_csv(OUT / "smoke_steps.csv", index=False)
    finally:
        iterator = None
        loader_iter = getattr(loader, "_iterator", None)
        if loader_iter is not None:
            loader_iter._shutdown_workers()
        del loader


if __name__ == "__main__":
    main()
