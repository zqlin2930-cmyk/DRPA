#!/usr/bin/env python3
"""Frozen 50% DRPA projection-rank ablation (r=4, r=8, or r=16).

This runner creates fresh r=4/r=8/r=16 runs under the frozen placement
AMP-forward protocol.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import atexit
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
AUDIT = BASE / "quality_audit/drpa_data_capacity_scaling"
PILOT = BASE / "quality_audit/voxtell_mtl_drpa8_pilot"
PEFT = BASE / "quality_audit/voxtell_mtl_peft"
PREP = BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop"
PLACEMENT = BASE / "quality_audit/placement_ablation_50"
MODEL = BASE / "VoxTell_weights/voxtell_v1.1"
BANK = BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
CACHE = PEFT / "text_embedding_cache.npz"
DEFAULT_SEED = 20260809
SEED = DEFAULT_SEED
DEVICE = torch.device("cuda")
EXPECTED_R8 = 10_969_696
_RANK_LAST_STEP = 0

sys.path[:0] = [str(PILOT), str(PEFT), str(PREP), str(PLACEMENT),
                str(BASE / "scripts/train"), str(BASE / "scripts/analysis")]
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
from text_embedding_cache import TextEmbeddingCache
import evaluate_drpa8 as evaluator
from rank_ablation_wrapper import RankAblationDRPAWrapper
from pro6000_pipeline_benchmark import TimedReaderDataset, collate, worker_init, close_loader


class EpochOrderSampler(Sampler[int]):
    """Mutable sampler preserving the frozen per-epoch canonical index order."""
    def __init__(self) -> None:
        self.indices: list[int] = []
    def set_indices(self, indices) -> None:
        self.indices = [int(index) for index in indices]
    def __iter__(self):
        return iter(self.indices)
    def __len__(self) -> int:
        return len(self.indices)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def close_loader_after_artifact_commit(loader: DataLoader, out: Path, phase: str) -> None:
    """Best-effort worker cleanup that cannot invalidate completed artifacts.

    A worker may abort while PyTorch joins persistent workers during process
    teardown.  Once the complete training and validation artifact set has
    already been atomically written, this is a runtime-cleanup warning rather
    than a training/result failure.  The warning remains auditable on disk.
    """
    try:
        close_loader(loader)
    except Exception as exc:
        atomic_json(out / "post_completion_dataloader_shutdown_warning.json", {
            "status": "POST_COMPLETION_DATALOADER_SHUTDOWN_WARNING",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "last_completed_step": int(globals().get("_RANK_LAST_STEP", 0)),
            "timestamp_unix": time.time(),
        })


def atomic_torch_save(path: Path, payload: dict) -> None:
    """Write one replaceable runtime-recovery checkpoint atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def trainable_model_state(wrapper: RankAblationDRPAWrapper) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in wrapper.model.named_parameters()
        if parameter.requires_grad
    }


def save_runtime_resume(
    path: Path,
    wrapper: RankAblationDRPAWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
    global_step: int,
    completed_epoch: int,
    epoch_order: list[int],
    steps: list[dict],
    epochs: list[dict],
    validations: list[dict],
    sampler_offset: int | None = None,
) -> None:
    """Save full epoch-boundary state for runtime recovery, never selection."""
    if sampler_offset is None:
        sampler_offset = len(epoch_order)
    payload = {
        "format": "drpa_rank_ablation_runtime_resume_v2",
        "purpose": "RUNTIME_RECOVERY_ONLY_NOT_MODEL_SELECTION",
        "projection_rank": int(wrapper.projection_rank),
        "global_step": int(global_step),
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(completed_epoch + 1),
        "sampler_state": {
            "completed_epoch_order": [int(index) for index in epoch_order],
            "completed_epoch_offset": int(sampler_offset),
            "next_epoch_seed": int(SEED + completed_epoch + 1),
        },
        "trainable_model_state": trainable_model_state(wrapper),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "scheduler_state": None,
        "scheduler_contract": "NO_SCHEDULER_CONSTANT_LR_GROUPS",
        "rng_state": capture_rng_state(),
        "history": {"steps": steps, "epochs": epochs, "validations": validations},
        "config": config,
        "saved_at_unix": time.time(),
    }
    atomic_torch_save(path, payload)


def load_runtime_resume(
    path: Path,
    wrapper: RankAblationDRPAWrapper,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict,
) -> tuple[int, int, list[dict], list[dict], list[dict], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "drpa_rank_ablation_runtime_resume_v2":
        raise RuntimeError("unsupported Rank runtime-resume format")
    if payload.get("purpose") != "RUNTIME_RECOVERY_ONLY_NOT_MODEL_SELECTION":
        raise RuntimeError("resume artifact purpose mismatch")
    if int(payload.get("projection_rank", -1)) != int(wrapper.projection_rank):
        raise RuntimeError("resume projection-rank mismatch")
    frozen_keys = (
        "experiment", "projection_rank", "seed", "max_optimizer_steps",
        "train_manifest_sha256", "val_manifest_sha256", "batch_size",
        "amp_forward", "autocast_dtype", "data_source", "num_workers",
        "prefetch_factor", "multiprocessing_context",
    )
    saved_config = payload.get("config", {})
    mismatches = {
        key: (saved_config.get(key), config.get(key))
        for key in frozen_keys
        if saved_config.get(key) != config.get(key)
    }
    if mismatches:
        raise RuntimeError(f"resume protocol mismatch: {mismatches}")
    named = {
        name: parameter
        for name, parameter in wrapper.model.named_parameters()
        if parameter.requires_grad
    }
    saved_state = payload.get("trainable_model_state", {})
    if set(saved_state) != set(named):
        raise RuntimeError("resume trainable-model key mismatch")
    for name, parameter in named.items():
        parameter.data.copy_(saved_state[name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload["grad_scaler_state"])
    if payload.get("scheduler_state") is not None:
        raise RuntimeError("rank protocol has no scheduler but resume state contains one")
    restore_rng_state(payload["rng_state"])
    history = payload.get("history", {})
    return (
        int(payload["global_step"]),
        int(payload["completed_epoch"]),
        list(history.get("steps", [])),
        list(history.get("epochs", [])),
        list(history.get("validations", [])),
        dict(payload.get("sampler_state", {})),
    )


def worker_snapshot(loader: DataLoader) -> dict:
    iterator = getattr(loader, "_iterator", None)
    workers = list(getattr(iterator, "_workers", []) or [])
    return {
        "worker_count": len(workers),
        "worker_pids": [int(worker.pid) for worker in workers if worker.pid is not None],
        "workers_alive": [bool(worker.is_alive()) for worker in workers],
        "all_workers_alive": bool(workers) and all(worker.is_alive() for worker in workers),
    }


def refuse_busy_gpu() -> None:
    used = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if used:
        raise RuntimeError(f"GPU is occupied: {used}")


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path, dtype=str)
    return [CaseRecord(str(row.case_id), str(row.image_path), str(row.label_path))
            for row in frame.itertuples()]


def loss_fp32(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, probability.ndim))
    inter = (probability * target).sum(dims)
    den = probability.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * inter + 1e-5) / (den + 1e-5)).mean()
    return dice + bce


def parameter_groups(wrapper: RankAblationDRPAWrapper) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for _, parameter, group in wrapper.trainable_parameter_groups():
        groups.setdefault(group, []).append(parameter)
    expected = {"cross_attention_lora", "projection_adapter", "decoder_stages"}
    if set(groups) != expected:
        raise RuntimeError(f"unexpected trainable groups: {set(groups)}")
    return groups


def grad_stats(groups: dict[str, list[torch.nn.Parameter]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for group, parameters in groups.items():
        none = zero = nonzero = 0; sq = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                none += 1; continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"non-finite gradient in {group}")
            if torch.count_nonzero(parameter.grad).item() == 0:
                zero += 1
            else:
                nonzero += 1
            sq += float(torch.sum(parameter.grad.float() ** 2).cpu())
        result[group] = {"parameter_tensors": len(parameters), "grad_none_tensors": none,
                         "zero_grad_tensors": zero, "nonzero_grad_tensors": nonzero,
                         "grad_norm": sq ** 0.5}
    return result


def validate(wrapper, val, val_ds, cache, step: int, out: Path, train_len: int) -> dict:
    rows = evaluator.evaluate_condition(f"RANK_ABLATION_r{wrapper.projection_rank}@step{step:05d}",
                                        wrapper, val, val_ds, cache)
    frame = pd.DataFrame(rows)
    summary = evaluator.summarize(frame)
    summary.insert(1, "step", step)
    frame.to_csv(out / f"validation_rows_step_{step:05d}.csv", index=False)
    summary.to_csv(out / f"validation_summary_step_{step:05d}.csv", index=False)
    raw = summary[summary.lcc == 0]
    overall = raw[raw.scope == "overall"].iloc[0]
    result = {"step": step, "equivalent_epoch": step / train_len,
              "mean_dice": float(overall.dice), "hd95_mm": float(overall.hd95_mm),
              "surface_dice_2mm": float(overall.surface_dice_2mm),
              "components": float(overall.connected_components),
              "fp_volume_ml": float(overall.false_positive_volume_ml)}
    for scope, key in [("hippocampus", "hipp"), ("entorhinal cortex", "ec"),
                       ("parahippocampal gyrus", "phg"), ("amygdala", "amy")]:
        result[f"{key}_dice"] = float(raw[raw.scope == scope].iloc[0].dice)
    return result


def main() -> None:
    global _RANK_LAST_STEP
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Frozen whole-run seed; paired r8/r16 runs must share it.")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume-state", default=None,
                        help="epoch-boundary runtime recovery only; never model selection")
    args = parser.parse_args()
    global SEED
    SEED = int(args.seed)
    if args.max_steps != 6000:
        raise ValueError("rank ablation is frozen at 6000 optimizer steps")
    train_path = AUDIT / "manifests/train_50pct.csv"
    val_path = AUDIT / "manifests/val_100pct_frozen.csv"
    train_frame, val_frame = pd.read_csv(train_path, dtype=str), pd.read_csv(val_path, dtype=str)
    if len(train_frame) != 461 or train_frame.ptid.nunique() != 169:
        raise RuntimeError("50% manifest is not 461 visits / 169 PTIDs")
    if len(val_frame) != 247 or val_frame.ptid.nunique() != 85:
        raise RuntimeError("frozen validation manifest mismatch")
    refuse_busy_gpu(); seed_all()
    wrapper = RankAblationDRPAWrapper(str(MODEL), str(BANK), projection_rank=args.rank, device=DEVICE)
    wrapper.set_training_mode()
    groups = parameter_groups(wrapper)
    counts = {name: sum(p.numel() for p in parameters) for name, parameters in groups.items()}
    total = sum(counts.values())
    if args.rank == 8 and total != EXPECTED_R8:
        raise RuntimeError(f"r8 audit count {total} != canonical {EXPECTED_R8}")
    config = {"experiment": "DRPA_RANK_ABLATION_50_AMP_MATCHED", "projection_rank": args.rank,
              "projection_alpha": 8.0, "cross_attention_rank": 4, "seed": SEED,
              "max_optimizer_steps": 6000, "train_visits": len(train_frame),
              "train_ptids": int(train_frame.ptid.nunique()), "val_visits": len(val_frame),
              "val_ptids": int(val_frame.ptid.nunique()), "batch_size": 1,
              "loss": "FP32 Dice+BCE", "amp_forward": True, "autocast_dtype": "float16",
              "grad_scaler": True, "weight_decay": 1e-5, "clip_norm": 1.0,
              "data_pipeline": "BEST_BATCH1_PIPELINE",
              "data_source": "DISK_CACHE", "num_workers": 32,
              "prefetch_factor": 2, "pin_memory": True,
              "persistent_workers": True, "non_blocking_h2d": True,
              "multiprocessing_context": "spawn",
              "pipeline_equivalence": "PIPELINE_EQUIVALENCE_PASS",
              "sample_order_contract": "np.random.default_rng(seed + epoch).permutation(train_len)",
              "parameter_groups": counts, "trainable_parameters": total,
              "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path)}
    if args.preflight:
        print(json.dumps({**config, "preflight": "PASS"}, indent=2)); return
    out = Path(args.output_dir)
    if args.resume_state:
        if not out.is_dir():
            raise RuntimeError("resume output directory does not exist")
        if not (out / "config.json").is_file():
            raise RuntimeError("resume output directory lacks config.json")
    else:
        out.mkdir(parents=True, exist_ok=False); (out / "checkpoints").mkdir()
        (out / "config.json").write_text(json.dumps(config, indent=2))
    def exception_hook(exc_type, exc, tb) -> None:
        atomic_json(out / "failure_context.json", {
            "error_type": getattr(exc_type, "__name__", str(exc_type)),
            "error": repr(exc),
            "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            "last_completed_step": int(globals().get("_RANK_LAST_STEP", 0)),
            "timestamp_unix": time.time(),
        })
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = exception_hook
    pd.DataFrame([{"module_group": name, "parameter_count": count}
                  for name, count in counts.items()]).to_csv(out / "parameter_summary.csv", index=False)
    train, val = records(train_path), records(val_path)
    spec = load_crop_spec(PILOT / "crop_spec.json")
    train_ds = TimedReaderDataset(train, spec, "disk")
    val_ds = BilateralGroupedPatchDataset(val, spec, cache_cases=False)
    epoch_sampler = EpochOrderSampler()
    train_loader = DataLoader(train_ds, batch_size=1, sampler=epoch_sampler,
                              num_workers=32, pin_memory=True, persistent_workers=True,
                              prefetch_factor=2, collate_fn=collate, worker_init_fn=worker_init,
                              multiprocessing_context="spawn")
    loader_cleanup_done = False

    def cleanup_loader(phase: str) -> None:
        nonlocal loader_cleanup_done
        if loader_cleanup_done:
            return
        loader_cleanup_done = True
        close_loader_after_artifact_commit(train_loader, out, phase)

    atexit.register(cleanup_loader, "atexit")
    text_cache = TextEmbeddingCache(str(BANK), str(MODEL), str(CACHE))
    optimizer = torch.optim.AdamW([
        {"params": groups["cross_attention_lora"], "lr": 1e-4},
        {"params": groups["projection_adapter"], "lr": 1e-4},
        {"params": groups["decoder_stages"], "lr": 1e-5},
    ], weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    steps: list[dict] = []
    epochs: list[dict] = []
    validations: list[dict] = []
    global_step = 0
    epoch = 0
    if args.resume_state:
        global_step, epoch, steps, epochs, validations, sampler_state = load_runtime_resume(
            Path(args.resume_state), wrapper, optimizer, scaler, config
        )
        if len(steps) != global_step:
            raise RuntimeError(f"resume history/global-step mismatch: {len(steps)} != {global_step}")
        atomic_json(out / "resume_event.json", {
            "status": "RUNTIME_RESUME_LOADED", "resume_state": str(args.resume_state),
            "global_step": global_step, "completed_epoch": epoch,
            "sampler_state": sampler_state,
            "timestamp_unix": time.time(),
        })
    started = time.time()
    while global_step < args.max_steps:
        epoch += 1; epoch_started = time.time(); losses = []
        epoch_order = np.random.default_rng(SEED + epoch).permutation(len(train_ds)).tolist()
        epoch_sampler.set_indices(epoch_order)
        atomic_json(out / "epoch_boundary_status.json", {
            "event": "EPOCH_ITERATOR_ABOUT_TO_START", "epoch": epoch,
            "last_completed_step": global_step, "timestamp_unix": time.time(),
            **worker_snapshot(train_loader),
        })
        for item in train_loader:
            if global_step >= args.max_steps: break
            image = item["image"].to(DEVICE, non_blocking=True)
            target = item["mask"].to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True); prompt_losses = []
            for prompt_index, prompt in enumerate(PROMPTS):
                embedding = text_cache.get([prompt], DEVICE)
                with torch.autocast("cuda", dtype=torch.float16): logits = wrapper(image, embedding)
                with torch.autocast("cuda", enabled=False): loss = loss_fp32(logits, target[:, [prompt_index]])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step={global_step+1}, case={item['case_id'][0]}")
                scaler.scale(loss / len(PROMPTS)).backward(); prompt_losses.append(float(loss.detach().cpu()))
                del embedding, logits, loss
            scaler.unscale_(optimizer); before = grad_stats(groups)
            all_params = [p for ps in groups.values() for p in ps]
            torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
            after = grad_stats(groups); scale_before = float(scaler.get_scale())
            scaler.step(optimizer); scaler.update(); step = global_step + 1
            steps.append({"step": step, "epoch": epoch, "case_id": item["case_id"][0],
                          "loss": float(np.mean(prompt_losses)), "pre_clip_grad_json": json.dumps(before),
                          "post_clip_grad_json": json.dumps(after), "scaler_before": scale_before,
                          "scaler_after": float(scaler.get_scale()), "step_seconds": time.time()-epoch_started})
            _RANK_LAST_STEP = step
            global_step = step
            losses.extend(prompt_losses); del image, target; gc.collect(); torch.cuda.empty_cache()
            if step == 6000:
                result = validate(wrapper, val, val_ds, text_cache, step, out, len(train_ds))
                validations.append(result); pd.DataFrame(validations).to_csv(out / "validation_metrics.csv", index=False)
                state = {name: p.detach().cpu() for name, p in wrapper.model.named_parameters() if p.requires_grad}
                torch.save({"format": "drpa_rank_ablation_amp_v1", "global_step": step,
                            "projection_rank": args.rank, "trainable_model_state": state,
                            "optimizer_state": optimizer.state_dict(), "rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all(), "config": config},
                           out / "checkpoints/step_06000.pt")
                print(json.dumps({"step": step, "validation": result}), flush=True)
        epochs.append({"epoch": epoch, "steps_end": global_step, "train_loss": float(np.mean(losses)),
                       "elapsed_sec": time.time()-epoch_started, "equivalent_epoch": global_step/len(train_ds)})
        pd.DataFrame(steps).to_csv(out / "training_dynamics.csv", index=False)
        pd.DataFrame(epochs).to_csv(out / "training_curve.csv", index=False)
        save_runtime_resume(
            out / "checkpoints/latest_resume.pt", wrapper, optimizer, scaler, config,
            global_step, epoch, epoch_order, steps, epochs, validations,
        )
        boundary = {"event": "EPOCH_COMPLETE", "epoch": epoch,
                    "last_completed_step": global_step, "timestamp_unix": time.time(),
                    "latest_resume": str(out / "checkpoints/latest_resume.pt"),
                    **worker_snapshot(train_loader)}
        atomic_json(out / "epoch_boundary_status.json", boundary)
        with (out / "epoch_boundary_events.jsonl").open("a") as handle:
            handle.write(json.dumps(boundary) + "\n")
        print(json.dumps(boundary), flush=True)
    (out / "run_summary.json").write_text(json.dumps({"status": "COMPLETE", "steps": global_step,
        "projection_rank": args.rank, "parameter_groups": counts, "validation": validations[-1],
        "total_wall_sec": time.time()-started}, indent=2))
    cleanup_loader("normal_completion")


if __name__ == "__main__":
    main()
