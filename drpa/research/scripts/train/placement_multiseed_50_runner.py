#!/usr/bin/env python3
"""Matched-pipeline 50% Placement Ablation runner.

Only the pre-registered trainable placement changes across B1, B1+Projection,
B1+Decoder and DRPA.  It reuses the hardened rank experiment's deterministic
warm-disk-cache, 32-worker pipeline and frozen evaluation contract.
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
DEVICE = torch.device("cuda")
EXPECTED = {"b1": 294_912, "b1_projection": 737_536,
            "b1_decoder": 10_527_072, "drpa": 10_969_696}

sys.path[:0] = [str(PILOT), str(PEFT), str(PREP), str(PLACEMENT),
                str(BASE / "scripts/train"), str(BASE / "scripts/analysis")]
from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
from text_embedding_cache import TextEmbeddingCache
from placement_wrappers import build_placement_wrapper
from rank_ablation_wrapper import RankAblationDRPAWrapper
from voxtell_peft_wrapper import VoxTellPEFTWrapper, _unique_named_parameters
from pro6000_pipeline_benchmark import TimedReaderDataset, collate, worker_init, close_loader
import evaluate_drpa8 as evaluator


class EpochOrderSampler(Sampler[int]):
    """Mutable sampler preserving the frozen per-epoch index order."""

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def refuse_busy_gpu() -> None:
    used = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=False).stdout.strip()
    if used:
        raise RuntimeError(f"GPU is occupied: {used}")


def records(path: Path) -> list[CaseRecord]:
    frame = pd.read_csv(path, dtype=str)
    return [CaseRecord(str(row.case_id), str(row.image_path), str(row.label_path)) for row in frame.itertuples()]


def make_wrapper(condition: str):
    if condition == "b1":
        wrapper = VoxTellPEFTWrapper(str(MODEL), str(BANK), rank=4, alpha=8.0, dropout=0.05, device=DEVICE)
        wrapper.model.eval()
        for layer in wrapper.model.transformer_decoder.layers:
            layer.multihead_attn.parametrizations.in_proj_weight[0].train()
            layer.multihead_attn.out_proj.parametrizations.weight[0].train()
        return wrapper
    if condition in {"b1_projection", "b1_decoder"}:
        return build_placement_wrapper(condition, str(MODEL), str(BANK), DEVICE)
    wrapper = RankAblationDRPAWrapper(str(MODEL), str(BANK), projection_rank=8, device=DEVICE)
    wrapper.set_training_mode()
    return wrapper


def parameter_groups(wrapper, condition: str) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {}
    if condition == "b1":
        iterable = ((name, parameter, "cross_attention_lora") for name, parameter in _unique_named_parameters(wrapper.model) if parameter.requires_grad)
    else:
        iterable = wrapper.trainable_parameter_groups()
    for _, parameter, group in iterable:
        groups.setdefault(group, []).append(parameter)
    expected = {
        "b1": {"cross_attention_lora"},
        "b1_projection": {"cross_attention_lora", "projection_adapter"},
        "b1_decoder": {"cross_attention_lora", "decoder_stages"},
        "drpa": {"cross_attention_lora", "projection_adapter", "decoder_stages"},
    }[condition]
    if set(groups) != expected:
        raise RuntimeError(f"{condition} groups {set(groups)} != {expected}")
    return groups


def loss_fp32(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, probability.ndim))
    inter = (probability * target).sum(dims)
    denom = probability.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
    return dice + bce


def finite_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("non-finite gradient")
        total += float(torch.sum(parameter.grad.float() ** 2).cpu())
    return total ** 0.5


def validate(wrapper, validation, val_dataset, cache, step: int, out: Path, condition: str, train_len: int) -> dict:
    rows = evaluator.evaluate_condition(f"PLACEMENT_MULTI_{condition}@step{step:05d}", wrapper, validation, val_dataset, cache)
    frame = pd.DataFrame(rows)
    summary = evaluator.summarize(frame)
    summary.insert(1, "step", step)
    frame.to_csv(out / f"validation_rows_step_{step:05d}.csv", index=False)
    summary.to_csv(out / f"validation_summary_step_{step:05d}.csv", index=False)
    raw = summary[summary.lcc.eq(0)]
    overall = raw[raw.scope.eq("overall")].iloc[0]
    result = {"step": step, "equivalent_epoch": step / train_len, "mean_dice": float(overall.dice),
              "hd95_mm": float(overall.hd95_mm), "surface_dice_2mm": float(overall.surface_dice_2mm),
              "components": float(overall.connected_components), "fp_volume_ml": float(overall.false_positive_volume_ml)}
    for scope, key in [("hippocampus", "hipp"), ("entorhinal cortex", "ec"),
                       ("parahippocampal gyrus", "phg"), ("amygdala", "amy")]:
        result[f"{key}_dice"] = float(raw[raw.scope.eq(scope)].iloc[0].dice)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight", action="store_true", help="Instantiate and audit the frozen contract without training.")
    args = parser.parse_args()
    if args.max_steps != 6000:
        raise ValueError("Placement multi-seed is frozen at 6000 optimizer steps")
    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {out}")
    train_path, val_path = AUDIT / "manifests/train_50pct.csv", AUDIT / "manifests/val_100pct_frozen.csv"
    train_frame, val_frame = pd.read_csv(train_path, dtype=str), pd.read_csv(val_path, dtype=str)
    if len(train_frame) != 461 or train_frame.ptid.nunique() != 169:
        raise RuntimeError("50% training manifest mismatch")
    if len(val_frame) != 247 or val_frame.ptid.nunique() != 85:
        raise RuntimeError("frozen validation manifest mismatch")
    refuse_busy_gpu(); seed_all(args.seed)
    wrapper = make_wrapper(args.condition)
    groups = parameter_groups(wrapper, args.condition)
    counts = {name: sum(parameter.numel() for parameter in parameters) for name, parameters in groups.items()}
    if sum(counts.values()) != EXPECTED[args.condition]:
        raise RuntimeError(f"{args.condition} parameters {sum(counts.values())} != {EXPECTED[args.condition]}")
    config = {
        "experiment": "PLACEMENT_MULTI_SEED_50_AMP_MATCHED", "condition": args.condition, "seed": args.seed,
        "max_optimizer_steps": 6000, "train_visits": 461, "train_ptids": 169, "val_visits": 247, "val_ptids": 85,
        "batch_size": 1, "loss": "FP32 Dice+BCE", "amp_forward": True, "autocast_dtype": "float16", "grad_scaler": True,
        "weight_decay": 1e-5, "clip_norm": 1.0, "data_pipeline": "BEST_BATCH1_PIPELINE", "data_source": "DISK_CACHE",
        "num_workers": 32, "prefetch_factor": 2, "pin_memory": True, "persistent_workers": True,
        "non_blocking_h2d": True, "multiprocessing_context": "spawn", "pipeline_equivalence": "PIPELINE_EQUIVALENCE_PASS",
        "sample_order_contract": "np.random.default_rng(seed + epoch).permutation(train_len)",
        "prompts": "8 fixed canonical prompts", "lcc_main": 0, "parameter_groups": counts,
        "trainable_parameters": sum(counts.values()), "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path),
    }
    if args.preflight:
        print(json.dumps({**config, "preflight": "PASS"}, indent=2))
        return
    out.mkdir(parents=True); (out / "checkpoints").mkdir()
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    pd.DataFrame([{"module_group": key, "parameter_count": value} for key, value in counts.items()]).to_csv(out / "parameter_summary.csv", index=False)
    (out / "initialization.json").write_text(json.dumps({"status": "FRESH_OFFICIAL_INITIALIZATION", **{key: config[key] for key in ["condition", "seed", "train_manifest_sha256", "val_manifest_sha256", "trainable_parameters"]}}, indent=2) + "\n")
    def exception_hook(exc_type, exc, tb):
        atomic_json(out / "failure_context.json", {"error_type": getattr(exc_type, "__name__", str(exc_type)), "error": repr(exc), "traceback": "".join(traceback.format_exception(exc_type, exc, tb)), "timestamp_unix": time.time()})
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = exception_hook
    train, validation = records(train_path), records(val_path)
    spec = load_crop_spec(PILOT / "crop_spec.json")
    dataset = TimedReaderDataset(train, spec, "disk")
    val_dataset = BilateralGroupedPatchDataset(validation, spec, cache_cases=False)
    sampler = EpochOrderSampler()
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=32, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2, collate_fn=collate, worker_init_fn=worker_init,
                        multiprocessing_context="spawn")
    loader_cleanup_done = False

    def cleanup_loader(phase: str) -> None:
        nonlocal loader_cleanup_done
        if loader_cleanup_done:
            return
        loader_cleanup_done = True
        try:
            close_loader(loader)
        except Exception as exc:
            atomic_json(out / "post_completion_dataloader_shutdown_warning.json", {
                "status": "POST_COMPLETION_DATALOADER_SHUTDOWN_WARNING",
                "phase": phase,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "timestamp_unix": time.time(),
            })
    atexit.register(cleanup_loader, "atexit")
    cache = TextEmbeddingCache(str(BANK), str(MODEL), str(CACHE))
    optimizer_groups = [{"params": groups["cross_attention_lora"], "lr": 1e-4}]
    if "projection_adapter" in groups: optimizer_groups.append({"params": groups["projection_adapter"], "lr": 1e-4})
    if "decoder_stages" in groups: optimizer_groups.append({"params": groups["decoder_stages"], "lr": 1e-5})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    steps, epochs, validations, global_step, epoch = [], [], [], 0, 0
    started = time.time()
    while global_step < 6000:
        epoch += 1; epoch_started = time.time(); losses = []
        order = np.random.default_rng(args.seed + epoch).permutation(len(dataset)).tolist(); sampler.set_indices(order)
        for item in loader:
            if global_step >= 6000: break
            image = item["image"].to(DEVICE, non_blocking=True); target = item["mask"].to(DEVICE, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True); prompt_losses = []
            for prompt_index, prompt in enumerate(PROMPTS):
                embedding = cache.get([prompt], DEVICE)
                with torch.autocast("cuda", dtype=torch.float16): logits = wrapper(image, embedding)
                with torch.autocast("cuda", enabled=False): loss = loss_fp32(logits, target[:, [prompt_index]])
                if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss step={global_step+1}, case={item['case_id'][0]}")
                scaler.scale(loss / len(PROMPTS)).backward(); prompt_losses.append(float(loss.detach().cpu()))
                del embedding, logits, loss
            scaler.unscale_(optimizer); parameters = [parameter for values in groups.values() for parameter in values]
            pre_norm = finite_grad_norm(parameters); torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            post_norm = finite_grad_norm(parameters); before_scale = float(scaler.get_scale()); scaler.step(optimizer); scaler.update()
            global_step += 1; losses.extend(prompt_losses)
            steps.append({"step": global_step, "epoch": epoch, "case_id": item["case_id"][0], "loss": float(np.mean(prompt_losses)), "pre_clip_grad_norm": pre_norm, "post_clip_grad_norm": post_norm, "scaler_before": before_scale, "scaler_after": float(scaler.get_scale()), "step_seconds": time.time() - epoch_started})
            del image, target; gc.collect(); torch.cuda.empty_cache()
            if global_step == 6000:
                result = validate(wrapper, validation, val_dataset, cache, 6000, out, args.condition, len(dataset)); validations.append(result)
                pd.DataFrame(validations).to_csv(out / "validation_metrics.csv", index=False)
                state = {name: parameter.detach().cpu() for name, parameter in wrapper.model.named_parameters() if parameter.requires_grad}
                torch.save({"format": "placement_multiseed_50_amp_v1", "global_step": 6000, "condition": args.condition, "trainable_model_state": state, "optimizer_state": optimizer.state_dict(), "grad_scaler_state": scaler.state_dict(), "config": config}, out / "checkpoints/step_06000.pt")
                print(json.dumps({"step": 6000, "validation": result}), flush=True)
        epochs.append({"epoch": epoch, "steps_end": global_step, "train_loss": float(np.mean(losses)), "elapsed_sec": time.time()-epoch_started, "equivalent_epoch": global_step/len(dataset)})
        pd.DataFrame(steps).to_csv(out / "training_dynamics.csv", index=False); pd.DataFrame(epochs).to_csv(out / "training_curve.csv", index=False)
        resume = {"format": "placement_multiseed_runtime_resume_v1", "purpose": "RUNTIME_RECOVERY_ONLY_NOT_MODEL_SELECTION", "condition": args.condition, "seed": args.seed, "global_step": global_step, "completed_epoch": epoch, "trainable_model_state": {name: parameter.detach().cpu() for name, parameter in wrapper.model.named_parameters() if parameter.requires_grad}, "optimizer_state": optimizer.state_dict(), "grad_scaler_state": scaler.state_dict(), "config": config}
        temporary = out / "checkpoints/latest_resume.tmp"; torch.save(resume, temporary); os.replace(temporary, out / "checkpoints/latest_resume.pt")
        print(json.dumps({"event": "EPOCH_COMPLETE", "epoch": epoch, "last_completed_step": global_step}), flush=True)
    (out / "run_summary.json").write_text(json.dumps({"status": "COMPLETE", "steps": 6000, "condition": args.condition, "parameter_groups": counts, "validation": validations[-1], "total_wall_sec": time.time()-started}, indent=2) + "\n")
    cleanup_loader("normal_completion")


if __name__ == "__main__":
    main()
