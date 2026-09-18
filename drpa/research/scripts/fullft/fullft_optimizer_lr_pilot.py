#!/usr/bin/env python3
"""No-checkpoint FullFT AdamW smoke test and pre-registered LR stability pilot.

This is deliberately an engineering gate, not a performance experiment.  It
uses only the frozen 10% training manifest and never reads validation data.
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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


SEED = 20260809
WEIGHT_DECAY = 1e-5
CLIP_NORM = 1.0
PILOT_STEPS = 20  # Fixed before launch; no validation or performance selection.
LR_CANDIDATES = (5e-6, 1e-5, 2e-5)


def load_gate_module(base: Path):
    path = base / "scripts/fullft/fullft_runtime_gate.py"
    spec = importlib.util.spec_from_file_location("fullft_runtime_gate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen runtime-gate helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for _, p in model.named_parameters(remove_duplicate=True))


def state_numel(optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
    entries = len(optimizer.state)
    elements = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                elements += value.numel()
    return entries, elements


def adamw_update_norm_proxy(optimizer: torch.optim.AdamW) -> float:
    """Reconstruct an update-norm proxy without cloning 440M parameters."""
    total_sq = 0.0
    for group in optimizer.param_groups:
        lr = float(group["lr"])
        wd = float(group["weight_decay"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue
            step = float(state["step"].item() if torch.is_tensor(state["step"]) else state["step"])
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            adaptive = (state["exp_avg"] / bias1) / ((state["exp_avg_sq"] / bias2).sqrt() + eps)
            # Decoupled weight decay uses pre-step weights. Current weights give a
            # first-order-identical reconstruction and avoid a multi-GB shadow copy.
            update = lr * (adaptive + wd * parameter.detach())
            total_sq += float(torch.sum(update.float() ** 2).item())
    return total_sq ** 0.5


def prepare(base: Path, case_indices: List[int]):
    """Build a fresh official FullFT model and fixed train-only data interface."""
    sys.path[:0] = [
        str(base / "VoxTell"),
        str(base / "quality_audit/voxtell_mtl_peft"),
        str(base / "quality_audit/voxtell_mtl_peft_pilot"),
        str(base / "quality_audit/voxtell_mtl_b1_bilateral_crop"),
        str(base / "quality_audit/voxtell_mtl_drpa8_pilot"),
    ]
    from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
    from text_embedding_cache import TextEmbeddingCache

    gate = load_gate_module(base)
    manifest = base / "quality_audit/drpa_data_capacity_scaling/manifests/train_10pct.csv"
    crop_spec = base / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
    model_dir = base / "VoxTell_weights/voxtell_v1.1"
    bank = base / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
    frame = pd.read_csv(manifest, dtype=str)
    records = [CaseRecord(str(r.case_id), str(r.image_path), str(r.label_path)) for r in frame.itertuples()]
    dataset = BilateralGroupedPatchDataset(records, load_crop_spec(crop_spec), cache_cases=False)
    model = gate.initialise_official_model(base)
    ledger = gate.parameter_ledger(model)
    if ledger["configured_trainable_numel"] != gate.EXPECTED_CONFIGURED:
        raise RuntimeError(f"FullFT parameter mismatch: {ledger['configured_trainable_numel']}")
    device = torch.device("cuda")
    model.to(device)
    model.train()
    cache = TextEmbeddingCache(str(bank), str(model_dir), None)
    return gate, frame, dataset, model, cache, PROMPTS, ledger, device


def one_optimizer_step(
    gate,
    dataset,
    cache,
    prompts,
    model,
    optimizer,
    device,
    case_index: int,
    capture_sentinel: bool,
) -> Dict[str, Any]:
    """One canonical optimizer step: 8 fixed prompt passes, FP32 Dice+BCE."""
    item = dataset[int(case_index)]
    image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target_all = item["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)
    unique = list(gate.unique_named_parameters(model))
    sentinel_name, sentinel = next((pair for pair in unique if pair[1].numel() > 32), unique[0])
    sentinel_before = sentinel.detach().reshape(-1)[:32].clone() if capture_sentinel else None
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    individual_losses = []
    forward_sec = 0.0
    backward_sec = 0.0
    for prompt_idx, prompt in enumerate(prompts):
        embedding = cache.get([prompt], device)
        torch.cuda.synchronize(device)
        f0 = time.perf_counter()
        logits = model(image, embedding)
        torch.cuda.synchronize(device)
        forward_sec += time.perf_counter() - f0
        loss = gate.canonical_loss(logits, target_all[:, [prompt_idx]])
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite loss for {item['case_id']} / {prompt}")
        torch.cuda.synchronize(device)
        b0 = time.perf_counter()
        (loss / len(prompts)).backward()
        torch.cuda.synchronize(device)
        backward_sec += time.perf_counter() - b0
        individual_losses.append(float(loss.detach().cpu()))
        del embedding, logits, loss
    grad = gate.gradient_ledger(model)
    if grad["nonfinite_gradient_parameter_names"]:
        raise FloatingPointError("non-finite gradient")
    all_parameters = [p for _, p in unique]
    pre_clip = float(torch.nn.utils.clip_grad_norm_(all_parameters, CLIP_NORM, error_if_nonfinite=True).item())
    post_clip = gate.gradient_ledger(model)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    state_entries, state_elements = state_numel(optimizer)
    sentinel_delta = float(torch.linalg.vector_norm(sentinel.detach().reshape(-1)[:32] - sentinel_before).item()) if sentinel_before is not None else None
    update_proxy = adamw_update_norm_proxy(optimizer)
    row = {
        "case_id": item["case_id"],
        "loss_mean": float(np.mean(individual_losses)),
        "loss_min": float(np.min(individual_losses)),
        "loss_max": float(np.max(individual_losses)),
        "loss_finite": True,
        "grad_finite": True,
        "parameter_finite_after_step": finite_parameters(model),
        "pre_clip_grad_norm": pre_clip,
        "post_clip_grad_active_numel": post_clip["runtime_gradient_active_numel"],
        "update_norm_proxy": update_proxy,
        "sentinel_parameter": sentinel_name,
        "sentinel_delta_l2": sentinel_delta,
        "optimizer_state_entries": state_entries,
        "optimizer_state_tensor_numel": state_elements,
        "forward_sec": forward_sec,
        "backward_sec": backward_sec,
        "step_sec": elapsed,
    }
    del image, target_all
    return row


def run_smoke(base: Path) -> Dict[str, Any]:
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    try:
        gate, frame, dataset, model, cache, prompts, ledger, device = prepare(base, [0])
        optimizer = torch.optim.AdamW(
            [p for _, p in gate.unique_named_parameters(model)], lr=1e-5,
            weight_decay=WEIGHT_DECAY,
        )
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        row = one_optimizer_step(gate, dataset, cache, prompts, model, optimizer, device, 0, True)
        row.update({
            "status": "FULLFT_OPTIMIZER_RUNTIME_READY" if row["parameter_finite_after_step"] and row["sentinel_delta_l2"] and row["sentinel_delta_l2"] > 0 else "FULLFT_RUNTIME_NUMERICAL_FAILURE",
            "lr": 1e-5, "weight_decay": WEIGHT_DECAY, "clip_norm": CLIP_NORM,
            "configured_trainable": ledger["configured_trainable_numel"],
            "optimizer_registered_numel": sum(p.numel() for group in optimizer.param_groups for p in group["params"]),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "optimizer_step_executed": True,
        })
        return row
    except RuntimeError as error:
        return {"status": "FULLFT_96GB_RUNTIME_OOM" if is_oom(error) else "FULLFT_RUNTIME_NUMERICAL_FAILURE", "error": repr(error), "traceback": traceback.format_exc()}
    except Exception as error:
        return {"status": "FULLFT_RUNTIME_NUMERICAL_FAILURE", "error": repr(error), "traceback": traceback.format_exc()}
    finally:
        for variable in ("model", "optimizer"):
            if variable in locals():
                del locals()[variable]
        gc.collect(); torch.cuda.empty_cache()


def run_lr(base: Path, lr: float, indices: List[int]) -> Dict[str, Any]:
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rows: List[Dict[str, Any]] = []
    try:
        gate, frame, dataset, model, cache, prompts, ledger, device = prepare(base, indices)
        optimizer = torch.optim.AdamW(
            [p for _, p in gate.unique_named_parameters(model)], lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        for step, index in enumerate(indices, start=1):
            row = one_optimizer_step(gate, dataset, cache, prompts, model, optimizer, device, int(index), step == 1)
            row["step"] = step
            rows.append(row)
            if not row["parameter_finite_after_step"]:
                raise FloatingPointError("non-finite parameter after optimizer step")
        stable = True
        status = "STABLE"
        error = None
    except RuntimeError as exc:
        stable = False; status = "OOM" if is_oom(exc) else "RUNTIME_ERROR"; error = repr(exc)
    except Exception as exc:
        stable = False; status = "NUMERICAL_FAILURE"; error = repr(exc)
    finally:
        peak_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        gc.collect(); torch.cuda.empty_cache()
    return {
        "lr": lr, "status": status, "stable": stable, "error": error,
        "steps_completed": len(rows), "steps_requested": len(indices), "rows": rows,
        "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
        "weight_decay": WEIGHT_DECAY, "clip_norm": CLIP_NORM,
        "train_manifest": str(base / "quality_audit/drpa_data_capacity_scaling/manifests/train_10pct.csv"),
        "validation_read": False,
    }


def markdown_smoke(result: Dict[str, Any]) -> str:
    return "\n".join([
        "# FullFT Optimizer Smoke Test", "", f"Status: `{result['status']}`.", "",
        "- One AdamW step only; no checkpoint and no validation.",
        f"- Loss finite / gradient finite / parameter finite: `{result.get('loss_finite')}` / `{result.get('grad_finite')}` / `{result.get('parameter_finite_after_step')}`.",
        f"- Optimizer state entries / tensor elements: `{result.get('optimizer_state_entries')}` / `{result.get('optimizer_state_tensor_numel')}`.",
        f"- Peak allocated/reserved: `{result.get('peak_allocated_bytes')}` / `{result.get('peak_reserved_bytes')}` bytes.",
        f"- Step seconds: `{result.get('step_sec')}`; sentinel update L2: `{result.get('sentinel_delta_l2')}`.",
        f"- Configured / optimizer-registered parameter numel: `{result.get('configured_trainable')}` / `{result.get('optimizer_registered_numel')}`.",
        "",
        "No formal checkpoint was created and this does not constitute FullFT training.",
    ]) + "\n"


def markdown_pilot(results: List[Dict[str, Any]], selected: str) -> str:
    lines = ["# FullFT LR Stability Pilot", "", "Train-only engineering pilot: fixed 10% manifest, seed 20260809, 20 optimizer steps, FP32, batch=1, eight fixed prompt passes per optimizer step. No validation was read.", "", "| LR | Status | Steps | Mean loss | Mean pre-clip grad norm | Mean update proxy | Mean step sec | Peak allocated |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        rows = result["rows"]
        mean = lambda key: float(np.mean([row[key] for row in rows])) if rows else float("nan")
        lines.append("| {:.1e} | {} | {}/{} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.3f} GiB |".format(result["lr"], result["status"], result["steps_completed"], result["steps_requested"], mean("loss_mean"), mean("pre_clip_grad_norm"), mean("update_norm_proxy"), mean("step_sec"), (result["peak_allocated_bytes"] or 0) / 1024 ** 3))
    lines += ["", f"Pre-registered formal LR decision: `{selected}`.", "Selection used only finite loss/gradients/parameters, no OOM, and bounded trajectory; validation Dice was never read."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__")))
    parser.add_argument("--mode", choices=("smoke", "pilot"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.resolve(); out = args.output_dir or base / "quality_audit/voxtell_fullft_baseline"; out.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        result = run_smoke(base)
        (out / "fullft_optimizer_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "FULLFT_OPTIMIZER_SMOKE_REPORT.md").write_text(markdown_smoke(result))
        print(json.dumps({"status": result["status"]}, indent=2))
        return
    frame = pd.read_csv(base / "quality_audit/drpa_data_capacity_scaling/manifests/train_10pct.csv", dtype=str)
    indices = np.random.default_rng(SEED).permutation(len(frame))[:PILOT_STEPS].astype(int).tolist()
    results = [run_lr(base, lr, indices) for lr in LR_CANDIDATES]
    stable = {result["lr"]: result["stable"] for result in results}
    selected = "1e-5" if stable.get(1e-5, False) else ("5e-6" if stable.get(5e-6, False) else "NO_STABLE_LR")
    verdict = "FULLFT_FORMAL_TRAINING_READY" if selected != "NO_STABLE_LR" else "FULLFT_LR_STABILITY_FAILED"
    payload = {"status": verdict, "pre_registered_steps": PILOT_STEPS, "indices": indices, "results": results, "selected_formal_lr": selected, "validation_read": False}
    (out / "fullft_lr_stability_pilot.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "FULLFT_LR_STABILITY_PILOT_REPORT.md").write_text(markdown_pilot(results, selected))
    print(json.dumps({"status": verdict, "selected_formal_lr": selected}, indent=2))


if __name__ == "__main__":
    main()
