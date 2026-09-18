#!/usr/bin/env python3
"""One-shot, no-update FP32 runtime gate for the frozen VoxTell-FullFT protocol.

This script is intentionally not a trainer: it builds the official VoxTell model,
uses one fixed canonical 192-cubed training microbatch, computes Dice+BCE, and
executes backward without constructing an optimizer or writing a checkpoint.
"""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import gc
import hashlib
import json
import os
import pydoc
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


EXPECTED_CONFIGURED = 440_029_541
SEED = 20260809


def unique_named_parameters(module: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    """Yield each Parameter once, explicitly guarding against aliases."""
    seen: set[int] = set()
    for name, parameter in module.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        yield name, parameter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exactly the frozen FP32 Dice+BCE objective used by the canonical runner."""
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return dice + bce


def group_for_parameter(name: str) -> str:
    if name.startswith("encoder."):
        return "image_encoder"
    if name.startswith("project_bottleneck_embed"):
        return "project_bottleneck_embed"
    if name.startswith("project_text_embed"):
        return "project_text_embed"
    if name.startswith("project_to_decoder_channels"):
        return "projections"
    if name.startswith("transformer_decoder"):
        if ".self_attn." in name:
            return "prompt_transformer.self_attn"
        if ".norm1." in name:
            return "prompt_transformer.norm1"
        if ".multihead_attn." in name:
            return "prompt_transformer.cross_attention"
        if ".linear1." in name or ".linear2." in name:
            return "prompt_transformer.ffn"
        if ".norm2." in name or ".norm3." in name:
            return "prompt_transformer.norm2_norm3"
        return "prompt_transformer.other"
    if name.startswith("decoder.stages"):
        return "decoder_stages"
    if name.startswith("decoder.transpconvs"):
        return "transpconvs"
    if name.startswith("decoder.seg_layers"):
        return "seg_layers"
    return "other"


def initialise_official_model(base: Path) -> torch.nn.Module:
    """Replicate VoxTellPredictor's official model construction without Qwen."""
    from nnunetv2.utilities.file_path_utilities import load_json
    from voxtell.model.voxtell_model import VoxTellModel

    model_dir = base / "VoxTell_weights/voxtell_v1.1"
    plans = load_json(str(model_dir / "plans.json"))
    arch_kwargs = dict(plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
    model = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )
    payload = torch.load(model_dir / "fold_0/checkpoint_final.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["network_weights"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def parameter_ledger(model: torch.nn.Module) -> Dict[str, Any]:
    raw = list(model.named_parameters(remove_duplicate=False))
    unique = list(unique_named_parameters(model))
    grouped: Dict[str, Dict[str, Any]] = {}
    for name, parameter in unique:
        group = group_for_parameter(name)
        row = grouped.setdefault(group, {"parameter_tensors": 0, "parameter_numel": 0})
        row["parameter_tensors"] += 1
        row["parameter_numel"] += parameter.numel()
    return {
        "raw_named_parameter_entries": len(raw),
        "unique_parameter_tensors": len(unique),
        "alias_duplicate_entries": len(raw) - len(unique),
        "configured_trainable_numel": sum(p.numel() for _, p in unique if p.requires_grad),
        "all_unique_parameters_trainable": all(p.requires_grad for _, p in unique),
        "module_parameter_ledger": grouped,
    }


def gradient_ledger(model: torch.nn.Module) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}
    active_numel = 0
    nonfinite = []
    for name, parameter in unique_named_parameters(model):
        group = group_for_parameter(name)
        row = grouped.setdefault(group, {
            "parameter_tensors": 0, "parameter_numel": 0, "grad_none_tensors": 0,
            "grad_none_numel": 0, "zero_grad_tensors": 0, "zero_grad_numel": 0,
            "nonzero_grad_tensors": 0, "nonzero_grad_numel": 0, "nonfinite_grad_tensors": 0,
            "grad_l2_sq": 0.0, "grad_abs_max": 0.0,
        })
        row["parameter_tensors"] += 1
        row["parameter_numel"] += parameter.numel()
        grad = parameter.grad
        if grad is None:
            row["grad_none_tensors"] += 1
            row["grad_none_numel"] += parameter.numel()
            continue
        if not bool(torch.isfinite(grad).all().item()):
            row["nonfinite_grad_tensors"] += 1
            nonfinite.append(name)
            continue
        norm = float(torch.linalg.vector_norm(grad.float()).item())
        abs_max = float(grad.detach().abs().max().item())
        row["grad_l2_sq"] += norm * norm
        row["grad_abs_max"] = max(row["grad_abs_max"], abs_max)
        if bool(torch.count_nonzero(grad).item() == 0):
            row["zero_grad_tensors"] += 1
            row["zero_grad_numel"] += parameter.numel()
        else:
            row["nonzero_grad_tensors"] += 1
            row["nonzero_grad_numel"] += parameter.numel()
            active_numel += parameter.numel()
    for row in grouped.values():
        row["grad_l2"] = row.pop("grad_l2_sq") ** 0.5
    return {
        "runtime_gradient_active_numel": active_numel,
        "nonfinite_gradient_parameter_names": nonfinite,
        "module_gradient_ledger": grouped,
    }


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def run_gate(base: Path, use_checkpointing: bool) -> Dict[str, Any]:
    # Imports use the same project-side preprocessing and fixed prompt cache as B1/DRPA/B3.
    sys.path[:0] = [
        str(base / "VoxTell"),
        str(base / "quality_audit/voxtell_mtl_peft"),
        str(base / "quality_audit/voxtell_mtl_peft_pilot"),
        str(base / "quality_audit/voxtell_mtl_b1_bilateral_crop"),
        str(base / "quality_audit/voxtell_mtl_drpa8_pilot"),
    ]
    from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, PROMPTS, load_crop_spec
    from text_embedding_cache import TextEmbeddingCache

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    train_manifest = base / "quality_audit/voxtell_mtl_drpa8_full_data/full_data_train_cases.csv"
    crop_spec = base / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json"
    model_dir = base / "VoxTell_weights/voxtell_v1.1"
    embedding_bank = base / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
    record_row = pd.read_csv(train_manifest, dtype=str).iloc[0]
    record = CaseRecord(str(record_row.case_id), str(record_row.image_path), str(record_row.label_path))
    dataset = BilateralGroupedPatchDataset([record], load_crop_spec(crop_spec), cache_cases=False)
    item = dataset[0]
    # One canonical prompt-wise microbatch matches the existing canonical runner's update path.
    prompt_index = 0
    prompt = PROMPTS[prompt_index]
    image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target = item["mask"][prompt_index:prompt_index + 1].unsqueeze(0).to(device=device, dtype=torch.float32)
    cache = TextEmbeddingCache(str(embedding_bank), str(model_dir), None)
    text_embedding = cache.get([prompt], device)

    model = initialise_official_model(base)
    params = parameter_ledger(model)
    if params["configured_trainable_numel"] != EXPECTED_CONFIGURED or not params["all_unique_parameters_trainable"]:
        return {
            "status": "FULLFT_PARAMETER_DEFINITION_MISMATCH",
            "expected_configured_trainable": EXPECTED_CONFIGURED,
            "parameter_ledger": params,
            "case_id": record.case_id,
            "prompt": prompt,
        }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    model.train()
    try:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if use_checkpointing:
            logits = checkpoint(lambda x, t: model(x, t), image, text_embedding, use_reentrant=False)
        else:
            logits = model(image, text_embedding)
        torch.cuda.synchronize(device)
        forward_sec = time.perf_counter() - t0
        loss = canonical_loss(logits, target)
        loss_finite = bool(torch.isfinite(loss).item())
        if not loss_finite:
            return {
                "status": "FULLFT_RUNTIME_NUMERICAL_FAILURE",
                "case_id": record.case_id, "prompt": prompt,
                "checkpointing": use_checkpointing, "logits_shape": list(logits.shape),
                "loss": float("nan"), "loss_finite": False, "parameter_ledger": params,
            }
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(device)
        backward_sec = time.perf_counter() - t1
        gradients = gradient_ledger(model)
        status = "FULLFT_96GB_RUNTIME_READY" if not gradients["nonfinite_gradient_parameter_names"] else "FULLFT_RUNTIME_NUMERICAL_FAILURE"
        return {
            "status": status,
            "checkpointing": use_checkpointing,
            "case_id": record.case_id,
            "ptid": str(record_row.ptid),
            "prompt": prompt,
            "input_shape": list(image.shape),
            "target_shape": list(target.shape),
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu()),
            "loss_finite": loss_finite,
            "forward_sec": forward_sec,
            "backward_sec": backward_sec,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "parameter_ledger": params,
            "gradient_ledger": gradients,
        }
    except RuntimeError as error:
        if is_oom(error):
            return {
                "status": "FULLFT_96GB_RUNTIME_OOM",
                "checkpointing": use_checkpointing,
                "case_id": record.case_id,
                "prompt": prompt,
                "error": str(error),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "parameter_ledger": params,
            }
        return {
            "status": "FULLFT_RUNTIME_NUMERICAL_FAILURE",
            "checkpointing": use_checkpointing,
            "case_id": record.case_id,
            "prompt": prompt,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "parameter_ledger": params,
        }
    finally:
        if "model" in locals():
            for parameter in model.parameters():
                parameter.grad = None
        del model, image, target, text_embedding
        gc.collect()
        torch.cuda.empty_cache()


def write_report(path: Path, result: Dict[str, Any], base: Path) -> None:
    def gib(value: Any) -> str:
        return "n/a" if value is None else f"{int(value) / 1024 ** 3:.3f} GiB"

    lines = [
        "# VoxTell-FullFT 96 GB FP32 Runtime Gate",
        "",
        f"Status: `{result['status']}`.",
        "",
        "## Fixed contract",
        "- Official VoxTell v1.1 canonical checkpoint; Qwen3-Embedding-4B is not loaded and remains frozen.",
        "- One canonical preprocessing microbatch: batch=1, 192x192x192, FP32, fixed canonical prompt, Dice+BCE.",
        "- No optimizer was created; no optimizer.step, checkpoint write, thresholding, or evaluation was performed.",
        f"- Case/prompt: `{result.get('case_id', 'n/a')}` / `{result.get('prompt', 'n/a')}`.",
        "",
        "## Runtime",
        f"- Checkpointing: `{result.get('checkpointing', False)}`.",
        f"- Input/logits shape: `{result.get('input_shape', 'n/a')}` / `{result.get('logits_shape', 'n/a')}`.",
        f"- Loss / finite: `{result.get('loss', 'n/a')}` / `{result.get('loss_finite', 'n/a')}`.",
        f"- Forward/backward seconds: `{result.get('forward_sec', 'n/a')}` / `{result.get('backward_sec', 'n/a')}`.",
        f"- Peak allocated/reserved: `{gib(result.get('peak_allocated_bytes'))}` / `{gib(result.get('peak_reserved_bytes'))}`.",
        "",
        "## Parameter definition",
    ]
    params = result.get("parameter_ledger", {})
    lines += [
        f"- Expected configured unique trainable: `{EXPECTED_CONFIGURED:,}`.",
        f"- Observed configured unique trainable: `{params.get('configured_trainable_numel', 'n/a')}`.",
        f"- Raw alias duplicate entries: `{params.get('alias_duplicate_entries', 'n/a')}`.",
    ]
    grad = result.get("gradient_ledger")
    if grad:
        lines += ["", "## Runtime gradient activity", f"- Runtime nonzero-gradient unique parameters: `{grad['runtime_gradient_active_numel']:,}`.", f"- Non-finite gradient tensors: `{len(grad['nonfinite_gradient_parameter_names'])}`.", "", "| Module | Param numel | grad=None | zero-grad | nonzero-grad | grad L2 | grad abs max |", "|---|---:|---:|---:|---:|---:|---:|"]
        for group, row in grad["module_gradient_ledger"].items():
            lines.append("| {} | {} | {} | {} | {} | {:.6g} | {:.6g} |".format(group, row["parameter_numel"], row["grad_none_numel"], row["zero_grad_numel"], row["nonzero_grad_numel"], row["grad_l2"], row["grad_abs_max"]))
    if result.get("error"):
        lines += ["", "## Error", "```text", str(result["error"]), "```"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__")))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    base = args.base.resolve()
    output_dir = args.output_dir or base / "quality_audit/voxtell_fullft_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass is always ordinary FP32. Checkpointing is a strict OOM fallback.
    result = run_gate(base, use_checkpointing=False)
    if result["status"] == "FULLFT_96GB_RUNTIME_OOM":
        fallback = run_gate(base, use_checkpointing=True)
        result = fallback
        if fallback["status"] == "FULLFT_96GB_RUNTIME_READY":
            result["status"] = "FULLFT_96GB_CHECKPOINTING_REQUIRED"
    result["protocol"] = {
        "configured_trainable_expected": EXPECTED_CONFIGURED,
        "precision": "FP32",
        "input": "batch=1, 192x192x192",
        "loss": "canonical Dice+BCE",
        "optimizer_step": False,
        "checkpointing_policy": "only after ordinary FP32 OOM",
        "base_checkpoint_sha256": sha256(base / "VoxTell_weights/voxtell_v1.1/fold_0/checkpoint_final.pth"),
    }
    (output_dir / "fullft_96gb_runtime_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(output_dir / "VOXTELL_FULLFT_96GB_RUNTIME_GATE_REPORT.md", result, base)
    print(json.dumps({"status": result["status"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
