#!/usr/bin/env python3
"""Frozen OASIS-TRT-20 external evaluation for VoxTell-family models.

The script consumes only the pre-frozen DKT31+CMA-to-internal-8ROI contract.
It neither updates models nor performs any external-data model selection.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pandas as pd
import torch


BASE = Path("__DRPA_WORKSPACE__")
OASIS = Path("__DRPA_OASIS_ROOT__")
OUT = BASE / "quality_audit/oasis_trt20_external_frozen_test"
MODEL_DIR = BASE / "VoxTell_weights/voxtell_v1.1"
BANK = BASE / "VoxTell_weights/embeddings/voxtell_v1.1/text_embeddings.npz"
TEXT_CACHE = BASE / "quality_audit/voxtell_mtl_peft/text_embedding_cache.npz"
CHECKPOINTS = {
    "FrozenVoxTell": MODEL_DIR / "fold_0/checkpoint_final.pth",
    "DRPA8": BASE / "quality_audit/voxtell_mtl_drpa8_full_data/formal_training/formal_6000_sparse_val_cache_false_20260812/checkpoints/step_06000.pt",
    "B3Canonical": BASE / "quality_audit/voxtell_full_data_b3_vs_drpa_efficiency/b3_full_fp32_formal_3000_6000_rerun/checkpoints/step_06000.pt",
    "B1": BASE / "quality_audit/canonical_data_capacity_completion/task3_b1_100pct/checkpoints/step_06000.pt",
    "FullFT": OUT / "checkpoints/voxtell_fullft_100pct_step06000.pt",
}
EXPECTED = {
    "DRPA8": ("voxtell_mtl_drpa8_v1", 10_969_696),
    "B3Canonical": ("voxtell_mtl_b3_full_data_v1", 81_930_624),
    "B1": ("b1_canonical_trainable_model_state_v1", 294_912),
    "FullFT": (None, 440_029_541),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def imports():
    paths = [
        BASE / "VoxTell", BASE / "quality_audit/voxtell_mtl_peft",
        BASE / "quality_audit/voxtell_mtl_peft_pilot",
        BASE / "quality_audit/voxtell_mtl_b1_bilateral_crop",
        BASE / "quality_audit/voxtell_mtl_drpa8_pilot",
        BASE / "quality_audit/voxtell_mtl_b3_decoder_capacity_upper_bound",
        BASE / "quality_audit/voxtell_full_data_b3_vs_drpa_efficiency",
    ]
    sys.path[:0] = [str(path) for path in paths]
    from b1_preprocessing import BilateralGroupedPatchDataset, CaseRecord, load_crop_spec
    from text_embedding_cache import TextEmbeddingCache
    from drpa8_wrapper import DRPA8Wrapper
    from voxtell_peft_wrapper import VoxTellPEFTWrapper
    from b3_full_data_wrapper import B3FullDataWrapper
    from voxtell.inference.predictor import VoxTellPredictor
    import evaluate_drpa8 as evaluator
    return SimpleNamespace(
        Dataset=BilateralGroupedPatchDataset, CaseRecord=CaseRecord, load_crop_spec=load_crop_spec,
        TextEmbeddingCache=TextEmbeddingCache, DRPA8Wrapper=DRPA8Wrapper,
        B1Wrapper=VoxTellPEFTWrapper, B3Wrapper=B3FullDataWrapper,
        Predictor=VoxTellPredictor, evaluator=evaluator,
    )


def audit_payloads(selected: list[str]) -> pd.DataFrame:
    rows = []
    for name in selected:
        path = CHECKPOINTS[name]
        if not path.is_file():
            rows.append({"model": name, "checkpoint": str(path), "status": "MISSING", "sha256": None})
            continue
        row = {"model": name, "checkpoint": str(path), "sha256": sha256(path), "status": "PASS"}
        if name == "FrozenVoxTell":
            row.update({"format": "official_voxtell_checkpoint_final", "step": 0, "configured_trainable": 0,
                        "provenance": "original frozen VoxTell base; no adapters attached"})
        else:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            fmt = payload.get("format") if isinstance(payload, dict) else None
            meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            config = payload.get("config", {}) if isinstance(payload, dict) else {}
            step = payload.get("global_step", meta.get("optimizer_step", meta.get("step")))
            params = (payload.get("configured_trainable") if isinstance(payload, dict) else None)
            if params is None:
                params = meta.get("trainable_parameters") if isinstance(meta, dict) else None
            # The canonical B1-100 runner stores only the explicitly trainable
            # LoRA tensors under trainable_model_state rather than adapter_state_dict.
            if name == "B1" and isinstance(payload, dict) and "trainable_model_state" in payload:
                fmt = "b1_canonical_trainable_model_state_v1"
                params = sum(t.numel() for t in payload["trainable_model_state"].values())
            row.update({"format": fmt, "step": step, "configured_trainable": params,
                        "metadata_experiment": meta.get("experiment") if isinstance(meta, dict) else None,
                        "config_experiment": config.get("experiment") if isinstance(config, dict) else None})
            expected_fmt, expected_params = EXPECTED[name]
            if name == "FullFT":
                if int(params or -1) != expected_params or int(step or -1) != 6000:
                    row["status"] = "FAIL_IDENTITY"
            elif fmt != expected_fmt or int(params or -1) != expected_params or int(step or -1) != 6000:
                row["status"] = "FAIL_IDENTITY"
        rows.append(row)
    frame = pd.DataFrame(rows)
    audit_path = OUT / "MODEL_PROVENANCE_AUDIT.csv"
    if audit_path.is_file():
        existing = pd.read_csv(audit_path)
        existing = existing[~existing.model.isin(selected)].copy()
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(audit_path, index=False)
    return frame


class NetworkWrapper:
    def __init__(self, model): self.model = model
    def __call__(self, image, text_embedding): return self.model(image, text_embedding)


def load_fullft_model(device):
    gate_path = BASE / "scripts/fullft/fullft_runtime_gate.py"
    spec = importlib.util.spec_from_file_location("external_fullft_gate", gate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.initialise_official_model(BASE)
    payload = torch.load(CHECKPOINTS["FullFT"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return NetworkWrapper(model)


def load_model(name: str, mod, device):
    if name == "FrozenVoxTell":
        predictor = mod.Predictor(model_dir=str(MODEL_DIR), device=torch.device("cpu"), embedding_bank=str(BANK), use_precomputed_embeddings=True)
        predictor.network.to(device).eval()
        return NetworkWrapper(predictor.network)
    if name == "DRPA8":
        wrapper = mod.DRPA8Wrapper(str(MODEL_DIR), str(BANK), device=device)
        wrapper.load_checkpoint(str(CHECKPOINTS[name])); wrapper.eval(); return wrapper
    if name == "B1":
        wrapper = mod.B1Wrapper(str(MODEL_DIR), str(BANK), device=device)
        payload = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=False)
        expected = dict(wrapper.trainable_named_parameters())
        incoming = payload["trainable_model_state"]
        if set(expected) != set(incoming):
            raise RuntimeError("B1 trainable checkpoint keys do not match B1 wrapper")
        with torch.no_grad():
            for key, parameter in expected.items(): parameter.copy_(incoming[key].to(parameter.device, dtype=parameter.dtype))
        wrapper.eval(); return wrapper
    if name == "B3Canonical":
        wrapper = mod.B3Wrapper(str(MODEL_DIR), str(BANK), rank=4, alpha=8, dropout=0.05, device=device)
        payload = torch.load(CHECKPOINTS[name], map_location="cpu", weights_only=False)
        expected = dict(wrapper.trainable_named_parameters())
        incoming = payload["trainable_state_dict"]
        if set(expected) != set(incoming):
            raise RuntimeError("B3 trainable checkpoint keys do not match B3 wrapper")
        with torch.no_grad():
            for key, parameter in expected.items(): parameter.copy_(incoming[key].to(parameter.device, dtype=parameter.dtype))
        wrapper.eval(); return wrapper
    if name == "FullFT": return load_fullft_model(device)
    raise ValueError(name)


def save_raw_masks(name: str, case_id: str, masks: list[np.ndarray], affine: np.ndarray, prompts: list[str]) -> None:
    path = OUT / "raw_masks_lcc0" / name
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"{case_id}.npz", masks=np.stack(masks).astype(np.uint8), affine=np.asarray(affine), prompts=np.asarray(prompts))


def evaluate_one(name, wrapper, records, dataset, cache, evaluator, device) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        item = dataset[index]
        _, label, affine, _ = evaluator.load_official_reader_case(record)
        probabilities = evaluator.infer(wrapper, item, cache)
        spacing = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
        voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000.0
        masks = []
        for prompt in evaluator.PROMPTS:
            probability = evaluator.restore(probabilities[prompt], item, label.shape, affine)
            mask = probability >= 0.5
            masks.append(mask)
            gt = label == evaluator.ROI_LABEL_IDS[prompt]
            pred_ml, gt_ml = float(mask.sum() * voxel_ml), float(gt.sum() * voxel_ml)
            rows.append({
                "model": name, "lcc": 0, "case_id": record.case_id, "subject_id": record.case_id,
                "prompt": prompt, "structure": prompt.split(" ", 1)[1], "side": prompt.split(" ", 1)[0],
                "dice": evaluator.dice(mask, gt), "hd95_mm": evaluator.hd95(mask, gt, spacing),
                "surface_dice_2mm": evaluator.surface_dice(mask, gt, spacing),
                "connected_components": int(evaluator.component_stats(mask)[0]),
                "false_positive_volume_ml": float((mask & ~gt).sum() * voxel_ml),
                "false_negative_volume_ml": float((~mask & gt).sum() * voxel_ml),
                "pred_volume_ml": pred_ml, "gt_volume_ml": gt_ml,
                "volume_error_ml_signed": pred_ml - gt_ml, "volume_error_ml_absolute": abs(pred_ml - gt_ml),
                "empty_mask": int(not mask.any()),
            })
        save_raw_masks(name, record.case_id, masks, affine, evaluator.PROMPTS)
        print(f"external-evaluated {name} {index + 1}/{len(records)} {record.case_id}", flush=True)
    del wrapper
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_8_roi": lambda q: q,
        "hipp_amygdala": lambda q: q[q.structure.isin(["hippocampus", "amygdala"])],
        "ec_phg_protocol_sensitive": lambda q: q[q.structure.isin(["entorhinal cortex", "parahippocampal gyrus"])],
        "hippocampus": lambda q: q[q.structure.eq("hippocampus")],
        "amygdala": lambda q: q[q.structure.eq("amygdala")],
        "entorhinal_cortex": lambda q: q[q.structure.eq("entorhinal cortex")],
        "parahippocampal_gyrus": lambda q: q[q.structure.eq("parahippocampal gyrus")],
    }
    rows = []
    for model in sorted(frame.model.unique()):
        subset = frame[frame.model == model]
        for scope, select in scopes.items():
            q = select(subset)
            rows.append({"model": model, "lcc": 0, "scope": scope, "n_subjects": q.subject_id.nunique(), "n_rows": len(q),
                         **{metric: float(q[metric].mean()) for metric in ["dice", "hd95_mm", "surface_dice_2mm", "false_positive_volume_ml", "false_negative_volume_ml", "pred_volume_ml", "gt_volume_ml", "volume_error_ml_signed", "volume_error_ml_absolute", "connected_components"]}})
    return pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame, seed: int = 20260908, draws: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed); rows, pairs = [], []
    scopes = {"all_8_roi": lambda q: q, "hipp_amygdala": lambda q: q[q.structure.isin(["hippocampus", "amygdala"])], "ec_phg_protocol_sensitive": lambda q: q[q.structure.isin(["entorhinal cortex", "parahippocampal gyrus"])]}
    models = sorted(frame.model.unique())
    for scope, select in scopes.items():
        subject_table = select(frame).groupby(["model", "subject_id"], as_index=False).dice.mean().pivot(index="subject_id", columns="model", values="dice").dropna()
        ids = subject_table.index.to_numpy(); n = len(ids)
        idx = rng.integers(0, n, size=(draws, n))
        for model in models:
            values = subject_table[model].to_numpy(); sampled = values[idx].mean(axis=1)
            rows.append({"scope": scope, "model": model, "n_subjects": n, "dice_mean": float(values.mean()), "ci95_low": float(np.quantile(sampled, .025)), "ci95_high": float(np.quantile(sampled, .975)), "bootstrap_draws": draws, "seed": seed})
        if "FrozenVoxTell" in models:
            base = subject_table["FrozenVoxTell"].to_numpy()
            for model in models:
                if model == "FrozenVoxTell": continue
                values = subject_table[model].to_numpy() - base; sampled = values[idx].mean(axis=1)
                pairs.append({"scope": scope, "comparison": f"{model}-FrozenVoxTell", "n_subjects": n, "paired_dice_delta": float(values.mean()), "ci95_low": float(np.quantile(sampled, .025)), "ci95_high": float(np.quantile(sampled, .975)), "bootstrap_draws": draws, "seed": seed})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def report(provenance: pd.DataFrame, metrics: pd.DataFrame, summary_frame: pd.DataFrame, bootstrap_ci: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    lines = ["# OASIS-TRT-20 External Frozen-Test Report", "", "## Contract", "", "- 20/20 OASIS MNI152 MRI/manual DKT31+CMA pairs; all primary metrics use raw thresholded masks (`lcc=0`, threshold `>=0.5`).", "- External mapping was frozen before inference. OASIS DKT31+CMA versus internal MALP-EM is an annotation-protocol shift; EC/PHG are reported as protocol-sensitive.", "- No OASIS labels, prompt, threshold, preprocessing, model, or checkpoint was selected or tuned by external outcomes.", "", "## Model provenance", "", provenance.to_markdown(index=False), "", "## Mean external metrics", "", summary_frame.to_markdown(index=False), "", "## Subject-level bootstrap, Dice", "", bootstrap_ci.to_markdown(index=False), "", "## Paired subject-level Dice bootstrap versus frozen VoxTell", "", pairwise.to_markdown(index=False), "", "## Files", "", "- `OASIS_EXTERNAL_CASE_LEVEL_METRICS.csv`: per subject × ROI raw-mask metrics.", "- `raw_masks_lcc0/`: compressed per-model/per-subject, per-prompt thresholded masks; no LCC was applied.", "- `OASIS_EXTERNAL_BOOTSTRAP*.csv`: frozen subject-level uncertainty analysis."]
    (OUT / "OASIS_TRT20_EXTERNAL_FROZEN_TEST_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="FrozenVoxTell,DRPA8,B3Canonical,B1,FullFT")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--report-existing", action="store_true", help="report-only repair from immutable existing case rows")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if args.report_existing:
        metrics = pd.read_csv(OUT / "OASIS_EXTERNAL_CASE_LEVEL_METRICS.csv")
        provenance = pd.read_csv(OUT / "MODEL_PROVENANCE_AUDIT.csv")
        summ = summary(metrics); summ.to_csv(OUT / "OASIS_EXTERNAL_SUMMARY_METRICS.csv", index=False)
        ci, paired = bootstrap(metrics); ci.to_csv(OUT / "OASIS_EXTERNAL_BOOTSTRAP_DICE.csv", index=False); paired.to_csv(OUT / "OASIS_EXTERNAL_PAIRED_BOOTSTRAP_DICE.csv", index=False)
        report(provenance, metrics, summ, ci, paired)
        (OUT / "run_status.json").write_text(json.dumps({"status": "COMPLETE_PARTIAL_FULLFT_PENDING", "models": sorted(metrics.model.unique()), "subjects": 20, "rows": len(metrics), "lcc": 0, "threshold": ">=0.5", "report_recovered_without_rerun": True}, indent=2) + "\n")
        return
    selected = [x.strip() for x in args.models.split(",") if x.strip()]
    if any(x not in CHECKPOINTS for x in selected): raise ValueError(selected)
    provenance = audit_payloads(selected)
    if not provenance.status.eq("PASS").all(): raise RuntimeError(provenance.to_string(index=False))
    preflight = OUT / "preflight" / "oasis_external_preflight_summary.json"
    if not preflight.is_file() or json.loads(preflight.read_text()).get("status") != "OASIS_EXTERNAL_INPUT_PREFLIGHT_PASS":
        raise RuntimeError("OASIS input preflight not passed")
    if args.preflight_only:
        print(provenance.to_string(index=False)); return
    mod = imports(); device = torch.device("cuda")
    manifest = pd.read_csv(OUT / "preflight/external_subject_manifest_mapped.csv").sort_values("subject_id")
    records = [mod.CaseRecord(str(x.case_id), str(x.image_path), str(x.label_path)) for x in manifest.itertuples()]
    spec = mod.load_crop_spec(BASE / "quality_audit/voxtell_mtl_drpa8_pilot/crop_spec.json")
    dataset = mod.Dataset(records, spec, cache_cases=False)
    cache = mod.TextEmbeddingCache(str(BANK), str(MODEL_DIR), str(TEXT_CACHE))
    frames = []
    for name in selected:
        wrapper = load_model(name, mod, device)
        frames.append(evaluate_one(name, wrapper, records, dataset, cache, mod.evaluator, device))
    # A later admissible model (e.g. staged FullFT) may be evaluated without
    # rerunning already-frozen external inferences. Existing results are kept
    # only for models not requested in this invocation.
    metrics_path = OUT / "OASIS_EXTERNAL_CASE_LEVEL_METRICS.csv"
    prior = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    if not prior.empty:
        prior = prior[~prior.model.isin(selected)].copy()
    metrics = pd.concat([prior, *frames], ignore_index=True)
    metrics.to_csv(OUT / "OASIS_EXTERNAL_CASE_LEVEL_METRICS.csv", index=False)
    summ = summary(metrics); summ.to_csv(OUT / "OASIS_EXTERNAL_SUMMARY_METRICS.csv", index=False)
    ci, paired = bootstrap(metrics); ci.to_csv(OUT / "OASIS_EXTERNAL_BOOTSTRAP_DICE.csv", index=False); paired.to_csv(OUT / "OASIS_EXTERNAL_PAIRED_BOOTSTRAP_DICE.csv", index=False)
    report(provenance, metrics, summ, ci, paired)
    (OUT / "run_status.json").write_text(json.dumps({"status": "COMPLETE", "models": selected, "subjects": 20, "rows": len(metrics), "lcc": 0, "threshold": ">=0.5"}, indent=2) + "\n")


if __name__ == "__main__": main()
