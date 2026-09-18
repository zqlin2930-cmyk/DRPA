#!/usr/bin/env python3
"""Read-only, PTID-clustered audit for the 50% r4/r8/r16 matched pipeline."""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIRS = {
    "r4": "rank4_hardened_batch1_step0",
    "r8": "rank8_hardened_batch1_step0",
    "r16": "rank16_hardened_batch1_step0",
}
STRUCTURES = ["hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"]
CONFIG_FIELDS = [
    "seed", "max_optimizer_steps", "train_visits", "train_ptids", "val_visits",
    "val_ptids", "batch_size", "loss", "amp_forward", "autocast_dtype", "grad_scaler",
    "data_pipeline", "data_source", "num_workers", "prefetch_factor", "pin_memory",
    "persistent_workers", "non_blocking_h2d", "multiprocessing_context", "pipeline_equivalence",
    "train_manifest_sha256", "val_manifest_sha256",
]


def die(message: str) -> None:
    raise RuntimeError(message)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """Dependency-free GitHub Markdown table formatter."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            elif pd.isna(value):
                values.append("NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_run(input_root: Path, rank: str) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    run = input_root / RUN_DIRS[rank]
    with (run / "config.json").open() as f:
        config = json.load(f)
    with (run / "run_summary.json").open() as f:
        summary = json.load(f)
    rows = pd.read_csv(run / "validation_rows_step_06000.csv")
    evaluator_summary = pd.read_csv(run / "validation_summary_step_06000.csv")
    return config, summary, rows, evaluator_summary


def ptid_metrics(rows_lcc0: pd.DataFrame) -> pd.DataFrame:
    frame = rows_lcc0.copy()
    # V_gt = TP + FN and V_pred = TP + FP, so FN = V_gt - V_pred + FP.
    frame["false_negative_volume_ml"] = (
        frame["gt_volume_ml"] - frame["pred_volume_ml"] + frame["false_positive_volume_ml"]
    )
    frame["false_negative_volume_ml"] = frame["false_negative_volume_ml"].clip(lower=0.0)

    result = pd.DataFrame(index=sorted(frame["ptid"].unique()))
    result.index.name = "ptid"
    all_metrics = {
        "mean_dice": "dice",
        "hd95_mm": "hd95_mm",
        "surface_dice_2mm": "surface_dice_2mm",
        "fp_volume_ml": "false_positive_volume_ml",
        "fn_volume_ml": "false_negative_volume_ml",
        "components": "connected_components",
    }
    for output, source in all_metrics.items():
        result[output] = frame.groupby("ptid")[source].mean()
    for structure, prefix in [
        ("hippocampus", "hipp_dice"),
        ("entorhinal cortex", "ec_dice"),
        ("parahippocampal gyrus", "phg_dice"),
        ("amygdala", "amy_dice"),
    ]:
        result[prefix] = frame.loc[frame["structure"] == structure].groupby("ptid")["dice"].mean()
    return result.reset_index()


def bootstrap_contrast(left: pd.DataFrame, right: pd.DataFrame, metric: str, rng: np.random.Generator, n_boot: int) -> dict:
    merged = left[["ptid", metric]].merge(right[["ptid", metric]], on="ptid", suffixes=("_left", "_right"), validate="one_to_one")
    # Contrast spelling is left minus right. All PTIDs carry their visits together.
    delta = (merged[f"{metric}_left"] - merged[f"{metric}_right"]).to_numpy(dtype=float)
    n = len(delta)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = delta[draws].mean(axis=1)
    return {
        "metric": metric,
        "n_ptids": n,
        "paired_mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "ptid_left_win_fraction": float((delta > 0).mean()),
        "ptid_right_win_fraction": float((delta < 0).mean()),
        "ptid_exact_tie_fraction": float((delta == 0).mean()),
        "bootstrap_replicates": n_boot,
        "bootstrap_seed": 20260908,
    }


def evaluator_overall(summary: pd.DataFrame) -> pd.Series:
    rows = summary.loc[(summary["lcc"] == 0) & (summary["scope"] == "overall")]
    if len(rows) != 1:
        die("Expected exactly one LCC=0 overall evaluator summary row.")
    return rows.iloc[0]


def write_markdown(out: Path, provenance: pd.DataFrame, final: pd.DataFrame, paired: pd.DataFrame) -> None:
    overall = paired.loc[paired["metric"] == "mean_dice"].copy()
    lines = [
        "# Rank Ablation @50% — Matched-Pipeline Final Audit",
        "",
        "## Scope",
        "",
        "This read-only analysis compares rank-4, the newly rerun rank-8, and rank-16 under the retained 50% AMP-forward protocol. The primary table reproduces the frozen LCC=0, visit-weighted evaluator output (247 visits). Paired uncertainty is clustered at PTID (85 PTIDs), so its paired mean can differ slightly from the visit-weighted difference because PTIDs have unequal visit counts. No voxel-level test is used.",
        "",
        "## Protocol identity",
        "",
        markdown_table(provenance),
        "",
        "## LCC=0 visit-weighted final metrics",
        "",
        markdown_table(final),
        "",
        "## Overall Dice paired PTID bootstrap",
        "",
        markdown_table(overall[["contrast", "paired_mean_delta", "ci95_low", "ci95_high", "ptid_left_win_fraction", "ptid_right_win_fraction"]]),
        "",
        "## Interpretation boundary",
        "",
        "The one-seed rank-8 minus rank-4 Dice difference and rank-16 minus rank-8 Dice difference quantify rank sensitivity under the new matched pipeline. They do not establish a stable cross-seed rank optimum. The planned multi-seed stage remains blocked until the user explicitly freezes seed2 and seed3; r8/r16 at seed 20260809 will be reused and not rerun.",
    ]
    (out / "RANK_ABLATION_50_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")


def write_multiseed_plan(out: Path) -> None:
    yaml = """study: RANK_MULTISEED_50
status: MULTISEED_BLOCKED_WAITING_FOR_USER_SEED_FREEZE
purpose: Validate rank-8 versus rank-16 across three fixed seeds at 50% data.
reused_seed:
  seed: 20260809
  r8_artifact: rank8_hardened_batch1_step0
  r16_artifact: rank16_hardened_batch1_step0
jobs:
  - {seed_name: seed2, seed: null, rank: 8, status: BLOCKED_WAITING_FOR_USER_SEED_FREEZE}
  - {seed_name: seed2, seed: null, rank: 16, status: BLOCKED_WAITING_FOR_USER_SEED_FREEZE}
  - {seed_name: seed3, seed: null, rank: 8, status: BLOCKED_WAITING_FOR_USER_SEED_FREEZE}
  - {seed_name: seed3, seed: null, rank: 16, status: BLOCKED_WAITING_FOR_USER_SEED_FREEZE}
frozen_protocol:
  train_ptids: 169
  train_visits: 461
  batch_size: 1
  optimizer_steps: 6000
  amp_forward: true
  loss: FP32 Dice+BCE
  cross_attention_lora_rank: 4
  decoder_stages_trainable: true
  data_pipeline: BEST_BATCH1_PIPELINE
  data_source: DISK_CACHE
  num_workers: 32
  prefetch_factor: 2
  evaluator_lcc: 0
forbidden:
  - rank32
  - r4_multiseed
  - new_adapter_loss_or_readout
  - automatic_seed_generation
"""
    (out / "RANK_MULTISEED_50_PLAN.yaml").write_text(yaml)
    protocol = """# Rank Multi-seed @50% Protocol\n\nStatus: `MULTISEED_BLOCKED_WAITING_FOR_USER_SEED_FREEZE`.\n\nSeed `20260809` is reused from the completed new-pipeline r8 and r16 artifacts. The only future runs are r8/r16 for the user-supplied `seed2` and `seed3`, producing four 6000-step jobs. The training and evaluation protocol is exactly the one in `RANK_MULTISEED_50_PLAN.yaml`; these files do not authorize a launch until both seeds are explicitly frozen.\n\nThe final analysis will report every seed's Overall/Hipp/EC/PHG/Amy Dice, HD95, Surface Dice, r16-r8 paired difference, and three-seed mean ± SD.\n"""
    (out / "RANK_MULTISEED_50_PROTOCOL.md").write_text(protocol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    configs, summaries, evaluator_by_rank, lcc0_by_rank, ptid_by_rank = {}, {}, {}, {}, {}
    identity = None
    provenance_rows = []
    for rank in RUN_DIRS:
        config, run_summary, rows, evaluator = load_run(args.input_root, rank)
        if run_summary.get("status") != "COMPLETE" or run_summary.get("steps") != 6000:
            die(f"{rank}: incomplete status or non-6000 step count.")
        if config.get("projection_rank") != int(rank[1:]):
            die(f"{rank}: projection rank in config does not match directory.")
        if rows["lcc"].isin([0, 1]).all() is False:
            die(f"{rank}: unexpected LCC value.")
        lcc0 = rows.loc[rows["lcc"] == 0].copy()
        if len(lcc0) != 1976 or lcc0["case_id"].nunique() != 247 or lcc0["ptid"].nunique() != 85:
            die(f"{rank}: LCC=0 validation population is not 1976 rows / 247 visits / 85 PTIDs.")
        if lcc0.duplicated(["case_id", "prompt", "structure", "side"]).any():
            die(f"{rank}: duplicate evaluator identity rows.")
        current_identity = lcc0[["case_id", "ptid", "prompt", "structure", "side"]].sort_values(["case_id", "prompt"]).reset_index(drop=True)
        if identity is None:
            identity = current_identity
        elif not identity.equals(current_identity):
            die(f"{rank}: validation visit/prompt identity differs from the other ranks.")
        summary_overall = evaluator_overall(evaluator)
        raw_overall = lcc0["dice"].mean()
        if not np.isclose(raw_overall, summary_overall["dice"], atol=1e-12):
            die(f"{rank}: raw LCC=0 Dice disagrees with evaluator summary.")
        configs[rank] = config
        summaries[rank] = summary_overall
        evaluator_by_rank[rank] = evaluator
        lcc0_by_rank[rank] = lcc0
        ptid_by_rank[rank] = ptid_metrics(lcc0)
        provenance = {"rank": rank, "artifact_dir": RUN_DIRS[rank], "status": "MATCHED_PIPELINE_PASS"}
        provenance.update({field: config.get(field) for field in CONFIG_FIELDS})
        provenance.update({
            "lcc": 0, "lcc0_rows": len(lcc0), "lcc0_visits": lcc0["case_id"].nunique(),
            "lcc0_ptids": lcc0["ptid"].nunique(), "trainable_parameters": config.get("trainable_parameters"),
            "raw_summary_dice_abs_diff": abs(raw_overall - summary_overall["dice"]),
        })
        provenance_rows.append(provenance)

    provenance_df = pd.DataFrame(provenance_rows)
    shared = [field for field in CONFIG_FIELDS if field not in {"seed"}]
    mismatch = [field for field in shared if provenance_df[field].nunique(dropna=False) != 1]
    if mismatch:
        die(f"Protocol mismatch across r4/r8/r16: {mismatch}")
    provenance_df.to_csv(args.out / "RANK_ABLATION_50_PROVENANCE_AUDIT.csv", index=False)
    (args.out / "RANK_ABLATION_50_PROVENANCE_AUDIT.md").write_text(
        "# Rank Ablation @50% Provenance Audit\n\n"
        "All three retained artifacts pass the requested matched-pipeline predicates. The table records raw config and evaluator facts; LCC=0 row identity is identical across ranks.\n\n"
        + markdown_table(provenance_df) + "\n"
    )

    metric_order = ["mean_dice", "hipp_dice", "ec_dice", "phg_dice", "amy_dice", "hd95_mm", "surface_dice_2mm", "fp_volume_ml", "fn_volume_ml", "components"]
    final_rows = []
    for rank in RUN_DIRS:
        metrics, summary, raw = ptid_by_rank[rank], evaluator_by_rank[rank], lcc0_by_rank[rank]
        summary0 = summary.loc[summary["lcc"] == 0].set_index("scope")
        row = {
            "rank": rank,
            "trainable_parameters": configs[rank]["trainable_parameters"],
            "n_visits": raw["case_id"].nunique(),
            "n_ptids": len(metrics),
            "mean_dice": summary0.loc["overall", "dice"],
            "hipp_dice": summary0.loc["hippocampus", "dice"],
            "ec_dice": summary0.loc["entorhinal cortex", "dice"],
            "phg_dice": summary0.loc["parahippocampal gyrus", "dice"],
            "amy_dice": summary0.loc["amygdala", "dice"],
            "hd95_mm": summary0.loc["overall", "hd95_mm"],
            "surface_dice_2mm": summary0.loc["overall", "surface_dice_2mm"],
            "fp_volume_ml": summary0.loc["overall", "false_positive_volume_ml"],
            "fn_volume_ml": (raw["gt_volume_ml"] - raw["pred_volume_ml"] + raw["false_positive_volume_ml"]).clip(lower=0.0).mean(),
            "components": summary0.loc["overall", "connected_components"],
            "ptid_cluster_mean_dice": metrics["mean_dice"].mean(),
        }
        final_rows.append(row)
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(args.out / "RANK_ABLATION_50_FINAL.csv", index=False)

    rng = np.random.default_rng(20260908)
    contrasts = [("r8-r4", "r8", "r4"), ("r16-r8", "r16", "r8"), ("r16-r4", "r16", "r4")]
    paired_rows = []
    for name, left, right in contrasts:
        for metric in metric_order:
            result = bootstrap_contrast(ptid_by_rank[left], ptid_by_rank[right], metric, rng, args.bootstrap)
            result.update({"contrast": name, "left_rank": left, "right_rank": right})
            paired_rows.append(result)
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(args.out / "RANK_ABLATION_50_PAIRED_BOOTSTRAP.csv", index=False)
    write_markdown(args.out, provenance_df, final_df, paired_df)
    write_multiseed_plan(args.out)
    print("RANK_MATCHED_PIPELINE_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
