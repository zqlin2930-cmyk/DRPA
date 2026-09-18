#!/usr/bin/env python3
"""Finalize the frozen rank-4/rank-8/rank-16 comparison from stored rows only."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
ROOT = BASE / "quality_audit/rank_ablation_50"
R4 = ROOT / "rank4_hardened_batch1_step0/validation_rows_step_06000.csv"
R16 = ROOT / "rank16_hardened_batch1_step0/validation_rows_step_06000.csv"
PLACEMENT = BASE / "quality_audit/placement_ablation_50"
R8_PTID = PLACEMENT / "PLACEMENT_UNIFIED_PTID_METRICS.csv"
R8_SUMMARY = PLACEMENT / "PLACEMENT_UNIFIED_COMPARISON.csv"
SEED = 20260907
DRAW_COUNT = 10_000
SCOPES = ["overall", "hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"]
METRICS = ["dice", "hd95_mm", "surface_dice_2mm", "false_positive_volume_ml",
           "fn_volume_ml", "connected_components"]
PARAMS = {"r4": 10_748_384, "r8": 10_969_696, "r16": 11_412_320}


def add_fn(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.loc[frame["lcc"].eq(0)].copy()
    tp_ml = x["dice"] * (x["pred_volume_ml"] + x["gt_volume_ml"]) / 2.0
    x["fn_volume_ml"] = (x["gt_volume_ml"] - tp_ml).clip(lower=0.0)
    return x


def scope_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    return frame if scope == "overall" else frame.loc[frame["structure"].eq(scope)]


def from_raw(path: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    counts = raw.groupby("lcc").size().to_dict()
    if counts != {0: 1976, 1: 1976}:
        raise RuntimeError(f"{model} raw-row contract mismatch: {counts}")
    frame = add_fn(raw)
    if frame["ptid"].nunique() != 85 or frame["case_id"].nunique() != 247:
        raise RuntimeError(f"{model} cohort contract mismatch")
    summaries, ptids = [], []
    for scope in SCOPES:
        part = scope_rows(frame, scope)
        summary = {"rank": model, "scope": scope, "trainable_params": PARAMS[model],
                   "n_rows": len(part), "n_visits": part.case_id.nunique(),
                   "n_ptids": part.ptid.nunique()}
        summary.update({metric: float(part[metric].mean()) for metric in METRICS})
        summaries.append(summary)
        grouped = part.groupby("ptid", as_index=False)[METRICS].mean()
        grouped.insert(1, "scope", scope)
        grouped.insert(2, "rank", model)
        ptids.append(grouped)
    return pd.DataFrame(summaries), pd.concat(ptids, ignore_index=True)


def load_r8() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(R8_SUMMARY)
    summary = summary.loc[summary["condition"].eq("P3_DRPA_50")].copy()
    summary = summary.rename(columns={"n_mask_rows": "n_rows", "condition": "rank"})
    summary["rank"] = "r8"
    summary["trainable_params"] = PARAMS["r8"]
    summary = summary[["rank", "scope", "trainable_params", "n_rows", "n_visits", "n_ptids", *METRICS]]
    ptid = pd.read_csv(R8_PTID)
    ptid = ptid.loc[ptid["condition"].eq("P3_DRPA_50")].drop(columns="condition")
    ptid.insert(2, "rank", "r8")
    if len(summary) != 5 or ptid.ptid.nunique() != 85:
        raise RuntimeError("r8 canonical PTID provenance incomplete")
    return summary, ptid


def bootstrap(base: pd.DataFrame, variant: pd.DataFrame, comparison: str) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for scope in SCOPES:
        b = base.loc[base.scope.eq(scope)].set_index("ptid")
        v = variant.loc[variant.scope.eq(scope)].set_index("ptid")
        common = b.index.intersection(v.index).sort_values()
        if len(common) != 85:
            raise RuntimeError(f"{comparison}/{scope} has {len(common)} paired PTIDs")
        for metric in METRICS:
            diff = (v.loc[common, metric] - b.loc[common, metric]).to_numpy(float)
            draws = diff[rng.integers(0, len(diff), size=(DRAW_COUNT, len(diff)))].mean(axis=1)
            rows.append({
                "comparison": comparison, "scope": scope, "metric": metric,
                "n_ptids": len(diff), "paired_mean_difference": float(diff.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "variant_win_fraction": float((diff > 1e-12).mean()) if metric == "dice" else np.nan,
                "baseline_win_fraction": float((diff < -1e-12).mean()) if metric == "dice" else np.nan,
                "tie_fraction": float((np.abs(diff) <= 1e-12).mean()) if metric == "dice" else np.nan,
            })
    return rows


def main() -> None:
    r4_summary, r4_ptid = from_raw(R4, "r4")
    r8_summary, r8_ptid = load_r8()
    r16_summary, r16_ptid = from_raw(R16, "r16")
    final = pd.concat([r4_summary, r8_summary, r16_summary], ignore_index=True)
    paired = pd.DataFrame(
        bootstrap(r4_ptid, r8_ptid, "r8-r4")
        + bootstrap(r8_ptid, r16_ptid, "r16-r8")
        + bootstrap(r4_ptid, r16_ptid, "r16-r4")
    )
    final.to_csv(ROOT / "RANK_ABLATION_50_FINAL.csv", index=False)
    paired.to_csv(ROOT / "RANK_ABLATION_50_PAIRED_BOOTSTRAP.csv", index=False)
    lines = [
        "# Rank Ablation @ 50% — Final Report", "",
        "Status: `RANK_ABLATION_50_COMPLETE`", "",
        "All results use the frozen 50% cohort, 6000-step AMP-forward + FP32-loss protocol and canonical raw-mask (`lcc=0`) evaluator. Rank-8 is the pre-existing canonical DRPA-50 result; it was not retrained.", "",
        "## Overall", "",
        "| Rank | Trainable params | Dice | HD95 mm | Surface Dice@2mm | FP ml | FN ml | Components |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in final.loc[final.scope.eq("overall")].itertuples():
        lines.append(f"| {row.rank} | {row.trainable_params:,} | {row.dice:.6f} | {row.hd95_mm:.6f} | {row.surface_dice_2mm:.6f} | {row.false_positive_volume_ml:.6f} | {row.fn_volume_ml:.6f} | {row.connected_components:.6f} |")
    lines += ["", "## ROI Dice", "", "| Rank | Hipp | EC | PHG | Amy |", "|---|---:|---:|---:|---:|"]
    for rank in ("r4", "r8", "r16"):
        x = final.loc[final["rank"].eq(rank)].set_index("scope")
        lines.append(f"| {rank} | {x.loc['hippocampus','dice']:.6f} | {x.loc['entorhinal cortex','dice']:.6f} | {x.loc['parahippocampal gyrus','dice']:.6f} | {x.loc['amygdala','dice']:.6f} |")
    lines += ["", "## PTID-paired Dice bootstrap", "", "| Comparison | Scope | Mean difference | 95% CI |", "|---|---|---:|---|"]
    for row in paired.loc[paired.metric.eq("dice")].itertuples():
        lines.append(f"| {row.comparison} | {row.scope} | {row.paired_mean_difference:+.6f} | [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] |")
    lines += ["", "Bootstrap unit: PTID; 10,000 resamples; seed 20260907. FN is reconstructed from stored Dice/predicted-volume/GT-volume rows using the same placement-analysis formula; no inference was rerun."]
    (ROOT / "RANK_ABLATION_50_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    (ROOT / "RANK_ABLATION_50_FINAL_STATUS.json").write_text(json.dumps({
        "status": "RANK_ABLATION_50_COMPLETE", "ranks": [4, 8, 16],
        "ptids": 85, "visits": 247, "bootstrap_draws": DRAW_COUNT,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
