#!/usr/bin/env python3
"""Read-only P0/P1/P2/P3 placement comparison from existing case-level CSVs."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260907
METRICS = ["dice", "hd95_mm", "surface_dice_2mm", "false_positive_volume_ml", "fn_volume_ml", "connected_components"]
SCOPES = ["overall", "hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala"]


def load_condition(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["lcc"].eq(0)].copy()
    frame["condition_id"] = condition
    numerator = frame["dice"] * (frame["pred_volume_ml"] + frame["gt_volume_ml"])
    true_positive = numerator / 2.0
    frame["fn_volume_ml"] = (frame["gt_volume_ml"] - true_positive).clip(lower=0.0)
    return frame


def scoped_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "overall":
        return frame
    return frame[frame["structure"].eq(scope)]


def aggregate_condition(frame: pd.DataFrame, condition: str) -> list[dict]:
    rows = []
    for scope in SCOPES:
        part = scoped_rows(frame, scope)
        row = {"condition": condition, "scope": scope, "n_mask_rows": len(part), "n_visits": part["case_id"].nunique(), "n_ptids": part["ptid"].nunique()}
        row.update({metric: float(part[metric].mean()) for metric in METRICS})
        rows.append(row)
    return rows


def ptid_table(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    parts = []
    for scope in SCOPES:
        part = scoped_rows(frame, scope)
        x = part.groupby("ptid", as_index=False)[METRICS].mean()
        x.insert(1, "scope", scope)
        x.insert(2, "condition", condition)
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def bootstrap_pair(base: pd.DataFrame, variant: pd.DataFrame, comparison: str, draws: int) -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows = []
    for scope in SCOPES:
        b = base[base.scope.eq(scope)].set_index("ptid")[METRICS]
        v = variant[variant.scope.eq(scope)].set_index("ptid")[METRICS]
        common = b.index.intersection(v.index).sort_values()
        if len(common) == 0:
            continue
        b, v = b.loc[common], v.loc[common]
        for metric in METRICS:
            diff = (v[metric] - b[metric]).to_numpy(float)
            idx = rng.integers(0, len(diff), size=(draws, len(diff)))
            boot = diff[idx].mean(axis=1)
            item = {
                "comparison": comparison,
                "scope": scope,
                "metric": metric,
                "n_ptids": len(diff),
                "paired_mean_difference": float(diff.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
            }
            if metric == "dice":
                item.update({
                    "variant_win_fraction": float((diff > 1e-12).mean()),
                    "baseline_win_fraction": float((diff < -1e-12).mean()),
                    "tie_fraction": float((np.abs(diff) <= 1e-12).mean()),
                })
            rows.append(item)
    return rows


def render_report(summary: pd.DataFrame, paired: pd.DataFrame, provenance: dict, output: Path) -> None:
    lines = ["# Canonical Placement Ablation @ 50%", "", "## Protocol", "", "- Training: 169 PTIDs / 461 visits; validation: 85 PTIDs / 247 visits.", "- All four conditions use seed 20260809, 6000 optimizer steps, AMP-forward, and FP32 Dice+BCE loss.", "- Results are recalculated only from pre-existing `lcc=0` case-level validation rows; no training or inference was run.", "- P2 completed training and 247/247 validation, but its post-validation checkpoint is missing because the original report writer failed on an unavailable optional field.", "", "## Overall metrics", "", "| Condition | Params | Dice | HD95 mm | Surface Dice@2mm | FP ml | FN ml | Components |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    all_rows = summary[summary.scope.eq("overall")]
    for _, r in all_rows.iterrows():
        lines.append(f"| {r.condition} | {provenance[r.condition]['params']:,} | {r.dice:.6f} | {r.hd95_mm:.6f} | {r.surface_dice_2mm:.6f} | {r.false_positive_volume_ml:.6f} | {r.fn_volume_ml:.6f} | {r.connected_components:.6f} |")
    lines += ["", "## ROI Dice", "", "| Condition | Hipp | EC | PHG | Amy |", "|---|---:|---:|---:|---:|"]
    for condition in provenance:
        x = summary[summary.condition.eq(condition)].set_index("scope")
        lines.append(f"| {condition} | {x.loc['hippocampus','dice']:.6f} | {x.loc['entorhinal cortex','dice']:.6f} | {x.loc['parahippocampal gyrus','dice']:.6f} | {x.loc['amygdala','dice']:.6f} |")
    lines += ["", "## PTID-level paired Dice", "", "| Comparison | Scope | Mean difference | 95% CI | Variant win / baseline win / tie |", "|---|---|---:|---|---:|"]
    for _, r in paired[paired.metric.eq("dice")].iterrows():
        win = f"{r.get('variant_win_fraction', np.nan):.3f} / {r.get('baseline_win_fraction', np.nan):.3f} / {r.get('tie_fraction', np.nan):.3f}"
        lines.append(f"| {r.comparison} | {r.scope} | {r.paired_mean_difference:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] | {win} |")
    lines += ["", "## Provenance", ""]
    for key, value in provenance.items():
        lines.append(f"- `{key}`: source `{value['source']}`; checkpoint status `{value['checkpoint_status']}`.")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", required=True, type=Path)
    ap.add_argument("--p1", required=True, type=Path)
    ap.add_argument("--p2", required=True, type=Path)
    ap.add_argument("--p3", required=True, type=Path)
    ap.add_argument("--p0-provenance")
    ap.add_argument("--p1-provenance")
    ap.add_argument("--p2-provenance")
    ap.add_argument("--p3-provenance")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {"P0_B1_50": args.p0, "P1_B1_PROJECTION_50": args.p1, "P2_B1_DECODER_50": args.p2, "P3_DRPA_50": args.p3}
    provenance = {
        "P0_B1_50": {"params": 294912, "source": args.p0_provenance or str(args.p0), "checkpoint_status": "available"},
        "P1_B1_PROJECTION_50": {"params": 737536, "source": args.p1_provenance or str(args.p1), "checkpoint_status": "available"},
        "P2_B1_DECODER_50": {"params": 10527072, "source": args.p2_provenance or str(args.p2), "checkpoint_status": "P2_METRICS_COMPLETE_CHECKPOINT_MISSING"},
        "P3_DRPA_50": {"params": 10969696, "source": args.p3_provenance or str(args.p3), "checkpoint_status": "available"},
    }
    frames = {name: load_condition(path, name) for name, path in files.items()}
    summary = pd.DataFrame([item for name, frame in frames.items() for item in aggregate_condition(frame, name)])
    ptids = pd.concat([ptid_table(frame, name) for name, frame in frames.items()], ignore_index=True)
    paired_rows = []
    comparisons = [
        ("P0_B1_50", "P1_B1_PROJECTION_50"),
        ("P0_B1_50", "P2_B1_DECODER_50"),
        ("P0_B1_50", "P3_DRPA_50"),
        ("P1_B1_PROJECTION_50", "P2_B1_DECODER_50"),
        ("P1_B1_PROJECTION_50", "P3_DRPA_50"),
        ("P2_B1_DECODER_50", "P3_DRPA_50"),
    ]
    for base, name in comparisons:
        paired_rows += bootstrap_pair(
            ptids[ptids.condition.eq(base)],
            ptids[ptids.condition.eq(name)],
            f"{name} - {base}",
            args.bootstrap_draws,
        )
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(args.out_dir / "PLACEMENT_UNIFIED_COMPARISON.csv", index=False)
    ptids.to_csv(args.out_dir / "PLACEMENT_UNIFIED_PTID_METRICS.csv", index=False)
    paired.to_csv(args.out_dir / "PLACEMENT_UNIFIED_PAIRED_BOOTSTRAP.csv", index=False)
    render_report(summary, paired, provenance, args.out_dir / "PLACEMENT_UNIFIED_COMPARISON.md")


if __name__ == "__main__":
    main()
