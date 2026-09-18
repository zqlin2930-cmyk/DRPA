#!/usr/bin/env python3
"""Artifact-gated serial completion of the missing r4 seeds at 50% data."""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("__DRPA_WORKSPACE__")
ROOT = BASE / "quality_audit/rank_ablation_50/multiseed_50"
RUNNER = BASE / "scripts/train/rank_ablation_50_runner.py"
PYTHON = Path("__DRPA_PYTHON__")
REUSED = BASE / "quality_audit/rank_ablation_50/rank4_hardened_batch1_step0"
JOBS = [(3407, 4), (2026, 4)]
STATE = ROOT / "rank4_supplement_supervisor_state.json"


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def artifact_gate(out: Path, seed: int) -> tuple[bool, str]:
    required = [
        out / "run_summary.json",
        out / "config.json",
        out / "checkpoints/step_06000.pt",
        out / "validation_rows_step_06000.csv",
        out / "validation_summary_step_06000.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing_artifacts={missing}"
    if (out / "failure_context.json").exists():
        return False, "failure_context_present"

    config = json.loads((out / "config.json").read_text())
    summary = json.loads((out / "run_summary.json").read_text())
    expected = {
        "projection_rank": 4,
        "seed": seed,
        "max_optimizer_steps": 6000,
        "train_visits": 461,
        "train_ptids": 169,
        "val_visits": 247,
        "val_ptids": 85,
        "batch_size": 1,
        "amp_forward": True,
        "loss": "FP32 Dice+BCE",
        "data_pipeline": "BEST_BATCH1_PIPELINE",
        "data_source": "DISK_CACHE",
        "num_workers": 32,
        "prefetch_factor": 2,
        "pipeline_equivalence": "PIPELINE_EQUIVALENCE_PASS",
        "trainable_parameters": 10_748_384,
        "train_manifest_sha256": "0204f74426531a5c61705750eeb45c519fac237d0e7f4657220101f8c84c752a",
        "val_manifest_sha256": "ef27630cddb3546e9ec50fb131b81a2de0b6115700764ab3e853f9c295656dcf",
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        return False, f"config_mismatch={mismatch}"
    if summary.get("status") != "COMPLETE" or summary.get("steps") != 6000:
        return False, f"run_summary={summary.get('status')}/{summary.get('steps')}"

    rows = pd.read_csv(out / "validation_rows_step_06000.csv")
    required_columns = {"case_id", "ptid", "prompt", "lcc", "dice", "hd95_mm", "surface_dice_2mm"}
    missing_columns = sorted(required_columns.difference(rows.columns))
    if missing_columns:
        return False, f"validation_schema_missing={missing_columns}"
    main = rows.loc[rows.lcc == 0]
    if len(main) != 1976 or main.case_id.nunique() != 247 or main.ptid.nunique() != 85:
        return False, "validation_population_mismatch"
    metrics = main[["dice", "hd95_mm", "surface_dice_2mm"]].to_numpy(dtype=float)
    if not np.isfinite(metrics).all():
        return False, "nonfinite_or_missing_validation_metric"
    if main.duplicated(["case_id", "prompt", "lcc"]).any():
        return False, "duplicate_validation_rows"
    expected_prompts = {
        f"{side} {structure}"
        for side in ("left", "right")
        for structure in ("hippocampus", "entorhinal cortex", "parahippocampal gyrus", "amygdala")
    }
    if main[["case_id", "ptid", "prompt"]].isna().any().any():
        return False, "missing_validation_identity"
    if not main.groupby("case_id")["prompt"].agg(lambda values: set(values) == expected_prompts).all():
        return False, "validation_prompt_coverage_mismatch"
    return True, "ARTIFACT_GATE_PASS"


def run_one(seed: int) -> None:
    out = ROOT / f"seed{seed}_r4"
    if out.exists():
        ok, detail = artifact_gate(out, seed)
        if ok:
            print(json.dumps({"event": "REUSED_COMPLETED_RUN", "seed": seed, "artifact_gate": detail}), flush=True)
            return
        raise RuntimeError(f"existing output refuses overwrite: {out}; {detail}")

    log_path = ROOT / f"seed{seed}_r4.launcher.log"
    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--rank", "4", "--seed", str(seed),
        "--max-steps", "6000", "--output-dir", str(out),
    ]
    started = time.time()
    atomic_json(STATE, {
        "status": "RUNNING",
        "current_job": {"seed": seed, "rank": 4, "output_dir": str(out)},
        "jobs": [{"seed": s, "rank": r} for s, r in JOBS],
        "updated_unix": started,
    })
    with log_path.open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    ok, detail = artifact_gate(out, seed) if process.returncode == 0 else (False, "runner_exit_nonzero")
    atomic_json(STATE, {
        "status": "RUNNING" if ok else "BLOCKED_ARTIFACT_GATE",
        "last_job": {
            "seed": seed,
            "rank": 4,
            "output_dir": str(out),
            "returncode": process.returncode,
            "artifact_gate": detail,
            "wall_seconds": time.time() - started,
        },
        "updated_unix": time.time(),
    })
    if not ok:
        raise RuntimeError(f"artifact gate failed for seed={seed}, r4: {detail}")


def finalize() -> None:
    runs = [(20260809, REUSED), (3407, ROOT / "seed3407_r4"), (2026, ROOT / "seed2026_r4")]
    rows = []
    for seed, run_dir in runs:
        ok, detail = artifact_gate(run_dir, seed)
        if not ok:
            raise RuntimeError(f"final gate failed for seed={seed}: {detail}")
        table = pd.read_csv(run_dir / "validation_summary_step_06000.csv")
        main = table.loc[table.lcc.eq(0)]
        overall = main.loc[main.scope.eq("overall")].iloc[0]
        roi = {row.scope: row for _, row in main.loc[~main.scope.eq("overall")].iterrows()}
        rows.append({
            "seed": seed,
            "projection_rank": 4,
            "trainable_parameters": 10_748_384,
            "mean_dice": overall.dice,
            "hipp_dice": roi["hippocampus"].dice,
            "ec_dice": roi["entorhinal cortex"].dice,
            "phg_dice": roi["parahippocampal gyrus"].dice,
            "amy_dice": roi["amygdala"].dice,
            "hd95_mm": overall.hd95_mm,
            "surface_dice_2mm": overall.surface_dice_2mm,
            "fp_volume_ml": overall.false_positive_volume_ml,
            "components": overall.connected_components,
            "run_dir": str(run_dir),
            "artifact_gate": detail,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "RANK_R4_MULTI_SEED_RESULTS.csv", index=False)
    numeric = [
        "mean_dice", "hipp_dice", "ec_dice", "phg_dice", "amy_dice",
        "hd95_mm", "surface_dice_2mm", "fp_volume_ml", "components",
    ]
    summary = pd.DataFrame({
        "metric": numeric,
        "mean": [frame[col].mean() for col in numeric],
        "sample_sd": [frame[col].std(ddof=1) for col in numeric],
        "n_seeds": 3,
    })
    summary.to_csv(ROOT / "RANK_R4_MULTI_SEED_SUMMARY.csv", index=False)
    dice = summary.loc[summary.metric.eq("mean_dice")].iloc[0]
    report = f"""# Rank-4 multi-seed supplement at 50% data

Status: `RANK_R4_MULTI_SEED_COMPLETE`

- Seeds: 20260809 (reused), 3407, 2026.
- Frozen protocol: 169 PTIDs / 461 visits, 6000 optimizer steps, batch 1, AMP-forward with FP32 Dice+BCE, LoRA rank 4, projection rank 4, decoder stages trainable, BEST_BATCH1 DISK_CACHE pipeline, 32 spawn workers, prefetch 2, LCC=0.
- Validation completeness: 247/247 visits and 85/85 PTIDs for every seed; no duplicate ROI rows; all artifact gates passed.
- Trainable parameters: 10,748,384.
- Mean Dice across three seeds: {dice['mean']:.6f} ± {dice['sample_sd']:.6f} (sample SD).

The original seed-20260809 run was not repeated. The two added seeds complete the r4 replication set without changing the frozen rank-ablation protocol.
"""
    (ROOT / "RANK_R4_MULTI_SEED_SUPPLEMENT.md").write_text(report)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for seed, _ in JOBS:
        run_one(seed)
    finalize()
    atomic_json(STATE, {
        "status": "COMPLETE",
        "jobs": [{"seed": s, "rank": r} for s, r in JOBS],
        "completed_unix": time.time(),
    })


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        previous_state = json.loads(STATE.read_text()) if STATE.exists() else None
        atomic_json(STATE, {
            "status": "BLOCKED_SUPERVISOR_ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "previous_state": previous_state,
            "updated_unix": time.time(),
        })
        raise
