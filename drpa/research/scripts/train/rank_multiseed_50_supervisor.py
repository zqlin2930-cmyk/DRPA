#!/usr/bin/env python3
"""Artifact-gated serial launcher for frozen r8/r16 50% multi-seed runs."""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("__DRPA_WORKSPACE__")
ROOT = BASE / "quality_audit/rank_ablation_50/multiseed_50"
RUNNER = BASE / "scripts/train/rank_ablation_50_runner.py"
PYTHON = Path("__DRPA_PYTHON__")
JOBS = [(3407, 8), (3407, 16), (2026, 8), (2026, 16)]


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def artifact_gate(out: Path, seed: int, rank: int) -> tuple[bool, str]:
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
    fields = {
        "projection_rank": rank, "seed": seed, "max_optimizer_steps": 6000,
        "train_visits": 461, "train_ptids": 169, "val_visits": 247,
        "val_ptids": 85, "batch_size": 1, "amp_forward": True,
        "loss": "FP32 Dice+BCE", "data_pipeline": "BEST_BATCH1_PIPELINE",
        "data_source": "DISK_CACHE", "num_workers": 32, "prefetch_factor": 2,
        "pipeline_equivalence": "PIPELINE_EQUIVALENCE_PASS",
    }
    mismatch = {key: (config.get(key), value) for key, value in fields.items() if config.get(key) != value}
    if mismatch:
        return False, f"config_mismatch={mismatch}"
    if summary.get("status") != "COMPLETE" or summary.get("steps") != 6000:
        return False, f"run_summary={summary.get('status')}/{summary.get('steps')}"
    rows = pd.read_csv(out / "validation_rows_step_06000.csv")
    main = rows.loc[rows.lcc == 0]
    if len(main) != 1976 or main.case_id.nunique() != 247 or main.ptid.nunique() != 85:
        return False, "validation_population_mismatch"
    if not np.isfinite(main[["dice", "hd95_mm", "surface_dice_2mm"]].to_numpy(dtype=float)).all():
        return False, "nonfinite_or_missing_validation_metric"
    return True, "ARTIFACT_GATE_PASS"


def run_one(seed: int, rank: int) -> None:
    out = ROOT / f"seed{seed}_r{rank}"
    if out.exists():
        ok, detail = artifact_gate(out, seed, rank)
        if ok:
            return
        raise RuntimeError(f"existing output refuses overwrite: {out}; {detail}")
    log_path = ROOT / f"seed{seed}_r{rank}.launcher.log"
    command = [str(PYTHON), "-u", str(RUNNER), "--rank", str(rank), "--seed", str(seed),
               "--max-steps", "6000", "--output-dir", str(out)]
    started = time.time()
    with log_path.open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    ok, detail = artifact_gate(out, seed, rank) if process.returncode == 0 else (False, "runner_exit_nonzero")
    position = JOBS.index((seed, rank))
    atomic_json(ROOT / "supervisor_state.json", {
        "last_job": {"seed": seed, "rank": rank, "output_dir": str(out), "returncode": process.returncode,
                     "artifact_gate": detail, "wall_seconds": time.time() - started},
        "status": "RUNNING" if ok else "BLOCKED_ARTIFACT_GATE",
        "next_jobs": [{"seed": s, "rank": r} for s, r in JOBS[position + 1:]],
    })
    if not ok:
        raise RuntimeError(f"artifact gate failed for seed={seed}, r{rank}: {detail}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "supervisor_state.json", {"status": "RUNNING", "jobs": JOBS, "started_unix": time.time()})
    for seed, rank in JOBS:
        run_one(seed, rank)
    atomic_json(ROOT / "supervisor_state.json", {"status": "COMPLETE", "jobs": JOBS, "completed_unix": time.time()})


if __name__ == "__main__":
    main()
