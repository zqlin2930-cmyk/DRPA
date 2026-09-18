#!/usr/bin/env python3
"""Artifact-gated serial supervisor for the matched 50% Placement multi-seed study."""
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
ROOT = BASE / "quality_audit/placement_multiseed_50"
RUNNER = BASE / "scripts/train/placement_multiseed_50_runner.py"
PYTHON = Path("__DRPA_PYTHON__")
SEEDS = (20260809, 3407, 2026)
CONDITIONS = ("b1", "b1_projection", "b1_decoder", "drpa")
EXPECTED = {"b1": 294_912, "b1_projection": 737_536, "b1_decoder": 10_527_072, "drpa": 10_969_696}
JOBS = [(seed, condition) for seed in SEEDS for condition in CONDITIONS]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def artifact_gate(out: Path, seed: int, condition: str) -> tuple[bool, str]:
    required = [out / "run_summary.json", out / "config.json", out / "checkpoints/step_06000.pt",
                out / "validation_rows_step_06000.csv", out / "validation_summary_step_06000.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing_artifacts={missing}"
    if (out / "failure_context.json").exists():
        return False, "failure_context_present"
    config, summary = json.loads((out / "config.json").read_text()), json.loads((out / "run_summary.json").read_text())
    fields = {"experiment": "PLACEMENT_MULTI_SEED_50_AMP_MATCHED", "condition": condition, "seed": seed,
              "max_optimizer_steps": 6000, "train_visits": 461, "train_ptids": 169, "val_visits": 247,
              "val_ptids": 85, "batch_size": 1, "amp_forward": True, "loss": "FP32 Dice+BCE", "lcc_main": 0,
              "data_pipeline": "BEST_BATCH1_PIPELINE", "data_source": "DISK_CACHE", "num_workers": 32,
              "prefetch_factor": 2, "pipeline_equivalence": "PIPELINE_EQUIVALENCE_PASS",
              "trainable_parameters": EXPECTED[condition]}
    mismatch = {key: (config.get(key), value) for key, value in fields.items() if config.get(key) != value}
    if mismatch:
        return False, f"config_mismatch={mismatch}"
    if summary.get("status") != "COMPLETE" or summary.get("steps") != 6000:
        return False, f"summary_status={summary.get('status')}/{summary.get('steps')}"
    rows = pd.read_csv(out / "validation_rows_step_06000.csv")
    rows = rows[rows.lcc.eq(0)]
    if len(rows) != 1976 or rows.case_id.nunique() != 247 or rows.ptid.nunique() != 85:
        return False, "validation_population_mismatch"
    if not np.isfinite(rows[["dice", "hd95_mm", "surface_dice_2mm"]].to_numpy(float)).all():
        return False, "nonfinite_validation_metric"
    return True, "ARTIFACT_GATE_PASS"


def run_one(seed: int, condition: str) -> None:
    out = ROOT / f"seed{seed}_{condition}"
    if out.exists():
        ok, detail = artifact_gate(out, seed, condition)
        if ok:
            return
        raise RuntimeError(f"refusing to overwrite {out}: {detail}")
    log = ROOT / f"seed{seed}_{condition}.launcher.log"
    command = [str(PYTHON), "-u", str(RUNNER), "--condition", condition, "--seed", str(seed),
               "--max-steps", "6000", "--output-dir", str(out)]
    started = time.time()
    with log.open("w") as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    ok, detail = artifact_gate(out, seed, condition) if process.returncode == 0 else (False, "runner_exit_nonzero")
    position = JOBS.index((seed, condition))
    atomic_json(ROOT / "supervisor_state.json", {
        "status": "RUNNING" if ok else "BLOCKED_ARTIFACT_GATE",
        "last_job": {"seed": seed, "condition": condition, "returncode": process.returncode,
                     "artifact_gate": detail, "wall_seconds": time.time() - started, "output_dir": str(out)},
        "next_jobs": [{"seed": other_seed, "condition": other_condition} for other_seed, other_condition in JOBS[position + 1:]],
    })
    if not ok:
        raise RuntimeError(f"artifact gate failed for seed={seed}, condition={condition}: {detail}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "supervisor_state.json", {"status": "RUNNING", "jobs": [{"seed": seed, "condition": condition} for seed, condition in JOBS], "started_unix": time.time()})
    for seed, condition in JOBS:
        run_one(seed, condition)
    atomic_json(ROOT / "supervisor_state.json", {"status": "COMPLETE", "jobs": [{"seed": seed, "condition": condition} for seed, condition in JOBS], "completed_unix": time.time()})


if __name__ == "__main__":
    main()
