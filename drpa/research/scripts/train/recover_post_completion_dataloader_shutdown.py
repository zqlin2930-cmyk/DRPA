#!/usr/bin/env python3
"""Recover an otherwise complete rank run from a post-artifact worker shutdown.

This script never runs model inference or training.  It verifies that the
complete 6000-step and 247-visit artifact contract exists, then archives only
the teardown-only failure context so the artifact-gated supervisor can proceed.
"""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--rank", required=True, type=int)
    args = parser.parse_args()
    run = Path(args.run_dir)
    step = 6000
    required = [
        run / "config.json",
        run / "run_summary.json",
        run / "training_dynamics.csv",
        run / "checkpoints/step_06000.pt",
        run / f"validation_rows_step_{step:05d}.csv",
        run / f"validation_summary_step_{step:05d}.csv",
        run / "failure_context.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"recovery blocked: missing artifacts: {missing}")

    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "run_summary.json").read_text())
    failure = json.loads((run / "failure_context.json").read_text())
    training = pd.read_csv(run / "training_dynamics.csv")
    raw = pd.read_csv(run / f"validation_rows_step_{step:05d}.csv")
    checkpoint = torch.load(run / "checkpoints/step_06000.pt", map_location="cpu", weights_only=False)

    if config.get("projection_rank") != args.rank or config.get("max_optimizer_steps") != step:
        raise RuntimeError("recovery blocked: config identity mismatch")
    if summary.get("status") != "COMPLETE" or summary.get("steps") != step:
        raise RuntimeError("recovery blocked: run summary is not complete")
    if checkpoint.get("global_step") != step or checkpoint.get("projection_rank") != args.rank:
        raise RuntimeError("recovery blocked: checkpoint identity mismatch")
    if len(training) != step or not np.isfinite(training.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("recovery blocked: training dynamics incomplete or non-finite")
    if raw.groupby("lcc").size().to_dict() != {0: 1976, 1: 1976}:
        raise RuntimeError("recovery blocked: validation row contract mismatch")

    failure_text = (failure.get("error", "") + "\n" + failure.get("traceback", "")).lower()
    if failure.get("last_completed_step") != step or "close_loader" not in failure_text:
        raise RuntimeError("recovery blocked: failure is not a post-completion DataLoader shutdown")
    if any(token in failure_text for token in ("non-finite", "nan", "overflow", "out of memory")):
        raise RuntimeError("recovery blocked: numerical/resource failure cannot be reclassified")

    archive = run / "post_completion_dataloader_shutdown_failure_context.json"
    if archive.exists():
        raise RuntimeError(f"recovery blocked: archive already exists: {archive}")
    os.replace(run / "failure_context.json", archive)
    recovery = {
        "status": "RANK_POST_COMPLETION_DATALOADER_SHUTDOWN_RECOVERED",
        "rank": args.rank,
        "step": step,
        "validation_rows": int(len(raw)),
        "validation_lcc_counts": {str(k): int(v) for k, v in raw.groupby("lcc").size().items()},
        "checkpoint": str(run / "checkpoints/step_06000.pt"),
        "archived_failure_context": str(archive),
        "reason": "training, validation, summary, and checkpoint were complete before worker shutdown",
        "timestamp_unix": time.time(),
    }
    atomic_json(run / "post_completion_recovery.json", recovery)
    print(json.dumps(recovery, indent=2))


if __name__ == "__main__":
    main()
