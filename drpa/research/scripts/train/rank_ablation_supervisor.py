#!/usr/bin/env python3
"""Artifact-gated rank-4 -> rank-16 continuation for the frozen 50% study."""
from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

BASE = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__"))
ROOT = BASE / "quality_audit/rank_ablation_50"
RUNNER = BASE / "scripts/train/rank_ablation_50_runner.py"
PYTHON = Path("__DRPA_PYTHON__")
R4 = ROOT / "rank4_hardened_batch1_step0"
R16 = ROOT / "rank16_hardened_batch1_step0"
LAUNCHER = BASE / "scripts/train/launch_rank_ablation_detached.sh"
STATE = ROOT / "RANK_ABLATION_SUPERVISOR_STATE.json"
FINALIZER = BASE / "scripts/analysis/rank_ablation_finalize.py"
RESUME_HISTORY = ROOT / "RANK_RUNTIME_RESUME_HISTORY.jsonl"


def gpu_idle() -> bool:
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False).stdout.strip()
    return not output


def rank_complete(run: Path, rank: int) -> tuple[bool, str]:
    required = [run / "config.json", run / "run_summary.json", run / "checkpoints/step_06000.pt",
                run / "validation_rows_step_06000.csv", run / "validation_summary_step_06000.csv",
                run / "training_dynamics.csv"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False, f"rank{rank} artifact gate incomplete"
    try:
        config = json.loads((run / "config.json").read_text())
        summary = json.loads((run / "run_summary.json").read_text())
        raw = pd.read_csv(run / "validation_rows_step_06000.csv")
        training = pd.read_csv(run / "training_dynamics.csv")
        checkpoint = torch.load(run / "checkpoints/step_06000.pt", map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"rank{rank} artifact unreadable: {exc!r}"
    if (config.get("projection_rank") != rank or config.get("max_optimizer_steps") != 6000
            or config.get("data_pipeline") != "BEST_BATCH1_PIPELINE"
            or config.get("data_source") != "DISK_CACHE"
            or config.get("num_workers") != 32
            or config.get("prefetch_factor") != 2
            or config.get("multiprocessing_context") != "spawn"
            or config.get("pipeline_equivalence") != "PIPELINE_EQUIVALENCE_PASS"):
        return False, f"rank{rank} protocol mismatch"
    if summary.get("status") != "COMPLETE" or summary.get("steps") != 6000:
        return False, f"rank{rank} summary not complete"
    if checkpoint.get("global_step") != 6000 or checkpoint.get("projection_rank") != rank:
        return False, f"rank{rank} checkpoint identity mismatch"
    expected_params = {4: 10_748_384, 16: 11_412_320}[rank]
    observed_params = sum(t.numel() for t in checkpoint.get("trainable_model_state", {}).values())
    if observed_params != expected_params:
        return False, f"rank{rank} checkpoint parameter mismatch {observed_params} != {expected_params}"
    if raw.groupby("lcc").size().to_dict() != {0: 1976, 1: 1976}:
        return False, f"rank{rank} validation row contract mismatch"
    numeric = training.select_dtypes(include=[np.number])
    if len(training) != 6000 or not np.isfinite(numeric.to_numpy()).all():
        return False, f"rank{rank} training dynamics incomplete/non-finite"
    if (run / "failure_context.json").exists():
        return False, f"rank{rank} unresolved failure_context exists"
    return True, f"rank{rank} artifact gate pass"


def launcher_active(run: Path) -> bool:
    path = Path(f"{run}.launcher_status.json")
    if not path.is_file():
        return False
    try:
        status = json.loads(path.read_text())
        child = int(status.get("child_pid"))
    except Exception:
        return False
    return status.get("state") == "RUNNING" and Path(f"/proc/{child}").exists()


def finite_tree(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def resume_state_valid(run: Path, rank: int) -> tuple[bool, str, Path | None]:
    path = run / "checkpoints/latest_resume.pt"
    if not path.is_file() or path.stat().st_size == 0:
        return False, "latest_resume missing", None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"latest_resume unreadable: {exc!r}", path
    step = int(payload.get("global_step", -1))
    if (payload.get("format") != "drpa_rank_ablation_runtime_resume_v2"
            or payload.get("purpose") != "RUNTIME_RECOVERY_ONLY_NOT_MODEL_SELECTION"
            or int(payload.get("projection_rank", -1)) != rank
            or not (0 < step < 6000)):
        return False, "latest_resume identity/step mismatch", path
    if not finite_tree(payload.get("trainable_model_state", {})):
        return False, "latest_resume has non-finite model state", path
    if not finite_tree(payload.get("optimizer_state", {})):
        return False, "latest_resume has non-finite optimizer state", path
    return True, f"latest_resume PASS at step {step}", path


def numerical_failure(run: Path) -> bool:
    path = run / "failure_context.json"
    if not path.is_file():
        return False
    text = path.read_text(errors="replace").lower()
    return any(token in text for token in ("non-finite", "nan", "inf", "overflow", "outofmemory", "out of memory"))


def append_resume_event(payload: dict) -> None:
    with RESUME_HISTORY.open("a") as handle:
        handle.write(json.dumps({"timestamp_unix": time.time(), **payload}) + "\n")


def launch(rank: int, run: Path, resume: Path | None = None) -> int:
    command = [str(LAUNCHER), str(rank), str(run)]
    if resume is not None:
        command.append(str(resume))
    submission = subprocess.run(command, capture_output=True, text=True, check=True)
    return int(submission.stdout.strip().splitlines()[-1])


def maybe_resume(run: Path, rank: int) -> tuple[bool, str]:
    if launcher_active(run):
        return False, f"rank{rank} running"
    if numerical_failure(run):
        return False, f"rank{rank} numerical failure; automatic resume prohibited"
    valid, reason, resume = resume_state_valid(run, rank)
    if not valid:
        return False, f"rank{rank} stopped; {reason}"
    failure = run / "failure_context.json"
    if failure.exists():
        archived = run / f"failure_context.pre_resume.{int(time.time())}.json"
        os.replace(failure, archived)
    pid = launch(rank, run, resume)
    append_resume_event({"rank": rank, "run": str(run), "resume": str(resume),
                         "reason": reason, "detached_pid": pid})
    return True, f"rank{rank} exact runtime resume started at PID {pid}; {reason}"


def update(**fields) -> None:
    payload = {"updated_at_unix": time.time(), **fields}
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, STATE)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        ok, reason = rank_complete(R4, 4)
        if not ok:
            resumed, resume_reason = maybe_resume(R4, 4)
            update(status="RANK4_RUNTIME_RESUMED" if resumed else "WAITING_FOR_RANK4_ARTIFACT_GATE",
                   reason=resume_reason)
            time.sleep(60)
            continue
        r16_ok, r16_reason = rank_complete(R16, 16)
        if r16_ok:
            final_status = ROOT / "RANK_ABLATION_50_FINAL_STATUS.json"
            if not final_status.is_file():
                subprocess.run([str(PYTHON), str(FINALIZER)], check=True)
            update(status="RANK_ABLATION_50_COMPLETE", reason=r16_reason,
                   final_report=str(ROOT / "RANK_ABLATION_50_FINAL_REPORT.md"))
            return
        if R16.exists():
            resumed, resume_reason = maybe_resume(R16, 16)
            update(status="RANK16_RUNTIME_RESUMED" if resumed else "WAITING_FOR_RANK16_ARTIFACT_GATE",
                   reason=resume_reason)
            time.sleep(60)
            continue
        if not gpu_idle():
            update(status="WAITING_FOR_IDLE_GPU", reason=reason)
            time.sleep(60)
            continue
        pid = launch(16, R16)
        update(status="RANK16_STARTED", reason=reason, pid=pid,
               launcher="detached_nohup_setsid")
        time.sleep(60)


if __name__ == "__main__":
    main()
