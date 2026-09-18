#!/usr/bin/env python3
"""Serial gate: finish FullFT 10% validation, then launch FullFT 100%.

This supervisor never changes scientific settings and never selects by
validation performance. It only checks completion/integrity and starts the
already-frozen 100% protocol. Any failed gate is terminal and is not retried.
"""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch


POLL_SECONDS = 60


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_metrics(path: Path) -> tuple[bool, str]:
    try:
        rows = pd.read_csv(path / "validation_rows_step_06000.csv")
        summary = pd.read_csv(path / "validation_summary_step_06000.csv")
        # The frozen evaluator emits two rows per visit-ROI (raw and LCC);
        # completeness is therefore checked as 247 unique visits and eight
        # structures, while accepting the evaluator's 2x diagnostic rows.
        if rows.case_id.nunique() != 247:
            return False, f"unique validation visits={rows.case_id.nunique()}, expected 247"
        if rows.structure.nunique() != 4 or rows.side.nunique() != 2 or len(rows) != 247 * 8 * 2:
            return False, f"validation rows={len(rows)}, expected canonical 3952 (247x8x2)"
        if len(summary) != 10:
            return False, f"validation summary rows={len(summary)}, expected 10"
        numeric = rows.select_dtypes(include="number")
        if numeric.replace([float("inf"), float("-inf")], float("nan")).isna().any().any():
            return False, "non-finite case-level metric"
        return True, "247/247 validation rows and 10 summary rows present"
    except Exception as exc:
        return False, repr(exc)


def check_10pct(root: Path) -> tuple[str, str]:
    path = root / "quality_audit/voxtell_fullft_baseline/formal_10pct"
    pid_file = path / "fullft_10pct.pid"
    pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
    if pid is not None and alive(pid):
        return "WAITING", f"10% PID {pid} still running"
    progress = json.loads((path / "progress.json").read_text()) if (path / "progress.json").exists() else {}
    if (path / "failure_context.json").exists() or progress.get("status") in {"FULLFT_10P_RUNTIME_FAILURE", "FULLFT_10P_NUMERICAL_FAILURE"}:
        return "FAILED", "10% failure_context or failure status exists"
    required = [
        path / "run_summary.json",
        path / "validation_metrics.csv",
        path / "validation_rows_step_06000.csv",
        path / "validation_summary_step_06000.csv",
        path / "checkpoints/voxtell_fullft_10pct_step06000.pt",
        path / "checkpoint_manifest.csv",
    ]
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        return "FAILED", "10% process ended before required artifacts: " + "; ".join(missing)
    ok, detail = check_metrics(path)
    if not ok:
        return "FAILED", detail
    try:
        checkpoint = torch.load(required[4], map_location="cpu", weights_only=False)
        if int(checkpoint["global_step"]) != 6000:
            return "FAILED", "step6000 checkpoint global_step mismatch"
        del checkpoint
    except Exception as exc:
        return "FAILED", "step6000 checkpoint unreadable: " + repr(exc)
    return "PASS", detail


def launch_100pct(root: Path) -> dict:
    path = root / "quality_audit/voxtell_fullft_baseline/formal_100pct"
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing 100% output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    log = path / "training.log"
    command = [
        sys.executable,
        str(root / "scripts/fullft/train_fullft_10pct_formal.py"),
        "--base", str(root),
        "--subset", "100pct",
        "--output-dir", str(path),
    ]
    handle = log.open("w")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    (path / "fullft_100pct.pid").write_text(str(process.pid) + "\n")
    handle.close()
    return {"pid": process.pid, "command": command, "started_unix": time.time()}


def monitor_100pct(root: Path, state_path: Path) -> None:
    path = root / "quality_audit/voxtell_fullft_baseline/formal_100pct"
    while True:
        pid_file = path / "fullft_100pct.pid"
        pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
        progress = json.loads((path / "progress.json").read_text()) if (path / "progress.json").exists() else {}
        if (path / "run_summary.json").exists() and progress.get("status") == "FULLFT_100P_COMPLETE":
            atomic_json(state_path, {"status": "FULLFT_100P_COMPLETE", "updated_unix": time.time(), "progress": progress})
            return
        if (path / "failure_context.json").exists() or (pid is not None and not alive(pid)):
            atomic_json(state_path, {"status": "FULLFT_100P_RUNTIME_FAILURE", "updated_unix": time.time(), "progress": progress})
            return
        atomic_json(state_path, {"status": "FULLFT_10P_COMPLETE__FULLFT_100P_RUNNING", "pid": pid, "updated_unix": time.time(), "progress": progress})
        time.sleep(POLL_SECONDS)


def main() -> None:
    root = Path(os.environ.get("MTL_MODEL_ROOT", "__DRPA_WORKSPACE__")).resolve()
    state_path = root / "quality_audit/voxtell_fullft_baseline/fullft_queue_state.json"
    while True:
        status, detail = check_10pct(root)
        atomic_json(state_path, {"status": "FULLFT_10P_WAITING" if status == "WAITING" else ("FULLFT_10P_VALIDATION_FAILED" if status == "FAILED" else "FULLFT_10P_GATE_PASS"), "detail": detail, "updated_unix": time.time()})
        if status == "FAILED":
            return
        if status == "PASS":
            launched = launch_100pct(root)
            atomic_json(state_path, {"status": "FULLFT_10P_COMPLETE__FULLFT_100P_RUNNING", "launch": launched, "updated_unix": time.time()})
            monitor_100pct(root, state_path)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
