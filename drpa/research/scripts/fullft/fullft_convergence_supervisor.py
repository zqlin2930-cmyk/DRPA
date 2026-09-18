#!/usr/bin/env python3
"""One-shot supervisor for the pre-authorized 9000 -> 12000 extension."""

from __future__ import annotations

# Distribution guard: execute prepared copies via python -m drpa.
if __name__ == "__main__" and not __import__("os").environ.get("DRPA_PREPARED_WORKSPACE"):
    raise SystemExit("Use python -m drpa prepare/run; archived scripts are not direct launchers.")


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--next-dir", type=Path, required=True)
    parser.add_argument("--extension-script", type=Path, required=True)
    args = parser.parse_args()
    extension_dir = args.extension_dir.resolve()
    next_dir = args.next_dir.resolve()
    state_path = next_dir.parent / "fullft_12000_supervisor_state.json"
    launch_log = next_dir.parent / "fullft_12000_extension_launch.log"
    next_dir.mkdir(parents=True, exist_ok=True)

    while Path(f"/proc/{args.pid}").exists():
        time.sleep(30)

    failure_path = extension_dir / "failure_context.json"
    summary_path = extension_dir / "run_summary.json"
    metrics_path = extension_dir / "validation_metrics.csv"
    checkpoint_path = extension_dir / "checkpoints/voxtell_fullft_100pct_step09000.pt"
    required = [summary_path, metrics_path, checkpoint_path]
    if failure_path.exists() or not all(path.exists() for path in required):
        atomic_json(state_path, {
            "status": "FULLFT_9000_EXTENSION_FAILED_NOT_STARTING_12000",
            "extension_dir": str(extension_dir),
            "failure_path": str(failure_path) if failure_path.exists() else None,
            "missing": [str(path) for path in required if not path.exists()],
            "updated_unix": time.time(),
        })
        return 1

    next_checkpoint = extension_dir / "checkpoints/voxtell_fullft_100pct_step09000.pt"
    command = [
        sys.executable,
        str(args.extension_script.resolve()),
        "--base", str(args.base.resolve()),
        "--input-checkpoint", str(next_checkpoint),
        "--output-dir", str(next_dir),
        "--start-step", "9000",
        "--target-step", "12000",
        "--baseline-metrics", str(metrics_path),
    ]
    with launch_log.open("a") as stream:
        child = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    atomic_json(state_path, {
        "status": "FULLFT_9000_COMPLETE__FULLFT_12000_RUNNING",
        "source_checkpoint": str(next_checkpoint),
        "target_dir": str(next_dir),
        "pid": child.pid,
        "command": command,
        "updated_unix": time.time(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
