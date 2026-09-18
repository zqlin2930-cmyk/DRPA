#!/usr/bin/env bash
: "${DRPA_PREPARED_WORKSPACE:?Use python -m drpa prepare/run to configure a workspace first}"
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <rank:4|8|16> <output-dir> [runtime-resume-checkpoint]" >&2
  exit 64
fi

rank="$1"
output_dir="$2"
resume_state="${3:-}"
project_root="${MTL_MODEL_ROOT:-__DRPA_WORKSPACE__}"
python_bin="__DRPA_PYTHON__"
runner="$project_root/scripts/train/rank_ablation_50_runner.py"

mkdir -p "$(dirname "$output_dir")"
status="${output_dir}.launcher_status.json"
log="${output_dir}.launcher.log"
heartbeat="${output_dir}.launcher_heartbeat.json"
child_pid=""
launcher_pid="$$"
launcher_parent_pid="$PPID"
launcher_session_id="$(ps -o sid= -p $$ | tr -d ' ')"
received_signal=""
started_at="$(date --iso-8601=seconds)"

write_status() {
  local state="$1"
  local exit_code="$2"
  local signal_name="$3"
  local tmp="${status}.tmp"
  printf '{\n  "state": "%s",\n  "rank": %s,\n  "launcher_pid": %s,\n  "launcher_parent_pid": %s,\n  "launcher_session_id": "%s",\n  "child_pid": %s,\n  "resume_state": "%s",\n  "started_at": "%s",\n  "updated_at": "%s",\n  "exit_code": %s,\n  "signal": "%s"\n}\n' \
    "$state" "$rank" "$launcher_pid" "$launcher_parent_pid" "$launcher_session_id" "${child_pid:-null}" "$resume_state" "$started_at" "$(date --iso-8601=seconds)" "$exit_code" "$signal_name" > "$tmp"
  mv "$tmp" "$status"
}

forward_signal() {
  local signal_name="$1"
  received_signal="$signal_name"
  write_status "SIGNAL_RECEIVED" null "$signal_name"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -s "$signal_name" "$child_pid" 2>/dev/null || true
  fi
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

write_status "STARTING" null ""
runner_args=(--rank "$rank" --max-steps 6000 --output-dir "$output_dir")
if [[ -n "$resume_state" ]]; then
  runner_args+=(--resume-state "$resume_state")
fi
"$python_bin" -u "$runner" "${runner_args[@]}" >> "$log" 2>&1 &
child_pid=$!
write_status "RUNNING" null ""

(
  while kill -0 "$child_pid" 2>/dev/null; do
    tmp="${heartbeat}.tmp"
    printf '{"timestamp":"%s","child_pid":%s,"alive":true}\n' "$(date --iso-8601=seconds)" "$child_pid" > "$tmp"
    mv "$tmp" "$heartbeat"
    sleep 30
  done
) &
heartbeat_pid=$!

set +e
wait "$child_pid"
exit_code=$?
set -e
kill "$heartbeat_pid" 2>/dev/null || true
wait "$heartbeat_pid" 2>/dev/null || true

if [[ "$exit_code" -eq 0 ]]; then
  state="COMPLETED"
elif [[ "$exit_code" -ge 128 ]]; then
  state="SIGNAL_EXIT"
else
  state="NONZERO_EXIT"
fi
write_status "$state" "$exit_code" "$received_signal"
exit "$exit_code"
