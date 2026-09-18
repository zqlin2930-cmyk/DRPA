#!/usr/bin/env bash
: "${DRPA_PREPARED_WORKSPACE:?Use python -m drpa prepare/run to configure a workspace first}"
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <rank:4|16> <output-dir> [runtime-resume-checkpoint]" >&2
  exit 64
fi

rank="$1"
output_dir="$2"
resume_state="${3:-}"
project_root="${MTL_MODEL_ROOT:-__DRPA_WORKSPACE__}"
launcher="$project_root/scripts/train/run_rank_ablation_with_forensics.sh"
detached_log="${output_dir}.detached_launcher.log"
pid_file="${output_dir}.detached_launcher.pid"

if [[ -e "$output_dir" && -z "$resume_state" ]]; then
  echo "refusing fresh run because output exists: $output_dir" >&2
  exit 73
fi
if [[ -n "$resume_state" && ! -s "$resume_state" ]]; then
  echo "runtime-resume checkpoint missing or empty: $resume_state" >&2
  exit 66
fi

args=("$rank" "$output_dir")
if [[ -n "$resume_state" ]]; then
  args+=("$resume_state")
fi
nohup setsid "$launcher" "${args[@]}" </dev/null >>"$detached_log" 2>&1 &
detached_pid=$!
printf '%s\n' "$detached_pid" > "$pid_file"
printf '{"state":"DETACHED_LAUNCH_SUBMITTED","rank":%s,"detached_pid":%s,"caller_pid":%s,"caller_parent_pid":%s,"timestamp":"%s"}\n' \
  "$rank" "$detached_pid" "$$" "$PPID" "$(date --iso-8601=seconds)" > "${output_dir}.detached_submission.json"
echo "$detached_pid"
