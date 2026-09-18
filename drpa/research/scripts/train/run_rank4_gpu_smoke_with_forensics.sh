#!/usr/bin/env bash
: "${DRPA_PREPARED_WORKSPACE:?Use python -m drpa prepare/run to configure a workspace first}"
set -uo pipefail

output_dir="${1:?output directory required}"
project_root="${MTL_MODEL_ROOT:-__DRPA_WORKSPACE__}"
python_bin="__DRPA_PYTHON__"
script="$project_root/scripts/analysis/rank4_spawn_gpu_smoke.py"
status="${output_dir}.launcher_status.json"
log="${output_dir}.launcher.log"
mkdir -p "$(dirname "$output_dir")"

write_status() {
  local state="$1" rc="$2" sig="${3:-}"
  local tmp="${status}.tmp"
  printf '{"state":"%s","child_pid":%s,"updated_at":"%s","exit_code":%s,"signal":"%s"}\n' \
    "$state" "${child_pid:-null}" "$(date --iso-8601=seconds)" "$rc" "$sig" > "$tmp"
  mv "$tmp" "$status"
}

on_signal() {
  local sig="$1"
  write_status "SIGNAL_RECEIVED" null "$sig"
  [[ -n "${child_pid:-}" ]] && kill -s "$sig" "$child_pid" 2>/dev/null || true
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

write_status STARTING null
RANK4_SMOKE_OUTPUT="$output_dir" RANK4_SMOKE_STEPS=20 \
  "$python_bin" -u "$script" > "$log" 2>&1 &
child_pid=$!
write_status RUNNING null
set +e
wait "$child_pid"
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  write_status COMPLETED "$rc"
elif [[ "$rc" -ge 128 ]]; then
  write_status SIGNAL_EXIT "$rc"
else
  write_status NONZERO_EXIT "$rc"
fi
exit "$rc"
