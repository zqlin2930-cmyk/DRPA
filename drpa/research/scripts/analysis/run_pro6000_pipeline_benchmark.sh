#!/usr/bin/env bash
: "${DRPA_PREPARED_WORKSPACE:?Use python -m drpa prepare/run to configure a workspace first}"
set -u

audit_dir=__DRPA_WORKSPACE__/quality_audit/pro6000_pipeline_tuning
python_bin=__DRPA_PYTHON__
benchmark_script=__DRPA_WORKSPACE__/scripts/analysis/pro6000_pipeline_benchmark.py

record_signal() {
  signal_name=$1
  python_status=$2
  printf '%s\n' "$signal_name" > "$audit_dir/pipeline_benchmark_retry4.signal"
  printf '%s\n' "$python_status" > "$audit_dir/pipeline_benchmark_retry4.exit_code"
  exit "$python_status"
}

trap 'record_signal SIGTERM 143' TERM
trap 'record_signal SIGINT 130' INT
trap 'record_signal SIGHUP 129' HUP

"$python_bin" "$benchmark_script" --run
python_status=$?
printf '%s\n' "$python_status" > "$audit_dir/pipeline_benchmark_retry4.exit_code"
exit "$python_status"
