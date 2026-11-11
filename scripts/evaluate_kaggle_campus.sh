#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_id="${1:?Supply the completed run ID}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
episodes="${2:-16}"
run="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}/$run_id"
test "$(cat "$run/exit-code.txt")" = 0
checkpoint=$(find "$run" -mindepth 3 -type f -name 'model_*.pt' -print | sort -V | tail -n 1)
test -n "$checkpoint"
policy="${checkpoint%/*}/policy.onnx"

run_suite() {
  local suite="$1" gpu="$2" status=0
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/kaggle_stairs.sh evaluate-campus --suite "$suite" \
    --checkpoint "$checkpoint" --onnx "$policy" --episodes-per-cell "$episodes" \
    --output "$run/eval-$suite.json" > "$run/eval-$suite.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$run/eval-$suite-exit-code.txt"
  return "$status"
}
run_suite clean 0 &
clean_pid=$!
run_suite stress 1 &
stress_pid=$!
status=0
wait "$clean_pid" || status=$?
wait "$stress_pid" || status=$?
exit "$status"
