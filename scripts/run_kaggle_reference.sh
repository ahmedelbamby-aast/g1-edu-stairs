#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_id="${1:?Supply a unique run ID}"
shift
run="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}/$run_id"
STAIRS_PROFILE=experiments/profiles/kaggle-reference.env bash scripts/kaggle_stairs.sh train "$run_id" \
  --init-actor-onnx deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --agent.algorithm.learning-rate 0.0003 --agent.algorithm.schedule fixed \
  --agent.algorithm.entropy-coef 0.003 "$@"

checkpoint=$(find "$run" -name 'model_*.pt' -print | sort -V | tail -n 1)
test -n "$checkpoint"
status=0
bash scripts/kaggle_stairs.sh evaluate --checkpoint "$checkpoint" \
  --onnx "$(dirname "$checkpoint")/policy.onnx" \
  --output "$run/evaluation.json" > "$run/evaluation.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$run/evaluation-exit-code.txt"
exit "$status"
