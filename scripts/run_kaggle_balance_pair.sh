#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
prefix="${1:?Supply a unique balance run prefix}"
checkpoint="${2:?Supply the mixed-skill checkpoint to resume}"
[[ "$prefix" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
results="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}"

train_leg() {
  local leg="$1" gpu="$2" seed="$3"
  local run_id="$prefix-$leg" run="$results/$prefix-$leg" final
  CUDA_VISIBLE_DEVICES="$gpu" GPU_IDS='[0]' SEED="$seed" NUM_ENVS="${NUM_ENVS:-1200}" \
    ITERATIONS="${ITERATIONS:-100}" STAIRS_TASK="Unitree-G1-EDU-Balance-${leg^}" \
    bash scripts/run_kaggle_campus.sh "$run_id" --resume-checkpoint "$checkpoint"
  final=$(find "$run" -mindepth 3 -type f -name 'model_*.pt' -print | sort -V | tail -n 1)
  for suite in clean stress; do
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/kaggle_stairs.sh evaluate-campus \
      --checkpoint "$final" --onnx "${final%/*}/policy.onnx" --drill "$leg" \
      --suite "$suite" --episodes-per-cell 24 --seed 913 --output "$run/eval-$suite.json" \
      > "$run/eval-$suite.log" 2>&1
  done
}
train_leg left 0 42 &
left_pid=$!
train_leg right 1 43 &
right_pid=$!
status=0
wait "$left_pid" || status=$?
wait "$right_pid" || status=$?
exit "$status"
