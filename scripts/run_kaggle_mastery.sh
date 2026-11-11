#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_id="${1:?Supply a unique run ID}"
shift
export STAIRS_TASK="${STAIRS_TASK:-Unitree-G1-EDU-Stair-Mastery}"
export STAIRS_PROFILE="${STAIRS_PROFILE:-experiments/profiles/kaggle-mastery.env}"
bash scripts/run_kaggle_campus.sh "$run_id" "$@"
