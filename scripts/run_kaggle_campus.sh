#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_id="${1:?Supply a unique run ID}"
shift
export STAIRS_TASK="${STAIRS_TASK:-Unitree-G1-EDU-Campus}"
export STAIRS_PROFILE="${STAIRS_PROFILE:-experiments/profiles/kaggle-campus.env}"
bash scripts/kaggle_stairs.sh train "$run_id" --randomize-initial-episode-length False "$@"
