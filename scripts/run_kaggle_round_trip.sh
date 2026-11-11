#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export STAIRS_TASK=Unitree-G1-EDU-Stair-Round-Trip
export STAIRS_PROFILE="${STAIRS_PROFILE:-experiments/profiles/kaggle-round-trip.env}"
bash scripts/run_kaggle_campus.sh "$@"
