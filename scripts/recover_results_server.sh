#!/usr/bin/env bash
# From a cloned checkout. GH_TOKEN, gh login, or the private token file is required.
# No training command appears in this recovery path.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/bootstrap_kaggle.sh
export MPLBACKEND=Agg MPLCONFIGDIR=/codex_download_here_work_here/cache/matplotlib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m scripts.restore_github_results
.venv/bin/python -m scripts.server_results_service start
.venv/bin/python -m scripts.server_results_service status
