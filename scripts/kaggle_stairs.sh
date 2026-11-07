#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
repo="$PWD"
scratch="${STAIRS_SCRATCH:-/codex_download_here_work_here}"
results="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}"
export UV_CACHE_DIR="$scratch/uv-cache"
export TMPDIR="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

mode="${1:-}"
if [[ "$mode" == install ]]; then
  uv venv --allow-existing --python "${STAIRS_PYTHON:-/usr/bin/python3.12}"
  uv pip install --torch-backend=cu128 -r experiments/requirements-kaggle.txt -e .
  uv pip check
  exit
fi

if [[ "$mode" != train && "$mode" != smoke && "$mode" != benchmark ]]; then
  echo "Usage: bash scripts/kaggle_stairs.sh install | {smoke,benchmark,train} RUN_ID [training flags]" >&2
  exit 2
fi
run_id="${2:-}"
if [[ ! "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "Use a unique run ID made of letters, digits, hyphens or underscores." >&2
  exit 2
fi
shift 2

# Environment overrides let a benchmark vary one parameter at a time.
profile="${STAIRS_PROFILE:-experiments/profiles/kaggle-t4.env}"
declare -A overrides
for key in NUM_ENVS GPU_IDS CPU_THREADS ITERATIONS SAVE_INTERVAL SEED; do
  if [[ -v "$key" ]]; then overrides[$key]="${!key}"; fi
done
source "$profile"
case "$mode" in
  smoke) NUM_ENVS=32; ITERATIONS=2; SAVE_INTERVAL=1 ;;
  benchmark) ITERATIONS=12; SAVE_INTERVAL=100 ;;
esac
for key in "${!overrides[@]}"; do printf -v "$key" '%s' "${overrides[$key]}"; done
read -r -a gpu_ids <<< "$GPU_IDS"

run_dir="$results/$run_id"
mkdir -p "$results"
mkdir "$run_dir"
export MJLAB_LOG_ROOT="$run_dir"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
# These T4s communicate over PCIe. Disable P2P on Kaggle's virtual PCI bridge.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export WANDB_MODE=disabled

status=0
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code.txt"' EXIT
cp "$profile" "$run_dir/profile.env"
git rev-parse HEAD > "$run_dir/git-commit.txt"
uv pip freeze --exclude-editable > "$run_dir/requirements.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run_dir/gpus.csv"
printf 'cpu_threads=%s\nnum_cpus=%s\nnccl_p2p_disable=%s\n' \
  "$CPU_THREADS" "$(nproc)" "$NCCL_P2P_DISABLE" > "$run_dir/runtime.txt"

command=(
  .venv/bin/python -m scripts.train Unitree-G1-EDU-Stairs
  --gpu-ids "${gpu_ids[@]}"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.seed "$SEED"
  --agent.max-iterations "$ITERATIONS"
  --agent.save-interval "$SAVE_INTERVAL"
  "$@"
)
printf '%q ' "${command[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
"${command[@]}" 2>&1 | tee "$run_dir/console.log"
