#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
repo="$PWD"
scratch="${STAIRS_SCRATCH:-/codex_download_here_work_here}"
results="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}"
export UV_CACHE_DIR="$scratch/uv-cache"
export UV_PYTHON_INSTALL_DIR="$scratch/uv-python"
export PATH="$scratch/bin:$PATH"
export TMPDIR="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

mode="${1:-}"
if [[ "$mode" == install ]]; then
  python="${STAIRS_PYTHON:-3.12}"
  if [[ -z "${STAIRS_PYTHON:-}" && -x /usr/bin/python3.12 ]]; then
    python=/usr/bin/python3.12
  fi
  uv venv --allow-existing --python "$python"
  requirements=experiments/requirements-kaggle.txt
  if [[ -f experiments/requirements-kaggle.lock.txt ]]; then
    requirements=experiments/requirements-kaggle.lock.txt
  fi
  uv pip install --torch-backend=cu128 -r "$requirements" -e .
  uv pip check
  exit
fi

if [[ "$mode" == evaluate ]]; then
  shift
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  exec .venv/bin/python -m scripts.evaluate_stairs "$@"
fi

if [[ "$mode" == render ]]; then
  shift
  export MUJOCO_GL="${STAIRS_RENDER_GL:-osmesa}"
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
  export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
  exec .venv/bin/python -m scripts.render_stairs "$@"
fi

if [[ "$mode" != train && "$mode" != smoke && "$mode" != benchmark ]]; then
  echo "Usage: bash scripts/kaggle_stairs.sh install | {smoke,benchmark,train} RUN_ID [flags] | {evaluate,render} [flags]" >&2
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
run_dir="$results/$run_id"
mkdir -p "$results"
mkdir "$run_dir"
export MJLAB_LOG_ROOT="$run_dir"
export OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS"
# These T4s communicate over PCIe. Disable P2P on Kaggle's virtual PCI bridge.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export WANDB_MODE=disabled

gpu_monitor=
finish() {
  local status=$?
  if [[ -n "$gpu_monitor" ]]; then
    kill "$gpu_monitor" 2>/dev/null || true
    wait "$gpu_monitor" 2>/dev/null || true
  fi
  printf '%s\n' "$status" > "$run_dir/exit-code.txt"
}
trap finish EXIT
cp "$profile" "$run_dir/profile.env"
git rev-parse HEAD > "$run_dir/git-commit.txt"
uv pip freeze --exclude-editable > "$run_dir/requirements.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run_dir/gpus.csv"
printf 'cpu_threads=%s\nnum_cpus=%s\nnccl_p2p_disable=%s\n' \
  "$CPU_THREADS" "$(nproc)" "$NCCL_P2P_DISABLE" > "$run_dir/runtime.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv,nounits --loop=2 > "$run_dir/gpu-samples.csv" &
gpu_monitor=$!

command=(
  .venv/bin/python -m scripts.train Unitree-G1-EDU-Stairs
  --gpu-ids "$GPU_IDS"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.seed "$SEED"
  --agent.max-iterations "$ITERATIONS"
  --agent.save-interval "$SAVE_INTERVAL"
  "$@"
)
printf '%q ' "${command[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
"${command[@]}" 2>&1 | tee "$run_dir/console.log"
