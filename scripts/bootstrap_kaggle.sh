#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
scratch="${STAIRS_SCRATCH:-/codex_download_here_work_here}"
results="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}"
mkdir -p "$scratch" "$results" "$scratch/bin"
export PATH="$scratch/bin:$PATH"

packages=()
command -v curl >/dev/null || packages+=(curl ca-certificates)
command -v ffmpeg >/dev/null || packages+=(ffmpeg)
if ! ldconfig -p | awk '/libOSMesa\.so/ { found=1 } END { exit !found }'; then
  packages+=(libosmesa6)
fi
if (( ${#packages[@]} )); then
  apt-get update
  apt-get install -y --no-install-recommends "${packages[@]}"
fi

if ! command -v uv >/dev/null; then
  curl --fail --silent --show-error --location https://astral.sh/uv/install.sh \
    --output "$scratch/uv-install.sh"
  UV_UNMANAGED_INSTALL="$scratch/bin" sh "$scratch/uv-install.sh"
fi

bash scripts/kaggle_stairs.sh install
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python - <<'PY'
import torch
import mujoco
import mujoco_warp
import mjlab

devices = torch.cuda.device_count()
if not devices:
    raise SystemExit("PyTorch cannot see a CUDA GPU. Select a GPU accelerator in Kaggle.")
for index in range(devices):
    torch.zeros(1, device=f"cuda:{index}")
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
with mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 64, 64):
    print("Headless rendering is available.")
PY
