"""Readiness checks only: construct the full training world and infer; never learn."""

import argparse
import hashlib
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks
from mjlab.utils.os import dump_yaml

import src.tasks
from scripts.training_metrics import atomic_json
from src.tasks.stair_mastery import mastery_env_cfg, mastery_runner_cfg
from src.tasks.stair_mastery.runner import MasteryRunner


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--teacher-onnx", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=1080)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  if args.num_envs < 1 or not args.teacher_onnx.is_file():
    parser.error("A real source policy and a positive environment count are required")
  import matplotlib
  import tensorboard
  if torch.cuda.device_count() < 2:
    raise RuntimeError("The prepared profile needs two CUDA GPUs")
  free_disk = shutil.disk_usage(args.output).free
  if free_disk < 10 * 1024**3:
    raise RuntimeError("Keep at least 10 GiB of results disk space free")
  cfg = mastery_env_cfg(student=True)
  cfg.scene.num_envs = args.num_envs
  agent = mastery_runner_cfg()
  dump_yaml(args.output / "env.yaml", asdict(cfg))
  dump_yaml(args.output / "agent.yaml", asdict(agent))
  torch.cuda.set_device(args.device)
  torch.empty(1, device=args.device)  # Initialize this device before allocator statistics.
  torch.cuda.reset_peak_memory_stats(args.device)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MasteryRunner(wrapped, asdict(agent), device=args.device)
    transfer = runner.initialize_actor(args.teacher_onnx)
    actor = runner.get_inference_policy(args.device)
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    observation = wrapped.get_observations()
    with torch.no_grad():
      for _ in range(4):
        observation, reward, _, _ = wrapped.step(actor(observation))
        if not torch.isfinite(reward).all() or not all(torch.isfinite(value).all() for value in observation.values()):
          raise RuntimeError("Nonfinite full-profile inference")
    if any(not torch.equal(value, before[key]) for key, value in actor.state_dict().items()):
      raise RuntimeError("Inference changed policy parameters or statistics")
    free, total = torch.cuda.mem_get_info(args.device)
    if free < 2 * 1024**3:
      raise RuntimeError("Less than 2 GiB remains for training workspace")
    result = {"status": "passed", "optimizer_steps": 0, "device": args.device,
              "num_envs": env.num_envs, "actor_width": observation["actor"].shape[1],
              "inference_steps": 4, "cuda_devices": torch.cuda.device_count(),
              "gpu": torch.cuda.get_device_name(args.device), "gpu_free_bytes": free,
              "gpu_total_bytes": total, "peak_torch_bytes": torch.cuda.max_memory_allocated(args.device),
              "disk_free_bytes": free_disk, "tasks_registered": list_tasks(), "actor_transfer": transfer,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "teacher_sha256": hashlib.sha256(args.teacher_onnx.read_bytes()).hexdigest(),
              "versions": {"torch": torch.__version__, "tensorboard": tensorboard.__version__, "matplotlib": matplotlib.__version__},
              "wall_time": time.time(), "limitation": "No optimizer/backward pass; training throughput is unmeasured."}
    atomic_json(args.output / f'readiness-{args.device.replace(":", "-")}.json', result)
    print(f"PASS {args.device}: {env.num_envs} worlds, zero optimizer steps, {free / 1024**3:.2f} GiB free")
  finally:
    env.close()


if __name__ == "__main__":
  main()
