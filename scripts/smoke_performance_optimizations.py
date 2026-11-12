"""Compare optimized history and physics observations with the original backend."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.buffers.circular_buffer import CircularBuffer

from scripts.training_metrics import atomic_json
from src.tasks.campus.skills import skill
from src.tasks.stair_mastery import mastery_env_cfg
from src.tasks.stair_mastery.performance import ORIGINAL_APPEND, branchless_append, set_branchless_history


def buffers(device="cpu"):
  comparisons = 0
  for length in (1, 2, 4, 5):
    for shape in ((6,), (6, 3), (6, 2, 3)):
      reference = CircularBuffer(length, 6, device)
      candidate = CircularBuffer(length, 6, device)
      for tick in range(16):
        if tick % 3 == 2:
          ids = torch.tensor([0, 3, 5], device=device)
          reference.reset(ids)
          candidate.reset(ids)
        data = torch.arange(np.prod(shape), dtype=torch.float32, device=device).reshape(shape) + tick
        ORIGINAL_APPEND(reference, data)
        branchless_append(candidate, data)
        assert torch.equal(reference._buffer, candidate._buffer)
        assert torch.equal(reference._num_pushes, candidate._num_pushes)
        for lag in range(length + 2):
          assert torch.equal(reference[lag], candidate[lag])
          comparisons += 1
  return comparisons


def rollout(device):
  set_branchless_history(True)
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  cfg = mastery_env_cfg(stress=False)
  cfg.seed, cfg.scene.num_envs = 42, 6
  cfg.commands["skill"].fast_foot_positions = True
  cfg.observations.pop("teacher")
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    for _ in range(8):
      state = skill(env)
      torch.testing.assert_close(env.sim.data.site_xpos[:, state.global_foot_ids],
                                 state.robot.data.site_pos_w[:, state.foot_ids], rtol=0, atol=0)
      observation, reward, _, _, _ = env.step(torch.zeros((6, 29), device=device))
      assert set(observation) == {"actor", "critic"}
      assert all(torch.isfinite(tensor).all() for tensor in (*observation.values(), reward))
  finally:
    env.close()
    set_branchless_history(False)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--cpu-only", action="store_true")
  args = parser.parse_args()
  result = {"buffer_comparisons": buffers(), "optimizer_steps": 0}
  if not args.cpu_only:
    result["gpu_buffer_comparisons"] = buffers(args.device)
    rollout(args.device)
    result["same_state_foot_positions_exact"] = True
    result["optimized_physics_rollout_finite"] = True
  result["status"] = "passed"
  atomic_json(args.output, result)
  print(result)


if __name__ == "__main__":
  main()
