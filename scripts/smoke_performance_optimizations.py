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


def buffers():
  comparisons = 0
  for length in (1, 2, 4, 5):
    for shape in ((6,), (6, 3), (6, 2, 3)):
      reference = CircularBuffer(length, 6, "cpu")
      candidate = CircularBuffer(length, 6, "cpu")
      for tick in range(16):
        if tick % 3 == 2:
          ids = torch.tensor([0, 3, 5])
          reference.reset(ids)
          candidate.reset(ids)
        data = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) + tick
        ORIGINAL_APPEND(reference, data)
        branchless_append(candidate, data)
        assert torch.equal(reference._buffer, candidate._buffer)
        assert torch.equal(reference._num_pushes, candidate._num_pushes)
        for lag in range(length + 2):
          assert torch.equal(reference[lag], candidate[lag])
          comparisons += 1
  return comparisons


def rollout(optimized, device):
  set_branchless_history(optimized)
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  cfg = mastery_env_cfg(stress=False)
  cfg.seed, cfg.scene.num_envs = 42, 6
  cfg.commands["skill"].fast_foot_positions = optimized
  if optimized:
    cfg.observations.pop("teacher")
  env = ManagerBasedRlEnv(cfg, device=device)
  values = []
  try:
    observation, _ = env.reset()
    for _ in range(8):
      state = skill(env)
      torch.testing.assert_close(env.sim.data.site_xpos[:, state.global_foot_ids],
                                 state.robot.data.site_pos_w[:, state.foot_ids], rtol=0, atol=0)
      observation, reward, done, timeout, _ = env.step(torch.zeros((6, 29), device=device))
      values.append({"actor": observation["actor"].cpu().clone(), "critic": observation["critic"].cpu().clone(),
                     "reward": reward.cpu().clone(), "done": done.cpu().clone(), "timeout": timeout.cpu().clone()})
  finally:
    env.close()
    set_branchless_history(False)
  return values


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--cpu-only", action="store_true")
  args = parser.parse_args()
  result = {"buffer_comparisons": buffers(), "optimizer_steps": 0}
  if not args.cpu_only:
    reference, optimized = rollout(False, args.device), rollout(True, args.device)
    error = {}
    for left, right in zip(reference, optimized):
      for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=1e-6, atol=1e-6)
        if left[name].dtype != torch.bool:
          error[name] = max(error.get(name, 0), float((left[name] - right[name]).abs().max()))
    result["maximum_absolute_error"] = error
    result["physics_trajectory_parity"] = "passed"
  result["status"] = "passed"
  atomic_json(args.output, result)
  print(result)


if __name__ == "__main__":
  main()
