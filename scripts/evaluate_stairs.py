"""Evaluate one checkpoint on every stair flight and difficulty without a viewer."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from src.tasks.stairs import TASK_ID, stairs_env_cfg, stairs_runner_cfg
from src.tasks.stairs.terrain import FINISH_X, FLIGHTS, LEVELS


def check_onnx(path: Path, actor_obs: torch.Tensor, actions: torch.Tensor) -> float:
  import numpy as np
  import onnxruntime as ort

  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
  input_name = session.get_inputs()[0].name
  obs = actor_obs[:8].detach().cpu().numpy()
  expected = actions[:8].detach().cpu().numpy()
  actual = np.concatenate([session.run(None, {input_name: row[None, :]})[0] for row in obs])
  np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
  return float(np.max(np.abs(actual - expected)))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict:
  configure_torch_backends()
  cfg = stairs_env_cfg(play=True)
  cfg.seed = args.seed
  cfg.scene.terrain.terrain_generator.seed = args.terrain_seed
  cfg.scene.num_envs = args.episodes_per_cell * LEVELS * len(FLIGHTS)
  agent = stairs_runner_cfg()
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    terrain = env.scene.terrain
    assert terrain is not None
    levels = torch.arange(LEVELS, device=args.device).repeat_interleave(
      len(FLIGHTS) * args.episodes_per_cell
    )
    columns = torch.arange(len(FLIGHTS), device=args.device).repeat_interleave(
      args.episodes_per_cell
    ).repeat(LEVELS)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = terrain.terrain_origins[levels, columns]
    obs, _ = wrapper.reset()
    runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    parity_error = check_onnx(args.onnx, obs["actor"], policy(obs)) if args.onnx else None

    completed = torch.zeros(env.num_envs, device=args.device, dtype=torch.bool)
    success = completed.clone()
    fall = completed.clone()
    off_lane = completed.clone()
    timeout = completed.clone()
    max_progress = torch.zeros(env.num_envs, device=args.device)
    for _ in range(env.max_episode_length):
      obs, _, done, _ = wrapper.step(policy(obs))
      done = done.bool()
      manager = env.termination_manager
      first_done = done & ~completed
      landed = first_done & manager.get_term("landing")
      success |= landed
      failed = first_done & ~landed
      fell = manager.get_term("fell_over") | manager.get_term("low_root") | manager.get_term("body_hit")
      fall |= failed & fell
      left_lane = manager.get_term("out_of_lane")
      off_lane |= failed & ~fell & left_lane
      timeout |= failed & ~fell & ~left_lane & manager.time_outs

      x = env.scene["robot"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
      progress = (x / FINISH_X).clamp(0.0, 1.0)
      max_progress = torch.where(
        ~completed & ~done, torch.maximum(max_progress, progress), max_progress
      )
      max_progress[landed] = 1.0
      completed |= done
      if completed.all():
        break
    if not completed.all():
      raise RuntimeError("Evaluation ended with unfinished first episodes")

    rows = []
    for level in range(LEVELS):
      for column, name in enumerate(FLIGHTS):
        mask = (levels == level) & (columns == column)
        successes = int(success[mask].sum())
        rows.append({
          "level": level,
          "flight": name,
          "episodes": args.episodes_per_cell,
          "successes": successes,
          "success_rate": successes / args.episodes_per_cell,
          "falls": int(fall[mask].sum()),
          "out_of_lane": int(off_lane[mask].sum()),
          "timeouts": int(timeout[mask].sum()),
          "mean_max_progress_fraction": round(float(max_progress[mask].mean()), 4),
        })
    return {
      "task": TASK_ID,
      "checkpoint": args.checkpoint.name,
      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "seed": args.seed,
      "terrain_seed": args.terrain_seed,
      "device": args.device,
      "onnx_max_abs_error": parity_error,
      "cells": rows,
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--onnx", type=Path)
  parser.add_argument("--episodes-per-cell", type=int, default=16)
  parser.add_argument("--seed", type=int, default=913)
  parser.add_argument("--terrain-seed", type=int, default=79)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.episodes_per_cell < 1:
    parser.error("--episodes-per-cell must be positive")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  report = json.dumps(evaluate(args), indent=2) + "\n"
  args.output.write_text(report, encoding="utf-8")
  print(f"Saved {len(FLIGHTS) * LEVELS} evaluation cells to {args.output}")


if __name__ == "__main__":
  main()
