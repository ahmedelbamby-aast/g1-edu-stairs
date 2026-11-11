"""Score every round-trip geometry, speed band and difficulty."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from scripts.evaluate_stairs import check_onnx
from src.tasks.campus.skills import skill
from src.tasks.campus.terrain import DIFFICULTIES
from src.tasks.round_trip import TASK_ID, round_trip_env_cfg, round_trip_runner_cfg
from src.tasks.round_trip.terrain import COURSES, DEPTHS, PACE_BANDS


@torch.inference_mode()
def evaluate(args):
  configure_torch_backends()
  cells = [
    (level, column, speed)
    for level in range(len(DIFFICULTIES))
    for column in range(len(COURSES))
    for speed in range(len(PACE_BANDS))
  ]
  count = args.episodes_per_cell
  cfg = round_trip_env_cfg(play=True, stress=args.suite == "stress", terrain_seed=args.terrain_seed)
  cfg.seed = args.seed
  cfg.scene.num_envs = len(cells) * count
  command = cfg.commands["skill"]
  command.fixed_level = None
  command.promote_after = command.demote_after = 10_000
  agent = round_trip_runner_cfg()
  dump_yaml(args.output.with_suffix(".env.yaml"), asdict(cfg))
  dump_yaml(args.output.with_suffix(".agent.yaml"), asdict(agent))
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    state = skill(env)
    levels, columns, speeds = torch.tensor(cells, device=args.device).repeat_interleave(count, dim=0).unbind(dim=1)
    state.terrain.terrain_levels[:] = levels
    state.terrain.terrain_types[:] = columns
    state.terrain.env_origins[:] = state.terrain.terrain_origins[levels, columns]
    state.speed_band[:] = speeds
    state.turn_side[:] = (state.ids % count) % 2 * 2 - 1
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = wrapper.get_observations()
    runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    parity = check_onnx(args.onnx, obs["actor"], policy(obs)) if args.onnx else None
    done_before = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
    completed = done_before.clone()
    falls = done_before.clone()
    exits = done_before.clone()
    timeouts = done_before.clone()
    top = done_before.clone()
    turned = done_before.clone()
    target = torch.zeros_like(state.speed_target)
    actual = torch.zeros_like(state.last_speed)
    measured = torch.zeros_like(state.last_speed_steps)
    turn_side = torch.zeros_like(state.last_turn_side)
    for _ in range(env.max_episode_length):
      obs, _, done, _ = wrapper.step(policy(obs))
      first = done.bool() & ~done_before
      manager = env.termination_manager
      landed = manager.get_term("landing")
      fell = manager.get_term("fell_over") | manager.get_term("body_hit") | manager.get_term("low_root")
      left = manager.get_term("out_of_lane")
      completed |= first & landed
      falls |= first & ~landed & fell
      exits |= first & ~landed & ~fell & left
      timeouts |= first & ~landed & ~fell & ~left & manager.time_outs
      top |= first & state.last_ascended
      turned |= first & state.last_turned
      target = torch.where(first, state.last_speed_target, target)
      actual = torch.where(first[:, None], state.last_speed, actual)
      measured = torch.where(first[:, None], state.last_speed_steps, measured)
      turn_side = torch.where(first, state.last_turn_side, turn_side)
      done_before |= first
      if done_before.all():
        break
    if not done_before.all():
      raise RuntimeError("Some round-trip trials did not terminate")
    speed_match = ((actual - target[:, None]).abs() <= command.speed_tolerance * target[:, None])
    speed_match = completed & (speed_match & (measured > 0)).all(dim=1)
    labels = tuple(PACE_BANDS)
    depths = tuple(DEPTHS)
    rows = []
    for index, (level, column, speed) in enumerate(cells):
      sample = slice(index * count, (index + 1) * count)
      finished = completed[sample]
      n = int(finished.sum())
      actual_complete = actual[sample][finished]
      rows.append({
        "level": level, "course": COURSES[column], "depth": depths[column % len(depths)],
        "speed_band": labels[speed], "episodes": count, "top_landings": int(top[sample].sum()),
        "turns": int(turned[sample].sum()), "completed_round_trips": n,
        "completed_at_target_pace_both_directions": int(speed_match[sample].sum()),
        "clockwise_attempts": int((turn_side[sample] < 0).sum()),
        "clockwise_turns_completed": int(((turn_side[sample] < 0) & turned[sample]).sum()),
        "counterclockwise_turns_completed": int(((turn_side[sample] > 0) & turned[sample]).sum()),
        "falls": int(falls[sample].sum()), "lane_exits": int(exits[sample].sum()),
        "timeouts": int(timeouts[sample].sum()), "mean_target_mps": float(target[sample].mean()),
        "mean_up_mps_on_completion": float(actual_complete[:, 0].mean()) if n else None,
        "mean_down_mps_on_completion": float(actual_complete[:, 1].mean()) if n else None,
      })
    return {
      "task": TASK_ID, "suite": args.suite, "seed": args.seed, "terrain_seed": args.terrain_seed,
      "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "checkpoint": args.checkpoint.name, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "onnx_max_abs_error": parity, "pace_relative_tolerance": command.speed_tolerance, "cells": rows,
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--onnx", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--episodes-per-cell", type=int, default=8)
  parser.add_argument("--seed", type=int, default=913)
  parser.add_argument("--terrain-seed", type=int, default=79)
  parser.add_argument("--suite", choices=("clean", "stress"), default="stress")
  args = parser.parse_args()
  if args.episodes_per_cell < 2 or args.episodes_per_cell % 2:
    parser.error("Use an even number of at least two trials per cell to test both turn directions")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  result = evaluate(args)
  args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(f"Saved {len(result['cells'])} cells to {args.output}")


if __name__ == "__main__":
  main()
