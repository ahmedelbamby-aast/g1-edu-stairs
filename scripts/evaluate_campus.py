"""Score every campus course, difficulty and isolated skill headlessly."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml

from scripts.evaluate_stairs import check_onnx
from src.tasks.campus import TASK_ID, campus_env_cfg, campus_runner_cfg
from src.tasks.campus.skills import Drill, Phase, skill
from src.tasks.campus.terrain import COURSES, DIFFICULTIES


def assignments():
  return [
    (level, column, drill)
    for level in range(len(DIFFICULTIES))
    for drill in Drill
    for column in (range(len(COURSES)) if drill >= Drill.STEP else (0,))
  ]


@torch.inference_mode()
def evaluate(args):
  configure_torch_backends()
  cells = assignments()
  cfg = campus_env_cfg(play=True, stress=args.suite == "stress", terrain_seed=args.terrain_seed)
  cfg.seed = args.seed
  cfg.scene.num_envs = len(cells) * args.episodes_per_cell
  command = cfg.commands["skill"]
  command.fixed_level = None
  command.promote_after = command.demote_after = 10_000
  agent = campus_runner_cfg()
  dump_yaml(args.output.with_suffix(".env.yaml"), asdict(cfg))
  dump_yaml(args.output.with_suffix(".agent.yaml"), asdict(agent))
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    state = skill(env)
    level, column, drill = torch.tensor(cells, device=args.device).repeat_interleave(
      args.episodes_per_cell, dim=0
    ).unbind(dim=1)
    state.drill[:] = drill
    state.balance[:] = drill < Drill.STEP
    state.step_drill[:] = drill == Drill.STEP
    state.flight[:] = drill == Drill.FLIGHT
    state.terrain.terrain_levels[:] = level
    state.terrain.terrain_types[:] = column
    state.terrain.env_origins[:] = state.terrain.terrain_origins[level, column]
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs = wrapper.get_observations()
    runner = MjlabOnPolicyRunner(wrapper, asdict(agent), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    parity = check_onnx(args.onnx, obs["actor"], policy(obs)) if args.onnx else None
    complete = torch.zeros(env.num_envs, dtype=torch.bool, device=args.device)
    success = complete.clone()
    fall = complete.clone()
    lane_exit = complete.clone()
    timeout = complete.clone()
    reached = torch.zeros_like(state.reached)
    risers = torch.zeros_like(state.riser)
    hold = torch.zeros_like(state.max_hold)
    for _ in range(env.max_episode_length):
      obs, _, done, _ = wrapper.step(policy(obs))
      first = done.bool() & ~complete
      manager = env.termination_manager
      landed = manager.get_term("landing")
      fell = manager.get_term("fell_over") | manager.get_term("body_hit") | manager.get_term("low_root")
      left = manager.get_term("out_of_lane")
      success |= first & landed
      fall |= first & ~landed & fell
      lane_exit |= first & ~landed & ~fell & left
      timeout |= first & ~landed & ~fell & ~left & manager.time_outs
      reached |= first[:, None] & state.last_reached
      risers = torch.where(first, state.last_risers, risers)
      hold = torch.where(first, state.last_hold, hold)
      complete |= first
      if complete.all():
        break
    if not complete.all():
      raise RuntimeError("Some evaluation worlds did not finish their first episode")

    rows = []
    for index, (tier, course, kind) in enumerate(cells):
      count = args.episodes_per_cell
      sample = slice(index * count, (index + 1) * count)
      completed = int(success[sample].sum())
      falls = int(fall[sample].sum())
      rows.append({
        "level": tier, "course": COURSES[course], "drill": kind.name.lower(),
        "episodes": count, "completions": completed, "falls": falls,
        "lane_exits": int(lane_exit[sample].sum()), "timeouts": int(timeout[sample].sum()),
        "phase_reached_counts": {phase.name.lower(): int(reached[sample, phase].sum()) for phase in Phase},
        "mean_risers_completed": float(risers[sample].float().mean()),
        "mean_best_single_support_s": float(hold[sample].float().mean()) * env.step_dt,
        "passes_sim_gate": completed / count >= args.min_success and falls / count <= args.max_falls,
      })
    return {
      "task": TASK_ID, "suite": args.suite, "seed": args.seed, "terrain_seed": args.terrain_seed,
      "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "device": args.device,
      "checkpoint": args.checkpoint.name,
      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "onnx_max_abs_error": parity, "minimum_completion_fraction": args.min_success,
      "maximum_fall_fraction": args.max_falls,
      "all_cells_pass_sim_gate": all(row["passes_sim_gate"] for row in rows), "cells": rows,
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--onnx", type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--episodes-per-cell", type=int, default=16)
  parser.add_argument("--seed", type=int, default=913)
  parser.add_argument("--terrain-seed", type=int, default=79)
  parser.add_argument("--suite", choices=("clean", "stress"), default="stress")
  parser.add_argument("--min-success", type=float, default=.95)
  parser.add_argument("--max-falls", type=float, default=.0)
  parser.add_argument("--require-pass", action="store_true")
  args = parser.parse_args()
  if args.episodes_per_cell < 1 or not 0 <= args.min_success <= 1 or not 0 <= args.max_falls <= 1:
    parser.error("Use positive episode counts and gate fractions between zero and one")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  result = evaluate(args)
  args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(f"Saved {len(result['cells'])} cells to {args.output}")
  if args.require_pass and not result["all_cells_pass_sim_gate"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
