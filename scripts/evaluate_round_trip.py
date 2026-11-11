"""Score every round-trip geometry, speed band and difficulty."""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from scripts.evaluate_stairs import check_onnx
from src.tasks.campus.skills import Phase, skill
from src.tasks.campus.terrain import DIFFICULTIES
from src.tasks.round_trip import TASK_ID, round_trip_env_cfg, round_trip_runner_cfg
from src.tasks.round_trip.terrain import COURSES, DEPTHS, PACE_BANDS
from src.tasks.stair_mastery import TASK_ID as MASTERY_ID, TASKS, mastery_env_cfg, mastery_runner_cfg


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
  if args.task == TASK_ID:
    cfg = round_trip_env_cfg(play=True, stress=args.suite == "stress", terrain_seed=args.terrain_seed)
    agent = round_trip_runner_cfg()
  else:
    cfg = mastery_env_cfg(TASKS[args.task][0], play=True, student=args.task.endswith("-Student"),
                          stress=args.suite == "stress", terrain_seed=args.terrain_seed)
    # Evaluation assigns every cell explicitly. Adaptive reset sampling would
    # otherwise replace those assignments even on the initial wrapper reset.
    cfg.curriculum = {}
    agent = mastery_runner_cfg()
  cfg.seed = args.seed
  cfg.scene.num_envs = len(cells) * count
  command = cfg.commands["skill"]
  command.fixed_level = None
  command.promote_after = command.demote_after = 10_000
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
    reached = torch.zeros((env.num_envs, len(Phase)), device=args.device, dtype=torch.bool)
    risers, hold = torch.zeros_like(target), torch.zeros_like(target)
    slip_sum, contact_count = torch.zeros_like(target), torch.zeros_like(target)
    force_sum, landing_count = torch.zeros_like(target), torch.zeros_like(target)
    for _ in range(env.max_episode_length):
      # Sample before stepping so automatic episode resets never contaminate first-trial statistics.
      active = ~done_before
      contact = env.scene["feet_ground_contact"]
      found = contact.data.found > 0
      speed = state.robot.data.site_lin_vel_w[:, state.foot_ids, :2].norm(dim=-1)
      impact = contact.compute_first_contact(dt=env.step_dt)
      magnitude = contact.data.force.norm(dim=-1)
      slip_sum += (speed * found).sum(dim=1) * active
      contact_count += found.sum(dim=1) * active
      force_sum += (magnitude * impact).sum(dim=1) * active
      landing_count += impact.sum(dim=1) * active
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
      reached |= first[:, None] & state.last_reached
      risers = torch.where(first, state.last_risers.float(), risers)
      hold = torch.where(first, state.last_hold.float() * env.step_dt, hold)
      done_before |= first
      if done_before.all():
        break
    if not done_before.all():
      raise RuntimeError("Some round-trip trials did not terminate")
    full_route = args.task not in TASKS or TASKS[args.task][0] == "round_trip"
    if full_route and ((completed & ~turned).any() or (turned & ~top).any()):
      raise RuntimeError("A trial advanced without completing the preceding route phase")
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
      up_valid, down_valid = measured[sample, 0] > 0, measured[sample, 1] > 0
      slip_n, landing_n = int(contact_count[sample].sum()), int(landing_count[sample].sum())
      row = {
        "level": level, "course": COURSES[column], "depth": depths[column % len(depths)],
        "speed_band": labels[speed], "episodes": count, "top_landings": int(top[sample].sum()),
        "turns": int(turned[sample].sum()), "completions": n,
        "clockwise_trials": int((turn_side[sample] < 0).sum()),
        "clockwise_turns_completed": int(((turn_side[sample] < 0) & turned[sample]).sum()),
        "counterclockwise_turns_completed": int(((turn_side[sample] > 0) & turned[sample]).sum()),
        "falls": int(falls[sample].sum()), "lane_exits": int(exits[sample].sum()),
        "timeouts": int(timeouts[sample].sum()), "mean_target_mps": float(target[sample].mean()),
        "mean_up_mps_on_completion": float(actual_complete[:, 0].mean()) if n else None,
        "mean_down_mps_on_completion": float(actual_complete[:, 1].mean()) if n else None,
        "phase_reached_counts": {phase.name.lower(): int(reached[sample, phase].sum()) for phase in Phase},
        "mean_risers_completed": float(risers[sample].mean()),
        "mean_best_single_support_s": float(hold[sample].mean()),
        "slip_velocity_sum_mps": float(slip_sum[sample].sum()), "contact_samples": slip_n,
        "landing_force_sum_n": float(force_sum[sample].sum()), "landing_samples": landing_n,
        "mean_slip_velocity_mps": float(slip_sum[sample].sum()) / slip_n if slip_n else None,
        "mean_landing_force_n": float(force_sum[sample].sum()) / landing_n if landing_n else None,
        "mean_up_speed_error_mps": float((actual[sample, 0] - target[sample]).abs()[up_valid].mean()) if up_valid.any() else None,
        "mean_down_speed_error_mps": float((actual[sample, 1] - target[sample]).abs()[down_valid].mean()) if down_valid.any() else None,
      }
      if full_route:
        row.update(completed_round_trips=n, completed_at_target_pace_both_directions=int(speed_match[sample].sum()))
      rows.append(row)
    return {
      "task": args.task, "suite": args.suite, "seed": args.seed, "terrain_seed": args.terrain_seed,
      "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
      "checkpoint": args.checkpoint.name, "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "onnx_max_abs_error": parity, "pace_relative_tolerance": command.speed_tolerance, "cells": rows,
      "wall_time": time.time(), "evaluation_goal": "round_trip" if full_route else TASKS[args.task][0],
    }
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", choices=(TASK_ID, *TASKS), default=TASK_ID)
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
