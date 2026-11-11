"""Inference-only smoke checks for every stair phase and the additive mastery tasks."""

import argparse
import json
from dataclasses import asdict
from math import pi
from pathlib import Path
from types import SimpleNamespace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.os import dump_yaml

import src.tasks
from scripts.evaluate_stairs import check_onnx
from src.tasks.campus.skills import Phase, skill
from src.tasks.campus.terrain import DIFFICULTIES
from src.tasks.round_trip import mdp
from src.tasks.round_trip.improved import adaptive_curriculum, descent_overspeed, next_footholds, speed_clock
from src.tasks.round_trip.skills import Route
from src.tasks.round_trip.terrain import COURSES, PACE_BANDS
from src.tasks.stair_mastery import TASKS, mastery_env_cfg, mastery_runner_cfg
from src.tasks.stair_mastery.distillation import ExpertPolicies, imitation_loss
from src.tasks.stair_mastery.runner import MasteryRunner


def finite(observation):
  if not all(torch.isfinite(value).all() for value in observation.values()):
    raise AssertionError("An observation contains NaN or infinity")


def phase_checks(env):
  """Controlled contact/pose fixtures exercise the same phase code used in simulation."""
  state = skill(env)
  original_robot, original_feet = state.robot, state._feet
  original_subtree = env.sim.data.subtree_com.clone()
  origin = state.terrain.env_origins
  n = env.num_envs
  feet = torch.zeros((n, 2, 3), device=env.device)
  found = torch.ones((n, 2), device=env.device, dtype=torch.bool)
  force = torch.zeros((n, 2), device=env.device)
  position = torch.zeros((n, 3), device=env.device)
  velocity = torch.zeros_like(position)
  angular = torch.zeros_like(position)
  gravity = torch.zeros_like(position)
  gravity[:, 2] = -1
  heading = torch.zeros(n, device=env.device)
  quaternion = torch.zeros((n, 4), device=env.device)
  quaternion[:, 0] = 1
  data = SimpleNamespace(
    root_link_pos_w=position, root_link_lin_vel_w=velocity, root_link_ang_vel_w=angular,
    projected_gravity_b=gravity, heading_w=heading, root_link_quat_w=quaternion,
  )
  state.robot = SimpleNamespace(data=data, indexing=original_robot.indexing)
  state._feet = lambda: (feet, found, force)
  checks = []

  def setup(route, phase):
    state.route.fill_(route)
    state.start_mode.fill_(Route.ASCEND)
    state.forward_sign.fill_(-1 if route == Route.DESCEND else 1)
    state.riser.zero_()
    state.phase.fill_(phase)
    state.support.zero_()
    state.hold.zero_()
    state.success.zero_()
    state.ascended.zero_()
    state.turned.zero_()
    state.turn_hold.zero_()
    feet[:] = origin[:, None, :]
    feet[:, 0, 1] += .10
    feet[:, 1, 1] -= .10
    position[:] = origin
    position[:, 2] += .80
    found.fill_(True)
    force[:, 0] = 180
    force[:, 1] = 30
    velocity.zero_()
    angular.zero_()
    heading.fill_(pi if route == Route.DESCEND else 0.)
    env.sim.data.subtree_com[:, original_robot.indexing.root_body_id, 1] = feet[:, 0, 1]

  def expect(label, phase):
    if not (state.phase == phase).all():
      raise AssertionError(f"{label} did not reach {phase.name}")
    checks.append(label)

  try:
    for route in (Route.ASCEND, Route.DESCEND):
      setup(route, Phase.APPROACH)
      _, edge, _, _, _, _, _, _, _ = state.targets()
      position[:, 0] = origin[:, 0] + (edge - .30) * state.forward_sign
      state.advance()
      expect(f"{route.name}:approach", Phase.SHIFT)
      setup(route, Phase.SHIFT)
      state.advance()
      expect(f"{route.name}:shift", Phase.LIFT)
      setup(route, Phase.LIFT)
      _, _, _, start, _, *_ = state.targets()
      found[:, 1] = False
      feet[:, 1, 2] = start + .04
      state.advance()
      expect(f"{route.name}:lift", Phase.CLEAR)
      setup(route, Phase.CLEAR)
      _, edge, _, start, goal, *_ = state.targets()
      found[:, 1] = False
      feet[:, 1, 0] = origin[:, 0] + edge * state.forward_sign
      feet[:, 1, 2] = torch.maximum(start, goal) + .06
      state.advance()
      expect(f"{route.name}:clear", Phase.PLANT)
      setup(route, Phase.PLANT)
      target = state.targets()[0]
      feet[:, 1] = target
      force[:, 1] = 200.
      state.advance()
      expect(f"{route.name}:plant", Phase.TRANSFER)
      setup(route, Phase.TRANSFER)
      feet[:, 1] = state.targets()[0]
      force[:, 1] = 250.
      force[:, 0] = 30.
      state.hold.fill_(2)
      state.advance()
      if not (state.riser == 1).all():
        raise AssertionError("Load transfer did not count one riser")
      checks.append(f"{route.name}:transfer")

      setup(route, Phase.FINISH)
      state.riser[:] = state.tiers[state.terrain.terrain_levels, 0].long()
      _, edge, _, _, goal_z, *_ = state.targets()
      position[:, 0] = origin[:, 0] + (edge + .60) * state.forward_sign
      feet[:, :, 0] = origin[:, 0:1] + (edge[:, None] + .30) * state.forward_sign[:, None]
      feet[:, :, 2] = goal_z[:, None]
      state.hold.fill_(4)
      state.advance()
      if route == Route.ASCEND:
        if not (state.route == Route.TURN).all():
          raise AssertionError("Ascent finish did not start the upper landing turn")
      elif not state.success.all():
        raise AssertionError("Descent finish did not complete the round trip")
      checks.append(f"{route.name}:finish")

    for direction in (-1., 1.):
      setup(Route.TURN, Phase.FINISH)
      _, last, top, _, _ = state.bounds()
      position[:, 0] = origin[:, 0] + last + .69
      feet[:, :, 0] = origin[:, 0:1] + last[:, None] + .69
      feet[:, :, 2] = top[:, None]
      heading.fill_(direction * pi)
      state.turn_side.fill_(direction)
      state.turn_angle.fill_(direction * pi)
      state.turn_yaw.fill_(direction * pi)
      state.turn_hold.fill_(4)
      state.advance()
      if not state.turned.all() or not (state.route == Route.DESCEND).all():
        raise AssertionError("Turn completion failed")
      checks.append(f"turn:{direction:+.0f}")

    setup(Route.DESCEND, Phase.APPROACH)
    velocity[:, 0] = -(state.speed_target + .20)
    fast = descent_overspeed(env)
    velocity[:, 0] = -state.speed_target
    on_target = descent_overspeed(env)
    if not (fast > on_target).all() or not (on_target == 0).all():
      raise AssertionError("Overspeed cost does not distinguish fast and target descent")
    checks.append("downhill_speed_penalty")
    state.riser.zero_()
    first_two = next_footholds(env, estimated=False)
    if first_two.shape != (n, 8) or not (first_two[:, (3, 7)] == 1).all():
      raise AssertionError("Two planned footholds are missing")
    checks.append("two_valid_footholds")
  finally:
    state.robot = original_robot
    state._feet = original_feet
    env.sim.data.subtree_com[:] = original_subtree
    env.reset()
  return checks


def balance_checks(env):
  state = skill(env)
  robot, feet_fn = state.robot, state._feet
  subtree = env.sim.data.subtree_com.clone()
  origin = state.terrain.env_origins
  feet = torch.zeros((env.num_envs, 2, 3), device=env.device)
  feet[:] = origin[:, None, :]
  feet[:, 0, 1] += .10
  feet[:, 1, 1] -= .10
  support = state.support
  swing = 1 - support
  feet[state.ids, swing, 2] += state.tiers[state.terrain.terrain_levels, 3]
  found = torch.ones((env.num_envs, 2), device=env.device, dtype=torch.bool)
  found[state.ids, swing] = False
  force = found.float() * 180.
  data = SimpleNamespace(
    root_link_pos_w=origin.clone(), root_link_lin_vel_w=torch.zeros_like(origin),
    projected_gravity_b=torch.tensor([0., 0., -1.], device=env.device).expand(env.num_envs, 3),
  )
  try:
    state.robot = SimpleNamespace(data=data, indexing=robot.indexing)
    state._feet = lambda: (feet, found, force)
    env.sim.data.subtree_com[:, robot.indexing.root_body_id, 1] = feet[state.ids, support, 1]
    state.phase.fill_(Phase.HOLD)
    state.hold[:] = torch.ceil(state.tiers[state.terrain.terrain_levels, 2] / env.step_dt).long() - 1
    state.advance()
    if not state.success.all():
      raise AssertionError("A continuous single-support hold was not recognized")
    state.phase.fill_(Phase.HOLD)
    state.hold.zero_()
    found.fill_(True)
    state.advance()
    if state.success.any():
      raise AssertionError("A two-foot stance was accepted as one-leg balance")
  finally:
    state.robot = robot
    state._feet = feet_fn
    env.sim.data.subtree_com[:] = subtree
    env.reset()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--teacher-onnx", type=Path, required=True)
  parser.add_argument("--teacher-checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  reports = []
  # Existing tasks are built through their normal registry entry points.
  legacy = [
    "Unitree-G1-EDU-Stairs", "Unitree-G1-EDU-Campus", "Unitree-G1-EDU-Balance-Left",
    "Unitree-G1-EDU-Balance-Right", "Unitree-G1-EDU-Step-Up", "Unitree-G1-EDU-Step-Down",
    "Unitree-G1-EDU-Campus-Flights", "Unitree-G1-EDU-Stair-Round-Trip",
  ]
  for task in legacy:
    if task not in list_tasks():
      raise AssertionError(f"Existing task was removed: {task}")
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 6
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      observation, _ = env.reset()
      finite(observation)
      for _ in range(3):
        observation, reward, _, _, _ = env.step(torch.zeros((6, 29), device=args.device))
        finite(observation)
        if not torch.isfinite(reward).all():
          raise AssertionError("Legacy reward is nonfinite")
      if "Balance-" in task:
        balance_checks(env)
      reports.append({"task": task, "inference_steps": 3, "balance_fixture": "Balance-" in task})
    finally:
      env.close()

  for task, (expert, student) in TASKS.items():
    cfg = mastery_env_cfg(expert, student=student)
    cfg.seed = 72
    cfg.scene.num_envs = 108
    # Smoke each geometry and level directly, without changing the curriculum definition.
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      state = skill(env)
      levels = torch.arange(6, device=args.device).repeat_interleave(len(COURSES))
      columns = torch.arange(len(COURSES), device=args.device).repeat(6)
      state.terrain.terrain_levels[:] = levels
      state.terrain.terrain_types[:] = columns
      state.terrain.env_origins[:] = state.terrain.terrain_origins[levels, columns]
      state.cfg.fixed_level = None
      wrapped = RslRlVecEnvWrapper(env, clip_actions=3.0)
      observation = wrapped.get_observations()
      if observation["actor"].shape != (108, 785) or observation["teacher"].shape != (108, 785):
        raise AssertionError(f"Unexpected actor/teacher shape for {task}")
      runner = MasteryRunner(wrapped, asdict(mastery_runner_cfg()), device=args.device)
      transfer = runner.initialize_actor(args.teacher_onnx)
      policy = runner.get_inference_policy(args.device)
      before = {key: value.clone() for key, value in policy.state_dict().items()}
      for _ in range(12):
        with torch.no_grad():
          action = policy(observation)
          observation, reward, _, _ = wrapped.step(action)
        finite(observation)
        if not torch.isfinite(reward).all():
          raise AssertionError("Mastery reward is nonfinite")
      if any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items()):
        raise AssertionError("Inference smoke changed policy weights or statistics")
      runner.write_run_audit(args.output / task, 0)
      dump_yaml(args.output / task / "env.yaml", asdict(cfg))
      report = {"task": task, "inference_steps": 12, "actor_width": 785,
                "actor_transfer": transfer, "optimizer_steps": 0}
      if task == "Unitree-G1-EDU-Stair-Mastery":
        # A phase fixture uses the full-route mode for all controlled cases.
        state.cfg.expert = "round_trip"
        state.terrain.terrain_levels[:] = levels
        state.terrain.terrain_types[:] = columns
        state.terrain.env_origins[:] = state.terrain.terrain_origins[levels, columns]
        report["phases"] = phase_checks(env)
        runner.export_policy_to_onnx(str(args.output / task), "smoke-only.onnx")
        observation = wrapped.get_observations()
        with torch.no_grad():
          report["onnx_error"] = check_onnx(args.output / task / "smoke-only.onnx",
                                             observation["actor"], policy(observation))
      if student:
        teachers = ExpertPolicies([args.teacher_checkpoint] * 3, observation,
                                   asdict(mastery_runner_cfg().actor), args.device)
        with torch.no_grad():
          label = teachers.actions(observation, state.route)
          loss = imitation_loss(policy, observation, label)
          if not torch.isfinite(loss):
            raise AssertionError("Imitation forward pass is nonfinite")
          report["imitation_forward_loss"] = float(loss)
      reports.append(report)
    finally:
      env.close()
  result = {"optimizer_steps": 0, "reports": reports}
  (args.output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
  print(f"Passed {len(reports)} task smoke checks with zero optimizer steps")


if __name__ == "__main__":
  main()
