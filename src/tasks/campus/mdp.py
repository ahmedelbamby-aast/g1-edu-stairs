"""Rewards, reset events and promotion for campus locomotion."""

import torch
from mjlab.envs import mdp as common

from .skills import Phase, STEP_SPAWN_X, skill
from .terrain import DIFFICULTIES, SPAWN_X, TILE_SIZE


def command_for_skill(env, command_name="twist"):
  state = skill(env)
  twist = env.command_manager.get_command(command_name)
  speed = torch.where(state.balance, 0.0, torch.where(state.step_drill, .28, twist[:, 0]))
  return torch.cat((speed[:, None], twist[:, 1:]), dim=1)


def gait_clock(env, period, command_name):
  cycle = env.episode_length_buf * (env.step_dt * 2 * torch.pi / period)
  clock = torch.stack((cycle.sin(), cycle.cos()), dim=1)
  return torch.where((command_for_skill(env, command_name)[:, :1] > .1), clock, 0.0)


def noisy_scan(env, sensor_name, offset, miss_value, dropout):
  heights = common.height_scan(env, sensor_name, offset, miss_value)
  if not dropout:
    return heights
  state = skill(env)
  probability = state.tiers[state.terrain.terrain_levels, 5:6]
  return torch.where(torch.rand_like(heights) < probability, miss_value, heights)


def reset_near_course(env, env_ids, pose_range, velocity_range):
  state = skill(env)
  step = state.step_drill[env_ids]
  common.reset_root_state_uniform(env, env_ids[~step], pose_range, velocity_range)
  step_pose = {**pose_range, "x": (STEP_SPAWN_X - .018, STEP_SPAWN_X + .018)}
  common.reset_root_state_uniform(env, env_ids[step], step_pose, velocity_range)


def course_push(env, env_ids):
  state = skill(env)
  bound = state.tiers[state.terrain.terrain_levels[env_ids], 4:5]
  delta = (2 * torch.rand((len(env_ids), 2), device=env.device) - 1) * bound
  velocity = state.robot.data.root_link_vel_w[env_ids]
  velocity[:, :2] += delta
  state.robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


def advance(env):
  state = skill(env)
  state.advance()
  # The fall and lane terms precede this term in the termination manager.
  state.success &= ~env.termination_manager.terminated
  return state.success


def lane_exit(env):
  state = skill(env)
  root = state.robot.data.root_link_pos_w - state.terrain.env_origins
  limit = state.tiers[state.terrain.terrain_levels, 1] / 2 - state.cfg.lane_margin_m
  return (root[:, 1].abs() > limit) | (root[:, 0] < -.40) | (root[:, 0] > TILE_SIZE[0] - SPAWN_X - .22)


def timeout(env):
  state = skill(env)
  drill_over = env.episode_length_buf * env.step_dt >= state.cfg.drill_timeout_s
  return common.time_out(env) | (~state.flight & drill_over)


def practice_reward(env):
  state = skill(env)
  phase = state.phase
  feet, found, force = state._feet()
  swing = feet[state.ids, 1 - state.support]
  support = feet[state.ids, state.support]
  com = env.sim.data.subtree_com[:, state.robot.indexing.root_body_id]
  lateral = (com[:, 1] - support[:, 1]).abs()
  load = force[state.ids, state.support] / force.sum(dim=1).clamp_min(1.0)
  shift = torch.exp(-(lateral / .12).square()) + .4 * load
  goal, _, _, start_z, goal_z, _, _, _, lift = state.targets()
  rise = (swing[:, 2] - state.terrain.env_origins[:, 2]).clamp_min(0)
  single_support = found[state.ids, state.support] & ~found[state.ids, 1 - state.support]
  balance = shift + (rise / lift).clamp(max=1.0) + .55 * single_support
  clearance = torch.maximum(start_z, goal_z) + .045
  clear_reward = torch.exp(-((swing[:, 2] - clearance) / .10).square())
  place_reward = torch.exp(-((swing - goal).square().sum(dim=1) / .16**2))
  step = torch.where(phase == Phase.SHIFT, shift, place_reward)
  step = torch.where((phase == Phase.LIFT) | (phase == Phase.CLEAR), .6 * clear_reward + .4 * place_reward, step)
  step = torch.where((phase == Phase.APPROACH) | (phase == Phase.FINISH), 0.0, step)
  return torch.where(state.balance, torch.where(phase == Phase.SHIFT, shift, balance), step)


def motion_reward(env):
  state = skill(env)
  speed = state.robot.data.root_link_lin_vel_w[:, :2]
  target = command_for_skill(env)[:, 0]
  moving = torch.exp(-((speed[:, 0] - target).square() + speed[:, 1].square()) / .30**2)
  standing = torch.exp(-speed.square().sum(dim=1) / .17**2)
  return torch.where(state.balance, standing, moving)


def phase_bonus(env):
  return skill(env).transition.float()


def course_curriculum(env, env_ids):
  state = skill(env)
  terrain = state.terrain
  levels = terrain.terrain_levels[env_ids]
  columns = terrain.terrain_types[env_ids]
  success = state.success[env_ids]
  finished = env.termination_manager.dones[env_ids]
  streak = state.streak[env_ids]
  streak = torch.where(finished & success, streak.clamp_min(0) + 1, streak)
  streak = torch.where(finished & ~success, streak.clamp_max(0) - 1, streak)
  promote = streak >= state.cfg.promote_after
  demote = streak <= -state.cfg.demote_after
  state.streak[env_ids] = torch.where(promote | demote, 0, streak)
  levels = (levels + promote.long() - demote.long()).clamp(0, len(DIFFICULTIES) - 1)
  if state.cfg.fixed_level is not None:
    levels.fill_(state.cfg.fixed_level)
  if state.cfg.fixed_column is not None:
    columns.fill_(state.cfg.fixed_column)
  elif state.cfg.mode in ("up_step", "down_step"):
    columns = columns - columns % 2 + int(state.cfg.mode == "down_step")
  terrain.terrain_levels[env_ids] = levels
  terrain.terrain_types[env_ids] = columns
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, columns]
  result = {"mean_level": terrain.terrain_levels.float().mean()}
  # Counts remain meaningful when the reset batch contains no episodes of a drill.
  for name, mask in (
    ("left_balance", state.drill[env_ids] == 0), ("right_balance", state.drill[env_ids] == 1),
    ("up_step", state.step_drill[env_ids] & (columns % 2 == 0)),
    ("down_step", state.step_drill[env_ids] & (columns % 2 == 1)),
    ("up_flight", state.flight[env_ids] & (columns % 2 == 0)),
    ("down_flight", state.flight[env_ids] & (columns % 2 == 1)),
  ):
    result[f"{name}_completed"] = (mask & finished & success).sum().float()
    result[f"{name}_attempted"] = (mask & finished).sum().float()
  return result
