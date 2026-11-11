"""Commands, rewards and reset geometry for a full stair round trip."""

from math import pi

import torch
from mjlab.envs import mdp as common
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from src.tasks.campus.mdp import next_levels
from src.tasks.campus.skills import Phase, skill
from src.tasks.campus.terrain import SPAWN_X

from .skills import Route
from .terrain import PACE_BANDS


def route_features(env):
  return skill(env).route_observation()


def velocity_command(env, command_name):
  del command_name  # Preserve the existing observation manager parameter contract.
  state = skill(env)
  turning = state.route == Route.TURN
  approach_stop = (2.0 * state.remaining()).clamp_min(0.)
  settle_speed = torch.minimum(state.speed_target, approach_stop)
  speed = torch.where(state.phase == Phase.FINISH, settle_speed, state.speed_target)
  speed = torch.where(turning, 0., speed)
  rate_limit = torch.where(turning, state.cfg.turn_max_rate, .65)
  yaw = torch.maximum(torch.minimum(1.5 * state.heading_error(), rate_limit), -rate_limit)
  return torch.stack((speed, torch.zeros_like(speed), yaw), dim=1)


def gait_clock(env, period, command_name):
  cycle = env.episode_length_buf * (env.step_dt * 2 * pi / period)
  clock = torch.stack((cycle.sin(), cycle.cos()), dim=1)
  command = velocity_command(env, command_name)
  return torch.where((command.norm(dim=1, keepdim=True) > .10), clock, 0.)


def reset_for_route(env, env_ids, pose_range):
  state = skill(env)
  practice = state.start_mode[env_ids] != Route.ASCEND
  descend = state.start_mode[env_ids] == Route.DESCEND
  _, last, top, _, _ = state.bounds()
  root = state.robot.data.default_root_state[env_ids].clone()
  root[:, :3] += state.terrain.env_origins[env_ids]
  jitter = torch.rand((len(env_ids), 3), device=env.device)
  limits = torch.tensor([pose_range[axis] for axis in ("x", "y", "yaw")], device=env.device)
  jitter = limits[:, 0] + jitter * (limits[:, 1] - limits[:, 0])
  root[:, 0] += jitter[:, 0] + torch.where(practice, last[env_ids] + state.cfg.finish_stop_m, 0.)
  root[:, 1] += jitter[:, 1]
  root[:, 2] += torch.where(practice, top[env_ids], 0.)
  yaw = jitter[:, 2] + descend * pi
  zero = torch.zeros_like(yaw)
  root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(zero, zero, yaw))
  state.robot.write_root_state_to_sim(root, env_ids=env_ids)


def lane_exit(env):
  state = skill(env)
  _, last, top, depth, _ = state.bounds()
  origin = state.terrain.env_origins
  root = state.robot.data.root_link_pos_w - origin
  half_width = state.tiers[state.terrain.terrain_levels, 1] / 2 - state.cfg.lane_margin_m
  outside = (root[:, 1].abs() > half_width) | (root[:, 0] < -SPAWN_X + .16)
  outside |= root[:, 0] > last + depth - .10
  feet, contacts, _ = state._feet()
  left_top = ((feet[..., 2] < top[:, None] - .055) & contacts).any(dim=1)
  return outside | ((state.route == Route.TURN) & ((root[:, 0] < last + .08) | left_top))


def timeout(env):
  state = skill(env)
  drill_over = env.episode_length_buf * env.step_dt >= state.cfg.drill_timeout_s
  return common.time_out(env) | ((state.start_mode != Route.ASCEND) & drill_over)


def velocity_reward(env):
  state = skill(env)
  twist = velocity_command(env, "twist")
  speed = state.robot.data.root_link_lin_vel_w
  along = speed[:, 0] * state.forward_sign
  yaw_rate = state.robot.data.root_link_ang_vel_w[:, 2]
  motion = torch.exp(-((along - twist[:, 0]).square() + speed[:, 1].square()) / .24**2)
  yaw = torch.exp(-((yaw_rate - twist[:, 2]) / .45).square())
  turn_motion = torch.exp(-speed[:, :2].square().sum(dim=1) / .22**2)
  turn_reward = .55 * turn_motion + .75 * yaw
  return torch.where(state.route == Route.TURN, turn_reward, motion + .30 * yaw)


def turn_progress(env):
  state = skill(env)
  yaw_rate = state.robot.data.root_link_ang_vel_w[:, 2]
  command = velocity_command(env, "twist")[:, 2]
  progress = (yaw_rate * command.sign() / state.cfg.turn_max_rate).clamp(-1., 1.)
  return torch.where(state.route == Route.TURN, progress, 0.)


def route_curriculum(env, env_ids):
  state = skill(env)
  terrain = state.terrain
  finished = env.termination_manager.dones[env_ids]
  success = state.success[env_ids]
  levels = next_levels(state, env_ids, finished, success)
  pinned = state.rehearsal_level[env_ids]
  levels = torch.where(pinned >= 0, pinned, levels)
  if state.cfg.fixed_level is not None:
    levels.fill_(state.cfg.fixed_level)
  columns = terrain.terrain_types[env_ids]
  if state.cfg.fixed_column is not None:
    columns.fill_(state.cfg.fixed_column)
  terrain.terrain_levels[env_ids] = levels
  terrain.terrain_types[env_ids] = columns
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, columns]
  result = {"mean_level": terrain.terrain_levels.float().mean()}
  full = state.start_mode[env_ids] == Route.ASCEND
  for index, label in enumerate(PACE_BANDS):
    pace = state.speed_band[env_ids] == index
    result[f"{label}_full_completed"] = (full & pace & finished & success).sum().float()
    result[f"{label}_full_attempted"] = (full & pace & finished).sum().float()
  for route in (Route.TURN, Route.DESCEND):
    drill = state.start_mode[env_ids] == route
    result[f"{route.name.lower()}_drill_completed"] = (drill & finished & success).sum().float()
    result[f"{route.name.lower()}_drill_attempted"] = (drill & finished).sum().float()
  return result
