"""Progress, failure and landing signals for straight stair flights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .terrain import FINISH_X

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

LANE_LIMIT = 0.51


def foot_height_from_pelvis(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  return robot.data.site_pos_w[:, asset_cfg.site_ids, 2] - robot.data.root_link_pos_w[:, 2:3]


def out_of_lane(env: ManagerBasedRlEnv) -> torch.Tensor:
  root = env.scene["robot"].data.root_link_pos_w
  pos = root - env.scene.env_origins
  return (pos[:, 1].abs() > LANE_LIMIT) | (pos[:, 0] < -0.40)


def body_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (env.scene["body_ground_contact"].data.found > 0).any(dim=1)


def reached_landing(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  feet = robot.data.site_pos_w[:, asset_cfg.site_ids, :] - env.scene.env_origins[:, None, :]
  on_landing = (feet[..., 0] > FINISH_X - 0.25) & (feet[..., 1].abs() < LANE_LIMIT)
  supported = (env.scene["feet_ground_contact"].data.found > 0).all(dim=1)
  upright = robot.data.projected_gravity_b[:, 2] < -0.85
  ahead = robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0] > FINISH_X
  return ahead & on_landing.all(dim=1) & supported & upright & ~out_of_lane(env) & ~body_hit(env)


def track_course_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Track horizontal speed along the flight, allowing necessary vertical motion."""
  robot = env.scene["robot"]
  target = env.command_manager.get_command(command_name)[:, 0]
  velocity = robot.data.root_link_lin_vel_w
  error = (velocity[:, 0] - target).square() + velocity[:, 1].square()
  return torch.exp(-error / 0.30**2)


def signed_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  speed = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  target = env.command_manager.get_command(command_name)[:, 0]
  return (speed / target).clamp(-1.0, 1.0)


def centerline_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  lateral = env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
  return lateral.square()


def success_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  manager = env.termination_manager
  if "landing" not in manager.active_terms:
    return torch.zeros_like(manager.terminated)
  return manager.get_term("landing")


def failure_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated & ~success_bonus(env)


def flight_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
  """Promote completed flights and repeat near misses before demoting."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  success = success_bonus(env)[env_ids]
  x = env.scene["robot"].data.root_link_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
  timed_out_early = env.termination_manager.time_outs[env_ids] & (x < 0.65 * FINISH_X)
  failure = failure_cost(env)[env_ids] | timed_out_early
  # Keep successful environments on the final row.
  terrain.update_env_origins(
    env_ids,
    success & (terrain.terrain_levels[env_ids] < terrain.max_terrain_level - 1),
    failure & ~success,
  )
  return {
    "level_mean": terrain.terrain_levels.float().mean(),
    "landing_fraction": success.float().mean(),
  }
