"""Use the same stair phase checks in both travel directions."""

from dataclasses import dataclass
from enum import IntEnum
from math import pi

import torch
import torch.nn.functional as F
from mjlab.utils.lab_api.math import wrap_to_pi

from src.tasks.campus.skills import CampusSkill, CampusSkillCfg, Phase
from src.tasks.campus.terrain import DIFFICULTIES, SPAWN_X

from .terrain import PACE_BANDS, RISERS, TURN_DEPTHS


class Route(IntEnum):
  ASCEND = 0
  TURN = 1
  DESCEND = 2


@dataclass(kw_only=True)
class RoundTripSkillCfg(CampusSkillCfg):
  mode: str = "flight"
  practice: bool = True
  fixed_speed: int | None = None
  turn_max_rate: float = .90
  turn_tolerance_rad: float = .23
  turn_dwell_s: float = .10
  turn_max_yaw_rate: float = .50
  turn_max_speed_mps: float = .23
  speed_tolerance: float = .35

  def build(self, env):
    return RoundTripSkill(self, env)


class RoundTripSkill(CampusSkill):
  cfg: RoundTripSkillCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if cfg.fixed_speed is not None and not 0 <= cfg.fixed_speed < len(PACE_BANDS):
      raise ValueError("fixed_speed must select slow (0), normal (1), or fast (2)")
    self.tiers[:, 0] = torch.tensor(RISERS, device=self.device)
    self.turn_depths = torch.tensor(TURN_DEPTHS, device=self.device)
    self.speed_bounds = torch.tensor(list(PACE_BANDS.values()), device=self.device)
    self.speed_band = self.ids % len(PACE_BANDS)
    if cfg.fixed_speed is not None:
      self.speed_band.fill_(cfg.fixed_speed)
    self.speed_target = self.speed_bounds[self.speed_band].mean(dim=1)
    slot = (self.ids // len(PACE_BANDS)) % 20
    self.start_mode = torch.zeros_like(slot)
    self.rehearsal_level = torch.full_like(slot, -1)
    if cfg.practice:
      self.start_mode[(slot >= 6) & (slot < 9)] = Route.TURN
      self.start_mode[(slot >= 9) & (slot < 12)] = Route.DESCEND
      self.rehearsal_level[slot < len(DIFFICULTIES)] = slot[slot < len(DIFFICULTIES)]
      drill = self.start_mode != Route.ASCEND
      self.rehearsal_level[drill] = (slot[drill] + self.terrain.terrain_types[drill]) % len(DIFFICULTIES)
    self.route = self.start_mode.clone()
    self.turn_side = (2 * ((self.ids // len(PACE_BANDS)) % 2) - 1).float()
    self.turn_angle = torch.zeros_like(self.speed_target)
    self.turn_yaw = torch.zeros_like(self.speed_target)
    self.turn_hold = torch.zeros_like(self.riser)
    self.ascended = torch.zeros_like(self.balance)
    self.turned = torch.zeros_like(self.balance)
    self.speed_sum = torch.zeros((self.num_envs, 2), device=self.device)
    self.speed_steps = torch.zeros_like(self.speed_sum)
    self.last_route = torch.zeros_like(self.route)
    self.last_ascended = torch.zeros_like(self.ascended)
    self.last_turned = torch.zeros_like(self.turned)
    self.last_turn_side = torch.zeros_like(self.turn_side)
    self.last_speed_target = torch.zeros_like(self.speed_target)
    self.last_speed = torch.zeros_like(self.speed_sum)
    self.last_speed_steps = torch.zeros_like(self.speed_steps)

  def bounds(self):
    rows, cols = self.terrain.terrain_levels, self.terrain.terrain_types
    count = self.tiers[rows, 0].long()
    return (
      self.edges[rows, cols, 0], self.edges[rows, cols, count - 1],
      self.floors[rows, cols, count], self.turn_depths[rows], count,
    )

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self.route[env_ids] = self.start_mode[env_ids]
    descending = self.route[env_ids] == Route.DESCEND
    self.forward_sign[env_ids] = torch.where(descending, -1., 1.)
    self.phase[env_ids] = torch.where(self.route[env_ids] == Route.TURN, Phase.FINISH, Phase.APPROACH)
    self.turn_side[env_ids] *= -1
    self.turn_angle[env_ids] = 0.
    self.turn_yaw[env_ids] = 0.
    self.turn_hold[env_ids] = 0
    self.ascended[env_ids] = False
    self.turned[env_ids] = False
    self.speed_sum[env_ids] = 0.
    self.speed_steps[env_ids] = 0.
    bounds = self.speed_bounds[self.speed_band[env_ids]]
    self.speed_target[env_ids] = bounds[:, 0] + torch.rand(len(env_ids), device=self.device) * (bounds[:, 1] - bounds[:, 0])

  def reset(self, env_ids):
    self.last_route[env_ids] = self.route[env_ids]
    self.last_ascended[env_ids] = self.ascended[env_ids]
    self.last_turned[env_ids] = self.turned[env_ids]
    self.last_turn_side[env_ids] = self.turn_side[env_ids]
    self.last_speed_target[env_ids] = self.speed_target[env_ids]
    self.last_speed[env_ids] = self.speed_sum[env_ids] / self.speed_steps[env_ids].clamp_min(1)
    self.last_speed_steps[env_ids] = self.speed_steps[env_ids]
    extras = {
      "top_landing_reached": self.ascended[env_ids].float().mean().item(),
      "turn_completed": self.turned[env_ids].float().mean().item(),
      "up_speed_mps": self.last_speed[env_ids, 0].mean().item(),
      "down_speed_mps": self.last_speed[env_ids, 1].mean().item(),
    }
    return {**extras, **super().reset(env_ids)}

  def targets(self):
    rows, cols = self.terrain.terrain_levels, self.terrain.terrain_types
    count, width, hold, lift, _, _ = self.tiers[rows].unbind(dim=1)
    count = count.long()
    step = torch.minimum(self.riser, count - 1)
    back = self.route == Route.DESCEND
    index = torch.where(back, count - 1 - step, step)
    following = torch.where(back, (index - 1).clamp_min(0), index + 1)
    edge = self.edges[rows, cols, index] * self.forward_sign
    next_edge = self.edges[rows, cols, following] * self.forward_sign
    next_edge = torch.where(back & (index == 0), SPAWN_X, next_edge)
    start_z = self.floors[rows, cols, index + back.long()]
    goal_z = self.floors[rows, cols, index + (~back).long()]
    x = (edge + ((next_edge - edge) / 2).clamp(max=.24)) * self.forward_sign
    y = torch.where(self.support == 0, -.10, .10) * self.forward_sign
    origin = self.terrain.env_origins
    turning = self.route == Route.TURN
    _, last, top, _, _ = self.bounds()
    x = torch.where(turning, last + self.cfg.finish_stop_m, x)
    goal_z = torch.where(turning, top, goal_z)
    goal = torch.stack((origin[:, 0] + x, origin[:, 1] + y, goal_z), dim=1)
    return goal, edge, next_edge, start_z, goal_z, count, width, hold, lift

  def heading_error(self):
    heading = self.robot.data.heading_w
    heading_goal = torch.where(self.route == Route.DESCEND, pi, 0.)
    error = wrap_to_pi(heading_goal - heading)
    return torch.where(self.route == Route.TURN, self.turn_side * pi - self.turn_angle, error)

  def remaining(self):
    first, last, *_ = self.bounds()
    final_edge = torch.where(self.route == Route.DESCEND, -first, last)
    distance = (self.robot.data.root_link_pos_w[:, 0] - self.terrain.env_origins[:, 0]) * self.forward_sign
    return final_edge + self.cfg.finish_stop_m - distance

  def route_observation(self):
    return torch.cat((
      F.one_hot(self.route, len(Route)),
      self.remaining().clamp(-1., 1.)[:, None],
      (self.heading_error() / pi).clamp(-1., 1.)[:, None],
    ), dim=1)

  def advance(self):
    old_route = self.route.clone()
    turn = old_route == Route.TURN
    first, last, top, depth, _ = self.bounds()
    origin = self.terrain.env_origins
    root = self.robot.data.root_link_pos_w - origin
    velocity = self.robot.data.root_link_lin_vel_w
    heading = self.robot.data.heading_w
    # Measure speed on the stairs, including any stall between the first and last edge.
    on_stairs = (root[:, 0] >= first - .04) & (root[:, 0] <= last + .06)
    up_sample = on_stairs & (old_route == Route.ASCEND)
    down_sample = on_stairs & (old_route == Route.DESCEND)
    self.speed_sum[:, 0] += up_sample * velocity[:, 0]
    self.speed_sum[:, 1] -= down_sample * velocity[:, 0]
    self.speed_steps[:, 0] += up_sample
    self.speed_steps[:, 1] += down_sample

    super().advance()
    ascend_done = (old_route == Route.ASCEND) & self.success
    descent_done = (old_route == Route.DESCEND) & self.success
    self.ascended |= ascend_done
    self.route[ascend_done] = Route.TURN
    self.turn_yaw[:] = torch.where(ascend_done, heading, self.turn_yaw)
    self.turn_angle[:] = torch.where(ascend_done, heading, self.turn_angle)
    yaw_delta = wrap_to_pi(heading - self.turn_yaw)
    self.turn_angle += torch.where(turn, yaw_delta, 0.)
    self.turn_yaw[:] = torch.where(turn, heading, self.turn_yaw)

    feet, found, _ = self._feet()
    foot_x = feet[..., 0] - origin[:, 0:1]
    feet_on_top = ((foot_x > last[:, None] + .07) & (foot_x < (last + depth - .08)[:, None])).all(dim=1)
    feet_on_top &= ((feet[..., 2] - top[:, None]).abs() < .055).all(dim=1) & found.all(dim=1)
    aligned = (wrap_to_pi(pi - heading).abs() <= self.cfg.turn_tolerance_rad)
    rotated = (self.turn_side * self.turn_angle >= pi - self.cfg.turn_tolerance_rad)
    settled = (
      (velocity[:, :2].norm(dim=1) <= self.cfg.turn_max_speed_mps)
      & (self.robot.data.root_link_ang_vel_w[:, 2].abs() <= self.cfg.turn_max_yaw_rate)
      & (self.robot.data.projected_gravity_b[:, 2] < -.87)
    )
    self.turn_hold[:] = torch.where(turn & feet_on_top & aligned & rotated & settled, self.turn_hold + 1, 0)
    turn_done = turn & (self.turn_hold * self._env.step_dt >= self.cfg.turn_dwell_s)
    self.turned |= turn_done
    start_return = turn_done & (self.start_mode == Route.ASCEND)
    self.route[start_return] = Route.DESCEND
    self.forward_sign[start_return] = -1.
    self.riser[start_return] = 0
    self.hold[start_return] = 0
    self.phase[start_return] = Phase.APPROACH
    self.transition |= ascend_done | turn_done
    self.success[:] = descent_done | (turn_done & (self.start_mode == Route.TURN))
    return self.success
