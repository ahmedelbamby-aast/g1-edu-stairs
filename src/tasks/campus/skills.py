"""Contact-driven skill phases shared by balance, step and flight training."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import torch
import torch.nn.functional as F
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from .terrain import COURSES, DIFFICULTIES, MAX_RISERS, SPAWN_X
from .terrain import course_layout


class Phase(IntEnum):
  APPROACH = 0
  SHIFT = 1
  LIFT = 2
  CLEAR = 3
  PLANT = 4
  TRANSFER = 5
  HOLD = 6
  FINISH = 7


class Drill(IntEnum):
  LEFT = 0
  RIGHT = 1
  STEP = 2
  FLIGHT = 3


SkillMode = Literal["mixed", "left", "right", "up_step", "down_step", "flight"]
MODES = {"left": Drill.LEFT, "right": Drill.RIGHT, "up_step": Drill.STEP,
         "down_step": Drill.STEP, "flight": Drill.FLIGHT}
STATE_SIZE = len(Phase) + 2 + 3 + 2 + len(Drill)
STEP_SPAWN_X = .51


@dataclass(kw_only=True)
class CampusSkillCfg(CommandTermCfg):
  mode: SkillMode = "mixed"
  fixed_level: int | None = None
  fixed_column: int | None = None
  promote_after: int = 2
  demote_after: int = 2
  support_force_n: float = 35.0
  load_fraction: float = .55
  transfer_s: float = .06
  finish_s: float = .10
  drill_timeout_s: float = 8.0
  lane_margin_m: float = .16

  def build(self, env):
    return CampusSkill(self, env)


class CampusSkill(CommandTerm):
  cfg: CampusSkillCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    terrain = env.scene.terrain
    self.terrain = terrain
    self.robot = env.scene["robot"]
    self.foot_ids, names = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    if tuple(names) != ("left_foot", "right_foot"):
      raise ValueError("Campus skills require left and right G1 foot sites in SDK order")
    if cfg.fixed_level is not None and not 0 <= cfg.fixed_level < len(DIFFICULTIES):
      raise ValueError("fixed_level is outside the campus curriculum")
    if cfg.fixed_column is not None and not 0 <= cfg.fixed_column < len(COURSES):
      raise ValueError("fixed_column is outside the campus course catalog")
    if cfg.fixed_column is not None and (
      (cfg.mode == "up_step" and cfg.fixed_column % 2)
      or (cfg.mode == "down_step" and not cfg.fixed_column % 2)
    ):
      raise ValueError("The selected course has the wrong direction for this drill")

    edges = torch.full((len(DIFFICULTIES), len(COURSES), MAX_RISERS + 1), 6.8)
    floors = torch.zeros_like(edges)
    for column, tile in enumerate(terrain.cfg.terrain_generator.sub_terrains.values()):
      for level in range(len(DIFFICULTIES)):
        course_edges, heights = course_layout(tile.seed, level, column)
        edges[level, column, :len(course_edges)] = torch.as_tensor(course_edges - SPAWN_X)
        floors[level, column, :len(heights)] = torch.as_tensor(heights)
    self.edges = edges.to(self.device)
    self.floors = floors.to(self.device)
    self.tiers = torch.tensor(
      [[d.risers, d.width_m, d.hold_s, d.lift_m, d.push_mps, d.scan_dropout] for d in DIFFICULTIES],
      device=self.device,
    )
    self.ids = torch.arange(self.num_envs, device=self.device)
    mixed = torch.tensor([Drill.LEFT, Drill.RIGHT, Drill.STEP, Drill.FLIGHT, Drill.FLIGHT], device=self.device)
    self.drill = mixed[self.ids % len(mixed)] if cfg.mode == "mixed" else torch.full_like(self.ids, MODES[cfg.mode])
    self.balance = self.drill < Drill.STEP
    self.step_drill = self.drill == Drill.STEP
    self.flight = self.drill == Drill.FLIGHT
    self.phase = torch.zeros_like(self.ids)
    self.support = torch.zeros_like(self.ids)
    self.riser = torch.zeros_like(self.ids)
    self.streak = torch.zeros_like(self.ids)
    self.hold = torch.zeros_like(self.ids)
    self.max_hold = torch.zeros_like(self.ids)
    self.success = torch.zeros_like(self.balance)
    self.transition = torch.zeros_like(self.balance)
    self.reached = torch.zeros((self.num_envs, len(Phase)), device=self.device, dtype=torch.bool)
    self.last_reached = torch.zeros_like(self.reached)
    self.last_risers = torch.zeros_like(self.riser)
    self.last_hold = torch.zeros_like(self.max_hold)

  def _resample_command(self, env_ids):
    self.phase[env_ids] = torch.where(self.flight[env_ids], Phase.APPROACH, Phase.SHIFT)
    # Consecutive step and flight worlds alternate their initial support side.
    self.support[env_ids] = torch.where(self.balance[env_ids], self.drill[env_ids], env_ids % 2)
    self.riser[env_ids] = 0
    self.hold[env_ids] = 0
    self.max_hold[env_ids] = 0
    self.success[env_ids] = False
    self.transition[env_ids] = False
    self.reached[env_ids] = False

  def _update_command(self):
    pass

  def _update_metrics(self):
    pass

  def reset(self, env_ids):
    self.last_reached[env_ids] = self.reached[env_ids]
    self.last_risers[env_ids] = self.riser[env_ids]
    self.last_hold[env_ids] = self.max_hold[env_ids]
    reached = self.reached[env_ids].float()
    extras = {f"reached_{p.name.lower()}": reached[:, p].mean().item() for p in Phase}
    extras["completed_risers"] = self.riser[env_ids].float().mean().item()
    extras["best_hold_s"] = (self.max_hold[env_ids].float().mean() * self._env.step_dt).item()
    return {**extras, **super().reset(env_ids)}

  def _feet(self):
    feet = self.robot.data.site_pos_w[:, self.foot_ids]
    sensor = self._env.scene["feet_ground_contact"].data
    force = sensor.force[..., 2].abs()
    return feet, (sensor.found > 0), force

  def targets(self):
    rows, columns = self.terrain.terrain_levels, self.terrain.terrain_types
    n, width, hold_s, lift, _, _ = self.tiers[rows].unbind(dim=1)
    current = torch.minimum(self.riser, n.long() - 1)
    edge = self.edges[rows, columns, current]
    next_edge = self.edges[rows, columns, current + 1]
    start_z = self.floors[rows, columns, current]
    goal_z = self.floors[rows, columns, current + 1]
    origin = self.terrain.env_origins
    x = edge + ((next_edge - edge) / 2).clamp(max=.24)
    y = torch.where(self.support == 0, -.10, .10)
    x = torch.where(self.balance, 0.04, x)
    z = torch.where(self.balance, origin[:, 2] + lift, goal_z)
    goal = torch.stack((origin[:, 0] + x, origin[:, 1] + y, z), dim=-1)
    return goal, edge, next_edge, start_z, goal_z, n.long(), width, hold_s, lift

  @property
  def command(self):
    goal, *_ = self.targets()
    feet, found, _ = self._feet()
    swing = feet[self.ids, 1 - self.support]
    delta = quat_apply_inverse(yaw_quat(self.robot.data.root_link_quat_w), goal - swing)
    return torch.cat((
      F.one_hot(self.phase, len(Phase)), F.one_hot(self.support, 2),
      delta.clamp(-1.0, 1.0), found, F.one_hot(self.drill, len(Drill)),
    ), dim=-1)

  def advance(self):
    old = self.phase.clone()
    origin = self.terrain.env_origins
    root = self.robot.data.root_link_pos_w - origin
    velocity = self.robot.data.root_link_lin_vel_w
    feet, found, force = self._feet()
    com = self._env.sim.data.subtree_com[:, self.robot.indexing.root_body_id]
    support_xyz = feet[self.ids, self.support]
    swing = feet[self.ids, 1 - self.support]
    supported = found[self.ids, self.support] & (force[self.ids, self.support] >= self.cfg.support_force_n)
    air = ~found[self.ids, 1 - self.support]
    load = force[self.ids, self.support] / force.sum(dim=1).clamp_min(1.0)
    _, edge, next_edge, start_z, goal_z, count, width, hold_s, lift = self.targets()
    upright = self.robot.data.projected_gravity_b[:, 2] < -.87
    swing_x = swing[:, 0] - origin[:, 0]
    near_support = (com[:, 1] - support_xyz[:, 1]).abs() < .09
    shifted = supported & near_support & (load >= self.cfg.load_fraction)
    shift_done = shifted | (~self.balance & supported & air)
    lift_height = torch.where(self.balance, origin[:, 2] + lift, start_z + .020)
    lifted = air & (swing[:, 2] >= lift_height - .01)
    clear = air & (swing_x >= edge - .06) & (swing[:, 2] >= torch.maximum(start_z, goal_z) + .015)
    contact = (~air & (force[self.ids, 1 - self.support] >= self.cfg.support_force_n)
               & (swing_x >= edge + .055) & (swing_x < next_edge - .025)
               & ((swing[:, 2] - goal_z).abs() < .055)
               & ((swing[:, 1] - origin[:, 1]).abs() < width / 2 - self.cfg.lane_margin_m))
    transferable = contact & (force[self.ids, 1 - self.support] >= force.sum(dim=1) * self.cfg.load_fraction)
    on_landing = ((feet[..., 0] - origin[:, 0:1]) > (edge[:, None] + .12)).all(dim=1)
    on_landing &= ((feet[..., 2] - goal_z[:, None]).abs() < .055).all(dim=1)
    finish = found.all(dim=1) & on_landing & (root[:, 0] > edge + .53) & upright
    finish &= root[:, 1].abs() < width / 2 - self.cfg.lane_margin_m
    holding = supported & lifted & near_support & upright & (velocity[:, :2].norm(dim=1) < .28)
    keep = ((old == Phase.HOLD) & holding) | ((old == Phase.TRANSFER) & transferable) | ((old == Phase.FINISH) & finish)
    self.hold[:] = torch.where(keep, self.hold + 1, 0)
    self.max_hold[:] = torch.where(old == Phase.HOLD, torch.maximum(self.max_hold, self.hold), self.max_hold)
    transferred = (old == Phase.TRANSFER) & (self.hold * self._env.step_dt >= self.cfg.transfer_s)
    self.success[:] = (
      ((old == Phase.HOLD) & (self.hold * self._env.step_dt >= hold_s))
      | (transferred & self.step_drill)
      | ((old == Phase.FINISH) & (self.hold * self._env.step_dt >= self.cfg.finish_s))
    ) & upright

    phase = torch.where((old == Phase.APPROACH) & (root[:, 0] >= edge - .40), Phase.SHIFT, old)
    phase = torch.where((old == Phase.SHIFT) & shift_done, Phase.LIFT, phase)
    phase = torch.where((old == Phase.LIFT) & lifted, torch.where(self.balance, Phase.HOLD, Phase.CLEAR), phase)
    phase = torch.where((old == Phase.CLEAR) & clear, Phase.PLANT, phase)
    phase = torch.where((old == Phase.PLANT) & contact, Phase.TRANSFER, phase)
    self.riser += transferred.long()
    phase = torch.where(transferred & self.flight, torch.where(self.riser == count, Phase.FINISH, Phase.APPROACH), phase)
    self.support[:] = torch.where(transferred & self.flight & (self.riser < count), 1 - self.support, self.support)
    self.transition[:] = phase != old
    self.phase[:] = phase
    self.reached[self.ids, old] = True
    return self.success


def skill(env) -> CampusSkill:
  return env.command_manager.get_term("skill")
