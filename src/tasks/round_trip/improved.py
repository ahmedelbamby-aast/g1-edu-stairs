"""Additive curriculum, timing and perception improvements for the round-trip task."""

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from src.tasks.campus.skills import Phase, skill
from src.tasks.campus.terrain import DIFFICULTIES

from . import mdp
from .skills import RoundTripSkill, RoundTripSkillCfg, Route
from .terrain import COURSES, PACE_BANDS


@dataclass(kw_only=True)
class ImprovedSkillCfg(RoundTripSkillCfg):
  expert: str = "round_trip"
  window_batches: int = 8
  minimum_trials: int = 64
  promotion_fraction: float = .85
  demotion_fraction: float = .40
  current_fraction: float = .60
  previous_fraction: float = .30
  hard_fraction: float = .10
  estimator_bias_m: float = .015
  estimator_noise_m: float = .010
  estimator_dropout: float = .02

  def build(self, env):
    return ImprovedSkill(self, env)


class ImprovedSkill(RoundTripSkill):
  cfg: ImprovedSkillCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if cfg.expert not in ("round_trip", "ascent", "turn", "descent"):
      raise ValueError("Unknown round-trip expert")
    if abs(cfg.current_fraction + cfg.previous_fraction + cfg.hard_fraction - 1) > 1e-6:
      raise ValueError("Curriculum allocation fractions must sum to one")
    if not 0 <= cfg.demotion_fraction < cfg.promotion_fraction <= 1:
      raise ValueError("Use ordered curriculum demotion and promotion fractions")
    if cfg.window_batches < 1 or cfg.minimum_trials < 1:
      raise ValueError("Curriculum window and minimum trial count must be positive")
    if not 0 <= cfg.estimator_dropout <= 1 or min(cfg.estimator_bias_m, cfg.estimator_noise_m) < 0:
      raise ValueError("Invalid perception error settings")
    self.rehearsal_level.fill_(-1)
    if cfg.expert in ("ascent", "turn", "descent"):
      self.start_mode.fill_({"ascent": Route.ASCEND, "turn": Route.TURN, "descent": Route.DESCEND}[cfg.expert])
    self.groups = len(COURSES) * len(PACE_BANDS) * len(Route)
    self.frontier = torch.zeros(self.groups, device=self.device, dtype=torch.long)
    shape = (self.groups * len(DIFFICULTIES), cfg.window_batches)
    self.result_success = torch.zeros(shape, device=self.device)
    self.result_trials = torch.zeros(shape, device=self.device)
    self.result_cursor = torch.zeros(shape[0], device=self.device, dtype=torch.long)
    self.gait_angle = torch.zeros(self.num_envs, device=self.device)
    self.estimator_bias = torch.zeros((self.num_envs, 3), device=self.device)
    self.episode_number = torch.zeros_like(self.ids)
    self.curriculum_rate = torch.zeros(self.groups, device=self.device)
    self.curriculum_trials = torch.zeros(self.groups, device=self.device)

  def group_ids(self, env_ids):
    columns = self.terrain.terrain_types[env_ids]
    return (columns * len(PACE_BANDS) + self.speed_band[env_ids]) * len(Route) + self.start_mode[env_ids]

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self.gait_angle[env_ids] = 0.
    self.episode_number[env_ids] += 1
    self.estimator_bias[env_ids] = (
      2 * torch.rand((len(env_ids), 3), device=self.device) - 1
    ) * self.cfg.estimator_bias_m

  def advance(self):
    ascent_before = self.ascended.clone()
    super().advance()
    if self.cfg.expert == "ascent":
      self.success |= self.ascended & ~ascent_before
    speed = mdp.velocity_command(self._env, "twist")[:, 0]
    # Integrate phase so changing speed does not cause a discontinuous phase jump.
    period = (.90 - .65 * speed).clamp(.42, .90)
    active = mdp.velocity_command(self._env, "twist").norm(dim=1) > .10
    self.gait_angle[:] = torch.remainder(
      self.gait_angle + active * self._env.step_dt * (2 * torch.pi / period), 2 * torch.pi
    )
    return self.success

  def reset(self, env_ids):
    extra = {
      "target_pace_mps": self.speed_target[env_ids].mean().item(),
      "frontier_mean": self.frontier.float().mean().item(),
    }
    return {**extra, **super().reset(env_ids)}


def adaptive_curriculum(env, env_ids):
  state = skill(env)
  terrain = state.terrain
  groups = state.group_ids(env_ids)
  levels = terrain.terrain_levels[env_ids]
  finished = env.termination_manager.dones[env_ids].clone()
  # A reset before any control steps is initialization, not a failed trial.
  finished &= env.episode_length_buf[env_ids] > 0
  cell = groups * len(DIFFICULTIES) + levels
  success = torch.zeros(state.groups * len(DIFFICULTIES), device=env.device)
  trials = torch.zeros_like(success)
  success.index_add_(0, cell, (finished & state.success[env_ids]).float())
  trials.index_add_(0, cell, finished.float())
  occupied = (trials > 0).nonzero().flatten()
  cursor = state.result_cursor[occupied] % state.cfg.window_batches
  state.result_success[occupied, cursor] = success[occupied]
  state.result_trials[occupied, cursor] = trials[occupied]
  state.result_cursor[occupied] += 1
  frontier_cell = torch.arange(state.groups, device=env.device) * len(DIFFICULTIES) + state.frontier
  n = state.result_trials[frontier_cell].sum(dim=1)
  rate = state.result_success[frontier_cell].sum(dim=1) / n.clamp_min(1)
  state.curriculum_rate[:] = rate
  state.curriculum_trials[:] = n
  enough = n >= state.cfg.minimum_trials
  promote = enough & (rate >= state.cfg.promotion_fraction) & (state.frontier < len(DIFFICULTIES) - 1)
  demote = enough & (rate <= state.cfg.demotion_fraction) & (state.frontier > 0)
  state.frontier += promote.long() - demote.long()
  changed = promote | demote
  # Every move to a frontier starts a fresh evidence window for that frontier.
  new_cell = torch.arange(state.groups, device=env.device) * len(DIFFICULTIES) + state.frontier
  state.result_trials[new_cell[changed]] = 0.
  state.result_success[new_cell[changed]] = 0.
  state.result_cursor[new_cell[changed]] = 0
  frontier = state.frontier[groups]
  draw = torch.rand(len(env_ids), device=env.device)
  easy = (torch.rand_like(draw) * frontier.clamp_min(1)).long()
  hard = frontier + 1 + (torch.rand_like(draw) * (len(DIFFICULTIES) - 1 - frontier).clamp_min(1)).long()
  levels = torch.where(draw < state.cfg.current_fraction, frontier, easy)
  levels = torch.where(draw >= state.cfg.current_fraction + state.cfg.previous_fraction, hard, levels)
  levels.clamp_(0, len(DIFFICULTIES) - 1)
  if state.cfg.fixed_level is not None:
    levels.fill_(state.cfg.fixed_level)
  columns = terrain.terrain_types[env_ids]
  if state.cfg.fixed_column is not None:
    columns.fill_(state.cfg.fixed_column)
  terrain.terrain_levels[env_ids] = levels
  terrain.terrain_types[env_ids] = columns
  terrain.env_origins[env_ids] = terrain.terrain_origins[levels, columns]
  return {
    "frontier_mean": state.frontier.float().mean(),
    "frontier_completion_fraction": rate[groups].mean(),
    "completed_trials_in_window": n[groups].mean(),
    "sampled_level_mean": terrain.terrain_levels.float().mean(),
    "completed": (finished & state.success[env_ids]).sum().float(),
    "attempted": finished.sum().float(),
  }


def speed_clock(env, period, command_name):
  del period, command_name
  state = skill(env)
  angle = state.gait_angle
  return torch.stack((angle.sin(), angle.cos()), dim=1)


def next_footholds(env, estimated):
  state = skill(env)
  rows, cols = state.terrain.terrain_levels, state.terrain.terrain_types
  count = state.tiers[rows, 0].long()
  back = state.route == Route.DESCEND
  origin = state.terrain.env_origins
  feet, _, _ = state._feet()
  result = []
  for offset in (0, 1):
    step = state.riser + offset
    valid = (step < count) & (state.route != Route.TURN)
    index = torch.where(back, count - 1 - step, step).clamp(0, state.edges.shape[-1] - 2)
    next_index = torch.where(back, (index - 1).clamp_min(0), index + 1)
    edge = state.edges[rows, cols, index] * state.forward_sign
    end = state.edges[rows, cols, next_index] * state.forward_sign
    end = torch.where(back & (index == 0), .65, end)
    x = (edge + ((end - edge) / 2).clamp(max=.24)) * state.forward_sign
    support = torch.where(torch.full_like(back, offset == 0), state.support, 1 - state.support)
    y = torch.where(support == 0, -.10, .10) * state.forward_sign
    z = state.floors[rows, cols, index + (~back).long()]
    goal = torch.stack((x + origin[:, 0], y + origin[:, 1], z), dim=1)
    swing = feet[state.ids, 1 - support]
    delta = quat_apply_inverse(yaw_quat(state.robot.data.root_link_quat_w), goal - swing)
    if estimated:
      delta = delta + state.estimator_bias + torch.randn_like(delta) * state.cfg.estimator_noise_m
      valid &= torch.rand(env.num_envs, device=env.device) >= state.cfg.estimator_dropout
    result.append(torch.cat((torch.where(valid[:, None], delta.clamp(-1.5, 1.5), 0.), valid[:, None]), dim=1))
  return torch.cat(result, dim=1)


def proprioception(env):
  state = skill(env)
  robot = state.robot
  _, found, _ = state._feet()
  return torch.cat((
    robot.data.root_link_ang_vel_b, robot.data.projected_gravity_b,
    robot.data.joint_pos - robot.data.default_joint_pos, robot.data.joint_vel,
    env.action_manager.action, found,
  ), dim=1)


def estimated_skill(env, estimated):
  state = skill(env)
  command = state.command.clone()
  if estimated:
    command[:, 10:13] += state.estimator_bias + torch.randn_like(command[:, 10:13]) * state.cfg.estimator_noise_m
    valid = torch.rand_like(command[:, 13:15]) >= state.cfg.estimator_dropout
    command[:, 13:15] *= valid
  return command


def descent_overspeed(env):
  state = skill(env)
  along = state.robot.data.root_link_lin_vel_w[:, 0] * state.forward_sign
  excess = ((along - state.speed_target) / state.speed_target.clamp_min(.15)).clamp_min(0.)
  return torch.where(state.route == Route.DESCEND, excess.square().clamp(max=9.), 0.)


def controlled_transfer(env):
  state = skill(env)
  _, found, force = state._feet()
  support = force[state.ids, state.support]
  swing = force[state.ids, 1 - state.support]
  share = swing / (support + swing).clamp_min(1)
  # Reward loading the new foot smoothly, with limited downward body velocity.
  vertical = state.robot.data.root_link_lin_vel_w[:, 2]
  settled = torch.exp(-(vertical / .30).square())
  return ((state.phase == Phase.TRANSFER) & found.all(dim=1)).float() * share * settled
