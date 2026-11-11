"""Independent, reproducible simulation observations for student validation MSE."""

import random
from contextlib import contextmanager

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.campus.skills import skill
from src.tasks.stair_mastery import mastery_env_cfg


@contextmanager
def isolated_rng(devices):
  python_state, numpy_state = random.getstate(), np.random.get_state()
  try:
    with torch.random.fork_rng(devices=devices):
      yield
  finally:
    random.setstate(python_state)
    np.random.set_state(numpy_state)


class ImitationValidator:
  def __init__(self, actor, teachers, device, num_envs=972, steps=32, seed=1913, terrain_seed=79):
    self.actor, self.teachers = actor, teachers
    self.device, self.steps, self.seed = device, steps, seed
    self.devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with isolated_rng(self.devices):
      cfg = mastery_env_cfg(student=True, play=True, stress=True, terrain_seed=terrain_seed)
      cfg.seed, cfg.scene.num_envs, cfg.curriculum = seed, num_envs, {}
      cfg.commands["skill"].fixed_level = None
      self.env = ManagerBasedRlEnv(cfg, device=device)
      state = skill(self.env)
      cells = torch.tensor([(level, column, pace, route) for level in range(6)
                            for column in range(18) for pace in range(3) for route in range(3)], device=device)
      levels, columns, paces, routes = cells[state.ids % len(cells)].unbind(dim=1)
      state.terrain.terrain_levels[:] = levels
      state.terrain.terrain_types[:] = columns
      state.terrain.env_origins[:] = state.terrain.terrain_origins[levels, columns]
      state.speed_band[:] = paces
      state.start_mode[:] = routes
      self.wrapper = RslRlVecEnvWrapper(self.env, clip_actions=3.)
    self.settings = {"seed": seed, "terrain_seed": terrain_seed, "worlds": num_envs,
                     "steps": steps, "coverage_cells": min(num_envs, 972),
                     "separate_environment": True, "optimizer_steps": 0}

  @torch.no_grad()
  def measure(self):
    # Validation neither changes training RNG streams nor updates normalization.
    with isolated_rng(self.devices):
      random.seed(self.seed)
      np.random.seed(self.seed)
      torch.manual_seed(self.seed)
      if self.devices:
        torch.cuda.manual_seed(self.seed)
      state = skill(self.env)
      state.turn_side[:] = (state.ids % 2) * 2 - 1
      self.env.reset()
      observation = self.wrapper.get_observations()
      total, count = 0., 0
      for _ in range(self.steps):
        target = self.teachers.actions(observation, skill(self.env).route)
        prediction = self.actor(observation)
        total += (prediction - target).square().sum().item()
        count += prediction.numel()
        observation, _, _, _ = self.wrapper.step(prediction)
      value = total / count
      if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("Validation MSE is nonfinite")
      return value

  def close(self):
    self.env.close()
