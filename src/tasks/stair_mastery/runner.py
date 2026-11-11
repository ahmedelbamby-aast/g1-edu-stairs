"""Export the enhanced input contract and persist adaptive curriculum evidence."""

from src.tasks.campus.skills import skill
from src.tasks.round_trip.runner import RoundTripRunner


class MasteryRunner(RoundTripRunner):
  observation_layout = (*RoundTripRunner.observation_layout, "footholds", "fine_scan", "proprioception_history")
  observation_sizes = (*RoundTripRunner.observation_sizes, 8, 117, 475)
  teacher_sizes = {**RoundTripRunner.teacher_sizes, RoundTripRunner.observation_layout: 185}

  def extra_audit_arrays(self, state):
    return {
      **super().extra_audit_arrays(state),
      "frontier": state.frontier.cpu().numpy(),
      "estimator_bias": state.estimator_bias.cpu().numpy(),
      "episode_number": state.episode_number.cpu().numpy(),
    }

  def extra_contract(self):
    return {
      **super().extra_contract(),
      "fine_scan": {"shape_yx": [9, 13], "spacing_m": .10, "offset_m": .8, "missing_value": 1.7},
      "footholds": {"count": 2, "values_each": ["heading_x_m", "heading_y_m", "vertical_m", "valid"]},
      "history": {"steps": 5, "step_s": self.env.unwrapped.step_dt,
                  "values_each": 95, "oldest_first": True,
                  "terms": ["gyro_xyz", "gravity_xyz", "joint_pos_relative_29", "joint_vel_29",
                            "previous_action_29", "left_right_contacts"]},
    }

  def save(self, path, infos=None):
    state = skill(self.env.unwrapped)
    saved = {
      name: getattr(state, name).detach().cpu()
      for name in ("frontier", "result_success", "result_trials", "result_cursor")
    }
    super().save(path, {**(infos or {}), "mastery_curriculum": saved})

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    if load_cfg is None or load_cfg.get("iteration", False):
      saved = (infos or {}).get("mastery_curriculum")
      if saved is not None:
        state = skill(self.env.unwrapped)
        for name, value in saved.items():
          destination = getattr(state, name)
          if destination.shape != value.shape:
            raise ValueError(f"Cannot resume a different curriculum shape for {name}")
          destination.copy_(value.to(destination.device))
    return infos
