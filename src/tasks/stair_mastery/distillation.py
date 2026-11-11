"""Teacher inference and action matching for the explicitly launched student workflow."""

from copy import deepcopy

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class ExpertPolicies:
  def __init__(self, checkpoints, observation, actor_cfg, device):
    self.models = []
    for checkpoint in checkpoints:
      saved = torch.load(checkpoint, weights_only=False, map_location=device)
      weights = saved["actor_state_dict"]
      width = weights["mlp.0.weight"].shape[1]
      if width not in (185, 785):
        raise ValueError("Teachers must use the 185-input round-trip or 785-input mastery contract")
      inputs = TensorDict({"teacher": observation["teacher"][:, :width]}, batch_size=observation.batch_size)
      cfg = deepcopy(actor_cfg)
      cfg.pop("class_name", None)
      cfg.pop("cnn_cfg", None)
      model = MLPModel(inputs, {"teacher": ["teacher"]}, "teacher", 29, **cfg).to(device)
      model.load_state_dict(weights, strict=True)
      model.eval()
      model.requires_grad_(False)
      self.models.append((width, model))
    if len(self.models) != 3:
      raise ValueError("Supply one teacher for ascent, turn, and descent")

  @torch.no_grad()
  def actions(self, observation, route):
    result = torch.zeros((len(route), 29), device=route.device)
    for index, (width, model) in enumerate(self.models):
      inputs = TensorDict({"teacher": observation["teacher"][:, :width]}, batch_size=observation.batch_size)
      result = torch.where((route == index)[:, None], model(inputs), result)
    return result


def imitation_loss(student, observation, target):
  return torch.nn.functional.mse_loss(student(observation), target.detach())
