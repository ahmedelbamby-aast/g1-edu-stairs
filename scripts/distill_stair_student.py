"""Explicit student training by imitating ascent, turn and descent teachers."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from src.tasks.campus.skills import skill
from src.tasks.stair_mastery import mastery_env_cfg, mastery_runner_cfg
from src.tasks.stair_mastery.distillation import ExpertPolicies, imitation_loss
from src.tasks.stair_mastery.runner import MasteryRunner


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ascent", type=Path, required=True)
  parser.add_argument("--turn", type=Path, required=True)
  parser.add_argument("--descent", type=Path, required=True)
  parser.add_argument("--init-actor-onnx", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=5000)
  parser.add_argument("--num-envs", type=int, default=540)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--learning-rate", type=float, default=.0001)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--check-only", action="store_true")
  args = parser.parse_args()
  if min(args.steps, args.num_envs) < 1 or args.learning_rate <= 0:
    parser.error("Use positive steps, environment count and learning rate")
  # Training output must be a fresh directory so retries never overwrite a run.
  args.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  cfg = mastery_env_cfg(student=True)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  agent = mastery_runner_cfg()
  dump_yaml(args.output / "params" / "env.yaml", asdict(cfg))
  dump_yaml(args.output / "params" / "agent.yaml", asdict(agent))
  manifest = {
    "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "sha256": {key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
               for key in ("ascent", "turn", "descent", "init_actor_onnx")},
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "WARP_CACHE_PATH")},
    "optimizer_steps": 0,
  }
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MasteryRunner(wrapped, asdict(agent), str(args.output), args.device)
    manifest["initial_actor"] = runner.initialize_actor(args.init_actor_onnx)
    runner.write_run_audit(args.output, 0)
    observation = wrapped.get_observations()
    teachers = ExpertPolicies((args.ascent, args.turn, args.descent), observation, asdict(agent.actor), args.device)
    actor = runner.alg.get_policy()
    actor.eval()
    if args.check_only:
      with torch.no_grad():
        target = teachers.actions(observation, skill(env).route)
        loss = imitation_loss(actor, observation, target)
        if not torch.isfinite(loss):
          raise RuntimeError("Teacher/student loss is nonfinite")
        manifest["check_loss"] = float(loss)
    else:
      optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
      with (args.output / "imitation.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.steps):
          target = teachers.actions(observation, skill(env).route)
          loss = imitation_loss(actor, observation, target)
          if not torch.isfinite(loss):
            raise RuntimeError("Teacher/student loss is nonfinite")
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
          optimizer.step()
          manifest["optimizer_steps"] += 1
          with torch.no_grad():
            mix = .75 - .60 * step / max(args.steps - 1, 1)
            use_teacher = torch.rand((env.num_envs, 1), device=args.device) < mix
            action = torch.where(use_teacher, target, actor(observation))
            observation, _, _, _ = wrapped.step(action)
          if step % 100 == 0:
            log.write(json.dumps({"step": step, "mse": float(loss.detach()), "teacher_fraction": mix}) + "\n")
            log.flush()
      runner.save(str(args.output / "model_distilled.pt"), {"distillation": manifest})
      torch.save(optimizer.state_dict(), args.output / "imitation-optimizer.pt")
  finally:
    env.close()
    (args.output / "distillation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
