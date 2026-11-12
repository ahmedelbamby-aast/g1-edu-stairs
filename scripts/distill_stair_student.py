"""Explicit student training by imitating ascent, turn and descent teachers."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict, deque
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
from src.tasks.stair_mastery.validation import ImitationValidator
from scripts.training_metrics import atomic_json, finalize_run, update_catalog


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
  parser.add_argument("--validation-every", type=int, default=100)
  parser.add_argument("--validation-envs", type=int, default=972)
  parser.add_argument("--validation-steps", type=int, default=32)
  parser.add_argument("--validation-seed", type=int, default=1913)
  parser.add_argument("--validation-terrain-seed", type=int, default=79)
  args = parser.parse_args()
  if min(args.steps, args.num_envs, args.validation_every, args.validation_envs, args.validation_steps) < 1 or args.learning_rate <= 0:
    parser.error("Use positive steps, environment count and learning rate")
  if args.seed == args.validation_seed or args.validation_terrain_seed == 41:
    parser.error("Validation seeds and terrain must differ from training")
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
  started = time.time()
  atomic_json(args.output / "experiment.json", {"experiment_id": args.output.name,
              "data_kind": "check-only" if args.check_only else "imitation", "started_at": started,
              "stage": "student", "seed": args.seed})
  env = None
  validator = None
  status = 1
  try:
    env = ManagerBasedRlEnv(cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MasteryRunner(wrapped, asdict(agent), str(args.output), args.device)
    manifest["initial_actor"] = runner.initialize_actor(args.init_actor_onnx)
    runner.write_run_audit(args.output, 0)
    observation = wrapped.get_observations()
    teachers = ExpertPolicies((args.ascent, args.turn, args.descent), observation, asdict(agent.actor), args.device)
    actor = runner.alg.get_policy()
    actor.eval()
    validator = ImitationValidator(actor, teachers, args.device, args.validation_envs,
                                  args.validation_steps, args.validation_seed, args.validation_terrain_seed)
    manifest["validation"] = validator.settings
    if args.check_only:
      with torch.no_grad():
        target = teachers.actions(observation, skill(env).route)
        loss = imitation_loss(actor, observation, target)
        if not torch.isfinite(loss):
          raise RuntimeError("Teacher/student loss is nonfinite")
        manifest["check_loss"] = float(loss)
        manifest["validation_mse"] = validator.measure()
        weights = {key: value.clone() for key, value in actor.state_dict().items()}
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(args.device).clone()
        repeated = validator.measure()
        if abs(repeated - manifest["validation_mse"]) > 1e-6:
          raise RuntimeError("Fixed-seed validation did not reproduce its MSE")
        if not torch.equal(torch.get_rng_state(), rng) or not torch.equal(torch.cuda.get_rng_state(args.device), cuda_rng):
          raise RuntimeError("Validation changed training RNG streams")
        if any(not torch.equal(value, weights[key]) for key, value in actor.state_dict().items()):
          raise RuntimeError("Validation changed actor parameters or normalization")
        manifest["validation_isolation"] = "passed"
        atomic_json(args.output / "check-metrics.json", {"train_mse": float(loss),
                    "validation_mse": manifest["validation_mse"], "optimizer_steps": 0})
    else:
      optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
      returns = torch.zeros(env.num_envs, device=args.device)
      lengths = torch.zeros_like(returns)
      recent_rewards, recent_lengths = deque(maxlen=100), deque(maxlen=100)
      metrics = defaultdict(list)
      interval_start, interval_steps = time.time(), 0
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
            observation, reward, done, info = wrapped.step(action)
            returns += reward
            lengths += 1
            finished = done.bool()
            recent_rewards.extend(returns[finished].cpu().tolist())
            recent_lengths.extend(lengths[finished].cpu().tolist())
            returns[finished], lengths[finished] = 0., 0.
            for key, value in info.get("log", {}).items():
              metrics[key].append(float(torch.as_tensor(value).float().mean()))
          interval_steps += env.num_envs
          if (step + 1) % args.validation_every == 0 or step + 1 == args.steps:
            # Account for rollout/optimization throughput separately from validation time.
            fps = interval_steps / max(time.time() - interval_start, 1e-6)
            validation_mse = validator.measure()
            values = {key: sum(items) / len(items) for key, items in metrics.items()}
            values.update({"Perf/total_fps": fps, "Loss/learning_rate": args.learning_rate})
            if recent_rewards:
              values["Train/mean_reward"] = sum(recent_rewards) / len(recent_rewards)
              values["Train/mean_episode_length"] = sum(recent_lengths) / len(recent_lengths)
            log.write(json.dumps({"step": step + 1, "wall_time": time.time(),
                                 "elapsed_s": time.time() - started, "train_mse": float(loss.detach()),
                                 "validation_mse": validation_mse, "teacher_fraction": mix,
                                 "metrics": values}, allow_nan=False) + "\n")
            log.flush()
            metrics.clear()
            interval_start, interval_steps = time.time(), 0
      runner.save(str(args.output / "model_distilled.pt"), {"distillation": manifest})
      torch.save(optimizer.state_dict(), args.output / "imitation-optimizer.pt")
    status = 0
  finally:
    if validator is not None:
      validator.close()
    if env is not None:
      env.close()
    (args.output / "distillation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "git-commit.txt").write_text(manifest["source_commit"] + "\n", encoding="utf-8")
    (args.output / "exit-code.txt").write_text(str(status) + "\n", encoding="utf-8")
    finalize_run(args.output)
    if not args.check_only:
      update_catalog(args.output.parent)


if __name__ == "__main__":
  main()
