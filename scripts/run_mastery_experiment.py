"""Explicitly launch a staged 500-update experiment with evaluation checkpoints."""

import argparse
import json
import os
import subprocess
from pathlib import Path

STAGES = {
  "ascent": "Unitree-G1-EDU-Stair-Expert-Ascent",
  "descent": "Unitree-G1-EDU-Stair-Expert-Descent",
  "turn": "Unitree-G1-EDU-Stair-Expert-Turn",
  "round_trip": "Unitree-G1-EDU-Stair-Mastery",
  "student": "Unitree-G1-EDU-Stair-Student",
}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=STAGES, default="round_trip")
  parser.add_argument("--seed", type=int, default=42)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--init-actor-onnx", type=Path)
  source.add_argument("--resume-checkpoint", type=Path)
  parser.add_argument("--updates", type=int, default=500)
  parser.add_argument("--evaluate-every", type=int, default=100)
  parser.add_argument("--episodes-per-cell", type=int, default=8)
  parser.add_argument("--plan-only", action="store_true")
  args = parser.parse_args()
  if not args.run_id.replace("-", "").replace("_", "").isalnum():
    parser.error("Use letters, numbers, hyphens and underscores for the run ID")
  if min(args.updates, args.evaluate_every) < 1 or args.episodes_per_cell < 2 or args.episodes_per_cell % 2:
    parser.error("Use positive update counts and an even episode count of at least two")
  print(json.dumps({
    "stage": args.stage, "task": STAGES[args.stage], "updates": args.updates,
    "evaluate_every": args.evaluate_every, "seed": args.seed,
    "run_prefix": args.run_id, "plan_only": args.plan_only,
  }, indent=2))
  if args.plan_only:
    return
  root = Path(os.environ.get("STAIRS_RESULTS", "/kaggle/working/g1-edu-stairs"))
  remaining = args.updates
  checkpoint = args.resume_checkpoint
  completed = 0
  while remaining:
    updates = min(remaining, args.evaluate_every)
    run_id = f"{args.run_id}-u{completed + updates:04d}"
    environment = {**os.environ, "STAIRS_TASK": STAGES[args.stage],
                   "ITERATIONS": str(updates), "SEED": str(args.seed)}
    initialization = (
      ["--resume-checkpoint", str(checkpoint)] if checkpoint else
      ["--init-actor-onnx", str(args.init_actor_onnx)]
    )
    subprocess.run(["bash", "scripts/run_kaggle_mastery.sh", run_id, *initialization],
                   env=environment, check=True)
    checkpoints = list((root / run_id).rglob("model_*.pt"))
    checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
    for suite in ("clean", "stress"):
      task = "Unitree-G1-EDU-Stair-Student" if args.stage == "student" else STAGES["round_trip"]
      subprocess.run([
        "bash", "scripts/kaggle_stairs.sh", "evaluate-round-trip", "--task", task,
        "--checkpoint", str(checkpoint), "--onnx", str(checkpoint.parent / "policy.onnx"),
        "--suite", suite, "--episodes-per-cell", str(args.episodes_per_cell),
        "--output", str(root / run_id / f"eval-{suite}.json"),
      ], check=True)
    remaining -= updates
    completed += updates


if __name__ == "__main__":
  main()
