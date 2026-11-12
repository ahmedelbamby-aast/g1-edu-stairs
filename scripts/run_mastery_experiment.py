"""Explicitly launch a staged 500-update experiment with evaluation checkpoints."""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.training_metrics import SAFE_ID, atomic_json, compare, finalize_run, queue, update_catalog

STAGES = {
  "ascent": "Unitree-G1-EDU-Stair-Expert-Ascent",
  "descent": "Unitree-G1-EDU-Stair-Expert-Descent",
  "turn": "Unitree-G1-EDU-Stair-Expert-Turn",
  "round_trip": "Unitree-G1-EDU-Stair-Mastery",
  "student": "Unitree-G1-EDU-Stair-Student",
}


def evaluate_checkpoint(task, checkpoint, segment, episodes, completed_updates):
  def run_suite(suite, gpu):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    output = segment / f"eval-{suite}.json"
    command = ["bash", "scripts/kaggle_stairs.sh", "evaluate-round-trip", "--task", task,
               "--checkpoint", str(checkpoint), "--onnx", str(checkpoint.parent / "policy.onnx"),
               "--suite", suite, "--episodes-per-cell", str(episodes), "--output", str(output)]
    with (segment / f"eval-{suite}.log").open("w", encoding="utf-8") as log:
      subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    result["completed_updates"] = completed_updates
    atomic_json(output, result)

  with ThreadPoolExecutor(max_workers=2) as pool:
    clean = pool.submit(run_suite, "clean", 0)
    stress = pool.submit(run_suite, "stress", 1)
    clean.result()
    stress.result()


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
  parser.add_argument("--performance-options", action="store_true")
  parser.add_argument("--plan-only", action="store_true")
  args = parser.parse_args()
  if not SAFE_ID.fullmatch(args.run_id):
    parser.error("Use letters, numbers, hyphens and underscores for the run ID")
  if min(args.updates, args.evaluate_every) < 1 or args.episodes_per_cell < 2 or args.episodes_per_cell % 2:
    parser.error("Use positive update counts and an even episode count of at least two")
  print(json.dumps({
    "stage": args.stage, "task": STAGES[args.stage], "updates": args.updates,
    "evaluate_every": args.evaluate_every, "seed": args.seed,
    "run_prefix": args.run_id, "plan_only": args.plan_only, "performance_options": args.performance_options,
  }, indent=2))
  if args.plan_only:
    return
  root = Path(os.environ.get("STAIRS_RESULTS", "/kaggle/working/g1-edu-stairs"))
  report = root / f"{args.run_id}-report"
  report.mkdir(parents=True, exist_ok=False)
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  (report / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
  atomic_json(report / "experiment.json", {"experiment_id": args.run_id, "stage": args.stage,
              "seed": args.seed, "planned_updates": args.updates, "started_at": time.time(),
              "data_kind": "experiment-comparison", "performance_options": args.performance_options,
              "initial_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
              "initial_actor": str(args.init_actor_onnx) if args.init_actor_onnx else None})
  remaining = args.updates
  checkpoint = args.resume_checkpoint
  completed = 0
  segments = []
  while remaining:
    updates = min(remaining, args.evaluate_every)
    run_id = f"{args.run_id}-u{completed + updates:04d}"
    environment = {**os.environ, "STAIRS_TASK": STAGES[args.stage],
                   "ITERATIONS": str(updates), "SEED": str(args.seed),
                   "STAIRS_EXPERIMENT_ID": args.run_id, "STAIRS_START_UPDATE": str(completed),
                   "STAIRS_END_UPDATE": str(completed + updates), "STAIRS_STAGE": args.stage,
                   "STAIRS_DEFER_REPORT": "1"}
    initialization = (
      ["--resume-checkpoint", str(checkpoint), "--resume-after-saved-update", "True"] if checkpoint else
      ["--init-actor-onnx", str(args.init_actor_onnx)]
    )
    status = 1
    performance = ["--record-performance", "True"]
    if args.performance_options:
      performance.extend(["--omit-unused-teacher-observations", "True", "--branchless-history", "True",
                          "--fast-foot-positions", "True"])
    try:
      subprocess.run(["bash", "scripts/run_kaggle_mastery.sh", run_id, *initialization, *performance],
                     env=environment, check=True)
      checkpoints = list((root / run_id).rglob("model_*.pt"))
      checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
      evaluate_checkpoint(STAGES[args.stage], checkpoint, root / run_id, args.episodes_per_cell, completed + updates)
      status = 0
    finally:
      segment = root / run_id
      if segment.exists():
        segments.append(segment)
        (segment / "workflow-exit-code.txt").write_text(str(status) + "\n", encoding="utf-8")
        finalize_run(segment)
        compare(segments, report / "metrics")
        (report / "exit-code.txt").write_text(str(status) + "\n", encoding="utf-8")
        queue(report)
        update_catalog(root)
    remaining -= updates
    completed += updates
  atomic_json(report / "workflow-finished.json", {"completed_updates": completed, "wall_time": time.time(), "status": "complete"})
  queue(report)


if __name__ == "__main__":
  main()
