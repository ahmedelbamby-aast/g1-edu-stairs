"""Repeat the entire no-training verification using scripts alone.

Default: local reporting/orchestration checks. --gpu adds VM-only physics and
hardware checks. --full-evaluation also measures two held-out evaluation suites.
No branch invokes an optimizer or a training launcher.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.training_metrics import atomic_json, queue, read_json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--historical-runs", type=Path, nargs="*", default=[])
  parser.add_argument("--gpu", action="store_true")
  parser.add_argument("--teacher-onnx", type=Path)
  parser.add_argument("--teacher-checkpoint", type=Path)
  parser.add_argument("--full-evaluation", action="store_true")
  args = parser.parse_args()
  if args.full_evaluation and not args.gpu:
    parser.error("Full physics evaluation requires --gpu")
  if args.gpu and (not args.teacher_onnx or not args.teacher_checkpoint):
    parser.error("GPU checks need the source ONNX and checkpoint")
  args.output.mkdir(parents=True, exist_ok=False)
  logs = args.output / "logs"
  logs.mkdir()
  environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
  if args.gpu:
    environment.update(MUJOCO_GL="egl", NCCL_P2P_DISABLE="1",
                       WARP_CACHE_PATH="/codex_download_here_work_here/cache/warp",
                       MPLCONFIGDIR="/codex_download_here_work_here/cache/matplotlib",
                       LD_LIBRARY_PATH="/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""))
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  (args.output / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
  checks = []
  status = 1

  def check(name, module, *arguments):
    print(f"Checking {name} (inference/fixtures only)", flush=True)
    command = [sys.executable, "-m", module, *map(str, arguments)]
    with (logs / f"{name}.log").open("w", encoding="utf-8") as stream:
      subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=True)
    checks.append(name)

  try:
    history = [item for path in args.historical_runs for item in [str(path)]]
    check("metrics", "scripts.smoke_training_metrics", "--output", args.output / "metric-checks",
          *( ["--historical-runs", *history] if history else []))
    check("workflow", "scripts.smoke_experiment_workflow", "--output", args.output / "workflow-checks")
    if args.gpu:
      # Concurrent processes check both GPUs at the real 1,080-world profile.
      processes = []
      streams = []
      try:
        for gpu in (0, 1):
          stream = (logs / f"capacity-{gpu}.log").open("w", encoding="utf-8")
          streams.append(stream)
          processes.append(subprocess.Popen([sys.executable, "-m", "scripts.preflight_training",
              "--output", str(args.output / "capacity"), "--teacher-onnx", str(args.teacher_onnx),
              "--device", f"cuda:{gpu}"], stdout=stream, stderr=subprocess.STDOUT, env=environment))
        for process in processes:
          if process.wait(timeout=240) != 0:
            raise RuntimeError("Concurrent GPU capacity check failed")
      finally:
        for process in processes:
          if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        for stream in streams:
          stream.close()
      checks.append("concurrent-capacity")
      check("collective", "scripts.check_gpu_collective", "--output", args.output / "capacity")
      check("phases", "scripts.smoke_stair_mastery", "--output", args.output / "phase-checks",
            "--teacher-onnx", args.teacher_onnx, "--teacher-checkpoint", args.teacher_checkpoint)
      check("student", "scripts.distill_stair_student", "--output", args.output / "student-checks",
            "--ascent", args.teacher_checkpoint, "--turn", args.teacher_checkpoint, "--descent", args.teacher_checkpoint,
            "--init-actor-onnx", args.teacher_onnx, "--num-envs", "12", "--check-only")
      if read_json(args.output / "student-checks" / "distillation.json")["optimizer_steps"] != 0:
        raise RuntimeError("A student check unexpectedly reported optimizer steps")
      if args.full_evaluation:
        directory = args.output / "phase-checks" / "Unitree-G1-EDU-Stair-Mastery" / "checkpoint-check"
        for goal in ("Mastery", "Expert-Turn"):
          check(f"evaluation-{goal}", "scripts.evaluate_round_trip", "--task", f"Unitree-G1-EDU-Stair-{goal}",
                "--checkpoint", directory / "model_smoke.pt", "--onnx", directory / "policy.onnx",
                "--episodes-per-cell", "2", "--suite", "clean", "--output", args.output / "evaluation" / f"eval-{goal}.json")
    status = 0
  finally:
    report = {"status": "passed" if status == 0 else "failed", "optimizer_steps": 0, "checks_passed": checks,
              "gpu_checks": "passed" if args.gpu and status == 0 else "not_completed",
              "training_ready": args.gpu and status == 0, "full_evaluation": args.full_evaluation,
              "source_commit": commit, "wall_time": time.time(), "github_upload": "queued_for_windows_publisher"}
    atomic_json(args.output / "readiness.json", report)
    atomic_json(args.output / "experiment.json", {"data_kind": "readiness-checks", "source_commit": commit})
    (args.output / "exit-code.txt").write_text(str(status) + "\n", encoding="utf-8")
    if (args.output / "metric-checks" / "metrics").exists():
      shutil.copytree(args.output / "metric-checks" / "metrics", args.output / "metrics")
    queue(args.output)
  print("Readiness checks completed; training was not started", flush=True)


if __name__ == "__main__":
  main()
