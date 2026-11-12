"""No-training integration checks, using labelled fixtures and existing run logs."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from scripts.training_metrics import (REQUIRED_RL, RL_FAMILIES, atomic_json, collect, compare,
                                      evaluation_metrics, export_run, pending, queue)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--historical-runs", nargs="*", type=Path, default=[])
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  fixtures = args.output / "fixtures"
  runs = []
  for run_index in range(2):
    run = fixtures / f"synthetic-{run_index}"
    run.mkdir(parents=True)
    atomic_json(run / "experiment.json", {"experiment_id": f"synthetic-experiment-{run_index}",
                "data_kind": "synthetic-smoke", "start_update": run_index * 100, "end_update": (run_index + 1) * 100})
    tags = (*REQUIRED_RL, *(prefix + "fixture" if prefix.endswith("/") or prefix.endswith("_") else prefix for prefix in RL_FAMILIES))
    with SummaryWriter(str(run)) as writer:
      for step in range(3):
        for tag in tags:
          writer.add_scalar(tag, step + .5 + run_index, step, walltime=1000 + run_index * 100 + step * 10)
    # A later event file replaces a duplicate tag/step, without dropping other metrics.
    with SummaryWriter(str(run / "restart")) as writer:
      writer.add_scalar("Loss/value", 42., 2, walltime=1090 + run_index * 100)
    cells = [{"episodes": 2, "completed_round_trips": 1, "falls": 1, "lane_exits": 0, "timeouts": 0},
             {"episodes": 10, "completed_round_trips": 9, "falls": 1, "lane_exits": 0, "timeouts": 0}]
    atomic_json(run / "eval-clean.json", {"suite": "clean", "cells": cells, "wall_time": 1200 + run_index * 100})
    summary = export_run(run)
    assert not summary["missing_expected"] and not summary["missing_rl_families"]
    assert abs(summary["evaluations"][0]["success_rate"] - 10 / 12) < 1e-12
    rows, _ = collect(run)
    losses = [row for row in rows if row["metric"] == "Loss/value"]
    assert len(losses) == 3 and losses[-1]["value"] == 42
    assert losses[0]["update"] == run_index * 100 + 1
    assert losses[-1]["elapsed_s"] == 90
    assert summary["plots"] == len(summary["metrics"])
    for path in (run / "metrics" / "plots").glob("*.png"):
      with Image.open(path) as image:
        image.verify()
    (run / "exit-code.txt").write_text("0\n", encoding="utf-8")
    queued = queue(run)
    assert pending(fixtures)
    # Simulate a failed upload: no acknowledgement means the same digest remains pending.
    assert next(item for item in pending(fixtures) if item["run_id"] == run.name)["digest"] == queued["digest"]
    command = [sys.executable, "-m", "scripts.training_metrics", "acknowledge", "--root", str(fixtures),
               "--run-id", run.name, "--digest", queued["digest"], "--url", "https://github.com/example/synthetic"]
    bad = subprocess.run([*command[:], "--digest", "0" * 64], capture_output=True)
    assert bad.returncode != 0 and not (run / "publish-receipt.json").exists()
    subprocess.run(command, check=True)
    assert not any(item["run_id"] == run.name for item in pending(fixtures))
    runs.append(run)
  # Cover the independent supervised validation curve schema without optimizer calls.
  student = fixtures / "synthetic-student"
  student.mkdir()
  atomic_json(student / "distillation.json", {"optimizer_steps": 0, "synthetic": True})
  atomic_json(student / "experiment.json", {"data_kind": "synthetic-smoke"})
  (student / "imitation.jsonl").write_text("\n".join(json.dumps({"step": step + 1, "wall_time": 1500 + step * 10,
      "train_mse": .2 / (step + 1), "validation_mse": .3 / (step + 1), "teacher_fraction": .75}) for step in range(3)) + "\n", encoding="utf-8")
  assert not export_run(student)["missing_expected"]
  compare([*runs, student], args.output / "synthetic-comparison")
  try:
    evaluation_metrics({"cells": [{"episodes": 0}]})
    raise AssertionError("Empty evaluation was accepted")
  except ValueError:
    pass
  specialized = evaluation_metrics({"cells": [{"episodes": 2, "completions": 2,
      "mean_up_mps_on_completion": None, "mean_down_mps_on_completion": None,
      "up_speed_completed_trials": 0, "down_speed_completed_trials": 0}]})
  assert specialized["success_rate"] == 1. and "mean_up_mps_on_completion" not in specialized
  weighted = evaluation_metrics({"cells": [
      {"episodes": 10, "completions": 1, "up_speed_trials": 1, "mean_up_speed_error_mps": .2},
      {"episodes": 10, "completions": 9, "up_speed_trials": 9, "mean_up_speed_error_mps": .8}]})
  assert abs(weighted["mean_up_speed_error_mps"] - .74) < 1e-12
  # A log-free check must report missing values, never fabricate a flat loss curve.
  empty = fixtures / "empty"
  empty.mkdir()
  missing = export_run(empty)
  assert missing["points"] == 0 and missing["missing_expected"] and missing["plots"] == 0
  history = []
  for source in args.historical_runs:
    destination = args.output / "historical" / source.name
    destination.mkdir(parents=True)
    for path in source.rglob("*"):
      if path.is_file() and (path.name.startswith("events.out.tfevents") or
                            (path.name.startswith("eval") and path.suffix == ".json")):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    atomic_json(destination / "experiment.json", {"experiment_id": source.name,
                "data_kind": "historical-replay", "source": str(source)})
    report = export_run(destination)
    assert report["points"] > 0 and not report["missing_expected"]
    history.append(destination)
  if history:
    compare(history, args.output / "metrics")
  atomic_json(args.output / "smoke.json", {"status": "passed", "optimizer_steps": 0,
      "synthetic_runs": 3, "historical_runs": [str(path) for path in history],
      "checks": ["all_metric_plots", "png_integrity", "weighted_validation_rates", "duplicate_event_merge",
                 "resumed_update_axis", "wall_time_axis", "train_validation_mse", "empty_missing_values",
                 "failed_upload_retained", "stale_ack_rejected", "successful_ack_deduplicated",
                 "unobserved_specialist_speed_omitted", "speed_error_trial_weighting"]})
  (args.output / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
  (args.output / "exit-code.txt").write_text("0\n", encoding="utf-8")
  queue(args.output)
  print("PASS metrics integration: zero optimizer steps")


if __name__ == "__main__":
  main()
