"""Exercise scheduling/failure/report hooks with mocked producers, never training."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.smoke_training_metrics import fixture_writer

from scripts import run_mastery_experiment
from scripts.training_metrics import atomic_json, collect, finalize_run, pending, read_json, update_catalog


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  original_run = subprocess.run
  seen = []
  failure = False

  def producer(command, **kwargs):
    nonlocal failure
    if command[0] == "git":
      return original_run(command, **kwargs)
    seen.append(command)
    if command[:2] == ["bash", "scripts/run_kaggle_mastery.sh"]:
      environment = kwargs["env"]
      run = args.output / command[2]
      run.mkdir()
      atomic_json(run / "experiment.json", {"experiment_id": environment["STAIRS_EXPERIMENT_ID"],
          "start_update": int(environment["STAIRS_START_UPDATE"]), "end_update": int(environment["STAIRS_END_UPDATE"]),
          "data_kind": "synthetic-smoke"})
      assert environment["STAIRS_TASK"] == "Unitree-G1-EDU-Stair-Expert-Ascent"
      assert environment["STAIRS_DEFER_REPORT"] == "1"
      checkpoint = run / "model_0.pt"
      checkpoint.write_text("Synthetic placeholder, not a policy", encoding="utf-8")
      (run / "exit-code.txt").write_text("1\n" if failure else "0\n", encoding="utf-8")
      (run / "gpu-samples.csv").write_text(
          "timestamp, index, memory.used [MiB], utilization.gpu [%], power.draw [W]\n"
          "1970/01/01 00:16:40.000, 0, 1234, 98, 55.5\n"
          "1970/01/01 00:16:40.001, 1, 2345, 97, 54.5\n", encoding="utf-8")
      with fixture_writer(run) as writer:
        writer.add_scalar("Loss/value", 1., 0, walltime=1000 + len(seen))
      if failure:
        raise subprocess.CalledProcessError(1, command)
    elif command[:3] == ["bash", "scripts/kaggle_stairs.sh", "evaluate-round-trip"]:
      assert command[command.index("--task") + 1] == "Unitree-G1-EDU-Stair-Expert-Ascent"
      path = Path(command[command.index("--output") + 1])
      atomic_json(path, {"suite": command[command.index("--suite") + 1], "wall_time": 1100 + len(seen),
                        "cells": [{"episodes": 2, "completions": 1, "falls": 1}]})
    else:
      raise AssertionError(f"Unexpected subprocess; refusing to execute: {command}")
    return subprocess.CompletedProcess(command, 0)

  def execute(run_id, updates=2):
    arguments = ["workflow", "--run-id", run_id, "--stage", "ascent", "--updates", str(updates),
                 "--evaluate-every", "1", "--episodes-per-cell", "2", "--init-actor-onnx", "synthetic.onnx"]
    with patch.dict(os.environ, {"STAIRS_RESULTS": str(args.output)}), patch.object(sys, "argv", arguments), \
         patch("subprocess.run", side_effect=producer):
      run_mastery_experiment.main()

  execute("synthetic-workflow")
  train_calls = [call for call in seen if call[1] == "scripts/run_kaggle_mastery.sh"]
  assert len(train_calls) == 2 and "--init-actor-onnx" in train_calls[0] and "--resume-checkpoint" in train_calls[1]
  assert len([call for call in seen if "evaluate-round-trip" in call]) == 4
  assert read_json(args.output / "synthetic-workflow-report" / "workflow-finished.json")["completed_updates"] == 2
  assert len(pending(args.output)) == 3
  telemetry, _ = collect(args.output / "synthetic-workflow-u0001")
  gpu = [row for row in telemetry if row["kind"] == "system"]
  assert len(gpu) == 6 and all(row["update"] == -1 for row in gpu)
  assert next(row["value"] for row in gpu if row["metric"] == "GPU/1/memory_mib") == 2345
  failure = True
  try:
    execute("synthetic-failure", 1)
    raise AssertionError("Workflow hid a failed producer")
  except subprocess.CalledProcessError:
    pass
  assert (args.output / "synthetic-failure-u0001" / "workflow-exit-code.txt").read_text().strip() == "1"
  assert len(pending(args.output)) == 5
  for name, metadata in (("readiness-container", {"data_kind": "readiness-checks"}), ("legacy-smoke-container", None)):
    container = args.output / name
    container.mkdir()
    (container / "exit-code.txt").write_text("0\n", encoding="utf-8")
    if metadata:
      atomic_json(container / "experiment.json", metadata)
    with fixture_writer(container / "fixtures" / "synthetic") as writer:
      writer.add_scalar("Loss/value", 1., 0, walltime=1000.)
  update_catalog(args.output)
  assert not (args.output / "experiment-comparisons").exists(), "Readiness fixtures leaked into the training catalog"
  damaged = args.output / "plot-error-backup"
  damaged.mkdir()
  (damaged / "exit-code.txt").write_text("1\n", encoding="utf-8")
  with patch("scripts.training_metrics.export_run", side_effect=RuntimeError("Synthetic plot failure")):
    try:
      finalize_run(damaged)
      raise AssertionError("Plot failure was hidden")
    except RuntimeError:
      pass
  assert (damaged / "publish-ready.json").exists() and (damaged / "metrics-error.json").exists()
  atomic_json(args.output / "smoke.json", {"status": "passed", "optimizer_steps": 0,
      "data_kind": "mocked orchestration", "checks": ["initialization_then_resume", "two_suites_each_segment",
      "expert_specific_evaluation", "comparison_report_queued", "failure_propagated_and_archived", "gpu_telemetry_csv",
      "nested_readiness_fixtures_excluded_from_training_catalog", "raw_results_queued_after_plot_failure"]})
  print("PASS mocked experiment orchestration; no training producer executed")


if __name__ == "__main__":
  main()
