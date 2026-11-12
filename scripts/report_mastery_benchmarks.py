"""Collect dual-T4 benchmark evidence and publish a compact, headless plot report."""

import argparse
import shutil
import subprocess
from pathlib import Path

from scripts.training_metrics import atomic_json, queue, read_json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--runs", type=Path, nargs="+", required=True)
  parser.add_argument("--selected-run", required=True)
  parser.add_argument("--parity", type=Path, required=True)
  args = parser.parse_args()
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = [row for run in args.runs for row in read_json(run / "benchmark.json")["trials"]]
  selected = next(row for row in rows if row["run"] == args.selected_run)
  measured = [row for row in rows if "median_transitions_per_s" in row]
  args.output.mkdir(parents=True, exist_ok=False)
  metrics = args.output / "metrics"
  plots = metrics / "plots"
  plots.mkdir(parents=True)
  atomic_json(metrics / "benchmarks.json", {"selected": selected, "trials": rows})
  shutil.copy2(args.parity, args.output / "optimization-parity.json")
  for row in measured:
    run = args.output.parent / row["run"]
    evidence = args.output / "trials" / row["run"]
    evidence.mkdir(parents=True)
    for name in ("effective-profile.env", "command.sh", "gpu-samples.csv", "runtime.txt", "git-commit.txt"):
      shutil.copy2(run / name, evidence / name)
    for samples in run.rglob("performance/rank-*.jsonl"):
      shutil.copy2(samples, evidence / samples.name)

  fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
  labels = [f'{row["worlds_per_gpu"]:,}\n{"optimized" if row["omit_unused_teacher"] else "baseline"}' for row in measured]
  colors = ["#036965" if row["run"] == args.selected_run else "#849fb7" for row in measured]
  for axis, values, title in zip(axes, (
      [row["median_transitions_per_s"] / 1000 for row in measured],
      [sum(row["mean_gpu_util_pct"]) / 2 if None not in row["mean_gpu_util_pct"] else 0 for row in measured],
      [max(row["peak_used_mib_by_rank"]) / 1024 for row in measured],
  ), ("Throughput (thousands of transitions/s)", "Mean GPU utilization (%)", "Peak sampled GPU memory (GiB)")):
    bars = axis.bar(range(len(labels)), values, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_xticks(range(len(labels)), labels, fontsize=8)
    axis.set_title(title, fontsize=9)
    axis.set_ylim(0, max(values) * 1.2)
    axis.set_xlabel("Simulated robots per GPU")
    axis.grid(axis="y", alpha=.2)
    axis.set_axisbelow(True)
  fig.suptitle("Measured PPO throughput on two Tesla T4s")
  fig.savefig(plots / "dual-t4-throughput.png", dpi=150)
  plt.close(fig)
  (plots / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Dual T4 PPO benchmarks</title>'
      '<h1>Dual T4 PPO benchmarks</h1><p>Median after three warmup updates. Green shows the selected run.</p>'
      '<img src="dual-t4-throughput.png" style="max-width:100%" alt="PPO throughput, GPU utilization and memory by workload">'
      '<p><a href="../benchmarks.json">Raw benchmark summary</a></p>', encoding="utf-8")
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  (args.output / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
  (args.output / "exit-code.txt").write_text("0\n", encoding="utf-8")
  atomic_json(args.output / "experiment.json", {"data_kind": "performance-benchmark", "selected": selected["run"]})
  queue(args.output)
  print(f"Benchmark plots and per-rank measurements queued at {args.output}")


if __name__ == "__main__":
  main()
