"""Real dual-GPU PPO throughput trials. This script performs training updates."""

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.training_metrics import atomic_json


def summarize(run, worlds, threads, omit_teacher, warmup):
  files = sorted(run.rglob("performance/rank-*.jsonl"))
  ranks = [list(map(json.loads, path.read_text().splitlines())) for path in files]
  if len(ranks) != 2 or not all(len(rank) > warmup for rank in ranks):
    raise ValueError("Missing dual-GPU performance samples")
  samples = [[row for row in rank if row["iteration"] >= warmup] for rank in ranks]
  count = min(map(len, samples))
  durations = [max(rank[i]["collection_s"] + rank[i]["update_s"] for rank in samples) for i in range(count)]
  first = min(rank[0]["wall_time"] for rank in samples)
  last = max(rank[-1]["wall_time"] for rank in samples)
  telemetry = []
  with (run / "gpu-samples.csv").open() as stream:
    for row in csv.DictReader(stream):
      row = {key.strip(): value.strip() for key, value in row.items()}
      stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
      if first <= stamp <= last:
        telemetry.append(row)
  result = {"run": run.name, "worlds_per_gpu": worlds, "total_worlds": worlds * 2,
            "cpu_threads_per_process": threads, "omit_unused_teacher": omit_teacher,
            "measured_iterations": count, "warmup_iterations_excluded": warmup,
            "median_iteration_s": statistics.median(durations),
            "median_transitions_per_s": statistics.median(worlds * 2 * 24 / value for value in durations),
            "collection_s_by_rank": [statistics.median(row["collection_s"] for row in rank) for rank in samples],
            "update_s_by_rank": [statistics.median(row["update_s"] for row in rank) for rank in samples],
            "peak_used_mib_by_rank": [max(row["gpu_used_mib"] for row in rank) for rank in ranks],
            "minimum_free_mib_by_rank": [min(row["gpu_free_mib"] for row in rank) for rank in ranks],
            "mean_gpu_util_pct": [statistics.mean(float(row["utilization.gpu [%]"]) for row in telemetry if int(row["index"]) == rank)
                                  if any(int(row["index"]) == rank for row in telemetry) else None for rank in range(2)],
            "profiling_instrumentation": False}
  result["memory_headroom_pass"] = min(result["minimum_free_mib_by_rank"]) >= 1536
  atomic_json(run / "throughput.json", result)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--prefix", required=True)
  parser.add_argument("--teacher-onnx", type=Path, required=True)
  parser.add_argument("--worlds", type=int, nargs="+", default=[1080, 3240, 6480])
  parser.add_argument("--threads", type=int, default=2)
  parser.add_argument("--updates", type=int, default=12)
  parser.add_argument("--omit-unused-teacher", action="store_true")
  parser.add_argument("--timeout", type=int, default=600)
  args = parser.parse_args()
  root = Path(os.environ.get("STAIRS_RESULTS", "/kaggle/working/g1-edu-stairs"))
  output = root / f"{args.prefix}-summary"
  output.mkdir(parents=True, exist_ok=False)
  results = []
  for worlds in args.worlds:
    run_id = f"{args.prefix}-n{worlds}-t{args.threads}"
    environment = {**os.environ, "NUM_ENVS": str(worlds), "CPU_THREADS": str(args.threads),
                   "ITERATIONS": str(args.updates), "SAVE_INTERVAL": str(args.updates),
                   "STAIRS_DEFER_REPORT": "1", "STAIRS_DATA_KIND": "performance-benchmark"}
    command = ["bash", "scripts/run_kaggle_mastery.sh", run_id, "--init-actor-onnx", str(args.teacher_onnx),
               "--record-performance", "True", "--omit-unused-teacher-observations", str(args.omit_unused_teacher)]
    with (output / f"{run_id}.launch.log").open("w") as log:
      process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
      try:
        status = process.wait(timeout=args.timeout)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=20)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
        status = 124
    if status:
      result = {"run": run_id, "status": "failed", "exit_code": status}
    else:
      result = summarize(root / run_id, worlds, args.threads, args.omit_unused_teacher, 3)
    results.append(result)
    atomic_json(output / "benchmark.json", {"trials": results})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
  main()
