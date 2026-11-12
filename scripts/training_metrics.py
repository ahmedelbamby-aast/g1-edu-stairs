"""Export real TensorBoard/JSONL measurements, plot runs, and queue recovery uploads.

This module never trains a model. Missing measurements remain missing.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ("experiment", "run", "kind", "metric", "source_step", "update", "wall_time", "elapsed_s", "value")
REQUIRED_RL = ("Loss/value", "Loss/surrogate", "Loss/entropy", "Loss/learning_rate",
               "Train/mean_reward", "Train/mean_episode_length", "Perf/total_fps")
RL_FAMILIES = ("Episode_Termination/", "Metrics/skill/reached_", "Metrics/skill/completed_risers",
               "Metrics/skill/best_hold_s", "Metrics/skill/up_speed_mps", "Metrics/skill/down_speed_mps",
               "Metrics/slip_velocity_mean", "Metrics/landing_force_mean", "Curriculum/")
SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def atomic_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  os.replace(temporary, path)


def read_json(path, default=None):
  return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def finite(value):
  value = float(value)
  if not math.isfinite(value):
    raise ValueError("A metric contains NaN or infinity")
  return value


def evaluation_metrics(report):
  """Pool counts, never average per-cell percentages with unequal trial counts."""
  cells = report["cells"]
  total = sum(int(cell["episodes"]) for cell in cells)
  if total < 1:
    raise ValueError("Evaluation has no trials")
  result = {"episodes": total}
  counts = {
    "success_rate": ("completed_round_trips", "completions", "successes"),
    "fall_rate": ("falls",), "lane_exit_rate": ("lane_exits", "out_of_lane"),
    "timeout_rate": ("timeouts",), "top_landing_rate": ("top_landings",),
    "turn_rate": ("turns",), "pace_success_rate": ("completed_at_target_pace_both_directions",),
  }
  for metric, keys in counts.items():
    if all(any(key in cell for key in keys) for cell in cells):
      numerator = sum(next(int(cell[key]) for key in keys if key in cell) for cell in cells)
      if not 0 <= numerator <= total:
        raise ValueError(f"Invalid evaluation count for {metric}")
      result[metric] = numerator / total
  for field in ("mean_risers_completed", "mean_best_single_support_s", "mean_slip_velocity_mps",
                "mean_landing_force_n", "mean_up_speed_error_mps", "mean_down_speed_error_mps"):
    available = [cell for cell in cells if cell.get(field) is not None]
    weight = {"mean_up_speed_error_mps": "up_speed_trials", "mean_down_speed_error_mps": "down_speed_trials"}.get(field, "episodes")
    n = sum(cell.get(weight, cell["episodes"]) for cell in available)
    if n:
      result[field] = sum(finite(cell[field]) * cell.get(weight, cell["episodes"]) for cell in available) / n
  for field in ("mean_up_mps_on_completion", "mean_down_mps_on_completion"):
    available = [cell for cell in cells if cell.get(field) is not None]
    weight = "up_speed_completed_trials" if field.startswith("mean_up") else "down_speed_completed_trials"
    n = sum(cell.get(weight, cell.get("completed_round_trips", cell.get("completions", 0))) for cell in available)
    if n:
      result[field] = sum(finite(cell[field]) * cell.get(weight, cell.get("completed_round_trips", cell.get("completions", 0))) for cell in available) / n
  for metric, numerator, denominator in (("mean_slip_velocity_mps", "slip_velocity_sum_mps", "contact_samples"),
                                         ("mean_landing_force_n", "landing_force_sum_n", "landing_samples")):
    if all(numerator in cell and denominator in cell for cell in cells):
      count = sum(cell[denominator] for cell in cells)
      if count:
        result[metric] = sum(finite(cell[numerator]) for cell in cells) / count
  phases = sorted({phase for cell in cells for phase in cell.get("phase_reached_counts", {})})
  for phase in phases:
    result[f"phase_{phase}_rate"] = sum(cell.get("phase_reached_counts", {}).get(phase, 0) for cell in cells) / total
  return result


def collect(run):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  meta = read_json(run / "experiment.json", {})
  experiment = meta.get("experiment_id", run.name)
  raw = []
  # Multiple event files from restarts are merged by tag/step; latest wall time wins.
  deduplicated = {}
  for path in sorted(run.rglob("events.out.tfevents*")):
    if "metrics" in path.relative_to(run).parts:
      continue
    events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    for tag in events.Tags()["scalars"]:
      if tag.endswith("/time"):
        continue  # These tags use seconds as their step axis; regular tags already have wall time.
      for point in events.Scalars(tag):
        key = (tag, point.step)
        if key not in deduplicated or point.wall_time > deduplicated[key][0]:
          deduplicated[key] = (point.wall_time, finite(point.value))
  first_step = min((step for _, step in deduplicated), default=0)
  offset = int(meta.get("start_update", 0))
  for (tag, step), (stamp, value) in deduplicated.items():
    raw.append(("training", tag, step, offset + step - first_step + 1, stamp, value))
  for path in run.rglob("imitation.jsonl"):
    for line in path.read_text(encoding="utf-8").splitlines():
      item = json.loads(line)
      stamp = finite(item["wall_time"])
      values = {"Imitation/train_mse": item.get("train_mse", item.get("mse")),
                "Imitation/validation_mse": item.get("validation_mse"),
                "Imitation/teacher_fraction": item.get("teacher_fraction"), **item.get("metrics", {})}
      for tag, value in values.items():
        if value is not None:
          raw.append(("validation" if tag.endswith("validation_mse") else "training", tag,
                      int(item["step"]), offset + int(item["step"]), stamp, finite(value)))
  fallback_update = max((row[3] for row in raw), default=offset)
  checks = read_json(run / "check-metrics.json", {})
  for tag, value in checks.items():
    if tag != "optimizer_steps":
      raw.append(("check-only", f"Check/{tag}", 0, 0, (run / "check-metrics.json").stat().st_mtime, finite(value)))
  evaluations = []
  for path in sorted(run.rglob("eval*.json")):
    report = read_json(path)
    if not isinstance(report, dict) or "cells" not in report:
      continue
    suite = report.get("suite", "heldout")
    stamp = finite(report.get("wall_time", path.stat().st_mtime))
    update = int(report.get("completed_updates", meta.get("end_update", fallback_update)))
    values = evaluation_metrics(report)
    evaluations.append({"file": str(path.relative_to(run)), "suite": suite, "update": update,
                        "seed": report.get("seed"), "terrain_seed": report.get("terrain_seed"), **values})
    for tag, value in values.items():
      raw.append(("validation", f"Validation/{suite}/{tag}", update, update, stamp, finite(value)))
  gpu = run / "gpu-samples.csv"
  if gpu.exists():
    with gpu.open(encoding="utf-8") as stream:
      for row in csv.DictReader(stream):
        row = {key.strip(): value.strip() for key, value in row.items()}
        stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        for field, tag in (("memory.used [MiB]", "memory_mib"), ("utilization.gpu [%]", "utilization_pct"),
                           ("power.draw [W]", "power_w")):
          if row.get(field) and row[field] not in ("[N/A]", "N/A"):
            raw.append(("system", f"GPU/{row['index']}/{tag}", -1, -1, stamp, finite(row[field])))
  start = min((row[4] for row in raw), default=time.time())
  rows = [dict(zip(FIELDS, (experiment, run.name, kind, tag, step, update, stamp, stamp - start, value)))
          for kind, tag, step, update, stamp, value in raw]
  rows.sort(key=lambda row: (row["metric"], row["wall_time"], row["source_step"]))
  tags = {row["metric"] for row in rows}
  expected = (("Check/train_mse", "Check/validation_mse") if checks else
              ("Imitation/train_mse", "Imitation/validation_mse")) if (run / "distillation.json").exists() else REQUIRED_RL
  summary = {"run": run.name, "experiment": experiment, "metadata": meta,
             "generated_at": time.time(), "points": len(rows), "metrics": sorted(tags),
             "missing_expected": [tag for tag in expected if tag not in tags], "evaluations": evaluations,
             "data_kind": meta.get("data_kind", "recorded"), "wall_start": start,
             "last_values": {tag: next(row["value"] for row in reversed(rows) if row["metric"] == tag) for tag in tags}}
  summary["missing_rl_families"] = [prefix for prefix in RL_FAMILIES if not any(tag.startswith(prefix) for tag in tags)]
  return rows, summary


def write_csv(path, rows):
  with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def plot(rows, output, title):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from html import escape

  output.mkdir(parents=True, exist_ok=True)
  groups = defaultdict(lambda: defaultdict(list))
  for row in rows:
    groups[row["metric"]][f'{row["experiment"]} / {row["run"]}'].append(row)
  links = []
  for index, (metric, series) in enumerate(sorted(groups.items())):
    filename = f"{index:03d}-" + re.sub(r"[^a-zA-Z0-9_-]", "_", metric)[:120] + ".png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    for label, points in series.items():
      points = sorted(points, key=lambda row: row["wall_time"])
      updates = [row for row in points if row["update"] >= 0]
      if updates:
        axes[0].plot([row["update"] for row in updates], [row["value"] for row in updates],
                     marker="." if len(updates) < 10 else None, label=label)
      axes[1].plot([row["elapsed_s"] / 60 for row in points], [row["value"] for row in points],
                   marker="." if len(points) < 10 else None, label=label)
    for axis, xlabel in zip(axes, ("Completed update (within experiment)", "Elapsed run time (minutes)")):
      axis.set_xlabel(xlabel)
      axis.set_ylabel(metric)
      axis.grid(alpha=.25)
      if axis.lines:
        axis.legend(fontsize=6)
    fig.suptitle(f"{title}: {metric}", fontsize=10)
    fig.savefig(output / filename, dpi=110)
    plt.close(fig)
    links.append(f'<figure><figcaption>{escape(metric)}</figcaption><a href="{filename}"><img loading="lazy" width="900" src="{filename}" alt="{escape(metric)}"></a></figure>')
  (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Training metrics</title>'
      '<style>body{font-family:system-ui;margin:24px}img{max-width:100%}figure{margin:24px 0}</style>'
      f'<h1>{escape(title)}</h1><p>Raw values; no smoothing. Missing measurements are omitted. '
      'Single points are measurements, not learning trends. Different tasks/seeds are identified in each legend.</p>'
      + "\n".join(links), encoding="utf-8")
  return len(links)


def export_run(run, output=None):
  run = Path(run)
  output = Path(output) if output else run / "metrics"
  output.mkdir(parents=True, exist_ok=True)
  rows, summary = collect(run)
  write_csv(output / "scalars.csv", rows)
  summary["plots"] = plot(rows, output / "plots", f'{summary["data_kind"]}: {run.name}')
  atomic_json(output / "summary.json", summary)
  print(f'Exported {run.name}: {summary["points"]} measurements, {summary["plots"]} plots', flush=True)
  return summary


def compare(runs, output):
  rows, summaries = [], []
  for run in runs:
    values, summary = collect(Path(run))
    rows.extend(values)
    summaries.append(summary)
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  write_csv(output / "comparison.csv", rows)
  count = plot(rows, output / "plots", "Run and experiment comparison")
  # Separate experiment-level elapsed time preserves gaps between resumed segments.
  starts = {}
  for row in rows:
    starts[row["experiment"]] = min(starts.get(row["experiment"], row["wall_time"]), row["wall_time"])
  experiment_rows = [{**row, "elapsed_s": row["wall_time"] - starts[row["experiment"]]} for row in rows]
  write_csv(output / "experiment-time.csv", experiment_rows)
  plot(experiment_rows, output / "experiment-plots", "Elapsed experiment time (all segments)")
  atomic_json(output / "comparison.json", {"runs": summaries, "plots": count})
  print(f"Compared {len(summaries)} runs: {count} metrics", flush=True)


def update_catalog(root):
  import subprocess
  catalog = root / "experiment-comparisons"
  runs = [run for run in sorted(root.iterdir()) if run.is_dir() and
          (run / "exit-code.txt").exists() and
          (any(run.rglob("events.out.tfevents*")) or (run / "imitation.jsonl").exists()) and
          read_json(run / "experiment.json", {}).get("data_kind") != "synthetic-smoke"]
  if not runs:
    return
  compare(runs, catalog / "metrics")
  commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
  (catalog / "git-commit.txt").write_text(commit + "\n", encoding="utf-8")
  (catalog / "exit-code.txt").write_text("0\n", encoding="utf-8")
  atomic_json(catalog / "experiment.json", {"data_kind": "cross-experiment-comparison", "run_ids": [run.name for run in runs]})
  queue(catalog)


def queue(run):
  """Publish only completed snapshots; acknowledge an exact content digest."""
  if not SAFE_ID.fullmatch(run.name) or not (run / "exit-code.txt").exists():
    raise ValueError("A publication needs a safe run ID and a completion marker")
  from zipfile import ZipFile, ZIP_DEFLATED
  with ZipFile(run / "metrics-report.zip", "w", compression=ZIP_DEFLATED) as archive:
    for name in ("readiness.json", "experiment.json", "README.md"):
      if (run / name).is_file():
        archive.write(run / name, name)
    for folder in (run / "metrics", run / "comparison"):
      for path in sorted(folder.rglob("*")):
        if path.is_file():
          archive.write(path, str(path.relative_to(run)))
  files = sorted(path for path in run.rglob("*") if path.is_file() and path.name not in
                 ("publish-ready.json", "publish-receipt.json"))
  digest = hashlib.sha256()
  for path in files:
    digest.update(str(path.relative_to(run)).encode())
    with path.open("rb") as stream:
      for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
  value = {"run_id": run.name, "digest": digest.hexdigest(), "queued_at": time.time()}
  atomic_json(run / "publish-ready.json", value)
  return value


def pending(root):
  result = []
  for path in sorted(root.glob("*/publish-ready.json")):
    item = read_json(path)
    receipt = read_json(path.parent / "publish-receipt.json", {})
    if item["digest"] != receipt.get("digest"):
      result.append(item)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  single = commands.add_parser("export")
  single.add_argument("--run", type=Path, required=True)
  single.add_argument("--output", type=Path)
  single.add_argument("--queue", action="store_true")
  single.add_argument("--catalog", action="store_true")
  combined = commands.add_parser("compare")
  combined.add_argument("--runs", type=Path, nargs="+", required=True)
  combined.add_argument("--output", type=Path, required=True)
  listing = commands.add_parser("pending")
  listing.add_argument("--root", type=Path, default=Path("/kaggle/working/g1-edu-stairs"))
  acknowledge = commands.add_parser("acknowledge")
  acknowledge.add_argument("--root", type=Path, default=Path("/kaggle/working/g1-edu-stairs"))
  acknowledge.add_argument("--run-id", required=True)
  acknowledge.add_argument("--digest", required=True)
  acknowledge.add_argument("--url", required=True)
  args = parser.parse_args()
  if args.command == "export":
    summary = export_run(args.run, args.output)
    if args.queue:
      queue(args.run)
    if args.catalog:
      update_catalog(args.run.parent)
    print(json.dumps({"points": summary["points"], "plots": summary["plots"],
                      "missing_expected": summary["missing_expected"]}))
  elif args.command == "compare":
    compare(args.runs, args.output)
  elif args.command == "pending":
    print(json.dumps(pending(args.root)))
  else:
    if not SAFE_ID.fullmatch(args.run_id) or not re.fullmatch(r"[a-f0-9]{64}", args.digest):
      parser.error("Invalid run ID or digest")
    run = args.root / args.run_id
    if read_json(run / "publish-ready.json")["digest"] != args.digest:
      raise RuntimeError("Run changed during upload; retry the new queued digest")
    atomic_json(run / "publish-receipt.json", {"digest": args.digest, "url": args.url, "uploaded_at": time.time()})


if __name__ == "__main__":
  main()
