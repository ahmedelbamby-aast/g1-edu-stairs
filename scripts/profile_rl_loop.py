"""Bounded CPU/CUDA profiling of the real RSL-RL loop, disabled for normal runs."""

import cProfile
import csv
import functools
import gzip
import pstats
import json
import resource
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.training_metrics import atomic_json


@contextmanager
def record_performance(runner, directory, rank, device):
  directory = Path(directory) / "performance"
  directory.mkdir(parents=True, exist_ok=True)
  original = runner.logger.log
  with (directory / f"rank-{rank}.jsonl").open("w", encoding="utf-8") as stream:
    def measured(*args, **kwargs):
      original(*args, **kwargs)
      free, total = torch.cuda.mem_get_info(device)
      usage = resource.getrusage(resource.RUSAGE_SELF)
      values = {"iteration": kwargs["it"], "wall_time": time.time(),
                "collection_s": kwargs["collect_time"], "update_s": kwargs["learn_time"],
                "gpu_used_mib": (total - free) / 1024**2,
                "torch_peak_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                "torch_reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
                "gpu_free_mib": free / 1024**2,
                "process_cpu_s": usage.ru_utime + usage.ru_stime, "loss": kwargs["loss_dict"]}
      stream.write(json.dumps(values, allow_nan=False) + "\n")
      stream.flush()
      writer = getattr(runner.logger, "writer", None)
      if writer is not None:
        for name in ("gpu_used_mib", "torch_peak_mib", "torch_reserved_mib", "gpu_free_mib"):
          writer.add_scalar("Performance/" + name, values[name], kwargs["it"])
    runner.logger.log = measured
    try:
      yield
    finally:
      runner.logger.log = original


@contextmanager
def profile_rl_loop(runner, directory, rank, wait=3, active=2):
  directory = Path(directory) / "profiler" / f"rank-{rank}"
  directory.mkdir(parents=True, exist_ok=True)
  originals = []
  python = cProfile.Profile()

  def wrap(owner, name, label):
    original = getattr(owner, name)
    @functools.wraps(original)
    def measured(*args, **kwargs):
      with torch.profiler.record_function(label):
        return original(*args, **kwargs)
    setattr(owner, name, measured)
    originals.append((owner, name, original))

  for owner, name, label in ((runner.env, "step", "rl/simulation_step"),
                              (runner.alg, "act", "rl/policy_action"),
                              (runner.alg, "process_env_step", "rl/store_transition"),
                              (runner.alg, "compute_returns", "rl/returns"),
                              (runner.alg, "update", "rl/ppo_update")):
    wrap(owner, name, label)

  def save_trace(profiler):
    python.disable()
    trace = directory / "trace.json"
    profiler.export_chrome_trace(str(trace))
    with trace.open("rb") as source, gzip.open(str(trace) + ".gz", "wb", compresslevel=1) as target:
      import shutil
      shutil.copyfileobj(source, target)
    trace.unlink()
    rows = [{"operator": event.key, "calls": event.count,
             "self_cpu_us": event.self_cpu_time_total,
             "self_device_us": event.self_device_time_total,
             "cpu_total_us": event.cpu_time_total} for event in profiler.key_averages()]
    rows.sort(key=lambda row: row["self_cpu_us"], reverse=True)
    with (directory / "operators.csv").open("w", newline="", encoding="utf-8") as stream:
      writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
      writer.writeheader()
      writer.writerows(rows)
    atomic_json(directory / "summary.json", {"wait_updates": wait, "active_updates": active,
                "activities": ["CPU", "CUDA"], "record_shapes": False, "with_stack": False,
                "top_cpu": rows[:20], "top_device": sorted(rows, key=lambda row: row["self_device_us"], reverse=True)[:20],
                "timing_note": "Profiled updates include instrumentation overhead; do not use them for throughput ranking."})

  profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
      schedule=torch.profiler.schedule(wait=wait, warmup=1, active=active, repeat=1),
      on_trace_ready=save_trace, record_shapes=False, with_stack=False, profile_memory=False)
  original_log = runner.logger.log
  def logged(*args, **kwargs):
    original_log(*args, **kwargs)
    python.disable()
    profiler.step()
    if profiler.current_action in (torch.profiler.ProfilerAction.RECORD, torch.profiler.ProfilerAction.RECORD_AND_SAVE):
      python.enable()
  runner.logger.log = logged
  try:
    with profiler:
      yield
  finally:
    python.disable()
    python.dump_stats(str(directory / "python.pstats"))
    with (directory / "python-top.txt").open("w", encoding="utf-8") as stream:
      pstats.Stats(python, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(65)
    runner.logger.log = original_log
    for owner, name, original in originals:
      setattr(owner, name, original)
