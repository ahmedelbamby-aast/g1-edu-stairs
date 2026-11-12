"""Low-priority headless plotting and GitHub delivery service on the GPU VM."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts.github_delivery import DEFAULT_REPO, DEFAULT_TOKEN, GitHub, github_environment, publish_run
from scripts.training_metrics import SAFE_ID, atomic_json, finalize_run, pending, read_json, update_catalog


def needs_export(run):
  if not (run / "exit-code.txt").exists() or (run / "metrics" / "summary.json").exists():
    return False
  kind = read_json(run / "experiment.json", {}).get("data_kind")
  if kind is not None and kind not in ("training", "imitation"):
    return False
  excluded = {"fixtures", "historical", "metrics", "metric-checks", "workflow-checks"}
  return (run / "imitation.jsonl").exists() or any(
      not excluded.intersection(path.relative_to(run).parts) for path in run.rglob("events.out.tfevents*"))


def rendering_check(scratch):
  import matplotlib
  matplotlib.use("Agg", force=True)
  import matplotlib.pyplot as plt
  directory = scratch / "services"
  directory.mkdir(parents=True, exist_ok=True)
  fig, axis = plt.subplots(figsize=(4, 2))
  axis.plot([0, 1, 2], [0, 1, 0])
  axis.set_title("Rendering smoke test; not training")
  fig.savefig(directory / "rendering-check.png", dpi=100)
  plt.close(fig)
  return {"backend": matplotlib.get_backend(), "png": "passed"}


def cycle(root, scratch, github, state, run_id=None):
  status_path = scratch / "services" / "results-status.json"
  changed = False
  for run in sorted(root.iterdir()):
    if run_id and run.name != run_id:
      continue
    if not run.is_dir() or not SAFE_ID.fullmatch(run.name) or not needs_export(run):
      continue
    item = state.setdefault("jobs", {}).setdefault(run.name, {})
    if item.get("retry_at", 0) > time.time():
      continue
    state.update(operation="plotting", active_run=run.name, heartbeat=time.time())
    atomic_json(status_path, state)
    try:
      finalize_run(run)
      changed = True
      item.pop("retry_at", None)
      item.pop("error", None)
    except Exception as error:
      item.update(error=str(error), retry_at=time.time() + 60)
      print(f"Plot retry queued: {run.name}: {error}", flush=True)
  if changed and run_id is None:
    state.update(operation="comparison", heartbeat=time.time())
    atomic_json(status_path, state)
    try:
      update_catalog(root)
    except Exception as error:
      state["comparison_error"] = str(error)
  for ready in pending(root):
    if run_id and ready["run_id"] != run_id and ready["run_id"] != "experiment-comparisons":
      continue
    item = state.setdefault("jobs", {}).setdefault(ready["run_id"], {})
    if item.get("digest") != ready["digest"]:
      item.clear()
      item["digest"] = ready["digest"]
    if item.get("retry_at", 0) > time.time():
      continue
    state.update(operation="uploading", active_run=ready["run_id"], heartbeat=time.time())
    atomic_json(status_path, state)
    try:
      receipt = publish_run(root / ready["run_id"], scratch, github)
      item.update(status="published", receipt=receipt["url"], uploaded_at=time.time(), attempts=0)
      item.pop("retry_at", None)
      item.pop("error", None)
      state["last_successful_upload"] = time.time()
      print(f'Published from GPU server: {receipt["url"]}', flush=True)
    except Exception as error:
      attempts = item.get("attempts", 0) + 1
      item.update(status="retry_pending", attempts=attempts, error=str(error),
                  retry_at=time.time() + min(300, 5 * 2 ** min(attempts - 1, 6)))
      print(f'Upload retry retained: {ready["run_id"]}: {error}', flush=True)
    atomic_json(status_path, state)
  state.update(operation="idle", active_run=None, heartbeat=time.time(), pending=len(pending(root)))
  atomic_json(status_path, state)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=("start", "watch", "once", "status"))
  parser.add_argument("--root", type=Path, default=Path("/kaggle/working/g1-edu-stairs"))
  parser.add_argument("--scratch", type=Path, default=Path("/codex_download_here_work_here"))
  parser.add_argument("--repo", default=DEFAULT_REPO)
  parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN)
  parser.add_argument("--interval", type=int, default=20)
  parser.add_argument("--run-id")
  args = parser.parse_args()
  if args.interval < 5:
    parser.error("Polling interval must be at least five seconds")
  service = args.scratch / "services"
  service.mkdir(parents=True, exist_ok=True)
  status_path = service / "results-status.json"
  state = read_json(status_path, {})
  if args.command == "status":
    try:
      if state.get("pid", 0) <= 0:
        raise ProcessLookupError()
      os.kill(state["pid"], 0)
      state["process_alive"] = True
    except OSError:
      state["process_alive"] = False
    print(json.dumps(state, indent=2))
    return
  if args.command == "start":
    github = GitHub(args.repo, args.token_file)
    github.auth_check()
    previous = state.get("pid", 0)
    if previous > 0 and Path(f"/proc/{previous}/cmdline").exists():
      command = Path(f"/proc/{previous}/cmdline").read_bytes()
      if b"scripts.server_results_service" in command and b"watch" in command:
        print(f"Results service already running as PID {previous}")
        return
    arguments = [sys.executable, "-m", "scripts.server_results_service", "watch", "--root", str(args.root),
                 "--scratch", str(args.scratch), "--repo", args.repo, "--token-file", str(args.token_file),
                 "--interval", str(args.interval)]
    with (service / "results-service.log").open("a", encoding="utf-8") as log:
      process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    deadline = time.time() + 45
    while time.time() < deadline:
      if process.poll() is not None:
        raise RuntimeError("Results service exited; inspect results-service.log")
      ready = read_json(status_path, {})
      if ready.get("pid") == process.pid and ready.get("rendering", {}).get("png") == "passed":
        print(f"Started independent GPU-server result service as PID {process.pid}")
        return
      time.sleep(.25)
    raise RuntimeError("Results service is still initializing; inspect its status before launching another worker")
  # Linux VM singleton, so reconnection cannot create duplicate workers.
  import fcntl
  lock = (service / "results-service.lock").open("a")
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  os.environ.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(args.scratch / "cache" / "matplotlib"),
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
  os.nice(10)
  args.root.mkdir(parents=True, exist_ok=True)
  github = GitHub(args.repo, args.token_file)
  state.update(pid=os.getpid(), started_at=time.time(), repository=args.repo,
               optimizer_steps=0, authentication=github.auth_check(), rendering=rendering_check(args.scratch))
  atomic_json(status_path, state)
  stopping = False
  def stop(signum, frame):
    nonlocal stopping
    stopping = True
  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  while not stopping:
    try:
      github.environment = github_environment(args.token_file)
      cycle(args.root, args.scratch, github, state, args.run_id)
    except Exception as error:
      state.update(operation="retry_pending", error=str(error), heartbeat=time.time())
      atomic_json(status_path, state)
      print(f"Service retry pending: {error}", flush=True)
    if args.command == "once":
      break
    for _ in range(args.interval):
      if stopping:
        break
      time.sleep(1)


if __name__ == "__main__":
  main()
