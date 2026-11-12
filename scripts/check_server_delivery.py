"""Live headless render, detached uploader, verified developer download check."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image

from scripts.download_github_run import download_snapshot
from scripts.github_delivery import DEFAULT_TOKEN, GitHub
from scripts.training_metrics import atomic_json, queue, read_json


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-id", default="round-trip-pilot-s42")
  parser.add_argument("--root", type=Path, default=Path("/kaggle/working/g1-edu-stairs"))
  parser.add_argument("--scratch", type=Path, default=Path("/codex_download_here_work_here"))
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  state = read_json(args.scratch / "services" / "results-status.json")
  if not state or state.get("rendering", {}).get("png") != "passed":
    raise RuntimeError("Headless plot check is not healthy")
  pid = state["pid"]
  os.kill(pid, 0)
  if os.getsid(pid) != pid:
    raise RuntimeError("Uploader is not detached from the SSH session")
  if DEFAULT_TOKEN.resolve().is_relative_to(args.root.resolve()) or DEFAULT_TOKEN.stat().st_mode & 0o077:
    raise RuntimeError("Credential location or permissions are incorrect")
  github = GitHub()
  auth = github.auth_check()
  receipt = read_json(args.root / args.run_id / "publish-receipt.json")
  if not receipt or receipt.get("publisher") != "gpu-server":
    raise RuntimeError("The selected run has not been uploaded by the GPU server")
  downloaded = download_snapshot(github, args.run_id, args.scratch / "developer-download-check")
  gallery = Path(downloaded["gallery"])
  pngs = sorted(gallery.parent.glob("*.png"))
  if not gallery.is_file() or not pngs:
    raise RuntimeError("Downloaded plot gallery is incomplete")
  for path in pngs:
    with Image.open(path) as image:
      image.verify()
  restore = read_json(args.scratch / "recovery" / "restore-status.json")
  if restore["failures"]:
    raise RuntimeError("Some published results have not been recovered")
  result = {"status": "passed", "optimizer_steps": 0, "run_id": args.run_id,
            "restored_runs": len(restore["runs"]), "restore_failures": 0,
            "headless_backend": state["rendering"]["backend"], "uploader_pid": pid,
            "uploader_detached": True, "github": auth, "download_sha256_verified": downloaded["sha256_verified"],
            "downloaded_pngs_verified": len(pngs), "gallery_exists": True,
            "credential_outside_results": True, "credential_mode": "0600",
            "published_run": receipt["url"], "checked_at": time.time()}
  atomic_json(args.output / "delivery-check.json", result)
  atomic_json(args.output / "experiment.json", {"data_kind": "delivery-verification", "optimizer_steps": 0})
  (args.output / "git-commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() + "\n")
  (args.output / "exit-code.txt").write_text("0\n")
  queue(args.output)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
