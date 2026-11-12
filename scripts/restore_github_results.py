"""Restore every published run after a VM reset, checking hashes before extraction."""

import argparse
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from scripts.github_delivery import DEFAULT_REPO, DEFAULT_TOKEN, GitHub, safe_members, verify_hash
from scripts.training_metrics import SAFE_ID, atomic_json, read_json


def restore_release(release, root, scratch, github):
  tag = release["tag_name"]
  if not tag.startswith("run-"):
    return None
  cache = scratch / "recovery" / tag
  names = {asset["name"] for asset in release["assets"]}
  if "downloads.json" in names:
    manifest = read_json(github.download(tag, "downloads.json", cache))
    run_id = manifest["run_id"]
    full, = [item for item in manifest["files"] if item["kind"] == "full_run"]
    filename, expected = full["name"], full["sha256"]
  else:
    archives = [name for name in names if name.endswith(".tar.gz")]
    if not archives:
      return None
    if len(archives) != 1:
      raise ValueError(f"Ambiguous legacy archive in {tag}")
    filename = archives[0]
    run_id = filename.removesuffix(".tar.gz").removesuffix("-checkpoint")
    checkfile = filename + ".sha256"
    if checkfile not in names:
      raise ValueError(f"No published checksum for {tag}")
    expected = github.download(tag, checkfile, cache).read_text(encoding="utf-8").strip().split()[0]
  if (not SAFE_ID.fullmatch(run_id) or "/" in filename or "\\" in filename
      or not filename.endswith(".tar.gz")):
    raise ValueError("Invalid release asset name")
  destination = root / run_id
  if destination.exists():
    receipt = read_json(destination / "restore-receipt.json", {})
    return {"run_id": run_id, "tag": tag, "status": "already_present", "verified_restore": receipt.get("sha256") == expected}
  archive = cache / filename
  if archive.exists():
    try:
      verify_hash(archive, expected)
    except ValueError:
      archive = github.download(tag, filename, cache)
  else:
    archive = github.download(tag, filename, cache)
  verify_hash(archive, expected)
  root.mkdir(parents=True, exist_ok=True)
  staging_root = scratch / "recovery" / "staging"
  staging_root.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(dir=staging_root) as staging:
    with tarfile.open(archive, "r:gz") as packed:
      members = safe_members(packed, run_id)
      required = sum(member.size for member in members)
      if required + 1024**3 > shutil.disk_usage(root).free:
        raise RuntimeError("Insufficient results-disk space to restore this archive")
      packed.extractall(staging, members=members, filter="data")
    staged = Path(staging) / run_id
    if not staged.is_dir():
      raise ValueError("Archive has no run directory")
    shutil.move(str(staged), destination)
  receipt = {"run_id": run_id, "tag": tag, "sha256": expected, "archive": filename,
             "restored_at": time.time(), "status": "restored", "verified_restore": True}
  atomic_json(destination / "restore-receipt.json", receipt)
  # Already published historical snapshots must not be uploaded merely because
  # they were restored. A fresh metric export creates a new queued digest.
  ready = read_json(destination / "publish-ready.json")
  if ready:
    atomic_json(destination / "publish-receipt.json", {"digest": ready["digest"], "publisher": "restored",
                "url": f"https://github.com/{github.repo}/releases/tag/{tag}", "restored_at": time.time()})
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", default=DEFAULT_REPO)
  parser.add_argument("--root", type=Path, default=Path("/kaggle/working/g1-edu-stairs"))
  parser.add_argument("--scratch", type=Path, default=Path("/codex_download_here_work_here"))
  parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN)
  parser.add_argument("--tags", nargs="*")
  args = parser.parse_args()
  github = GitHub(args.repo, args.token_file)
  releases = github.releases()
  tags = {release["tag_name"] for release in releases}
  selected = [release for release in releases if
              (not args.tags or release["tag_name"] in args.tags) and
              (not release["tag_name"].endswith("-checkpoint") or release["tag_name"].removesuffix("-checkpoint") not in tags)]
  report = {"repository": args.repo, "started_at": time.time(), "runs": [], "failures": []}
  for release in selected:
    try:
      restored = restore_release(release, args.root, args.scratch, github)
      if restored:
        report["runs"].append(restored)
        print(f'{restored["status"]}: {restored["run_id"]}', flush=True)
    except Exception as error:
      report["failures"].append({"tag": release["tag_name"], "error": str(error)})
      print(f'Failed restore: {release["tag_name"]}: {error}', flush=True)
    atomic_json(args.scratch / "recovery" / "restore-status.json", report)
  report["completed_at"] = time.time()
  atomic_json(args.scratch / "recovery" / "restore-status.json", report)
  if report["failures"]:
    raise SystemExit(1)
  print(f'Restored or retained {len(report["runs"])} published runs; no training started', flush=True)


if __name__ == "__main__":
  main()
