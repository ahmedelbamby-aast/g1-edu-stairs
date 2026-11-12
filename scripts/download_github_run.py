"""Developer download: select the published snapshot, verify SHA-256, unzip plots."""

import argparse
import json
import stat
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from scripts.github_delivery import DEFAULT_REPO, GitHub, verify_hash
from scripts.training_metrics import SAFE_ID, read_json


def download_snapshot(github, run_id, output, kind="metrics", extract=True):
  if not SAFE_ID.fullmatch(run_id):
    raise ValueError("Invalid run ID")
  tag = f"run-{run_id}"
  output.mkdir(parents=True, exist_ok=True)
  manifest = read_json(github.download(tag, "downloads.json", output))
  if manifest["run_id"] != run_id:
    raise ValueError("Download manifest belongs to a different run")
  items = [item for item in manifest["files"] if item["kind"] == kind]
  if len(items) != 1:
    raise ValueError("Requested artifact is not present in the download manifest")
  item = items[0]
  if "/" in item["name"] or "\\" in item["name"]:
    raise ValueError("Invalid artifact filename")
  path = github.download(tag, item["name"], output)
  verify_hash(path, item["sha256"])
  if path.stat().st_size != item["bytes"]:
    raise ValueError("Artifact size does not match manifest")
  result = {"path": str(path.resolve()), "sha256_verified": True, "kind": kind}
  if kind == "metrics" and extract:
    target = output / f'{run_id}-{manifest["content_digest"][:20]}'
    with ZipFile(path) as archive:
      for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in info.filename or ":" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
          raise ValueError("Report ZIP contains an unsafe path or link")
      archive.extractall(target)
    result["gallery"] = str((target / "metrics" / "plots" / "index.html").resolve())
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repo", default=DEFAULT_REPO)
  parser.add_argument("--kind", choices=("metrics", "full_run"), default="metrics")
  parser.add_argument("--no-extract", action="store_true")
  args = parser.parse_args()
  github = GitHub(args.repo)
  for attempt in range(3):
    try:
      result = download_snapshot(github, args.run_id, args.output, args.kind, not args.no_extract)
      print(json.dumps(result, indent=2))
      return
    except Exception:
      if attempt == 2:
        raise
      time.sleep(2 ** attempt)


if __name__ == "__main__":
  main()
