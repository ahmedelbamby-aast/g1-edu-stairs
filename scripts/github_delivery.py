"""GitHub release delivery primitives; credentials never enter result directories."""

import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

from scripts.training_metrics import SAFE_ID, atomic_json, read_json

DEFAULT_REPO = "ahmedelbamby-aast/g1-edu-stairs"
DEFAULT_TOKEN = Path("/codex_download_here_work_here/credentials/github-token")


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def github_environment(token_file=DEFAULT_TOKEN):
  environment = dict(os.environ)
  if not environment.get("GH_TOKEN") and not environment.get("GITHUB_TOKEN") and token_file.exists():
    if os.name != "nt" and token_file.stat().st_mode & 0o077:
      raise PermissionError("The GitHub credential file must have mode 0600")
    environment["GH_TOKEN"] = token_file.read_text(encoding="utf-8").strip()
  environment["GH_PROMPT_DISABLED"] = "1"
  environment["GH_PAGER"] = "cat"
  return environment


class GitHub:
  def __init__(self, repo=DEFAULT_REPO, token_file=DEFAULT_TOKEN):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
      raise ValueError("Invalid GitHub repository")
    self.repo = repo
    self.environment = github_environment(Path(token_file))

  def call(self, *arguments, check=True):
    result = subprocess.run(["gh", *map(str, arguments)], env=self.environment, text=True,
                            capture_output=True, timeout=600)
    if check and result.returncode:
      message = result.stderr.strip()
      for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        if self.environment.get(key):
          message = message.replace(self.environment[key], "[redacted]")
      raise RuntimeError(f"GitHub command failed: {message[:1000]}")
    return result

  def auth_check(self):
    data = json.loads(self.call("api", f"repos/{self.repo}").stdout)
    if not data.get("permissions", {}).get("push"):
      raise PermissionError("GitHub login lacks repository write permission")
    return {"repository": self.repo, "write_access": True}

  def release(self, tag):
    result = self.call("api", f"repos/{self.repo}/releases/tags/{tag}", check=False)
    if result.returncode:
      if "404" in result.stderr or "Not Found" in result.stderr:
        return None
      raise RuntimeError("Cannot read release metadata; retry when GitHub is available")
    return json.loads(result.stdout)

  def releases(self):
    pages = json.loads(self.call("api", f"repos/{self.repo}/releases?per_page=100", "--paginate", "--slurp").stdout)
    return [release for page in pages for release in page if not release["draft"]]

  def download(self, tag, name, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    self.call("release", "download", tag, "--repo", self.repo, "--pattern", name,
              "--dir", directory, "--clobber")
    return directory / name


def verify_hash(path, expected):
  if not re.fullmatch(r"[a-fA-F0-9]{64}", expected) or sha256(path) != expected.lower():
    raise ValueError(f"Checksum mismatch for {Path(path).name}")


def safe_members(archive, run_id):
  """A run archive may contain regular files/directories under one exact root."""
  if not SAFE_ID.fullmatch(run_id):
    raise ValueError("Invalid restore run ID")
  members = archive.getmembers()
  for member in members:
    path = PurePosixPath(member.name)
    if (path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name
        or not path.parts or path.parts[0] != run_id or not (member.isfile() or member.isdir())):
      raise ValueError("Archive contains an unsafe path, link or special file")
  return members


def create_snapshot(run, destination):
  """Make a regular-file-only archive, with stable gzip and tar metadata."""
  run, destination = Path(run), Path(destination)
  destination.parent.mkdir(parents=True, exist_ok=True)
  if not SAFE_ID.fullmatch(run.name):
    raise ValueError("Invalid run ID")
  with destination.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
    with tarfile.open(fileobj=zipped, mode="w") as archive:
      for path in [run, *sorted(run.rglob("*"))]:
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
          raise ValueError("Result snapshots cannot contain links or special files")
        if path.name in ("publish-receipt.json",):
          continue
        name = (Path(run.name) / path.relative_to(run)).as_posix()
        info = archive.gettarinfo(str(path), arcname=name)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        if path.is_file():
          with path.open("rb") as stream:
            archive.addfile(info, stream)
        else:
          archive.addfile(info)


def publish_run(run, scratch, github):
  """Upload immutable assets first, then a downloads.json pointer, then acknowledge."""
  run, scratch = Path(run), Path(scratch)
  ready = read_json(run / "publish-ready.json")
  if not ready or ready["run_id"] != run.name or not re.fullmatch(r"[a-f0-9]{64}", ready["digest"]):
    raise ValueError("Missing or invalid publication queue entry")
  receipt = read_json(run / "publish-receipt.json", {})
  if receipt.get("digest") == ready["digest"] and receipt.get("publisher") == "gpu-server":
    return receipt
  if not (run / "exit-code.txt").exists():
    raise ValueError("Only completed run snapshots can be published")
  commit = (run / "git-commit.txt").read_text(encoding="utf-8").strip()
  if not re.fullmatch(r"[a-f0-9]{40}", commit):
    raise ValueError("Missing valid source commit")
  suffix = ready["digest"][:20]
  tag = f"run-{run.name}"
  directory = scratch / "delivery" / run.name / suffix
  directory.mkdir(parents=True, exist_ok=True)
  full = directory / f"{run.name}-{suffix}.tar.gz"
  create_snapshot(run, full)
  payloads = [(full, "full_run")]
  report = run / "metrics-report.zip"
  if report.exists():
    import shutil
    quick = directory / f"{run.name}-metrics-{suffix}.zip"
    shutil.copy2(report, quick)
    payloads.append((quick, "metrics"))
  files = [{"name": path.name, "kind": kind, "bytes": path.stat().st_size, "sha256": sha256(path)}
           for path, kind in payloads]
  checksum = directory / f"{run.name}-{suffix}.sha256"
  checksum.write_text("".join(f'{item["sha256"]}  {item["name"]}\n' for item in files), encoding="utf-8")
  manifest = {"schema": 1, "run_id": run.name, "content_digest": ready["digest"], "source_commit": commit,
              "files": [{**item, "url": f'https://github.com/{github.repo}/releases/download/{tag}/{item["name"]}'}
                        for item in files], "checksum_file": checksum.name, "published_at": time.time(),
              "plot_gallery": "Unzip the metrics asset and open metrics/plots/index.html"}
  immutable_manifest = directory / f"{run.name}-{suffix}.manifest.json"
  atomic_json(immutable_manifest, manifest)
  pointer = directory / "downloads.json"
  atomic_json(pointer, manifest)
  if read_json(run / "publish-ready.json")["digest"] != ready["digest"]:
    raise RuntimeError("Run changed during snapshot creation; the new version remains queued")
  release = github.release(tag)
  if release is None:
    notes = directory / "notes.md"
    notes.write_text("Download the metrics ZIP for CSV files, plots and an offline HTML gallery. "
                     "The full archive contains logs, settings, checkpoints and other run artifacts. "
                     "Use downloads.json and the SHA-256 file to identify and verify the latest complete snapshot.\n",
                     encoding="utf-8")
    github.call("release", "create", tag, "--repo", github.repo, "--target", commit,
                "--title", f"G1 EDU | {run.name}", "--latest=false", "--notes-file", notes)
  payload_paths = [path for path, _ in payloads] + [checksum, immutable_manifest]
  github.call("release", "upload", tag, *payload_paths, "--repo", github.repo, "--clobber")
  uploaded = github.release(tag)
  sizes = {asset["name"]: asset["size"] for asset in uploaded["assets"]}
  if any(sizes.get(path.name) != path.stat().st_size for path in payload_paths):
    raise RuntimeError("GitHub asset size verification failed; snapshot remains pending")
  # This pointer is changed only after every payload is present on GitHub.
  github.call("release", "upload", tag, pointer, "--repo", github.repo, "--clobber")
  if read_json(run / "publish-ready.json")["digest"] != ready["digest"]:
    raise RuntimeError("Run changed during upload; the new snapshot remains pending")
  receipt = {"digest": ready["digest"], "publisher": "gpu-server", "uploaded_at": time.time(),
             "url": f"https://github.com/{github.repo}/releases/tag/{tag}", "files": files}
  atomic_json(run / "publish-receipt.json", receipt)
  return receipt
