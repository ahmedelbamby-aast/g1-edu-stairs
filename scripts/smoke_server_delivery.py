"""Local failure-path checks for VM recovery and server publication; no training."""

import argparse
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from scripts.download_github_run import download_snapshot
from scripts.github_delivery import create_snapshot, publish_run, safe_members, sha256, verify_hash
from scripts.restore_github_results import restore_release
from scripts.training_metrics import atomic_json, queue, read_json


class FakeGitHub:
  repo = "example/test"

  def __init__(self, root):
    self.root = root
    self.uploaded = {}
    self.created = False
    self.fail_upload = False
    self.commands = []

  def release(self, tag):
    return {"tag_name": tag, "assets": [{"name": name, "size": path.stat().st_size}
            for name, path in self.uploaded.items()]} if self.created else None

  def call(self, *arguments, **kwargs):
    self.commands.append(tuple(map(str, arguments)))
    if arguments[:2] == ("release", "create"):
      self.created = True
    elif arguments[:2] == ("release", "upload"):
      if self.fail_upload:
        self.fail_upload = False
        raise RuntimeError("Simulated offline GitHub")
      paths = arguments[3:arguments.index("--repo")]
      for path in paths:
        path = Path(path)
        target = self.root / path.name
        shutil.copy2(path, target)
        self.uploaded[path.name] = target
    else:
      raise AssertionError(f"Unexpected fake API call: {arguments}")
    return SimpleNamespace(stdout="", stderr="", returncode=0)

  def download(self, tag, name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    shutil.copy2(self.uploaded[name], target)
    return target


def checks(root):
  run = root / "source" / "delivery-fixture"
  (run / "metrics" / "plots").mkdir(parents=True)
  (run / "exit-code.txt").write_text("0\n")
  (run / "git-commit.txt").write_text("a" * 40 + "\n")
  (run / "metrics" / "plots" / "index.html").write_text("<h1>Labelled delivery fixture</h1>")
  (run / "raw.txt").write_text("Unmodified source data")
  credential = root / "credentials" / "github-token"
  credential.parent.mkdir()
  credential.write_text("synthetic-credential-outside-results")
  queued = queue(run)
  remote = root / "fake-github"
  remote.mkdir()
  github = FakeGitHub(remote)
  github.fail_upload = True
  try:
    publish_run(run, root / "scratch", github)
    raise AssertionError("Upload failure was hidden")
  except RuntimeError:
    pass
  assert not (run / "publish-receipt.json").exists() and "downloads.json" not in github.uploaded
  receipt = publish_run(run, root / "scratch", github)
  assert receipt["digest"] == queued["digest"]
  assert github.commands[-1][3].endswith("downloads.json"), "Pointer was not published last"
  command_count = len(github.commands)
  publish_run(run, root / "scratch", github)
  assert len(github.commands) == command_count, "Acknowledged upload was duplicated"
  manifest = read_json(github.uploaded["downloads.json"])
  for item in manifest["files"]:
    verify_hash(github.uploaded[item["name"]], item["sha256"])
  downloaded = download_snapshot(github, run.name, root / "developer")
  assert downloaded["sha256_verified"] and Path(downloaded["gallery"]).is_file()
  restored = restore_release(github.release("run-" + run.name), root / "restored", root / "recovery", github)
  assert restored["verified_restore"] and (root / "restored" / run.name / "raw.txt").read_text() == "Unmodified source data"
  again = restore_release(github.release("run-" + run.name), root / "restored", root / "recovery", github)
  assert again["status"] == "already_present"
  full = next(item for item in manifest["files"] if item["kind"] == "full_run")
  with tarfile.open(github.uploaded[full["name"]]) as archive:
    assert not any("credentials" in member.name for member in archive.getmembers())
  for bad_name, kind in (("../escape", tarfile.REGTYPE), (run.name + "/../escape", tarfile.REGTYPE),
                         (run.name + "/link", tarfile.SYMTYPE), (run.name + "/hard", tarfile.LNKTYPE)):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
      info = tarfile.TarInfo(bad_name)
      info.type, info.linkname = kind, "../../escape"
      archive.addfile(info)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as archive:
      try:
        safe_members(archive, run.name)
        raise AssertionError("Unsafe archive accepted")
      except ValueError:
        pass
  corrupted = root / "corrupted"
  corrupted.write_bytes(b"wrong bytes")
  try:
    verify_hash(corrupted, "0" * 64)
    raise AssertionError("Corruption was accepted")
  except ValueError:
    pass
  if os.name != "nt":
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
      archive.addfile(tarfile.TarInfo(run.name + "/logs/2026-11-12T12:01:00/worker.log"))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer) as archive:
      assert safe_members(archive, run.name)
  return {"status": "passed", "optimizer_steps": 0, "checks": ["upload_failure_retained", "retry_success",
          "pointer_published_last", "unchanged_upload_skipped", "developer_download_sha256",
          "fresh_vm_restore", "restore_idempotent", "archive_traversal_and_links_rejected",
          "corruption_rejected", "credentials_outside_archives"]}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  with tempfile.TemporaryDirectory(dir=args.output) as directory:
    result = checks(Path(directory))
  atomic_json(args.output / "smoke.json", result)
  print("PASS restore/server delivery/developer download; zero optimizer steps")


if __name__ == "__main__":
  main()
