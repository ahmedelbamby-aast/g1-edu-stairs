#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
run_id="${1:?Supply the completed run ID}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
shift
run="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}/$run_id"
output="${STAIRS_VIDEO_DIR:-/kaggle/content/g1-edu-stairs/videos}"
test "$(cat "$run/exit-code.txt")" = 0
checkpoint=$(find "$run" -name 'model_*.pt' -print | sort -V | tail -n 1)
test -n "$checkpoint"
bash scripts/kaggle_stairs.sh render --checkpoint "$checkpoint" --output "$output" "$@"

# Keep the requested video directory and include those media in the run archive.
mkdir -p "$run/media"
.venv/bin/python - "$output" "$run/media" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
manifest = source / "media.json"
for course in json.loads(manifest.read_text())["courses"]:
    for filename in (course["png"], course["mp4"]):
        if filename:
            shutil.copy2(source / filename, destination / filename)
shutil.copy2(manifest, destination / manifest.name)
PY
