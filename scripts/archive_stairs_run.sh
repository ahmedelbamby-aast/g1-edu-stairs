#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?Supply a run ID}"
mode="${2:-completed}"
[[ "$run_id" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]] || exit 2
results="${STAIRS_RESULTS:-/kaggle/working/g1-edu-stairs}"
scratch="${STAIRS_SCRATCH:-/codex_download_here_work_here}"
run="$results/$run_id"
bundles="$scratch/bundles"
archive="$run_id.tar.gz"

case "$mode" in
  completed)
    test -s "$run/exit-code.txt"
    files=("$run_id")
    ;;
  checkpoint)
    archive="$run_id-checkpoint.tar.gz"
    checkpoint=$(find "$run" -mindepth 3 -type f -name 'model_*.pt' -print | sort -V | tail -n 1)
    test -n "$checkpoint"
    files=("$run_id/git-commit.txt" "$run_id/profile.env" "$run_id/command.sh"
      "$run_id/requirements.txt" "$run_id/runtime.txt" "$run_id/gpus.csv"
      "${checkpoint#"$results/"}" "${checkpoint#"$results/"}")
    files[-1]="${files[-1]%/*}/params"
    if [[ -d "${checkpoint%/*}/audit" ]]; then files+=("${files[-1]%/*}/audit"); fi
    ;;
  *) exit 2 ;;
esac

mkdir -p "$bundles"
temporary=$(mktemp "$bundles/.archive.XXXXXX")
trap 'rm -f "$temporary"' EXIT
tar --sort=name -czf "$temporary" -C "$results" "${files[@]}"
mv "$temporary" "$bundles/$archive"
cd "$bundles"
sha256sum "$archive" > "$archive.sha256"
cat "$run/git-commit.txt"
