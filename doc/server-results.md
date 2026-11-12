# GPU-server plots, GitHub delivery and reset recovery

The GPU server now generates plots and uploads results directly. Its background Python service is separate from SSH and Windows. Closing the Windows terminal does not stop delivery. No agent or open Windows watcher is required.

## Recover a completely empty VM

From the Windows checkout, use the share and fingerprint printed by the new Kaggle notebook:

```powershell
./scripts/recover_kaggle_session.ps1 `
  -Share '<new-private-share>' `
  -Fingerprint 'SHA256:<new-notebook-fingerprint>'
```

The script verifies the host fingerprint, rebuilds pinned dependencies, tests headless rendering/plotting, provisions the existing GitHub CLI login over verified SSH, restores published run archives, and starts server-side delivery. It also handles replacement of the old zrok tunnel. It contains no training command.

The GitHub credential has private filesystem permissions and lives at `/codex_download_here_work_here/credentials/github-token`, outside the checkout and all result/archive directories. It is never printed or included in a result bundle. A fresh VM needs a fresh provision step; the script handles this using the existing Windows GitHub login.

For a VM already given a GitHub login or `GH_TOKEN` through its environment, run this from the cloned checkout:

```bash
bash scripts/recover_results_server.sh
```

Restore reads published release metadata, verifies SHA-256, rejects paths that escape the run directory and rejects archive links/special files. It stages each recovered run before exposing it to the plotter. It preserves existing local runs. The older fixed-name archive format and the new versioned download manifests are both supported. Restored historical snapshots are not immediately reuploaded unchanged; missing plots are regenerated from their real saved logs and become new published versions.

Code, environments, caches, downloads and service state stay under `/codex_download_here_work_here`. Training/results stay under `/kaggle/working/g1-edu-stairs`. Restored videos are copied, without re-encoding, under `/kaggle/content/g1-edu-stairs/videos/<run>/`.

## Automatic operation

The service runs at a lower CPU priority with one reporting thread so it can yield CPU time to training. Matplotlib uses the `Agg` backend and needs no display. It monitors completed runs and the publication queue every 20 seconds, generates missing reports, and uploads pending snapshots. The training and evaluation scripts also export/queue reports when they finish their configured segments.

Uploads are retried with increasing delays, up to five minutes between attempts. A result is acknowledged only after the assets exist on GitHub and the queued version still matches. Existing versioned payloads are retained during retries. The small `downloads.json` pointer is published after its payloads, so developers can identify a complete version.

Each release contains:

- A versioned full archive: logs, settings, checkpoints and other run artifacts.
- A smaller metrics ZIP: CSV files, plot PNGs, and an offline HTML gallery.
- A SHA-256 file and a versioned manifest.
- `downloads.json`, pointing at the latest complete snapshot and giving each payload's size and checksum.

The service status and log are in `/codex_download_here_work_here/services/`. Check them with:

```bash
.venv/bin/python -m scripts.server_results_service status
```

If the process was stopped, start it idempotently:

```bash
.venv/bin/python -m scripts.server_results_service start
```

The notebook session and internet connection must be available. Network failures remain queued. A reset can destroy results that were never uploaded; recovery restores the versions actually published on GitHub. Credentials can be reprovisioned without putting them into the repository. The service rereads its credential file between cycles.

## Developer download

Developers need Python and GitHub CLI access to the repository. The download helper does not require Torch, MuJoCo, a GPU or plotting packages:

```bash
python -m scripts.download_github_run \
  --run-id round-trip-pilot-s42 --output ./downloads
```

It reads the current manifest, downloads the smaller report, verifies SHA-256 and size, safely extracts it, and prints the local gallery path. Open that `metrics/plots/index.html` file in a browser. The PNGs and CSV files are ordinary downloadable files. For the full checkpoint/log archive, add `--kind full_run`.

The corresponding release can also be downloaded through the [GitHub release page](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-round-trip-pilot-s42).

## Training stays explicit

The recovery scripts and result service never train. The Windows training launcher now checks server-side delivery during preflight. It starts training only when `-StartTraining` is explicitly supplied, and no longer requires a Windows watcher to remain running. `-WatchFromWindows` enables the older watcher as an optional fallback.

The fresh-session [verification report](../experiments/results/server-recovery-20261112/) records restoration, headless plotting, direct upload, a developer download, and GPU/inference checks. All performed zero optimizer steps.
