# Rebuild the G1 EDU training VM

The GitHub fork contains the task, full dependency versions, training profiles, and startup scripts. GitHub Releases carry the binary checkpoints and experiment files. The Windows copy of the repository keeps the downloaded archives under `outputs/runs`.

## Connect and install

1. Start a Kaggle session with the T4 GPU accelerator. Run the notebook's SSH and private zrok share setup. Copy the new VM's **ED25519 fingerprint** from the notebook log.
2. In PowerShell in the local checkout, run the connection script with that fingerprint. Paste the current share into its prompt if the script needs to start a tunnel. It verifies the notebook fingerprint before sending SSH authentication.

```powershell
./scripts/connect_kaggle.ps1 -Fingerprint 'SHA256:<fingerprint-from-notebook>' -Bootstrap
```

The script clones the fork under `/codex_download_here_work_here/g1-edu-stairs` and installs its Python environment, CUDA-compatible packages, FFmpeg, and the headless renderer. It checks that PyTorch sees the GPUs and that MuJoCo can render an image. For a tunnel you started yourself, close its old window when replacing a reset VM. The helper can replace a stale tunnel that it started.

From an existing SSH session, you can run the same VM setup directly.

```bash
cd /codex_download_here_work_here/g1-edu-stairs
bash scripts/bootstrap_kaggle.sh
```

## Start or resume training

```bash
cd /codex_download_here_work_here/g1-edu-stairs
nohup bash scripts/run_kaggle_reference.sh reference-001 \
  > /kaggle/working/g1-edu-stairs/reference-001.launch.log 2>&1 </dev/null &
```

The reference launcher trains with two T4s, saves checkpoints under `/kaggle/working/g1-edu-stairs/reference-001`, and evaluates its final checkpoint headlessly. The main training log is `console.log` inside the run folder.

While a run is active, save a recovery copy to Windows and GitHub whenever a newer checkpoint is ready. After training and evaluation finish, sync the full run.

```powershell
./scripts/sync_stairs_run.ps1 -RunId reference-001 -Checkpoint
./scripts/sync_stairs_run.ps1 -RunId reference-001
```

If the VM resets, download the recovery release or use the local archive. Extract it under `/kaggle/working/g1-edu-stairs` in the new VM. Use a new run ID and pass the restored checkpoint as an absolute path.

```bash
STAIRS_PROFILE=experiments/profiles/kaggle-reference.env ITERATIONS=80 \
  bash scripts/kaggle_stairs.sh train reference-resumed \
  --resume-checkpoint /kaggle/working/g1-edu-stairs/reference-001/g1_edu_stairs/<timestamp>/model_N.pt \
  --agent.algorithm.learning-rate 0.0003 --agent.algorithm.schedule fixed \
  --agent.algorithm.entropy-coef 0.003
```

Replace the placeholders with the extracted path. Resume restores the policy, critic, optimizer, and update counter. Fresh simulation episodes begin at the configured initial difficulty.

## Render media headlessly

```bash
bash scripts/kaggle_stairs.sh render \
  --checkpoint /absolute/path/to/model_N.pt \
  --output /kaggle/content/g1-edu-stairs/videos
```

Omit the checkpoint for course screenshots at the starting pose. Add `--screenshots-only` with a checkpoint to capture a still from each rollout. For a single course, add flags such as `--course down_compact --level 5`. The output includes a small JSON file that identifies the course and records what happened in the selected episode.
