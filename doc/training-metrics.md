# Training metrics, plots, and automatic publishing

Reporting and uploading use ordinary scripts. They do not need an agent. These scripts never start training unless the Windows launcher is given the explicit `-StartTraining` switch.

## Measurements

| Measurement | Computed and saved by | Plot source |
| --- | --- | --- |
| PPO policy loss, value loss, entropy, learning rate | RSL-RL TensorBoard logger | `Loss/*` |
| Mean episode reward and length | RSL-RL episode logger | `Train/*` |
| Falls, lane exits, timeout and supported finish signals | Environment termination logger | `Episode_Termination/*` |
| Phase reach, risers completed, longest one-leg hold | Skill reset logger | `Metrics/skill/*` |
| Ascent/descent speed and velocity error | Command/skill logger | `Metrics/*` |
| Foot slip and landing force | Contact reward measurements | `Metrics/slip_velocity_mean`, `Metrics/landing_force_mean` |
| Frontier, completion evidence, sampled difficulty | Curriculum logger | `Curriculum/*` |
| Simulation throughput, collection and learning time | RSL-RL logger | `Perf/*` |
| Per-GPU memory, utilization and power | NVIDIA telemetry every two seconds | `GPU/<index>/*` |
| Held-out completion, falls, pace, phases, slip, landing force and speed errors | Clean/stress evaluator, including per-cell counts | `Validation/<suite>/*` |
| Student training MSE and independent validation MSE | Student imitation script | `Imitation/train_mse`, `Imitation/validation_mse` |

PPO does not have a conventional supervised validation loss. Its validation curves are measured task outcomes. The student has a genuine supervised validation MSE: its action predictions are compared with the selected expert's actions in a separate simulation environment.

Student validation uses seed 1913 and terrain seed 79, compared with training seed 42 and terrain seed 41 by default. Its 972 worlds cover all 324 geometry/depth/pace/level combinations at each of three route starts. Validation runs for 32 control ticks every 100 optimizer steps and at the last step. It restores Python, NumPy, CPU Torch and CUDA random streams and leaves policy weights and normalization unchanged. This loss is an action-imitation check; the full clean/stress episode suites separately measure successful robot movement.

Every logged TensorBoard scalar is exported, including individual reward terms. Native PPO episode statistics follow RSL-RL's primary-worker reporting; they should not be described as an independently pooled count over both workers. Held-out success and fall rates are pooled from explicit trial counts. Both GPUs have separate hardware telemetry and startup audits.

## Files and time axes

Each completed run saves:

- Original TensorBoard events, evaluation JSON, logs, source commit, package versions and resolved environment/agent settings.
- The original `profile.env`, the actual `effective-profile.env`, and the exact launch command.
- `metrics/scalars.csv`: experiment ID, run ID, metric, raw step, completed update, UTC timestamp, elapsed seconds and value.
- `metrics/summary.json`: available metrics, last values, evaluation results, and missing expected metrics.
- `metrics/plots/index.html` and a PNG for every measured metric, with update and elapsed-time axes.
- `metrics-report.zip`: a smaller download containing the reports and plots.

`<experiment>-report/metrics` compares the experiment's segments. `experiment-comparisons/metrics` compares recorded runs and experiments. Both include CSV tables, plots by completed update, and plots by elapsed experiment time. Different tasks may use different rewards; check task settings before comparing their reward values.

The exporter preserves raw source steps and rebases each resumed segment using its recorded starting update. It merges duplicate event entries using the latest timestamp. GPU telemetry uses its real time axis and is not assigned invented training updates. New evaluation files record their timestamp; older files use their saved modification time when no timestamp was recorded.

Values are plotted without smoothing. Missing measurements remain missing. A turn-only trial has no measured stair-climbing speed, so its unobserved speed is omitted. Counts and measured speed samples determine summary weights. A single evaluation point is shown as a point, not a learning trend. Synthetic test data is explicitly labelled and excluded from the automatic experiment catalog.

## Automatic GitHub delivery

The default is now the [independent GPU-server service](server-results.md), which restores prior published results on an empty VM and uploads directly. It does not require Windows to remain online. The watcher described below is an optional fallback. The fresh [reset-recovery verification](../experiments/results/server-recovery-20261112/) supersedes the previous session's offline status.

The VM exports plots and writes a content-digest publication queue after each completed training/evaluation segment. Failed runs also preserve diagnostics and queue available measurements. The reference and campus evaluation scripts refresh their reports too. Training runs do not require manually asking an agent to upload them.

The Windows watcher uses the existing SSH tunnel and GitHub CLI login. It downloads the full run archive, verifies SHA-256, uploads the archive/checksum and smaller metrics ZIP to a GitHub release, and acknowledges the exact queued digest. Interrupted transfers stay pending. Identical acknowledged snapshots are skipped; updated reports are uploaded again.

```powershell
./scripts/watch_stairs_results.ps1
```

Keep this PowerShell process running while results are produced. If the computer, tunnel or network is offline, rerun it after reconnection. The VM needs no copied GitHub token. The queue survives a connection interruption, but a VM reset can destroy any files that were not uploaded yet.

For future training, the following launcher first checks GitHub write access, both GPU profiles and NCCL communication. With no training switch it performs preflight only:

```powershell
./scripts/start_mastery_experiment.ps1 -RunId mastery-ascent-s42 `
  -Stage ascent -InitActorOnnx /absolute/path/to/policy.onnx
```

Adding `-StartTraining` is the explicit future action that starts training after GPU and server-delivery checks. Direct server publishing remains independent of Windows; add `-WatchFromWindows` only for the fallback watcher. Non-training preflight checks retry SSH transport errors up to three times. A training launch is never automatically retried. A unique launch lock prevents accidentally reusing the same run ID.

## Repeat verification without training

On Windows, this creates an isolated CPU reporting environment, runs the reporting and mocked workflow tests, checks launcher guards, then publishes the local report automatically:

```powershell
./scripts/verify_stairs_metrics.ps1
```

Use `-LocalOnly` if an upload is not wanted. CPU tests do not require Torch, MuJoCo or a GPU. They cannot establish that a disconnected VM is ready.

After [VM recovery](kaggle-recovery.md), run the full scripted verification on the VM:

```bash
.venv/bin/python -m scripts.verify_training_readiness \
  --output /kaggle/working/g1-edu-stairs/readiness-fresh-session \
  --gpu --full-evaluation \
  --teacher-onnx /absolute/path/to/round-trip/policy.onnx \
  --teacher-checkpoint /absolute/path/to/round-trip/model_159.pt
```

This tests concurrent 1,080-world capacity on each GPU, NCCL communication, all 13 task entries and controlled phases, actor transfer, history/curriculum recovery, independent student validation, and full-route/turn-specialist held-out evaluation. It calls no training producer and performs zero optimizer updates. Its queued results are picked up by the Windows watcher.

You can also regenerate plots or compare selected old runs without simulating:

```bash
.venv/bin/python -m scripts.training_metrics export --run /absolute/run --queue --catalog
.venv/bin/python -m scripts.training_metrics compare \
  --runs /absolute/run-a /absolute/run-b --output /absolute/comparison
```

## Current verification evidence

The [published reporting smoke archive](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-metrics-readiness-smoke-v1) contains 501 PNG plots, raw measurements from two earlier runs, labelled test fixtures and checks for duplicate events, resumed axes, pooled evaluation rates, and publication receipts. The watcher uploaded it successfully and skipped a second unchanged upload.

Before the VM went offline, concurrent capacity checks passed on both T4s with 1,080 worlds per GPU and about 13.5 GiB free per GPU. NCCL, the refreshed 13-task phase suite, independent 972-world student validation, and two held-out evaluations also passed their execution checks. No training was performed. Some later VM-only raw reports were not synchronized before disconnection; they are not claimed as recoverable artifacts. The published archive and local verified copy remain available.

The previous integrated launcher received both GPU pass messages, then SSH disconnected. That session is recorded in the [original verification record](../experiments/results/metrics-observability/). A new VM has since been rebuilt, 19 prior published runs restored, and fresh GPU checks and direct delivery verified; see the [current recovery record](../experiments/results/server-recovery-20261112/).
