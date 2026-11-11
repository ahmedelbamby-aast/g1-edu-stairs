# Additive stair training improvements

The mastery tasks add the discussed improvements to the existing stair work. Every original task, phase, course family, speed band and difficulty level remains available. The new tasks use the same 108 round-trip geometries and all 324 combinations of family, tread depth, pace and level.

| Task | Purpose |
| --- | --- |
| `Unitree-G1-EDU-Stair-Mastery` | Complete ascent, turn and return with the improved curriculum and observations |
| `Unitree-G1-EDU-Stair-Expert-Ascent` | Practice climbing and settling on the upper landing |
| `Unitree-G1-EDU-Stair-Expert-Descent` | Start at the upper landing and practice controlled descent |
| `Unitree-G1-EDU-Stair-Expert-Turn` | Practice each turn direction and settle facing downhill |
| `Unitree-G1-EDU-Stair-Student` | Use simulated terrain and contact estimates with error, delay and missing measurements |

The left-leg and right-leg tasks, isolated up and down steps, mixed campus training, campus full flights, original stairs and original round trips continue to use their existing entry points.

## How the improvements work

The new curriculum keeps separate evidence for every geometry, speed band and practice assignment. It records success and trial counts over eight rolling evidence batches for each level. Each batch accumulates at least eight trials before advancing, so robots finishing at different times still build enough evidence. Promotion needs at least 64 completed trials and an 85% completion rate at the current frontier. Demotion uses a 40% rate. After moving the frontier, its evidence window starts fresh.

New episodes sample 60% from the frontier, 30% from previous levels, and 10% from harder levels. At either end of the curriculum, samples are bounded to the six existing levels. Fixed-level evaluation bypasses level sampling. Initial resets never count as failures. The evidence and frontiers are saved with checkpoints so a resume retains the curriculum state.

Downhill overspeed has an additional penalty proportional to squared relative speed error. A new load-transfer reward encourages loading the next foot while controlling vertical body motion. The original motion, landing, phase and failure rewards are retained.

The gait clock now integrates a speed-dependent period between 0.42 and 0.90 seconds. Changing a speed command therefore changes the rate of the clock without jumping its phase. Turn commands also keep the clock active.

The actor receives two upcoming footholds, with a valid flag for each, plus a second terrain scan at 10 cm spacing. The second scan has 13 × 9 points. The existing 63-point scan remains. A five-sample history adds recent joint positions and velocities, IMU state, actions, and foot-contact flags. Each history sample has 95 values. The new actor has 785 inputs.

The student task applies episode-level foothold bias up to 1.5 cm per axis, Gaussian target error with 1 cm standard deviation, and 2% missing target/contact estimates. Fine terrain readings have up to 2.5 cm of added uniform noise, level-dependent dropout, and up to 40 ms of delay. History and foothold estimates can be delayed by 20 ms. The teacher observation group provides a separate accurate input stream for imitation. These errors are simulation parameters awaiting calibration from the physical robot.

## Smoke checks

`scripts/smoke_stair_mastery.py` runs inference and short physics rollouts. It checks all legacy task entries, all new tasks, observation sizes, finite values, transfer from the saved round-trip actor, and ONNX output parity. Controlled pose/contact fixtures check each ascent and descent phase, both turn directions, and positive and negative single-leg hold cases. Additional checks exercise rolling promotion/demotion and history reset isolation.

The smoke suite performs zero optimizer steps and checks that policy weights and observation normalization remain identical before and after inference. Its exported `smoke-only.onnx` is a transfer-validation artifact.

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64
export WARP_CACHE_PATH=/codex_download_here_work_here/cache/warp
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
.venv/bin/python -m scripts.smoke_stair_mastery \
  --teacher-onnx /absolute/path/to/round-trip/policy.onnx \
  --teacher-checkpoint /absolute/path/to/round-trip/model_159.pt \
  --output /kaggle/working/g1-edu-stairs/mastery-smoke-unique
```

## Prepared future training workflow

The [mastery profile](../experiments/profiles/kaggle-mastery.env) defines 500 updates with checkpoints every 100. The experiment launcher supports ascent, turn, descent, round-trip and student stages. It runs clean and stress evaluations between 100-update segments, then resumes the full training state for the next segment. Choose different run IDs and `--seed` values for repeated experiments.

This command prints the prepared schedule without launching training.

```bash
.venv/bin/python -m scripts.run_mastery_experiment \
  --stage round_trip --run-id mastery-seed42 --seed 42 \
  --init-actor-onnx /absolute/path/to/round-trip/policy.onnx --plan-only
```

The imitation entry point accepts separate ascent, turn and descent checkpoints. It selects teacher actions according to the route stage, trains the student on noisy inputs, and gradually reduces the share of teacher actions used during simulated rollouts. It saves the teacher checksums, resolved environment and algorithm settings, imitation loss log, audit and resulting policy. It accepts the existing 185-input round-trip teachers as well as the new 785-input experts. `--check-only` verifies teacher/student inference and finite loss with zero optimizer steps.

```bash
.venv/bin/python -m scripts.distill_stair_student \
  --ascent /absolute/path/to/ascent/model_N.pt \
  --turn /absolute/path/to/turn/model_N.pt \
  --descent /absolute/path/to/descent/model_N.pt \
  --init-actor-onnx /absolute/path/to/round-trip/policy.onnx \
  --output /kaggle/working/g1-edu-stairs/student-check-unique --check-only
```

Training and imitation must be launched explicitly. Smoke results validate that the added paths execute and enforce their phase conditions. Task completion rates and physical robot performance require subsequent training and evaluation.
