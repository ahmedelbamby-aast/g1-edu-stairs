# Unitree G1 EDU stairs

`Unitree-G1-EDU-Stairs` uses the 29-joint Unitree G1 model, action order, PD gains
and default joint positions from the official G1 velocity task. Physics runs at
200 Hz and the policy sets joint-position targets at 50 Hz.

## Courses and curriculum

Each level includes standard and compact courses in both directions. The robot
faces forward along the lane for ascent and descent. Descending episodes start
on the upper landing. Compact courses have shorter treads, narrower lanes,
and, from level 2, up to 8% variation in individual risers and treads.

| Level | Risers | Nominal riser | Standard tread | Compact tread |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0 | Flat | — | — |
| 1 | 2 | 4.5 cm | 36 cm | 31.5 cm |
| 2 | 3 | 7.5 cm | 35 cm | 30.5 cm |
| 3 | 4 | 10.5 cm | 34 cm | 29.5 cm |
| 4 | 5 | 14 cm | 33 cm | 28.5 cm |
| 5 | 6 | 17 cm | 32 cm | 27.5 cm |

Training starts on level 0. A traversal succeeds when the pelvis passes the
finish line and both feet are supported on the landing while the robot is
upright. Success promotes the environment one level. Falling, leaving the
lane, or timing out before 65% of the course drops it one level. A late timeout
repeats the same level. Successful environments on level 5 stay there.

## Train headlessly on Kaggle

Clone or update the fork under `/codex_download_here_work_here`, then run from
its root. The launcher puts the environment and caches under the scratch path
and writes run outputs to `/kaggle/working/g1-edu-stairs/<run-id>`.

```bash
bash scripts/kaggle_stairs.sh install
bash scripts/kaggle_stairs.sh smoke smoke-001
bash scripts/kaggle_stairs.sh train stairs-001 \
  --init-actor-onnx deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx \
  --agent.algorithm.learning-rate 0.0003
```

The launcher uses synchronous PPO across both T4s. Each GPU owns the configured
number of simulation environments and PPO averages gradients across ranks.
Two CPU threads per worker leave four threads total for the two workers.
Set `NUM_ENVS`, `GPU_IDS` (for example `'[0]'`), `ITERATIONS`, `SAVE_INTERVAL`,
or `SEED` in the shell to override the checked-in profile. The `benchmark`
mode runs a short trial and saves GPU memory and utilization samples.

Warm-starting copies the official flat actor and its proprioception
normalization. The first layer starts with zero weights for the 63 new
terrain inputs. The launcher checks the transferred action outputs against
the official ONNX before PPO updates begin. Omit the initialization flag
to start from random weights. To continue a run, use a new run ID and
`--resume-checkpoint /absolute/path/to/model_N.pt`. Resume restores the
policy, critic and optimizer. New episodes start at the configured initial
terrain level.

## Evaluate and publish

Run the headless evaluator with the selected checkpoint and, for the latest
checkpoint in a run, its matching ONNX export. It uses a separate terrain
seed and reports results for all 24 combinations of course and difficulty.

```bash
bash scripts/kaggle_stairs.sh evaluate \
  --checkpoint /kaggle/working/g1-edu-stairs/stairs-001/g1_edu_stairs/<timestamp>/model_N.pt \
  --onnx /kaggle/working/g1-edu-stairs/stairs-001/g1_edu_stairs/<timestamp>/policy.onnx \
  --output /kaggle/working/g1-edu-stairs/stairs-001/evaluation.json
```

Replace `<timestamp>` and `N` with the run's directory and checkpoint number.
Each completed run includes the launch command, source commit, dependency
versions, environment and PPO settings, console and TensorBoard logs, GPU
samples, and any produced checkpoints, exports and evaluations. From the
Windows checkout, upload the run and its SHA-256 checksum as a GitHub release
using the locally authenticated GitHub CLI.

```powershell
./scripts/sync_stairs_run.ps1 -RunId stairs-001
```

## SDK2 deployment interface

The exported actor uses the same motor command layout and proprioceptive
input order as Unitree's flat actor. The stair actor adds 63 terrain inputs
after those 98 values. The scan samples a 9-by-7 grid aligned to pelvis yaw
at 20 cm spacing, with x increasing within each y row. Each input represents
pelvis height minus measured terrain height minus 0.8 m. A missing ray is
encoded as 1.7 m.

The stock SDK2 controller supplies the original inputs. Deploying the stair
actor requires a calibrated terrain-height provider that fills the added
scan inputs and a heading command aligned with the flight. This project
validates policy execution in simulation. Physical G1 EDU testing is pending.
