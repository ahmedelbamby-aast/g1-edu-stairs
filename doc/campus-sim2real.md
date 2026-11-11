# G1 EDU campus stairs and single-leg skills

The campus task trains one policy across left-leg balance, right-leg balance, isolated steps, and full flights. The two balance modes can also run on their own. All tasks retain the same Unitree G1 29-joint model, joint order, and 50 Hz policy rate.

## Skill phases

| Phase | What the robot practices | How the simulator checks it |
| --- | --- | --- |
| Approach | Walk into range of the next step | Pelvis reaches the approach zone |
| Shift | Move weight toward the assigned support foot | Foot is loaded and the center of mass moves toward it |
| Lift | Raise the other foot | Swing foot leaves contact and reaches a clearance height |
| Clear | Move the swing foot past the edge | Swing foot reaches the edge above the higher surface |
| Plant | Put the swing foot on the target tread | Foot contact and load are within that tread's landing area |
| Transfer | Move weight onto the new support foot | New foot carries the required share of the load for consecutive ticks |
| Hold | Balance on the named leg | Other foot stays raised while the robot remains upright and nearly still |
| Finish | Slow down and stop on the final landing | Both feet land beyond the final riser and the body stays upright below the finish speed limit |

Balance drills run Shift → Lift → Hold. A left-leg drill requires the **left** foot to support the robot throughout the hold. A right-leg drill requires the **right** foot. Single-step drills end after Transfer. Full flights repeat the step sequence and finish with both feet on the far landing. The combined task allocates 20% of worlds to each support leg, 20% to isolated steps, and 40% to full flights. Both stair directions are represented in the step and flight groups.

## Campus stair designs

Each family runs in ascent and descent. Levels 4 and 5 use the full nominal riser and tread dimensions. These models reproduce the cited dimensions on straight stair flights. The source documents also describe building details such as nosings, handrails and drainage that this physics model does not represent.

| Family | Riser | Tread | Source |
| --- | ---: | ---: | --- |
| Georgia Tech plaza | 5 in / 12.70 cm | 15 in / 38.10 cm | [Georgia Tech Green Book, exterior stairs](https://facilities.gatech.edu/system/files/forms_files/greenbookfinal-20200810.pdf) |
| LSU building entry | 6 in / 15.24 cm | 14 in / 35.56 cm | [LSU Campus Site & Landscape Standards](https://www.lsu.edu/pdc/files/lsu_site_and_landscape_standards.pdf) |
| Wyoming exterior concrete | 6 in / 15.24 cm | 12 in / 30.48 cm | [University of Wyoming Design Guidelines, exterior concrete stairs](https://www.uwyo.edu/uwops/_files/construction_standards_guidelines/design-guidelines-march-2023.pdf) |
| Virginia Tech exterior | 7 in / 17.78 cm | 12 in / 30.48 cm | [Virginia Tech design manual, §4.5.4](https://www.facilities.vt.edu/content/dam/facilities_vt_edu/design-and-construction-standards/vt-design-and-construction-standards-manual-2021.pdf). This pair satisfies the manual's 26-inch formula. |
| UConn exterior | 7 in / 17.78 cm | 15 in / 38.10 cm | [UConn design guidelines, §6.7](https://updc.media.uconn.edu/wp-content/uploads/sites/1525/2025/10/Volume-One-DESIGN-GUIDELINES-September-2025.pdf) |
| Virginia Tech split-flight variation | 7 in / 17.78 cm | 12 in / 30.48 cm | Same nominal dimensions, with an extra metre of intermediate landing added in simulation. |

## Difficulty progression

The thresholds increase independently for every family. Each world advances one level after two consecutive completions and drops one after two consecutive failures. The final level keeps the full-height stairs and makes the disturbances and balance requirement harder.

| Level | Risers in a flight | Riser scale | Added tread depth | Lane width | One-leg hold | Minimum lift target | Push per horizontal axis | Terrain samples dropped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 30% | 15 cm | 2.15 m | 0.30 s | 4.5 cm | 0.00 m/s | 0% |
| 1 | 2 | 50% | 11 cm | 2.00 m | 0.55 s | 6.0 cm | 0.04 m/s | 1% |
| 2 | 3 | 72% | 7 cm | 1.85 m | 0.85 s | 7.5 cm | 0.08 m/s | 2% |
| 3 | 4 | 88% | 3 cm | 1.70 m | 1.20 s | 9.0 cm | 0.12 m/s | 3% |
| 4 | 6 | 100% | 0 cm | 1.60 m | 1.60 s | 10.5 cm | 0.16 m/s | 4% |
| 5 | 6 | 100% | 0 cm | 1.60 m | 2.10 s | 12.0 cm | 0.22 m/s | 7% |

The stress configuration also varies foot friction from 0.38 to 1.25, encoder bias up to 0.020 radians, motor stiffness from 88% to 112% of nominal, motor damping from 85% to 115%, and torso mass and inertia together. It models up to 15 ms of actuator delay and up to 40 ms of terrain scan delay. These are simulation ranges pending measurements from the physical G1 EDU.

## Train and evaluate on Kaggle

Run from the checkout after [VM setup](kaggle-recovery.md). The verified reference actor can initialize the campus policy. New skill inputs start with zero weights, and training checks that this transfer preserves the teacher's output on the same inputs.

```bash
cd /codex_download_here_work_here/g1-edu-stairs
mkdir -p ../teachers
curl -fL https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/download/run-reference-t4-s42/reference-policy.onnx \
  -o ../teachers/reference-policy.onnx
teacher="$PWD/../teachers/reference-policy.onnx"
bash scripts/run_kaggle_campus.sh campus-001 --init-actor-onnx "$teacher"
```

The [campus profile](../experiments/profiles/kaggle-campus.env) starts at 1,200 worlds per T4. To train a specific skill, set `STAIRS_TASK` to one of these task names and use a unique run ID.

- `Unitree-G1-EDU-Balance-Left`
- `Unitree-G1-EDU-Balance-Right`
- `Unitree-G1-EDU-Step-Up`
- `Unitree-G1-EDU-Step-Down`
- `Unitree-G1-EDU-Campus-Flights`

Run the evaluator twice with `--suite clean` and `--suite stress`. It uses held-out terrain and reports each balance leg, each isolated step, and each full flight at every level. By default, each of the 156 cells gets 16 episodes. The simulation gate requires at least 95% completion and no falls per cell. Add `--require-pass` when a failed gate should return a nonzero exit code.

```bash
bash scripts/kaggle_stairs.sh evaluate-campus --suite stress \
  --checkpoint /absolute/path/to/model_N.pt --onnx /absolute/path/to/policy.onnx \
  --output /kaggle/working/g1-edu-stairs/campus-001/eval-stress.json
```

## What every run records

The run directory contains the exact launch command, profile, source commit, package versions, GPU telemetry, resolved environment and PPO settings, seeds, and teacher checksum. The audit directory records the realized edges and heights of all 72 stair courses. Each GPU saves its startup physics values, initial poses and course assignments, and its effective process settings. TensorBoard records progress through the phases and outcomes for each drill. The evaluation JSON includes a separate count for each phase reached, completed risers and longest single-support hold by test cell. Per-tick random draws are driven by the recorded seeds and ranges.

## Physical G1 EDU integration

The run exports a 180-input ONNX policy and a `sim2real-contract.json` with the ordered signals, joint order, action scaling and control period. The policy requires live estimates of terrain, the next foothold, swing foot position and foot contacts. The simulator provides these estimates from its physics state. SDK2 supplies joint state and IMU readings, and a robot-side estimator must supply the added signals. [Unitree G1 SDK example](https://github.com/unitreerobotics/unitree_ros2/blob/master/example/src/src/g1/lowlevel/g1_low_level_example.cpp)

Before hardware acceptance, measure the actual motor and sensor delays, calibrate the terrain frame and foot-contact estimates, and replay recorded observations through the export. Then validate balance on each leg, an isolated low step in each direction, and finally full flights with supervised physical tests. The simulator's completion gate is one input to that decision.
