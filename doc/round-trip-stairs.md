# G1 EDU round-trip stairs

The robot starts below a flight, climbs it, turns about 180° on the upper landing, and descends **the same physical treads**. It finishes upright with both feet down on the lower landing. The ascent and descent use contact checks for approach, weight shift, lift, edge clearance, foot placement, and load transfer. During the turn the robot commands zero forward speed. It must settle on the upper landing facing downhill before descent can start.

## Course and pace diversity

The six campus families retain their riser proportions. Each has short, medium, and long tread variants in this task. The split-flight family also has an intermediate landing. These variants use the campus designs as a starting point for training geometry.

| Tread band | Nominal depth at levels 4–5 |
| --- | ---: |
| Short | 25–28 cm |
| Medium | 31–35 cm |
| Long | 38–44 cm |

Seeded depth variation changes individual steps by up to 1.2 cm. The easier tiers add up to 6 cm to each regular tread. Level 0 has two shallow risers so that tread length matters even on the easiest course.

| Level | Risers | Riser height relative to the campus family | Added regular tread depth | Lane width | Upper landing depth |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 30% | 6.0 cm | 2.15 m | 1.68 m |
| 1 | 2 | 50% | 4.5 cm | 2.00 m | 1.57 m |
| 2 | 3 | 72% | 3.0 cm | 1.85 m | 1.46 m |
| 3 | 4 | 88% | 1.5 cm | 1.70 m | 1.36 m |
| 4 | 6 | 100% | 0 cm | 1.60 m | 1.28 m |
| 5 | 6 | 100% | 0 cm | 1.60 m | 1.22 m |

Level 5 also uses the highest push and terrain dropout settings from the [campus curriculum](campus-sim2real.md). The same motor, sensor, foot-friction, inertia, and delay randomization remains active in stress training.

| Pace | Forward command on the stairs |
| --- | ---: |
| Slow | 0.20–0.30 m/s |
| Normal | 0.38–0.48 m/s |
| Fast | 0.57–0.70 m/s |

Each episode samples a speed within its band and uses it on both legs of the trip. The policy slows down on each landing. Turns alternate clockwise and counterclockwise. Evaluation measures actual speed along the stairs separately in each direction, including pauses or backtracking. A pace-matched result requires completing the entire trip and staying within 35% of the commanded average speed in both directions.

## How the worlds train

The two-T4 profile runs 1,080 worlds on each GPU. In each geometry and pace pair, six worlds rehearse the six fixed levels from the bottom. Eight start from the bottom and progress when they complete routes. Three practice the turn on the upper landing and three practice descent from the upper landing. The fixed-level worlds ensure the hardest cases appear in training from the start.

The held-out evaluation runs every **family × depth × pace × level** combination, for 324 cells. Every test starts at the bottom, and each cell includes both turn directions. It records upper landings, completed turns, full returns, falls, lane exits, timeouts, and actual ascent and descent speed. The pilot uses two trials per cell in each suite. The default evaluator uses eight.

## Start a run

After [Kaggle setup](kaggle-recovery.md), the combined campus policy can initialize the round-trip policy. Five new route inputs tell the policy which part of the trip it is on, how far it is from the next stop, and which way it needs to face. The original 180 inputs retain their order.

```bash
cd /codex_download_here_work_here/g1-edu-stairs
mkdir -p ../teachers
curl -fL https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/download/run-campus-refine-s42/campus-policy.onnx \
  -o ../teachers/campus-policy.onnx
bash scripts/run_kaggle_round_trip.sh round-trip-001 \
  --init-actor-onnx "$PWD/../teachers/campus-policy.onnx"
bash scripts/evaluate_kaggle_round_trip.sh round-trip-001
```

Every run saves its resolved PPO and environment settings, teacher checksum, GPU telemetry, actual stair geometry, phase metrics, and sampled startup physics values. The audit also records each world's initial pace, turn direction, and practice assignment. The archive includes the checkpoints and both evaluation configurations. [Experiment index](../experiments/README.md)
