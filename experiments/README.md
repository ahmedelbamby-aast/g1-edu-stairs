# G1 EDU stairs experiments

The [Kaggle profile](profiles/kaggle-t4.env) sets the default rollout size and
CPU budget. Each [run release](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases)
is tied to the source commit and carries the full run directory and its checksum.
Training outputs on Kaggle live under `/kaggle/working/g1-edu-stairs`.

## Verification and throughput

| Run | Work per GPU | Result |
| --- | ---: | --- |
| [Dual-T4 smoke](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-smoke-dual-t4-v4) | 32 environments | Two PPO iterations and 24 headless evaluation cells. ONNX maximum action error was 6.33e-8. |
| [Dual-T4 768](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-t4-dual-768) | 768 environments | 23.2k transitions/s at a median 1.59 s per iteration after iteration 0. |
| [Dual-T4 1536](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-t4-dual-1536) | 1,536 environments | 32.9k transitions/s at a median 2.24 s per iteration after iteration 0. |
| [Dual-T4 3072](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-t4-dual-3072) | 3,072 environments | 38.2k transitions/s at a median 3.86 s per iteration after iteration 0. |

[Setup diagnostics](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-bootstrap-diagnostics)
records the initial dependency and command-line fixes.

## Reference training

Run `reference-t4-s42` uses the 160-update reference profile and Unitree's
flat-walking actor. The [resolved settings](results/reference-t4-s42/) record
the exact environment, algorithm, launch command, and actor transfer.
The [recovery release](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-reference-t4-s42-checkpoint)
contains a saved checkpoint and its settings.

The [full reference release](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-reference-t4-s42)
contains all nine checkpoints, the ONNX export, evaluation, TensorBoard and
GPU logs, and four 720p clips. The final checkpoint completed 299 of 320
held-out stair episodes. [Training report](../doc/training-report.md)

## Campus balance and stair phases

The [campus curriculum](../doc/campus-sim2real.md) adds separate balance and step drills,
six stair families, and a held-out test for each level and skill.

| Run | Evidence |
| --- | --- |
| [Campus smoke test](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-smoke-v2) | Two-GPU rollout, actor transfer, ONNX parity, and a 156-cell headless test. The landing-command retest completed all 24 level 0 full flights with the same checkpoint. |
| [Campus pilot](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-pilot-s42) | First training stage, stopped after saving update 100 to continue with the revised landing command. |
| [Campus refinement recovery copy](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-refine-s42-checkpoint) | Latest saved checkpoint and audit. The [resolved settings](results/campus-refine-s42/) include the source checkpoint checksum and realized stair geometry. |
| [Campus refinement full run](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-refine-s42) | Combined policy, complete audit, and 156-cell clean and stress tests. It completed 519/576 isolated steps and 250/576 full flights under stress. |
| [Left-leg balance](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-balance-left) | Independent specialist checkpoint, audit and evaluation for the left support leg. |
| [Right-leg balance](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-campus-balance-right) | Independent specialist checkpoint, audit and evaluation for the right support leg. |

## Round-trip stairs

The [round-trip task](../doc/round-trip-stairs.md) trains ascent, a turn on the upper
landing, and descent on the same treads. The startup audit verifies full-route
practice in all 324 family, depth, speed and difficulty combinations on each GPU.

| Run | Evidence |
| --- | --- |
| [Round-trip smoke test](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-round-trip-smoke-s42) | Two-GPU PPO and verified actor transfer. |
| [Round-trip pilot recovery copy](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-round-trip-pilot-s42-checkpoint) | Saved checkpoint and run audit. [Resolved settings](results/round-trip-pilot-s42/) |
| [Round-trip pilot full run](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-round-trip-pilot-s42) | 160 PPO updates. Evaluation completed 263/648 clean and 223/648 stress routes. Of those, 144 clean and 109 stress trials met the requested pace in both directions. |
