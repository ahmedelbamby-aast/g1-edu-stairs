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
| [Flat actor warm-start smoke](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-warm-flat-smoke) | 32 environments | Two PPO iterations initialized from Unitree's flat actor. Maximum transfer error was 7.15e-7. |
