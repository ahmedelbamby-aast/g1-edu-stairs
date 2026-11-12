# Dual-T4 stair training performance

The round-trip mastery continuation uses 6,480 simulated robots per T4, two CPU threads per worker, and 24 control steps per robot per PPO update. The environment runs 12,960 robots across both GPUs. The measured configuration processes 27,256 transitions per second, 2.76 times the original 1,080-world-per-GPU profile. [Benchmark plots, run commands, and raw measurements](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-mastery-t4-throughput-20261112)

| Robots per GPU | Runtime path | Median transitions/s | Average utilization, T4 0 / 1 | Peak sampled GPU memory, T4 0 / 1 |
| ---: | --- | ---: | ---: | ---: |
| 1,080 | Original | 9,878 | 54% / 47% | 1.41 / 1.29 GiB |
| 3,240 | Original | 22,482 | 72% / 71% | 3.38 / 3.25 GiB |
| 6,480 | Original | 24,713 | 86% / 86% | 7.02 / 6.90 GiB |
| 6,480 | Optimized | 27,256 | 88% / 89% | 6.39 / 6.27 GiB |
| 9,720 | Optimized | Startup failed | — | — |

Throughput uses real dual-GPU PPO updates with three initial updates excluded. The remaining measurement window is nine updates for each original configuration and five for the optimized configuration. GPU utilization is averaged from two-second samples during that window. Peak memory is sampled per update, so short-lived allocations may differ. The 9,720-world trial was killed during startup with an out-of-memory kill recorded by the VM's 30 GiB system RAM cgroup. [Failed trial log and settings](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-mastery-bench-fast-n9720-t2)

The CPU and CUDA trace shows that rollout collection takes most of each update. Python-side launch and synchronization work includes repeated delay-buffer updates and foot-pose calculations. The tuned path skips the teacher observation group during PPO when no model consumes it, keeps history initialization on the GPU, and reads foot positions directly from the same simulation state. The parity smoke checked 960 history reads on each of CPU and GPU, exact foot-position agreement on the same state, and finite optimized rollout observations and rewards. [PyTorch CPU/CUDA and Python profiling artifacts](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-mastery-profile-1080-s42)

The production workflow, `mastery-t4-continue-s42`, resumes the mastery profiler run's `model_7.pt` with its actor, critic, optimizer, and saved curriculum state. The profiler run was initialized from the earlier round-trip pilot. The continuation schedules 500 additional PPO updates, saves checkpoints at interval 10, and evaluates 8 trials per cell in each clean and stress suite every 100 updates. The two suites run on separate GPUs. Larger batches supply more samples per update, so held-out evaluations measure the resulting policy quality. The configured physics, curriculum levels, route stages, and PPO coefficients are preserved.

```bash
NUM_ENVS=6480 CPU_THREADS=2 SAVE_INTERVAL=10 \
  .venv/bin/python -m scripts.run_mastery_experiment \
  --run-id mastery-t4-continue-s42 --stage round_trip --seed 42 \
  --resume-checkpoint /absolute/path/to/mastery/model_7.pt \
  --updates 500 --evaluate-every 100 --episodes-per-cell 8 --performance-options
```

Use a new run ID to repeat the experiment. For recovery after interruption, supply the latest compatible saved checkpoint and the number of updates still needed. Update indices resume after the checkpoint's recorded update. The server result service publishes each completed segment, comparison plots, raw metrics, and downloadable checkpoints. [Server delivery and recovery](server-results.md)
