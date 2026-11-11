# Mastery implementation verification

This run used source commit `0bbba97` and performed **zero PPO or imitation optimizer steps**. It smoke-tested eight existing task entries and five additive mastery entries on the headless Kaggle VM.

| Check | Result |
| --- | --- |
| Existing task registration and short physics rollouts | 8 passed |
| New task inference and physics rollouts | 5 passed; 324 coverage worlds each |
| Existing left and right single-leg balance | Correct single-support holds accepted; two-foot holds rejected |
| Ascent and descent phase fixtures | Approach, shift, lift, clear, plant, transfer, finish passed |
| Upper-landing turns | Both directions passed |
| Additional controls | Downhill overspeed and controlled load transfer passed |
| Two-step lookahead | Both footholds valid before the first riser |
| Adaptive curriculum | Promotion, demotion, asynchronous evidence accumulation and fixed-level handling passed |
| Curriculum checkpoint save/load | Passed using the real runner entry points |
| Student history reset isolation | Passed |
| Maximum transferred actor error | 9.5367e-7 |
| New ONNX/PyTorch maximum action difference | 5.3644e-7 |
| Student CLI check on second T4 | Passed with `--check-only`; zero optimizer steps |
| Future experiment launcher | `--plan-only` printed the 500-update schedule |

[Raw smoke result](smoke.json), [student CLI manifest](distillation.json), [prepared experiment plan](experiment-plan.json), and [exact source commit](git-commit.txt).

The phase fixtures supply controlled poses and contacts to test transitions. The short physics rollouts check execution and finite values. These checks do not measure learned success rates or validate physical robot operation. All 108 stair geometries and three paces remain available. No prior level or stage was removed.

The student check reused the saved legacy round-trip policy for all three teacher slots. This validates loading and inference; it does not create new expert policies. The smoke checkpoint is marked `smoke_only`. Do not treat it as a newly trained result.

The full [GitHub recovery archive](https://github.com/ahmedelbamby-aast/g1-edu-stairs/releases/tag/run-mastery-smoke-v5) contains configuration snapshots, startup audits, software versions, logs, and check artifacts. A checksum-verified local copy is stored under `outputs/runs/mastery-smoke-v5`.
