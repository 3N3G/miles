# Easy-question continuation: small training-score gain

Forty additional updates raised mean training reward from **5.291 to 5.428 out of seven** on 15 repeated questions.
Eleven question means increased and four decreased. This descriptive change does not establish a reliable learning trend.

The question was whether further updates improve comparable visits while moving the observed objective in the correct direction.
Qwen3.5-4B continued from 25 saved updates, with thinking, GRPO/CISPO, and a 90,000-token output cap.
Each update consumed 32 answers to one question. Learning rate stayed at 1e-6, with 96 concurrent answer tasks.
Luna medium scored the supplied rubrics in both anonymous answer orders. Their candidate scores were averaged.
Training consumed 1,280 answers across 20 easy questions from 17 papers.
Five questions appeared once, ten twice, and five three times. These were not two balanced passes.
FOCS 2026 and STOC 2026 remained untouched.

| Matched first/latest visits | First | Latest |
|---|---:|---:|
| Mean rubric points | 5.291/7 | 5.428/7 |
| Full-credit answers | 177/480 (36.9%) | 190/480 (39.6%) |
| Mean response tokens | 27,605 | 32,620 |
| Answers hitting the output cap | 6/480 (1.25%) | 16/480 (3.33%) |

All six scheduled before/after checks increased the fixed-batch surrogate objective.
Full-policy entropy ranged from 0.374 to 0.691 nats/token.
Sampled divergence from initialization ranged from 0.00168 to 0.01552 nats/token.
All 40 updates completed without skips or nonfinite metrics. The checkpoint contains 65 total updates.

Candidate scores differed across judge orders for **338/1,280 answers (26.4%)**.
One diagnostic extreme received **4/7 and 0/7** for the same answer.
The [six-answer mathematical review](JUDGE_ROLLOUT_REVIEW_20261122.md) separates sound proofs, errors, and inconsistent grading strictness.
These two-call disagreements do not isolate position effects from judge sampling variability.
The [private report](https://huggingface.co/datasets/3N3G/internalization-runs/blob/b944b262c244e09338c0f07664111d688802a732/miles/easy-overfit-v12-diagnostic-20261122/REPORT.md)
links that raw answer, both decisions, all per-question trajectories, and a seeded example from every group.

These are training visits from one seed, without a frozen-checkpoint evaluation or seed-level confidence interval.
Longer answers, uneven visits, and uncalibrated judging limit interpretation. No sole cause of limited growth was identified.
Human calibration and any further evaluation or training allocation remain researcher decisions.

The [approved plan](EASY_CONTINUATION_PLAN_20261122_APPROVED.md) defines the bounded experiment.
The [closure evidence](evidence/easy_continuation_outcome_20261122.json) records repairs, resources, checkpoint inventory, and verified archives.
The [finished W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/261122-050822-879-native)
contains all 40 new steps. Use `audit/rollout/episode_raw_reward` against `audit/optimizer_step`.
