# Easy and medium TCS reinforcement learning

The same-question comparison shows no score gain after the mixed-difficulty continuation.
Qwen3.5-4B generated 32 answers on each of six easy and six medium questions before and after training.
GPT-6 Luna, at high reasoning effort, graded final answers against labeled references and weighted Boolean rubrics.
Scores range from 0 to 7. Thinking, sampling and the 90,000-token generation cap matched.

| Difficulty | Mean before → after | Full-credit answers before → after | Capped answers before → after |
|---|---:|---:|---:|
| Easy,192 answers per checkpoint |5.760 →5.760 |127 →125 |4 →2 |
| Medium,192 answers per checkpoint |1.995 →1.969 |34 →33 |103 →77 |

The continuation completed 28 saved optimizer updates and 3,584 answers across 112 of 236 prepared question rows.
Each update used four questions with 32 answers each. Total RL allocation time was 11 h 27 m 39 s on eight AMD GPUs.
The pinned [TCS v3 corpus](https://huggingface.co/datasets/ian-and-friends/tcs_v3/tree/8e486f2d980d926fa5543efe7f1f2adf475ce3ea) contains 20 easy and 216 medium questions. It was not filtered by observed rewards.

Mean response length fell on all twelve screen questions. Medium empty finals fell from 74 to 54.
Completed zero-score medium answers rose from 14 to 31. Finishing more often did not increase full-credit answers.
One uniformly sampled sorting answer wrongly applies a worst-case information bound to every execution. Luna correctly gives zero.
A sampled auction proof correctly computes expected surplus as exactly two-thirds of optimum and earns seven.
[All twelve paired raw examples and plots](https://huggingface.co/datasets/3N3G/internalization-runs/tree/7da39fd097ddca0c46f8853b5cf8f3ffc1183f58/miles/checkpoint-final236-screen-v5/comparison) accompany the aggregates.

The baseline already received easy-question RL. This is not a comparison with the original pretrained model.
Six screen questions entered the new training updates. One training seed and twelve training-eligible questions do not establish generalization or seed uncertainty.
FOCS 2026 and STOC 2026 remain sealed. The judge has assistant audits, without independent human calibration.
The [process audit](PROCESS_AUDIT_20261126.md) separates verified behavior, reward limitations and proposed controls.
