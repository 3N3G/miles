# Four problems per optimizer update

Four-problem updates ran successfully. Nineteen repeated questions averaged **5.752 → 5.776/7**, a change of **+0.025 points**.
Six question means rose and thirteen fell. This does not establish a reliable learning gain.

The test asked whether accumulating more problems would improve the previously flat training rewards.
We continued Qwen3.5-4B for ten optimizer updates, each using four problems and 32 answers per problem.
The run consumed 1,280 answers across the same 20 easy questions from 17 papers.
Thinking remained enabled, with a 90,000-token output cap and learning rate 1e-6.
Luna medium used the unchanged seven-point rubrics and prompt version three, averaging both anonymous answer orders.
FOCS 2026 and STOC 2026 remained sealed.

The largest rise was the rooted-forest question: **1.406 → 3.141/7**.
Its capped-answer count fell from **24/32 to 15/32**.
That question asks whether an expected forest cost follows from marginal sampling probabilities, without independence.
The seed-selected latest example still reaches 90,000 tokens without a final answer, receiving **0/0** across orders.
The [raw example and original decisions](https://huggingface.co/datasets/3N3G/internalization-runs/blob/83b6d95ac83df247d9b50e5de0a7674d3378daf8/miles/easy-overfit-v17-batch128/seeded-proofs-108.json) show that the aggregate rise does not mean every answer succeeds.

All ten optimizer updates and rollout-weight publications passed runtime checks.
Dynamic scheduling accumulated **51–73 microbatches** per update.
All **811 W&B scalar values** matched native logs.
The run used **19.05 GPU-hours** and **$4.151** in estimated judge usage.
The final checkpoint preserves all ten new updates.

Judge orders disagreed on **177/1,280 answers**, including 72 gaps of at least two points.
These are provisional training scores, not measured proof accuracy.
Human agreement and independent-provider calibration remain unmeasured.
This is one continuation, with changing policies and asynchronous training visits.
No frozen-policy comparison, significance test or seed-level confidence interval was planned.

See the [matched-question plot, complete tables, raw proofs and audit scripts](https://huggingface.co/datasets/3N3G/internalization-runs/tree/83b6d95ac83df247d9b50e5de0a7674d3378daf8/miles/easy-overfit-v17-batch128),
[frozen plan](MULTIPROBLEM_BATCH_PLAN_20261123.md), and [execution record](RUN_V17_OPERATIONS_20261123.md).
