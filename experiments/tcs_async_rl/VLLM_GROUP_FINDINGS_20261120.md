# The compatible inference backend also gives a constant-zero reward group

Frozen Qwen3.5-4B returned 32 responses on the fixed hard polygon-packing question.
Twenty-eight contained final text. All 28 received zero of seven rubric points in both judge orders.
Nine responses reached the 90,000-token cap and ended in repetition. Four had no final answer.
This group fails completion and reward-variation requirements. No RL job was submitted.

The question asks for constructive approximation algorithms for packing convex polygons using translations.
It requires factors 51 for strip width, 11 for unit-square bins and 7.3 for container perimeter.
The question came from SODA 2023 in `ian-and-friends/tcs_v3`, revision `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
It was the original randomly selected training question, rather than a question chosen after observing answers.
FOCS 2026 and STOC 2026 remained sealed.

The generator used Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with thinking enabled.
vLLM 0.29.0 ran eager, target-only BF16 inference on one MI300 GPU using ROCm 7.2.3.
The GPU passed actual FP32 and BF16 kernel controls before model loading.
Generation used temperature 1, top-p 0.95, top-k 20 and seeds 20261118 through 20261149.
All 32 requests were admitted concurrently. The output ceiling was 90,000 tokens and total context was 98,304.
The formatted prompt had 374 tokens. No model parameters changed.

GPT-6 Luna at medium effort graded final text against the supplied six binary rubric criteria, totaling seven points.
Each answer was judged anonymously against the supplied reference in both presentation orders.
The reward is the mean candidate score across those orders. Missing final answers remained ungraded, rather than becoming zero.

| Measurement | Result |
|---|---:|
| Returned generations | 32 |
| Natural stops | 23 |
| Length-limit stops | 9 |
| Nonempty final answers | 28 |
| Graded final answers | 28 |
| Zero-point final answers | 28 |
| Observed variance of the 28 mean rewards | 0 |
| Generated tokens, minimum / median / maximum | 28,548 / 47,443 / 90,000 |
| Total generated tokens | 1,763,680 |
| GPU allocation elapsed time | 97 minutes |

The reference, empty and irrelevant *candidate controls* scored 7, 0 and 0 in both orders.
The paired reference received seven points in 61 of 62 judgments. One judgment gave it two points.
That judgment claimed a missing additive term in the reference's factored area bound.
Direct multiplication confirms the displayed factorization. Its stated criticism is algebraically incorrect.
The original verdict remains preserved. No reward was rewritten.

A separate read-only audit found two earlier non-seven reference verdicts in saved runs on this same question.
Their stated geometric and algebraic objections also fail the local checks recorded with the receipts.
These checks do not certify every step of the complete reference proof or establish a judge error rate.
The [private reference audit](https://huggingface.co/datasets/3N3G/internalization-runs/tree/c8377cac7ad4a3ccf7e9e8426ed1d4ee6e90964a/miles/judge-reference-audit-v1)
contains the exact objections, source hashes and independent checks. Human calibration remains necessary.

Reporting seed 20261118 selects response index 13. Both orders awarded it zero on all six criteria.
It proposes putting arbitrary polygons into unit-square bins before assembling the perimeter container.
A single allowed 2×2 square cannot fit in any unit-square bin under translation. This independently invalidates that step.
One judge explanation also overstates a different area-comparison claim in the answer.
The saved spot-check records that limitation separately from the actual infeasible unit-bin step.
The private archive contains the complete selected answer and its criterion decisions beside the aggregate plots.

The [native repaired group](REPAIRED_GROUP_FINDINGS_20261120.md) independently failed, with 28 zero-point finals and four missing finals.
Its five capped paths differ from this backend's nine capped paths. Shared seed numbers do not imply identical sample paths.
Do not pool these repeated generations as independent papers or infer a causal backend or sampling-filter effect.

These results concern one question. They do not estimate corpus-wide model capability or learning.
The grades remain provisional without human calibration and an independent provider.
This backend does not qualify native sampling replay, trainer backward computation, checkpoint conversion or optimizer updates.

The [immutable private run archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/7316dbeab2f14afbc00912c6efa0832fde73f6a9/miles/group-check-v13-mi300-vllm32)
contains inputs, outputs, both-order receipts, plots, calibration examples and the independent spot-checks.
Operational details remain in `jobs.jsonl` and the evidence records.
