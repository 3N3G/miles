# Easy-question RL after grading clarification

Clearer grading and a supervised resume did not establish a consistent reward increase.
Fourteen repeated questions averaged **5.067 → 4.969 points out of seven**.
Six increased, seven decreased, and one stayed unchanged.

The pilot used Qwen3.5-4B with thinking enabled and 90,000 output tokens.
It trained on 20 easy theoretical computer science questions from 17 papers.
Each update consumed 32 answers to one question using asynchronous reinforcement learning.
The judge was GPT-6 Luna at medium reasoning effort.
It answered binary rubric questions with the supplied weights, averaging two reversed answer orders.
Judge prompt version three clarified equivalent arguments, elementary implicit steps, and unfinished answers.

The latest allocation saved 14 updates and consumed 448 answers, totaling 21,182,426 response tokens.
Together with the preceding durable continuation, the comparable trajectory contains 34 updates and 1,088 answers.
The comparison excludes four unsaved predecessor updates and uses each repeated question's first and latest visit.
These are sampled training visits from one seed, without frozen-policy evaluation or between-seed confidence intervals.
FOCS/STOC 2026 remained sealed.

Matched answer length rose from 52,355 to 56,344 tokens.
The fraction reaching the token cap rose from 17.6% to 23.0%.
For example, the forest question declined from 2.672 to 1.531 points.
A [randomly selected zero](https://huggingface.co/datasets/3N3G/internalization-runs/blob/6eb6ab1a9781ec1392d18cf6144e21401c93d5db/miles/easy-overfit-v16-diagnostic-20261123/zero-example-step-88.json)
asserts false submodularity and probability claims.
A separate seed-selected answer gives a valid parent-edge deletion proof and receives seven in both orders.

All optimizer metrics were finite. Importance-weight effective sample-size ratios exceeded 0.9991, with zero clipping.
The observed update-direction check was positive. Matched entropy fell from 0.526 to 0.498 nats per token.
Sampled divergence from initialization rose from 0.00779 to 0.00928.
These observations do not establish improved answers or rule out every implementation problem.

Judge orders disagreed on 45/448 answers. One identical answer received [two versus seven points](https://huggingface.co/datasets/3N3G/internalization-runs/blob/6eb6ab1a9781ec1392d18cf6144e21401c93d5db/miles/easy-overfit-v16-diagnostic-20261123/maximum-order-gap-example.json).
Two paired references received a deduction in one order.
Human calibration remains unmeasured. Agreement is not proof of correctness.

The planned cost guard stopped training after a verified checkpoint containing 99 total updates.
Shared estimated judge spend settled at $4.689. All 1,127 final W&B scalar values matched native events.
The [complete comparison and plots](https://huggingface.co/datasets/3N3G/internalization-runs/tree/6eb6ab1a9781ec1392d18cf6144e21401c93d5db/miles/easy-overfit-v16-diagnostic-20261123)
preserve every sampled example. [Operational details](RUN_V16_OPERATIONS_20261123.md) record limits, repairs, and checkpoint verification.
Human rubric adjudication is the next unresolved step before interpreting another training variant.
