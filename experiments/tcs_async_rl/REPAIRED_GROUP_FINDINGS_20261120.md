# Bounded sampling still gives zero rewards on the fixed polygon-packing question

Frozen Qwen3.5-4B returned 32 responses on one hard TCS question.
Twenty-eight contained nonempty final text. All 28 received zero of seven rubric points in both judge orders.
Five responses reached 90,000 generated tokens and ended in repeated text. Four had no final answer.
This group fails the required completion and reward-variation checks. No RL update followed it.

The question asks for constructive proofs of three approximation bounds for packing convex polygons.
The requested factors are 51 for strip width, 11 for unit-square bins and 7.3 for container perimeter.
The question came from SODA 2023 in `ian-and-friends/tcs_v3`, revision `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
It was the original uniformly selected training question, rather than a question chosen after seeing model answers.
FOCS 2026 and STOC 2026 remained sealed.

The generator used Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with thinking enabled.
Native SGLang ran target-only decoding on one MI350 GPU. Generation used temperature 1, top-p 0.95 and top-k 20.
The output ceiling was 90,000 tokens and total context was 98,304. The formatted prompt had 374 tokens.
The group used seeds 20261118 through 20261149. There was no training.

GPT-6 Luna at medium effort graded final text against the supplied six binary rubric criteria.
Their weights sum to seven. Each answer was judged anonymously in both presentation orders against the reference.
The reward is the mean candidate score across those orders. Empty final answers remained ungraded in this diagnostic.
Reference, empty-answer and irrelevant-answer controls received 7, 0 and 0 points respectively, in both orders.

| Measurement | Result |
|---|---:|
| Returned generations | 32 |
| Natural stops | 27 |
| Length-limit stops | 5 |
| Nonempty final answers | 28 |
| Graded final answers | 28 |
| Zero-point final answers | 28 |
| Observed variance of the 28 mean rewards | 0 |
| Generated tokens, minimum / median / maximum | 30,132 / 44,048 / 90,000 |
| Total generated tokens | 1,647,186 |
| GPU allocation elapsed time | 17 minutes 44 seconds |

The previously unfinished seed now stopped naturally at 47,474 tokens. Other seeds still repeated until the cap.
The actual 16-token control also passed sampling-support capture and normalized behavior-log-probability checks.
Those successes qualify specific generation plumbing. They do not qualify trainer backward computation or produce a usable reward group.

One example was selected with reporting seed 20261118, giving response index 13.
Its final answer asserts that polygons fitting in a container implies their axis-aligned bounding boxes also fit.
This assertion is false, independently of the judge's verdict.

Take the parallelogram with vertices `(0,0), (0.01,0), (1.01,1), (1,1)`.
Take a second copy translated horizontally by `0.01`.
Their interiors have disjoint values of `x-y`, so both fit without overlap in a rectangle of width `1.02` and height `1`.
Each bounding box has width `1.01` and height `1`.
Two such boxes need width at least `2.02` in a height-one container. They cannot both fit in the original rectangle.
Exact rational arithmetic verified the coordinates, areas and width comparison.
Both judge orders awarded this response zero on all six criteria.
The private artifacts contain its complete raw text and every criterion decision beside the aggregate plots.

These are repeated generations of one question, not independent papers or a corpus sample.
Human judge calibration and an independent-provider audit are still pending. The grades remain provisional.
The exact counterexample checks one claim and does not establish general judge reliability.
Differences from previous attempts include decoding and batching settings, so these counts do not isolate a causal effect of the filters.

The [immutable private run archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/0d0e574257151feb5f3794ca3ef3e6dc73bedabb/miles/group-check-v12-mi350)
contains inputs, formatted prompts, outputs, both-order receipts, plots, calibration examples and the exact spot-check.
Source, submission and scheduler details remain in `jobs.jsonl` and the linked evidence records.
