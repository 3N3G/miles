# Fixed-question baseline under the accepted judge

We regraded384 saved Qwen3.5-4B answers with GPT-6 Luna at high reasoning effort and the accepted reference-guided prompt v5.
This establishes the baseline for a later comparison on the same questions. This stage generated no new solver answer and performed no optimizer update.

The saved policy is checkpoint108, after109 historical updates. The screen contains six easy and six medium training-eligible questions, with32 answers each.
Generation enabled thinking and allowed90,000 new tokens within98,304 total tokens. The original inputs, sampling, answers and generation provenance were preserved.
The judge sees the problem, original weighted binary rubric, labeled reference and candidate final answer. Scores range from0 to7.

| Difficulty | Answers | Mean score | Full credit | Zero | Token cap |
|---|---:|---:|---:|---:|---:|
| Easy | 192 | 5.760/7 | 127 | 16 | 4 |
| Medium | 192 | 1.995/7 | 34 | 101 | 103 |

All six medium questions have varied positive scores. Their means range from0.188 to6.781.
The seeded easy example earns7 by deriving expected auction surplus of two-thirds of optimal welfare.
The seeded medium example earns3: it correctly forces different path choices in separate regions, but assumes an unjustified nearby transition using an infimum.
The [private archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/664372cffe49cec49011cfed617f059484175858/miles/checkpoint108-screen-v5-baseline) contains both examples, all grades and original receipts. All1,051 artifacts were checksum-verified.

These are descriptive results from one generation per checkpoint. Easy questions had already entered training. FOCS2026 and STOC2026 remained sealed.
The accepted judge matched assistant reviews on34/40 exact totals and96.17% of rubric items. This is not independent human calibration.
The current RL corpus contains all236 easy/medium questions. This12-question screen measures fixed-question change after training.

[Exact per-question results and fingerprints](evidence/long236_fixed_baseline_20261125.json).
