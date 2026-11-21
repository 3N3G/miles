# The medium question produces constant-zero provisional rewards

Frozen Qwen3.5-4B generated 32 responses to one preselected medium TCS proof question.
All 28 nonempty final answers received zero of seven rubric points in both judge orders.
Four missing final answers remained ungraded. The automatic gate stopped, and no RL job was submitted.

The question asks whether a faster Walsh–Hadamard algorithm implies a faster algorithm for a recursively defined matrix.
It comes from STOC 2023 in `ian-and-friends/tcs_v3`.
Native SGLang ran on one MI350 GPU with thinking enabled, temperature 1, top-p 0.95 and top-k 20.
Output and total-context limits were 90,000 and 98,304 tokens.
GPT-6 Luna at medium effort used the supplied binary rubric, averaging anonymous judgments in both answer orders.
No optimizer ran. FOCS 2026 and STOC 2026 remained sealed.

| Measurement | Result |
|---|---:|
| Nonempty final answers / generations | 28 / 32 |
| Positive / zero graded answers | 0 / 28 |
| Length limits / missing finals | 5 / 4 |
| Generated tokens, median / total | 69,761 / 2,278,962 |
| GPU allocation elapsed time | 19 minutes 30 seconds |
| Reference / empty / irrelevant controls | 7 / 0 / 0 |
| Paired-reference scores of seven | 62 / 62 |

Reporting seed 20261118 selected answer 13.
It derives a recurrence allowing order-N-log-N cost, then asserts the required smaller asymptotic bound without establishing a reduction.
An exact countersequence confirms that its recurrence alone does not imply that conclusion.
Its suggested whole-matrix permutation also fails at dimension two: the modified base has zero entries, while the Walsh matrix has none.
The archive contains the full answer, both verdicts and exact checks.

These are provisional grades on one question, without human calibration or an independent-provider audit.
They cannot determine dataset sampling weights or general medium-question performance.
The easy question remains a candidate with varied grades. Its four capped missing finals still need a qualification decision.

[Private outputs, plots and checks](https://huggingface.co/datasets/3N3G/internalization-runs/tree/df899421fac85aa59009542bd86109eeee83a11e/miles/medium-check-v15-mi350).
Exact revisions, seeds and reproduction details are in `EASY_MEDIUM_CHECK_PLAN_20261120.md` and `evidence/medium_group_v15_outcome_20261121.json`.
