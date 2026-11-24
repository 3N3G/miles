# Higher-effort judging on forty frozen mathematical answers

GPT-6 Luna with high reasoning effort matched **36/40 audited total scores (90%)**.
Case-balanced binary-item agreement was **96.8%**.
It meets the declared observed gate of 90% exact-score and 95% item agreement.
All 51 separate controls passed across seventeen questions: references scored seven, empty and irrelevant answers zero.

The test kept the repaired labeled-reference prompt, answers, rubric weights and audit labels fixed.
Only reasoning effort changed from medium to high.
The packet contains 32 easy answers and eight answers from one medium question.
These are saved Qwen3.5-4B answers, not new solver generations.

| Grading recipe | Exact score matches | Item agreement | Mean absolute score error, out of 7 |
|---|---:|---:|---:|
| Anonymous pair, both orders, medium | 29/40 | 95.75% | 0.275 |
| Labeled reference, medium | 32/40 | 94.54% | 0.400 |
| Labeled reference, high | 36/40 | 96.79% | 0.225 |

One raw disagreement concerns a valid interval argument.
The answer compares both endpoints with sqrt(2n), proving the interval empty.
The audit awards seven. Luna high awards five because it requests a particular rearrangement from the rubric.
The other three disagreements concern oracle notation, rational bit-length accounting, and an alternative clipping proof.
Their frozen labels remain unchanged. High made no clear false full-credit award on this packet.

The descriptive question-cluster bootstrap interval for high exact agreement is 79.4%–97.8%.
There are seventeen question clusters and no independent human calibration.
The assistant wrote the audits and prompt. Earlier failures informed the effort choice.
Observed gate passage is therefore not a population-accuracy guarantee.

The 92 new paid calls, including one unfinished-verdict expansion, cost $0.0825564.
The shared experiment ledger now contains 536 settled calls, no pending reservations, and $0.5342506 total usage.
All 250 archived artifacts passed immutable hash verification in the
[private evidence archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/79887438e9e3e313184aaf9969d4130604892d2f/miles/judge-comparison-high-20261124).

The selected recipe grades one candidate against an explicitly labeled reference, using Luna high.
It has no measured answer-order gap.
The next stage is the approved six-easy/six-medium screen from the saved training checkpoint.
No GPU stage has run for this comparison.
