# Judge prompt and six saved-answer checks

The actual request and response name **gpt-6-luna**, with **medium** reasoning effort.
The [system prompt](rubric_judge_prompt.txt) requests independent Boolean decisions for each weighted rubric item.
The [message builder](rubric_judge.py) supplies the problem, dataset rubric, and two anonymous answers.
One is the generated final answer. The other is the supplied reference, without a reference label.
The client swaps their positions, grades again, and averages the candidate's weighted scores out of seven.
Reference scores do not normalize the candidate reward. Solver thinking text is excluded.

Requests start with 8,192 completion tokens. Only unfinished verdicts can receive a larger budget, up to 32,768.
Temperature and sampling seed are unset. Gemini was not used for this run.
The [private review](https://huggingface.co/datasets/3N3G/internalization-runs/blob/11a802f912aab51aea0effb85ff85a29c4f8a840/miles/easy-overfit-v12-graded-rollout-review-20261122/REPORT.md)
contains the exact request payloads, full proofs, rubrics, both decisions, and mathematical checks.

This review asks whether actual proofs and grading reasons support the training rewards.
Four saved group examples were sampled with seed 20261122. Two additional cases diagnose zero rewards and maximum disagreement.
Each original group example used a reward-independent within-group index.
These checks made no new inference or judge calls and changed no original scores or prompts.

| Answer | Two scores /7 | Assistant assessment |
|---|---:|---|
| Affine dimension | 7, 7 | Main proof strategy checks out. |
| Auction residual surplus | 7, 5 | Main calculation is valid. Calls differ over an implicit monotonicity justification. |
| Communication protocol | 7, 6 | Sound reduction with a domain typo. Calls differ on its severity. |
| Row-subset construction | 7, 7 | Explicit construction and both deletion cases check out. |
| Unary certification, diagnostic maximum gap | 4, 0 | Incomplete parameter verification. Calls apply different prerequisites for partial credit. |
| Hypergraph removal, diagnostic zero group | 0, 0 | Declares a false universal assertion true without proof. The supplied counterexample checks out. |

The examples show correct proofs receiving full credit, an incorrect answer receiving zero, and inconsistent strictness between calls.
They do not measure judge accuracy. These are assistant checks, not human calibration labels.
Swapped-call disagreement mixes answer-position effects with sampling variability. No same-order repeat control was run.
The [evidence record](evidence/judge_rollout_review_20261122.json) pins selection, request identity, and seven verified archive files.
Human calibration remains pending before interpreting the [small training-score gain](EASY_CONTINUATION_FINDINGS_20261122.md) as improved proof correctness.
