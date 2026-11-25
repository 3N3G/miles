# Luna high still leaves unresolved grading disagreements

Luna high and three further prompt revisions did not produce a grader that passes the final validation gate.
The final prompt matched **34/40 total scores (85%)** and **96.17% of binary rubric decisions**.
The declared thresholds were 90% and 95%, with no high-confidence false full-credit award.
It passed the latter two checks and all 132 reference, empty-answer and irrelevant-answer controls.
The twelve-hour reinforcement-learning continuation has not started.

We compared two fixed prompts using GPT-6 Luna with high reasoning effort.
Both received the problem, original weighted rubric, explicitly identified reference and one candidate's final answer.
The final revision added contrasting examples about implicit steps and independent component credit.
Forty previously unused saved Qwen3.5-4B answers received assistant labels before any comparison calls.
There were 32 easy answers and 8 medium answers across 22 questions, including all six screened medium questions.
The dataset was `ian-and-friends/tcs_v3`. FOCS 2026 and STOC 2026 remained sealed.
No labels changed after the verdicts arrived.

| Prompt | Exact totals | Item agreement | Mean absolute error, out of 7 |
|---|---:|---:|---:|
| Previous |32/40|94.33%|0.400|
| Final |34/40|96.17%|0.275|

The final prompt's descriptive 95% question-cluster bootstrap interval for exact agreement was 72.2%–95.1%.
This is a small diagnostic packet, with assistant labels and reused question identities.
It does not measure independent human accuracy or establish a significant improvement.

One answer states `n/sqrt(d) >= sqrt(2n)` and derives `delta >= 2`, proving the permitted interval empty.
The audit awards 7. Both prompts award 5 because it does not isolate `d >= n/delta`.
Another answer asserts an unrestricted sub-Gaussian moment bound using `g''(t) < 0`, which is false.
The audit and final prompt agree on 2/7 for its separate valid moment calculation and exponential Markov step.
Four of the six final disagreements carry pre-recorded audit uncertainty.
Some may reflect overly permissive or strict assistant labels, including the cube-face representation and recursive-call accounting.

The initial high-effort test passed 36/40 on an older packet.
Subsequent screening completed 384 answers from the already-trained checkpoint, with zero optimizer updates.
Live proof inspection then motivated these additional validation rounds.
Those 384 answers remain available for regrading without another GPU generation job.

This final comparison cost $0.2007948. All new stages together cost $1.7725486 against the approved $25 cap.
The next decision is whether to resolve the six disagreements by human review or test another judge.
Human review is the draft recommendation. Further tuning against this consumed packet would not be fresh validation.

The exact labels, verdicts, intervals and disagreements are in the linked [evidence summary](evidence/judge-comparison-item-v5-20261124-final-validation-summary.json).
Operational receipts and archive locations are in [the execution record](FINAL_LUNA_HIGH_OPERATIONS_20261124.md).
