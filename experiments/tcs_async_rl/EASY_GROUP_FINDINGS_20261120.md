# The easy question produces varied provisional rewards

Frozen Qwen3.5-4B produced 32 answers to one preselected easy TCS question.
Of 28 completed final answers, GPT-6 Luna gave 15 positive scores and 13 zeros.
Four generations reached the 90,000-token limit without a final answer. They remain ungraded.
This result supported the approved medium-question check. It does not establish performance across the dataset.
No optimizer update ran.

The question concerns a minimum-cost spanning forest with one specified root per tree.
It asks for an expected-cost bound after adding a correlated random set of roots.
It comes from SODA 2025 in `ian-and-friends/tcs_v3`, revision `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
It was selected uniformly within the eligible easy pool before generating any answers.
FOCS 2026 and STOC 2026 remained sealed.

The model revision was `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, with thinking enabled.
Native SGLang used target-only decoding on one MI350 GPU.
Sampling used temperature 1, top-p 0.95 and top-k 20.
The output limit was 90,000 tokens. Total context was 98,304 tokens, with a 195-token formatted prompt.
The 32 generation seeds were 20261118 through 20261149. No reference or rubric entered the generator prompt.

GPT-6 Luna at medium effort graded the final answers against three supplied binary rubric criteria.
The weights were three points for construction, two for the forest invariant and two for the expected-cost bound.
Each candidate appeared anonymously against the reference in both orders. Its reward is the mean across orders.
The reference, empty and irrelevant candidate controls received 7, 0 and 0 in both orders.

| Measurement | Result |
|---|---:|
| Generations | 32 |
| Natural stops with nonempty final answers | 28 |
| Length limits without final answers | 4 |
| Final-answer scores: 0 / 3 / 5 / 6 / 7 | 13 / 1 / 7 / 5 / 2 |
| Mean over graded final answers | 2.929 / 7 |
| Population variance over graded final answers | 7.923 points squared |
| Order pairs differing in point total | 8 / 28 |
| Order pairs agreeing on zero versus positive | 28 / 28 |
| Paired-reference scores of seven, including controls | 62 / 62 |
| Generated tokens, minimum / median / maximum | 53,699 / 73,896 / 90,000 |
| Total generated tokens | 2,361,916 |
| GPU allocation elapsed time | 21 minutes 17 seconds |

The reporting seed selected answer index 13. Both orders gave it zero.
Its argument incorrectly makes the forest cost depend only on the number of roots.
Consider a three-vertex path with edge costs 1 and 100.
Choosing the first two vertices as roots costs 100. Choosing the two endpoints costs 1.
Both root sets have size two. Exhaustive enumeration of all four edge subsets confirms the counterexample.
The private archive contains this answer's complete text, both verdicts and the exact calculation.

The positive answers differ in their construction, rather than only their wording.
All 15 received the three-point construction criterion in both judge orders.
For example, answer index 0 received five points in both orders.
It deletes each selected vertex's parent edge from an optimal rooted forest.
Each remaining component has exactly one root: its zero-indegree vertex.
Each retained edge is charged to its child being unselected, so the marginal probability bound suffices without independence.
This checks the central construction and cost argument. The answer still contains confused invariant prose and a false side remark.
This local mathematical review does not certify every sentence or all 15 positive proofs.

The scores remain provisional. Human calibration and an independent-provider audit are pending.
Eight order pairs disagree on the point total, despite agreeing on zero versus positive.
There are no error bars or significance test: these are generations of one selected question, without training.
The original RL qualification rule also requires 32 nonempty final answers, so this group fails that strict rule.
The user was asked whether recorded model limits may instead receive actual empty-answer judgments through the production adapter.
No missing-answer reward was imputed or changed.

The [immutable private archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/ff81a3d2b6e0f3b34a01bcc01cfa2258c90ed34c/miles/easy-check-v14-mi350)
contains exact inputs, formatted prompt, all raw outputs, criterion decisions, plots and mathematical checks.
Reproduction and scheduling details are in `jobs.jsonl` and `evidence/easy_group_v14_outcome_20261120.json`.
The approved sequence and its limits remain in `EASY_MEDIUM_CHECK_PLAN_20261120.md`.
