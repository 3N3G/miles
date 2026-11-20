# Qwen3.5-4B produced no reward variation on one hard TCS question

The frozen model finished 31 of 32 final answers. Luna scored all 31 answers zero in both presentation orders.
One answer reached 90,000 tokens while repeating `Wait.` without producing a final answer.
This group failed the required gate for reinforcement learning. These results concern one question only.

## Question and setup

The check asked whether a response group supplies complete answers and varied rubric rewards.
The question requires constructive approximation algorithms for packing convex polygons using translations only.
Its targets include strip width within factor 51, unit-square bins within factor 11, and perimeter within factor 7.3.
Every nontrivial geometric estimate needs a proof. An asserted packing theorem is insufficient.

The question is the seeded SODA 2023 example from `Online_Sorting_and_Translational_Packing_of_Convex_Polygons`, node 25.
It belongs to the training-eligible data. FOCS 2026 and STOC 2026 contents remained sealed.
The dataset revision is `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
The Qwen3.5-4B model revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The generator received the problem alone through the native user chat template, with thinking enabled.
It received neither the reference answer nor the rubric. The formatted prompt contained 374 tokens.
There were 32 candidates, with indexed seeds from 20261118 through 20261149.
Sampling used temperature 1, top-p 1 and no top-k cutoff.
The output cap was 90,000 tokens. Total context capacity was 98,304 tokens.

Generation used one MI350X with 287.98 GiB of memory, BF16 weights and tensor parallelism one.
SGLang 0.5.24.dev102+g3f7c4f02a used EAGLE acceleration, with step/top-k/draft settings 2/1/3.
The check performed no optimizer updates.

## Scoring

Luna medium judged each final answer beside the reference, with anonymous labels and both answer orders.
The candidate score is the mean across orders. The supplied rubric has six binary criteria:

| Criterion | Points |
|---|---:|
| Proved geometric packing mechanism | 2 |
| Quantitative rectangular-container estimate | 1 |
| Valid strip-packing guarantee | 1 |
| Constructive unit-square guarantee | 1 |
| Valid bin-packing guarantee | 1 |
| Perimeter guarantee and polynomial-time construction | 1 |

Reference, empty-answer and irrelevant-answer controls scored 7, 0 and 0 respectively in both orders.
The unfinished thinking response received no substituted zero score.
Human calibration and an independent-provider audit remain pending. All judge scores are provisional.

## Measured results and raw examples

| Measurement | Result |
|---|---:|
| Requests returned | 32/32 |
| Natural stops with nonempty final answers | 31/32 |
| Length-limited responses without a final answer | 1/32 |
| Graded final answers | 31 |
| Nonzero candidate scores, either judge order | 0 |
| Generated tokens, minimum / median / maximum | 24,388 / 36,754 / 90,000 |
| Total generated tokens, including the unfinished response | 1,216,469 |
| Allocation elapsed time, including startup | 28 minutes 50 seconds |
| Estimated judge cost at the configured rates, including controls | $0.1742 |

The reporting rule uniformly selects one response by sample index, using seed 20261118. It selects sample 13.
Both judge orders awarded it zero on every criterion.
Its raw answer claims that the optimal height-one strip area exceeds the sum of polygon diameters.
One rectangle disproves that claim. Use vertices `(0,0), (1/100,0), (1/100,1), (0,1)`.
Its optimal strip area is `1/100`. Its diameter squared is `10001/10000`, so its diameter exceeds one.
This is an exact check of one explicit claim, not a complete audit of the answer or all grades.

The unfinished sample is 25. Its raw tail repeatedly contains `Wait.` and never closes the thinking block.
The engine reported the length limit. It did not report a memory failure or request retraction.

The [private evidence archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/ee9d513a726ef0299ea9e295c4c4109e3bd9819c/miles/group-check-v9-mi350)
contains the complete formatted input, outputs, reference, rubric, both-order verdicts, plots and exact rectangle check.
Its manifest and logs contain reproduction details. The proposed inference repair has a [separate operational plan](REPETITION_REPAIR_20261120.md).

## Limits

This is one hard question, not a sample of independent papers or a corpus-wide accuracy estimate.
Different candidate seeds do not replace independent training seeds. There was no learning comparison or significance test.
The controls show that this grader distinguishes the supplied reference from empty or irrelevant answers.
They do not establish agreement with human grading or recognition of every valid alternative proof.
The rectangle check corroborates one error. It does not validate all 31 zero scores.
Generation at the observed lengths does not qualify trainer memory, backward computation or weight publication.
