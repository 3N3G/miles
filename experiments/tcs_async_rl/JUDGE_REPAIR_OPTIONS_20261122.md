# Repairing the TCS rubric judge: evidence and options

Prompt clarification, labeled examples, and stronger reasoning are all supported options.
Their effects must be measured against human labels. Agreement between repeated judge calls is not proof of correctness.
This is a literature review and a proposed comparison, not an executed judge experiment.
The active prompt, Luna medium, original rewards, and held-out split remain unchanged.

## What the primary sources support

| Intervention | Evidence | Limit |
|---|---|---|
| Clarify the rubric and partial credit | [Anthropic's evaluation guide](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) recommends structured criteria, partial credit, transcript review, and expert calibration. | The intended grading standard still needs a human decision. |
| Few-shot examples | [OpenAI's grader guidance](https://developers.openai.com/api/docs/guides/graders) recommends examples covering different answer qualities. [MT-Bench](https://arxiv.org/html/2306.05685v4#A4.SS2) increased GPT-4 position consistency from 65.0% to 77.5% with examples. | MT-Bench's few-shot human agreement remained similar to zero-shot. Consistency gains need not be accuracy gains. |
| Higher reasoning effort | [JudgeBench](https://arxiv.org/html/2410.12784v2) reports overall o3-mini accuracy of 70.57%, 76.57%, and 80.86% at low, medium, and high effort. | Different models, datasets, and scoring tasks. These numbers do not predict Luna's performance on TCS proofs. |
| Stronger judge model | JudgeBench found that larger models generally performed better within providers, while many strong judges still struggled. | A stronger model remains a fallible measurement tool. |
| Human calibration | [Anthropic's November 2026 guidance](https://www.anthropic.com/research/building-strong-safety-evaluations) recommends blinded human labels, chance-corrected agreement, and comparison with human-to-human agreement. | Our human labels and independent-provider checks remain pending. |

[OpenAI's reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning-best-practices) favors clear, direct instructions.
It recommends trying zero-shot first, then consistent few-shot examples when needed.
Adding a generic request to think step by step is not a demonstrated repair for an existing reasoning model.
[Luna's model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna) confirms support for high reasoning effort.

## Application to the saved failures

Our [current judge](JUDGE_ROLLOUT_REVIEW_20261122.md) already uses binary weighted items, references, anonymous answers, and order swaps.
The six-answer review found inconsistent treatment of implicit monotonicity, a domain typo, and prerequisites for partial credit.
It also found valid full-credit proofs and an incorrect zero-credit answer.
These observations motivate specific prompt clarification. They do not establish that prompt wording is the only limitation.

A prompt variant should require a concrete mathematical justification for each verdict.
It should apply only the item's stated requirements and explicit dependencies, and accept equivalent valid arguments.
Human-adjudicated examples should define treatment of implicit standard facts, notation slips, and independently correct partial work.
Use examples to encode that policy, rather than copying uncalibrated Luna grades as ground truth.

## Recommended comparison, not yet run

1. Label the existing ten-example blinded packet. Reserve separate examples for judging prompt revisions.
   Ten examples are a diagnostic start, not a precise estimate across the full corpus.
2. Compare the current prompt with one targeted revision at Luna medium.
3. With the selected prompt fixed, compare Luna medium, Luna high, and a stronger judge on the same saved answers.
4. Measure item agreement, false awards, false denials, total-score error, repeatability, latency, and cost.
   Include identical-order repeats as well as swapped calls to separate sampling variability from position effects.

Select by agreement with the intended human grading standard, not by higher rewards or smoother RL charts.
Keep demonstration examples separate from judge-validation examples. Leave FOCS/STOC 2026 sealed.
The exact comparison, stronger model, labels, and spending cap remain researcher decisions before paid calls.
