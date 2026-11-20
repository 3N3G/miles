# Experiment: check two eligible questions for a usable runtime reward group

Status: proposal only. No alternative-question inference is approved or submitted.
The data-scope question is pending in `QUESTIONS.md` and chat.

## TLDR

Check one easy and one medium training question, in that order.
Select each uniformly within its declared difficulty before generating any answer.
Stop when a complete 32-answer group meets the existing runtime gate.
Use that question only for the already requested two-update RL qualification.
This selection cannot support a claim about representative TCS performance or learning.

## Background / motivation

The original fixed hard question produced 31 completed answers with zero rewards.
Both repaired groups then returned 28 nonempty finals, all scored zero in both judge orders.
Each group also had four missing finals. The native group had five repetition caps, and the vLLM group had nine.
See [native findings](REPAIRED_GROUP_FINDINGS_20261120.md) and [vLLM findings](VLLM_GROUP_FINDINGS_20261120.md).
The saved reference-verdict audit also documents three incorrect local criticisms of the supplied reference.
Preserve all verdicts and inspect paired-reference scores when reviewing any proposed qualification.
These judge limitations reinforce the need for human calibration before interpreting learning results.

Changing the selected question needs the user's decision. Sampling and backend repairs alone have not opened the gate.

## Concrete question we are answering

Does either preselected eligible question yield completed model answers with genuine reward variation?
If yes, qualify the existing async RL runtime on that selected question.
If no, report both outcomes and preserve the closed gate.
Neither result establishes corpus-wide model capability.

The first cheap checks are each question's reference, empty and irrelevant-answer judge controls.
Native inference also runs the existing 16-token sampling-support control before the complete group.

## Methodology

- Models: retain the pinned Qwen3.5-4B generator and GPT-6 Luna judge at medium effort.
  Model revisions and scientific settings remain owned by `config.sampling_repair_v2.json`.
- Data: use only eligible cohorts at the pinned TCS revision. Keep FOCS 2026 and STOC 2026 sealed.
  `evidence/alternative_question_proposal_20261120.json` owns selected identities and input hashes.
  One candidate asks for an expected rooted-forest cost bound using vertex marginal probabilities.
  The other asks to transfer a sub-N-log-N Walsh–Hadamard algorithm to a recursively defined matrix.
  The private proposal directory contains both exact inputs, references, rubrics and formatted prompts.
  Both inputs and references were read. The rubrics permit equivalent correct proofs.
- Selection: for each of the easy and medium strata, use seed 20261118 on eligible rows
  sorted by cohort and original row. Do not select based on generated quality.
  The eligible pools contain 20 easy and 216 medium questions.
- Generation: use native target-only inference and the existing bounded sampling profile.
  Keep thinking enabled, temperature 1, top-p 0.95, top-k 20, 90,000 output tokens and 98,304 context.
  Generate a fresh group of 32 indexed seeds. Never combine old answers or different questions into one gate.
  Use one MI350 GPU per question, at most four hours each, sequentially. Stop after the first qualifying group.
  The two proposed groups have an eight-GPU-hour total ceiling. No optimizer runs during screening.
- Reward: use each candidate's supplied seven-point rubric and the existing anonymous two-order judge procedure.
  Do not change criterion weights, award partial criteria differently or score provider failures as zero.
  Keep exact model outputs and judge receipts. All grades remain provisional pending human and independent-provider calibration.
- Metric: require 32 nonempty final answers, finite real rubric grades and at least two reward values.
  Inspect each nonzero awarded criterion against the raw answer before accepting variation.
  Record truncation, finish reasons, response lengths and order gaps separately.
- Baselines and controls: the frozen base model is the baseline. Require reference 7, empty 0 and irrelevant 0
  on each question. These controls test judge plumbing, not general judge reliability.
  Reuse the existing $5 per-group judge cap. Provider credentials remain on the VM.
- Seeds: selection seed 20261118, generation seeds 20261118 through 20261149.
  Choose the reporting example uniformly with seed 20261118, before inspecting its verdict.
- Statistical test: none. This is a runtime feasibility check with deliberate question selection.
  Do not pool backend repeats as independent evidence or claim a learning effect.
- Exact plots: for each completed group, plot reward value versus answer count and generated tokens versus answer count.
  Mark the 90,000-token cap. Include the seed-selected raw answer and its rubric decisions beside the aggregates.
- RL handoff: create a fresh versioned gate-bound snapshot for the qualifying question.
  Validate the input hash, sampling signature and question-specific service health before submission.
  Retain the two-update resources and acceptance checks in `RL_SMOKE_PLAN.md`.
  Record changed question identity explicitly. Preserve every old config and result.

## Results

No alternative-question generation, grading or training has run.

## Global takeaways

Reserved for the user's interpretation.

## Next steps

1. Finish the current fixed-question checks.
2. Obtain the pending decision before submitting either alternative question.
3. If approved, pin fresh inputs/configs/source, run the controls and execute the proposed groups sequentially.
4. Submit and debug the bounded RL qualification only after a complete group passes.

## Questions

The user must decide whether the diagnostic can change training questions.
