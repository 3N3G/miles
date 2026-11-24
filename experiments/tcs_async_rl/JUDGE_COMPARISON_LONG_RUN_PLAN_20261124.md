# Experiment: compare graders, screen difficulties, then continue 4B training

## TLDR

The user approved comparing a labeled-reference grader with the existing anonymous two-order grader.
Choose by agreement with assistant mathematical audits, then screen more easy and medium questions.
The user approved **12 hours of RL, with screening extra**, and **$25 total paid judging** across all stages.
The user delegated agreement thresholds to the assistant's judgment.
No further approval is needed for these concrete stages within their bounds.
An unsuccessful grading gate does not authorize a twelve-hour training job with an unreliable reward.

## Background / motivation

The previous four-problem run completed ten saved updates.
Nineteen repeated-question means changed by +0.025/7, with six increasing and thirteen decreasing.
Its reference was anonymous, its live reference scores did not gate each reward, and human accuracy was unmeasured.
Earlier medium screening covered one question. Its zero rewards cannot establish general medium-difficulty capability.
See [the previous results](FOUR_PROBLEM_BATCH_RESULTS_20261123.md) and [judge qualification](JUDGE_PROMPT_QUALIFICATION_FINDINGS_20261122.md).

## Concrete question we are answering

Which grading recipe better reproduces a frozen, independently written assistant audit on the same saved answers?
Does the selected recipe provide varied rewards across several medium questions for the current 4B checkpoint?
Does a longer continuation improve comparable scores and completion under that fixed grader?
If neither grader is sufficiently reliable, diagnose its failures before spending training compute.
Agreement with this assistant remains a diagnostic proxy, not human mathematical accuracy.

## Methodology

### Fixed-answer comparison

- Keep GPT-6 Luna, medium effort, the dataset's original binary rubric items and weights, and final-only extraction.
- Recipe A is the unchanged prompt-v3 anonymous candidate/reference pair, judged in both orders and averaged.
- Recipe B clearly labels the reference and candidate and grades only the candidate in one request.
  Use a concise general rubric prompt. It accepts valid alternatives and requires mathematical evidence.
  This compares complete grading recipes, not the isolated causal effect of labeling or prompt length.
- Freeze 40 nonempty saved candidate answers before new judge calls.
  Sample 32 from 16 randomly selected easy questions, two per question, plus eight answers from the earlier medium question.
  Exclude previously assistant-reviewed sample identities. The easy selections use randomized old-score strata
  to cover full and partial/incorrect answers. Old scores guide selection, never the new labels.
  These are a diagnostic sample, not a representative estimate of the whole corpus.
- Inspect the complete problem, reference, rubric and candidate. Record a Boolean and mathematical reason for every item.
  Record confidence and unresolved rubric ambiguity. Keep old judge outputs out of the labeling view.
  Fingerprint the labels before calling either new comparison recipe. The assistant is also the prompt author.
- Score each recipe by exact agreement with the audit's weighted total on each answer, and mean absolute score error.
  Report item agreement as a case-balanced average, averaging the two presentations within a paired case.
  Never count two presentations or several items as independent proof examples.
- Select higher exact total-score agreement. Break a tie by lower mean absolute error, then lower measured cost.
  Report question-cluster bootstrap intervals descriptively. No significance claim is planned.
- The observed gate is at least **95% case-balanced item agreement and 90% exact total-score agreement**.
  Require no clear false full-credit award on a substantively incorrect audited proof.
  Ambiguous audit labels must be reported. Do not silently redefine them after seeing outputs.
- Check reference/empty/irrelevant controls separately. A failed control is a reviewable error, not a candidate zero.
- If both recipes fail, allow one documented prompt repair using those failures as development cases,
  followed by a fresh disjoint 40-answer validation packet. Preserve every prompt, label and result.
  If neither then passes, stop before GPU screening/training and report the unresolved cases.
- At most $2 of the shared $25 is reserved for comparison, controls and this one repair round.
  Unfinished verdict retries keep the same input and model, with the existing bounded token expansion.

### Broader inference screen

- Model: the preserved Qwen3.5-4B checkpoint at iteration 108, containing 109 historical updates.
  This assesses the current checkpoint, not the untouched pretrained model.
- Select six easy and six medium questions uniformly within the eligible difficulty strata,
  using seed 20261124 and distinct papers where possible. Record exact identities before generation.
  An easy example asks for an expected rooted-forest cost bound under correlated sampling.
  The previously tested medium example concerns transferring a faster Walsh–Hadamard algorithm to a recursive matrix.
  Record one fully formatted actual input and output per difficulty after selection.
- Read only the ten eligible cohorts of dataset revision `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
  FOCS 2026 and STOC 2026 remain sealed, with zero held-out reads.
- Generate 32 answers per question with thinking, temperature 1, top-p 0.95, top-k 20,
  90,000 output tokens and 98,304 total tokens. Use the selected fixed grader.
- Make no optimizer update. Verify restored engine checksums against the saved checkpoint publication.
  Explicitly initialize fresh sampling/queued-rollout state for the new question set.
  Preserve model, optimizer and scheduler state. Never train on pending samples graded by the old recipe.
- Report per-question exact full-credit, positive, zero and capped counts, reward variation, tokens and cost.
  Report difficulty aggregates with their actual question count. Read seeded examples from every screened question.
- If at least two of six medium questions have genuine within-question reward variation and a positive answer,
  include all six sampled medium questions alongside the existing twenty easy questions in the long run.
  Do not select only successful medium questions. Shuffle questions uniformly in that frozen 26-question set.
  Otherwise retain the twenty easy questions for the long run and report the medium evidence separately.
- Bound the initial screen to one eight-GPU allocation of at most two hours, including loading and repairs.
  This is at most 16 GPU-hours, in addition to the approved RL time.

### Longer continuation and final inference check

- Resume iteration 108 with the selected fixed grader and the dataset selected by the declared screen rule.
  Preserve all old model/checkpoint directories, configurations, scores and artifacts.
- Keep four questions ×32 answers per optimizer update, global batch 128, admission 96,
  dynamic token microbatches at 98,304 tokens/GPU, and the existing 2-actor/6-rollout GPU topology.
  Keep GRPO/CISPO, constant LR 1e-6, Adam parameters, support replay, MTP policy and maximum staleness six.
  Entropy and reference divergence remain observations, with zero regularization coefficients.
- One seed, 20261124. Set an upper iteration target that cannot force an allocation extension.
  The actual stopping limit is twelve hours on eight MI350 GPUs, at most 96 GPU-hours,
  or the shared judge/storage guard, whichever comes first.
- Save periodically, approximately every five optimizer updates, and at the deadline/budget stop.
  Keep the source checkpoint and document any retention of this new run's intermediate checkpoints.
  Reserve at least 30 minutes for a forced final save before walltime. Verify files, tracker and engine publication.
- A single append-only judge ledger owns the entire $25 cap. Carry its exact settled history between sequential phases.
  No ledger reset, silent provider switch or zero reward after a provider/budget failure is allowed.
  Request the final checkpoint at $22 spent plus reserved, preserving $3 for pending calls and closure.
  Stop if another safe checkpoint cannot fit. Do not delete protected source checkpoints.
- After training, run the same twelve-question screen on the final saved checkpoint,
  with the same sampling recipe and selected grader, under at most two additional eight-GPU hours.
  All paid calls still count toward $25. If the judge cap prevents this, report the missing final evaluation explicitly.
- Compare before/after per-question scores, full-credit and completion rates, and lengths on the fixed screen.
  The six medium questions may have entered training, so this is a training-question evaluation, not generalization.
  Keep training-visit curves separate. One continuation provides no seed-level learning guarantee.
- Retain a fixed initial-policy reference. Log full/sampling entropy, sampled reference and rollout mismatch,
  gradient norms, clipping, effective sample size, fixed-batch update direction, lengths/caps and weight versions.
  Log resolved configuration, data/grader hashes and actual microbatch counts to the single-writer W&B run.
- Supervise foreground through scheduler termination, checkpoint verification, immutable private HF uploads,
  W&B readback and process closure. Inspect actual proofs and grades at least hourly and diagnose failures.

## Results

Not run. Read-only AMD checks show no current job for this user, a 96-hour partition maximum,
and 1,095,083,216,896 bytes free at the latest check. The requested RL allocation remains capped at twelve hours.
No new judge call or GPU submission has occurred at this plan's creation.

### Planned figures

- Judge methods: exact score agreement and absolute score error by answer, with one raw case beside every aggregate.
- Full-credit false awards, zero-credit false denials and method costs on the fixed audit packet.
- Before/after per-question score, full-credit, completion and response length on the twelve-question screen.
- Training metrics against optimizer updates and consumed tokens, with question/difficulty identities.
- Entropy, divergence, gradients, clipping, accumulation, lag, judge budget and disk space separately.

## Global takeaways

Reserved for the user. An assistant-agreement winner is not an independently calibrated mathematical judge.

## Next steps

1. Freeze the comparison packet and assistant labels, implement both grading recipes, and validate CPU controls.
2. Compare methods and apply the declared agreement gate. Preserve failures and any single repair round.
3. Screen easy/medium questions, freeze the selected dataset and long-run configuration, then submit and supervise.

## Questions

The user selected twelve hours of RL with screening extra and a $25 total judge cap.
The user delegated agreement-threshold judgment. No resource-approval question remains for the listed stages.
