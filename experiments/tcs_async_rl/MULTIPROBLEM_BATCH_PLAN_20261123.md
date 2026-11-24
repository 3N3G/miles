# Experiment: accumulate four TCS problems per optimizer update

## TLDR

The user approved changing each update from one problem to four on November 23.
Keep 32 answers per problem and increase the global answer batch from 32 to 128.
Resume the saved iteration 98 of Qwen3.5-4B. Preserve every used configuration and result.
The proposed run has ten updates, consuming 1,280 answers, matching the original forty-update pilot's answer budget.
The approved limit is four hours on eight MI350 GPUs and a separate new $5 judge allowance.
The user confirmed that “Do it” approves the concrete run and resource choices already discussed.
The [frozen active configuration](rl_easy_overfit_v17.json) resolves those limits.
The [earlier draft](rl_easy_overfit_v17.proposed.json) remains unchanged.
A fresh AMD check has 1.27 TB free, so every source checkpoint can remain preserved.

## Background / motivation

The saved one-problem run completed 14 further updates with 32 answers each.
Its matched-question rewards did not show a reliable increase.
The [recorded diagnosis](evidence/learning_diagnosis_20261123.json) identifies question-specific updates as one possible noise source.
It also records judge disagreement, reward standardization, long responses and restricted sampling support.
This test changes the number of problem groups accumulated before an optimizer step. It does not repair those other issues.

## Concrete question we are answering

Does continuing with four problem groups per update yield steadier updates and higher comparable training rewards?
If results improve, inspect the actual proofs, judge decisions, response lengths and question coverage.
If results remain flat, larger batches alone have not resolved this implementation's observed training behavior.
Neither outcome isolates batch size causally: the comparison uses earlier training, different weights and one seed.
The cheapest checks are configuration equivalence, actual per-problem grouping, accumulation scheduling and monitor regression tests.
GPU loading and the first complete four-group update remain runtime qualification steps.

## Methodology

- Model: the same pinned Qwen3.5-4B with thinking enabled and one multi-token-prediction layer.
  Restore model, optimizer, scheduler, random state and pending rollouts from saved iteration 98.
  Keep the initial converted model as the frozen reference.
  Verify learning rate 1e-6 and scheduler consumed-sample count 3,168 before the first update.
  New steps each add 128 consumed answers to that count. Never substitute update count multiplied by the new batch size for historical samples.
- Data: the unchanged checksum-pinned 20 easy questions from 17 papers.
  One example asks for an expected rooted-forest cost bound under arbitrary correlated sampling.
  The exact private bundle, formatted inputs, supplied answers and seven-point rubrics remain unchanged.
  FOCS 2026 and STOC 2026 remain sealed, with zero held-out reads.
  Keep source order/shuffling and pending-rollout restoration. Record each consumed group's actual question identity.
- Batch: set `rollout_batch_size=4` and `global_batch_size=128`.
  Keep `n_samples_per_prompt=32`, generation admission 96 and token microbatch limit 98,304.
  Three problem groups can generate concurrently. Buffer four completed groups before training through dynamic microbatches.
  Compare rewards within each 32-answer group. Never normalize rewards across the four problems together.
  Random shuffling across epoch boundaries can revisit a question. Report actual unique question counts.
- Other settings: preserve GRPO/CISPO, support replay, temperature 1, top-p 0.95, top-k 20,
  90,000 output tokens, 98,304 total context tokens, Adam parameters, learning rate and staleness limit six.
  Keep entropy observation, zero-coefficient reference divergence and the existing fixed-batch direction schedule.
  No entropy or divergence penalty is introduced.
- Reward: keep Luna medium, prompt version three, supplied binary rubric items, final-answer extraction and anonymous swapped-order averaging.
  Existing human calibration and independent-provider checks remain unresolved. Scores are provisional training rewards.
  No provider fallback, rubric change, score correction or judge upgrade is part of this run.
  Reuse only verified controls from the same questions and exact judge configuration, with their provenance.
- Limits: run ten optimizer updates, global IDs 99–108, or the resource limit, whichever comes first.
  This is two nominal passes and 1,280 consumed answers. Async generation can produce unused answers beyond this count.
  All generated tokens, provider calls, reservations and unused samples count toward their resource allowances.
  Four hours on eight GPUs is at most 32 GPU-hours. The new judge ledger cap is $5.
  The old ledger remains unchanged. No resource extension or ledger reset is authorized merely by a timeout.
- Metrics: retain per-question mean raw reward, zero/full-credit rates, within-problem reward variation,
  both full-policy and sampling entropy, sampled initial-policy and rollout divergence, gradient norms,
  clipping, effective sample size, fixed-batch surrogate gain, response tokens, caps, weight lag and actual publications.
  Larger problem batches can change logged entropy and gradient norms through question mix alone.
  Use matched-question views and token/cost coordinates. Do not claim a smoother mixed reward curve proves better proofs.
- Baseline and controls: compare descriptively with the preserved one-problem trajectory.
  Reference/empty/irrelevant-answer controls must pass before submission.
  CPU monitoring controls must accept 128-answer updates and reject the obsolete 32-answer shape.
  Four constant-reward problems with different means must not count as within-problem learning signal.
  One varied group among the four must count. Verify all four groups and their 32-answer sizes in actual exports.
  Verify finite backward execution, intended fixed-batch direction and identical rollout-engine publications.
- Seed and statistics: retain seed 20261118. This is one continuation, with descriptive comparisons only.
  No significance test, seed-level interval or proof-correctness claim is planned.
- Saving: use the existing single-checkpoint policy, writing to the fresh v17 directory.
  Preserve iteration 98 and all source checkpoints. Request a final or budget/deadline checkpoint once.
  The current storage guard requires 90,696,450,048 free bytes before submission and at allocation.
  Recheck storage before submission and at allocation. No checkpoint deletion is needed with current free space.
  Request the one checkpoint if unfinished with 30 minutes left, or when judge spending plus reservations reaches $4.
  Finish the active batch, verify the saved tracker and its weight publication, then stop only this owned allocation if it saved early.
  The $1 judge reserve allows the larger pending batch to settle. A provider cap rejection remains an error, never a zero reward.
- Logging and supervision: use the existing geney/tcs-async-rl W&B project with one VM-native history writer.
  Keep explicit optimizer coordinates and the separate Return, Policy health and Generation panels.
  Watch native training, judge/tunnel health, weight publication, monitor heartbeat and disk space through terminal state.
  Inspect raw rollouts at least hourly. Upload results privately and verify immutable readbacks and W&B history before closure.

## Results

No GPU job has been submitted and no new judge call has been made.
All 114 TCS CPU tests pass. Black, isort and Ruff pass for the changed monitoring files.
The resolved training arguments change only the two batch settings, resume/output identities and checkpoint bookkeeping.
The installed Megatron argument-check function permits the changed batch with the actual disabled phase-transition setting.
Its negative control still rejects a changed batch when phase transitions require equality.
These are CPU checks. Loading the actual model and optimizer with the new batch still requires the allocation.
See [the exact validation](evidence/multiproblem_validation_20261123.json).

The read-only AMD check found 7,597,572,096 free bytes, below the 90,696,450,048-byte requirement.
Its shortfall is 83,098,877,952 bytes. All four protected checkpoints remain present, including iteration 98.
No account-owned GPU job is queued or running at that observation.
See [the scheduler and checkpoint inventories](evidence/multiproblem_resources_20261123.json).

### Planned figures

- Per-question raw reward against cumulative consumed answers and generated tokens.
- Full-policy and sampling entropy, separately from sampled reference and rollout divergence.
- Gradient norms and fixed-batch surrogate gain against optimizer update, with actual question identities.
- Response tokens, cap fraction, judge-order gaps, policy lag, waiting time and judge cost.
- Pair every reported aggregate with a seed-selected raw proof and both judge decisions.

All figures are descriptive. Do not invent confidence intervals from one training continuation.

## Global takeaways

Reserved for the user after results. No learning or variance reduction has been measured for the proposed batch.

## Next steps

1. Finish CPU qualification and verify the preserved iteration-98 checkpoint inventory.
2. Verify current resource availability and freeze the approved configuration and source.
3. Submit once, verify the first complete 128-answer update, and supervise through terminal archival.

## Questions

The answered resource decision and approval wording are recorded in the repository-root `QUESTIONS.md`.

## Artifact storage qualification

Hugging Face is the durable private artifact store. Existing authentication has verified read and write access.
The VM keeps small operational files and a bounded staging batch, rather than the durable rollout archive.
One larger atomic optimizer trace can stage alone when it fits the one-GiB free-space reserve.
The uploader verifies immutable Hub LFS SHA256 or Git blob hashes before recording eviction and removing a local staged trace.
A failed upload or readback preserves the local file and AMD original. Model checkpoints remain on AMD.
All 117 TCS CPU tests pass, including oversized transfer and failed-hash retention controls.
The new readback helper also verifies an actual 26.9 MB archived trace against its immutable Hub SHA256.
The 120 existing control receipts match the exact current model, effort, prompt, payload hashes and rubric outcomes.
They can qualify all 60 question/control pairs without new judge calls. Their provenance is preserved in the new run.
