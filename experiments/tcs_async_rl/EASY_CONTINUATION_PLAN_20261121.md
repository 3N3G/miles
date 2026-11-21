# Experiment: continue fitting the 20 easy TCS questions with complete diagnostics

Status: prepared for review. The user requested continued work until comparable rewards increase.
The proposed admission change and next bounded GPU allocation need approval under the updated AGENTS instructions.
The reviewable snapshot is `rl_easy_overfit_v10.draft.json`. No job has been submitted for this plan.

## TLDR

Resume the retained checkpoint after 25 optimizer updates. Request 40 additional updates, stopping at four hours.
Use the same Qwen3.5-4B, all 20 easy questions, thinking, 90k output limit, GRPO/CISPO, judge and optimizer.
Add full-policy entropy, sampling entropy, frozen-initial-policy divergence and fixed-batch update-direction observations.
Propose 96 concurrent answers instead of 32, allowing three groups in flight on three rollout engines.
Keep 32 answers per question and global batch 32. Keep learning rate 1e-6.

## Background / motivation

The first attempt completed 27 updates. Its latest retained checkpoint contains 25 updates.
The two later unsaved updates cannot be resumed. Old configs, checkpoints, logs and results remain unchanged.
The seven repeated questions averaged 4.464 then 4.536 points. This did not show a consistent increase.
Entropy observation was disabled. The fork rejected reference divergence with sampling support replay.
The audit also found independent offline W&B writers with colliding history counters.
See `RL_NOISE_AUDIT_20261121.md` and its private evidence links.

## Concrete question we are answering

Do further updates increase rewards on comparable visits while moving the fixed-batch objective in the correct direction?
If rewards increase, inspect per-question changes, sampled proofs, lengths, caps and both judge calls before reporting progress.
If they remain flat, use actual entropy, initial-policy drift and update-direction scores to distinguish competing explanations.
Do not select a favorable single group or treat shuffled-question means as a learning curve.
The earliest likely technical failure is the added reference pass or observer pass under the real ROCm runtime.
CPU controls check distribution semantics and gradient invariance before that GPU test.

## Methodology

- Models: the same pinned Qwen3.5-4B initialization and one MTP layer, with the unchanged BF16/FP32-gradient recipe.
  Load the full optimizer/RNG checkpoint at iteration 24. Write all new checkpoints to a distinct run directory.
  Keep the initial converted checkpoint as the frozen reference. Do not update that reference during training.
- Data: the same checksum-pinned 20-question easy bundle from the ten eligible TCS cohorts.
  It spans 17 papers, with 12 proof-writing and eight proof-strategy questions.
  One example asks for an expected rooted-forest cost bound under arbitrary correlated sampling.
  The exact formatted inputs, references and rubrics remain in the private bundle.
  FOCS 2026 and STOC 2026 remain sealed. No dataset content or difficulty weight changes.
- Prompts and sampling: retain the original problem-only chat prompt, thinking enabled, temperature 1,
  top-p 0.95, top-k 20, 90,000 generated tokens and 98,304 total context tokens.
  The updated prompt-repair authorization is recorded, but this continuation does not change prompts.
- Optimization: keep Adam, learning rate 1e-6, weight decay 0.1, betas 0.9/0.98, GRPO and CISPO unchanged.
  Keep batch 32 and 32 answers per question. Propose admission 96, leaving the existing staleness bound at six.
  Admission is the sole proposed non-observational setting change. Its throughput and lag effects must be measured.
- Reward: retain Luna medium, the supplied weighted binary rubric, both anonymous orders and final-only extraction.
  Retain recorded-cap grading and the existing failure/parser rules. No score correction or silent judge fallback.
  Use a separate continuation ledger capped at $5. Copy only completed question-specific controls and record their provenance.
  Human calibration and independent-provider verification remain pending. Training rewards remain provisional.
- Diagnostics: enable training entropy and add sampling-conditioned entropy as separate metrics.
  Observe reference divergence with `use_kl_loss=true`, coefficient zero and the preserved unfiltered actor score.
  This does not add a regularizer or alter GRPO rewards. Report sampled k3 divergence and signed log-probability change.
  These are sampled diagnostics on rollout prefixes, not exact full-vocabulary KL under an independently sampled policy.
  At global updates 25, 26, 30, 40, 50 and 60, score the exact consumed batch before and after optimization.
  Report detached-weight CISPO surrogate gain and mean probability changes for positive/negative advantages.
  These observations add forward passes but do not change gradients, optimizer state or answer selection.
- Metric: compare complete balanced passes over the same questions, plus every per-question trajectory.
  Report actual update count and coverage if walltime prevents a full second pass.
  Track zero/full-credit rates, cap and repetition rates, lengths, judge order gaps, loss, gradient norms, clipping,
  effective sample size, actual weight spans, reference-pass time, generation time, waiting time and checkpoint state.
- Controls: identical actor/reference scores must give zero divergence even with singleton sampling support.
  Singleton support must have zero sampling entropy while full-policy entropy can remain positive.
  Observational metrics must leave CISPO losses and gradients unchanged.
  Fixed-batch positive, reversed and zero update controls must report the corresponding surrogate-gain signs.
  The real checkpoint load, first observer passes, finite backward update and weight publication must pass in the allocation.
- Baseline: observed early versus later visits in this continuation, with the same prompt and sampling profile.
  This is a training-set fitting attempt. It is not a frozen base/checkpoint evaluation or a generalization experiment.
- Seed: keep 20261118. One continuation of one training seed. No significance test or seed-level error bars.
  Preserve all iterations and do not keep selecting variants until a statistical threshold is crossed.
- Resources: one eight-MI350 node, at most four hours and 32 GPU-hours for this continuation.
  Target 40 new updates, ending at global step 64 if all complete. This does not authorize another allocation after timeout.
  The training budget and any later hyperparameter changes remain researcher decisions.
- Saving and logging: retain the latest completed checkpoint, saving every five global updates and at normal final completion.
  Keep original checkpoint iteration 24 intact. Record the real starting iteration before generation.
  Publish complete native metrics from one VM writer during the run, with explicit optimizer-step coordinates.
  Keep AMD offline files privately archived without merging their independent counters into the online history.
  Supervise the GPU job, reward service, tunnel, disks, checkpoint saving, private uploads and W&B through terminal state.

## Results

No GPU job has been submitted. Ninety TCS tests and 19 CPU loss/diagnostic tests passed.
The two-process control covers a tensor-parallel rank with no allowed sampling tokens.
Live W&B controls verified explicit nonzero step coordinates, preserved running state and correct terminal state.
See [the software repair evidence](DIAGNOSTICS_REPAIR_20261121.md) for semantics and remaining qualification.
The native ROCm observation path, reference-pass memory and proposed three-group admission remain unmeasured.

### Planned figures

- Per-question reward versus global update, with first/second-pass means and seed-selected raw examples.
- Full-policy and sampling entropy versus global update, labeled in nats per token.
- Sampled initial-policy divergence versus update, separately from rollout/trainer mismatch.
- Fixed-batch surrogate gain and positive/negative-advantage log-probability changes at the scheduled observations.
- Caps, response tokens, judge-call gaps, gradient norms, clipping, waiting time and policy lag.
  Figures are descriptive, without inferential error bars. Report the complete trajectory.

## Global takeaways

Reserved for the user. Any proposed explanation after this continuation must distinguish measurements from hypotheses.

## Next steps

1. Complete CPU controls and live W&B verification. Review the admission change and bounded continuation.
2. Freeze the approved config and source, then submit and monitor the same easy-only experiment.
3. Inspect real update direction, divergence and raw proofs before choosing any further research change.

## Questions

Approve 96 concurrent answers for the proposed continuation, or retain admission 32?
The associated bounded continuation is 40 additional updates, four hours on eight GPUs and a $5 judge cap.
