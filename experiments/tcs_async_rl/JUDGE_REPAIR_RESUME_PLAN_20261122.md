# Experiment: clarify rubric grading and resume easy-question RL

Status: authorized by the user's 2026-11-22 request to adjudicate ambiguities, repair the prompt, and rerun or resume.
This plan uses assistant mathematical adjudications as diagnostic labels. They are not human calibration labels.
The next continuation retains the prior per-run ceiling: 40 new updates, four hours on eight GPUs, and $5 for judging.
No open-ended compute allocation or model replacement is inferred from the request to watch for reliable improvement.

## TLDR

Validate the versioned judge prompt, then resume Qwen3.5-4B from the checkpoint containing 65 updates.
Keep Luna medium, the same 20 easy questions, solver prompt, sampling, optimizer, topology, and seven-point rubric weights.
Only the grader's interpretation instructions change. Preserve all old prompts, configurations, grades, and checkpoints.

## Background and question

The preceding 40-update continuation moved 15 repeated-question means from 5.291 to 5.428 points out of seven.
Answers became 18.2% longer. Candidate scores differed across swapped calls for 338/1,280 answers.
The six-case review identified inconsistent treatment of implicit standard facts, domain typos, and dependent partial-credit items.
It also found valid full-credit proofs and an incorrect zero-credit answer. These findings do not isolate the cause of slow learning.

Does clarifying those rules produce defensible grading and continued matched-question improvement with healthy optimizer diagnostics?
If yes, retain the run through the approved limit and inspect the actual proofs behind gains.
If no, diagnose the observed failure. Do not raise rewards by weakening correctness requirements or select a favorable chart.
The cheapest likely failure test is rescoring saved proofs and reference/empty/irrelevant controls before allocating GPUs.

## Methodology

- Generator: the same pinned Qwen3.5-4B and one MTP layer, thinking enabled, full optimizer/RNG resume at iteration 64.
  Keep the original converted initialization as the observational reference.
- Data: the exact existing 20-question, 17-paper easy bundle. One example asks for a rooted-forest cost bound under correlated sampling.
  Its SHA-256 is `18ed636017d16c3af5c1448ee4967b4aa6b996037c73f5304bf4ce95469e5929`.
  FOCS 2026 and STOC 2026 remain sealed. No corpus mixture or solver-prompt change.
- Generation: 32 answers per question, global batch 32, admission 96, temperature 1, top-p 0.95, top-k 20.
  Preserve the 90,000-output-token and 98,304-total-token limits, support replay, and maximum weight staleness six.
- Optimization: unchanged GRPO/CISPO, constant learning rate 1e-6, Adam betas 0.9/0.98, weight decay 0.1.
  Keep entropy, sampled reference drift, separate MTP norm observations, and fixed-batch update-direction checks.
- Reward: Luna medium, both anonymous orders, original per-question binary rubric and weights, final-only extraction, and existing cap handling.
  Use `rubric_judge_prompt_v2.txt`. Its policy examples clarify elementary implicit inferences, harmless notation, substantive gaps,
  and item dependencies. They do not contain a new reference solution for the solver.
- Diagnostic labels: assistant-adjudicate saved answers independently of the revised judge outputs.
  Keep known prompt-development cases separate from fresh saved-answer validation cases. Record every item decision and justification.
  Require correct reference/empty/irrelevant controls and inspect false awards and false denials, not just score agreement.
  Retain every provider receipt and any unfinished-verdict retry. No silent judge fallback.
- Baselines: the original judge decisions on exactly the same saved candidates, and early versus later visits scored by the revised prompt.
  A score-scale shift after changing the prompt is not a model improvement. Do not compare unrescored old rewards directly with new rewards.
- Progress: inspect matched-question trajectories, full-credit and zero rates, lengths, caps, repetitions, judge gaps, entropy,
  initialization drift, rollout/trainer mismatch, gradients, clipping, weight hashes, version spans, and optimizer update direction.
  Call an observed gain provisional unless repeated comparable visits and raw mathematical checks support it.
  This remains one training seed, without a frozen-policy evaluation or a proof of reliable generalization.
- Seed and statistics: retain seed 20261118. Report descriptive paired-question changes and coverage. No significance threshold,
  optional-stopping test, or seed-level confidence interval is introduced by this repair.
- Compute: at most 40 additional updates, global steps 65–104, or four hours on eight MI350 GPUs, whichever comes first.
  Keep repairs within the combined 32 GPU-hour ceiling for this continuation. The judge ledger is capped at $5.
- Storage and supervision: verify space for retained source checkpoints and new atomic saves before submission.
  Observe the GPU job, service, tunnel, archive, and W&B independently. Review at least hourly and through terminal cleanup.
  Preserve iteration 64 and earlier results. Upload results without checkpoints and verify immutable archive checksums.

## Results

Not run. Prompt validation and storage preflight are in progress.

### Planned figures

Per-question rewards versus global update; matched-question visits; full/sampling entropy; sampled initialization drift;
fixed-batch update direction; gradient/clipping/staleness checks; response length/caps; and judge item disagreements.
Link a seed-selected raw answer beside each question trajectory. Label all plots as descriptive, single-seed training observations.

## Takeaways

Reserved for the user. Assistant interpretations will remain explicitly provisional.

## Next steps

Validate the judge repair, resolve storage safely, freeze and submit the continuation, then supervise through completion.
If the bounded run does not establish reliable growth, report that limit and the measured failure rather than claiming success.
