# Experiment: all236 easy and medium questions with the accepted Luna-high judge

## TLDR

The user explicitly approved all easy and medium questions and a launch on November25.
Use all20 easy and216 medium questions, uniformly shuffled, with no reward-based filtering.
Resume checkpoint108, use the unchanged final reference-guided Luna/high prompt v5, and train on8 AMD GPUs for at most12 hours.
Preserve the shared$25 paid-judge cap, including$1.7725486 already spent. Request a final save at11.5 hours or$22 spent/reserved.
The user accepted the remaining judge disagreement. Preserve the measured85% exact agreement and failed original numeric gate.
This approval supersedes the earlier20/26-question pilot restriction. Do not silently change optimizer hyperparameters.

## Background and question

Does the broader corpus and accepted fixed grader produce increasing comparable rewards within the approved runtime?
An increase requires matched-question or fixed-checkpoint evidence, not merely a rising average from different sampled questions.
If no increase appears, report coverage, within-question variation, completion rate and policy health. Do not infer a single cause.

The latest ten-update run changed matched-question reward from5.752 to5.776/7 across19 questions.
Its normalized reward metric stayed near zero by construction, while raw reward showed little gain too.
There were177/1280 order disagreements with the old paired-medium grader,29/40 groups averaged at least6/7,
and3/40 groups had constant reward. Answers averaged44,863 tokens and106/1280 reached the cap.
Gradient norms were0.084–0.126, importance-weight ESS ratios at least0.99933, and CISPO clipping zero.
Rollout and trainer weights matched. Full entropy varied0.427–0.533 and sampled reference divergence rose0.0095→0.0144.
These support concerns about score noise, saturation, question mix and long terminal-reward credit assignment.
They do not prove that the optimizer is broken or establish that a larger corpus will improve learning.
Exact sources are in [the metric review](evidence/long236_flat_reward_review_20261125.json).

## Methodology

- Model: pinned Qwen3.5-4B, thinking enabled, resume iteration108 with109 saved historical updates.
  Preserve model, optimizer, scheduler and RNG state. Reset rollout queues and data position for the new corpus and grader.
- Data: all236 easy/medium rows from `ian-and-friends/tcs_v3@8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
  Read only the ten eligible cohorts. FOCS2026 and STOC2026 remain sealed with zero held-out reads.
  An easy example converts squared-loss improvement to signed correlation by clipping and recentering.
  A medium example asks for an elementary mean-sensitive concentration proof without quoting a concentration theorem.
  Freeze each formatted input, rubric, reference and identity. The maximum prompt is1099 tokens.
- Reward: unchanged `rubric_judge_reference_v5.txt`, GPT-6 Luna/high, one explicitly labeled reference presentation.
  Grade the extracted final answer with the original weighted Boolean rubric. Code sums its zero-to-seven reward.
  Preserve every paid attempt. Provider failure never becomes reward zero. No silent provider switching.
- Training: fully asynchronous GRPO/CISPO,4 groups ×32 answers, global batch128, admission96, staleness6.
  Keep Adam LR1e-6, betas0.9/0.98, weight decay0.1, MTP weight0.2, BF16 and FP32 gradient accumulation.
  Keep temperature1, top-p0.95, top-k20,90000 new tokens including thinking,98304 total context and tokens/GPU.
  Actor uses2 GPUs, rollout uses6 GPUs in three TP2 engines. Observe entropy and reference divergence with zero penalties.
- Controls: check supplied reference, empty and irrelevant answers throughout the236-question corpus before submission.
  Reuse exact cached requests on restart. Record unexpected outcomes rather than silently changing scores or dropping data.
  Verify first actual backward update,128 consumed answers, dynamic microbatches, scheduler LR and three-engine weight publications.
- Metrics: raw0–7 reward by difficulty and question, within-group standard deviation and zero-variance fraction,
  full/zero-credit fractions, completion/cap rates, response tokens, entropy, reference and rollout divergence,
  effective sample size, clipping, gradient norm, surrogate gain, actual microbatches and weight staleness.
  Keep centered `rollout/rewards` labeled separately from raw `rollout/episode_raw_reward`.
  Read reproducible raw examples alongside aggregates. Track question coverage because a12-hour run may not revisit every question.
- Baseline: preserve the old ten-update traces. The saved384-answer checkpoint108 screen can be regraded with v5 without new generation.
  If budget permits, use the already approved same-question final screen to compare frozen policies under the same judge.
- Seed:20261124, one continuation. Comparisons are descriptive. No significance claim or seed-level confidence interval.
  A changing question mix and changing judge confound comparison with old mixed training means.
- Resources: one8-GPU AMD allocation,12-hour limit, maximum96 GPU-hours. Screening allowance remains separate and is not reset.
  Stop at the shared judge threshold or unsafe checkpoint storage. Preserve all source checkpoints and retain the latest new checkpoint.
  Save every5 updates, with an explicit graceful-stop trigger. The large update target is an upper bound, not an extra allocation.
- Logging: print and attach the exact judge prompt, safe parser defaults, source/model/data/container revisions and complete setup in W&B.
  Use the existing single VM writer and native optimizer-step axis. Archive full rollouts and judging receipts privately in HF.
  Foreground supervision continues through scheduler terminal state, checkpoint inspection, immutable archive checks and W&B verification.
  Inspect metrics every minute and raw proofs at least hourly. Repair diagnosed implementation failures within the approved scope.

## Results

Pending. No training result is claimed by data preparation or judging controls.

Planned plots: raw return by difficulty and matched question against optimizer step, coverage and zero-variance groups,
entropy/reference divergence, response length/cap rate, gradient/ESS/clipping, throughput/staleness and actual paid usage.
Use one claim per plot and include seeded raw examples. Report whether completion or proof quality changed.

## Global takeaways

Reserved for the user after measured results.

## Next steps

Finish full-corpus controls and CPU checks, freeze the accepted execution snapshot, submit once, and supervise through final verification.
