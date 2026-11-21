# Experiment: initial fitting attempt on the 20 easy TCS questions

Status: the user approved option 1 for recorded caps and selected 40 updates with a four-hour, eight-GPU limit.
The frozen run snapshot is `rl_easy_overfit_v8.json`. All earlier run snapshots remain unchanged.

## TLDR

Train Qwen3.5-4B on all 20 eligible easy questions using the existing asynchronous CISPO recipe.
Attempt 40 optimizer updates, or stop at the four-hour allocation limit.
Each update requests one prompt group of 32 answers. Forty updates target two nominal passes and 1,280 consumed answers.
Actual question coverage and policy versions must be measured. This is a first fitting pilot, not a generalization test.

## Background / motivation

The easy diagnostic had 15 positive grades, 13 zeros and four capped missing finals.
Actual Luna calls now give those four empty finals zero in both orders, yielding 32 real qualification grades.
The original 28 grades remain unchanged. The medium question had constant-zero rewards, and its automatic handoff stopped.
The user selected the complete easy subset and asked whether training can fit it before considering broader data.

## Concrete question we are answering

Does this bounded run complete native optimizer updates and increase observed rewards when the same training questions recur?
If rewards rise, report their per-question trajectories, lengths, caps and actual model versions for inspection.
If they do not rise, distinguish runtime failure, absent reward variation, truncation and limited optimization time.
A negative result within 40 updates or four hours does not establish that this initialization cannot fit the questions.
The earliest likely failures remain checkpoint conversion, backward kernels, memory and weight publication.
Qualify recorded-cap judging, every question's reward controls and the exact native arguments before relying on training results.

## Methodology

- Model: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, thinking enabled.
  Keep BF16 parameters, FP32 accumulated gradients, one trained MTP layer with factor 0.2, and target-only rollout decoding.
- Data: all and only the 20 `easy` questions in the ten eligible cohorts of
  `ian-and-friends/tcs_v3@8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
  They span 17 papers, with 12 proof-writing and eight proof-strategy tasks.
  An example asks for an expected rooted-forest cost bound under correlated sampling.
  A strategy example asks how weighted spanning-tree packing gives a tree crossing a specified cut few times.
  The private bundle stores every original prompt, answer, rubric, identity and exact chat-template rendering.
  The actor sees the problem only. References and rubrics belong to grading. FOCS/STOC 2026 remain sealed.
- Sampling: uniform shuffled input records, 32 answers per prompt, temperature 1, top-p 0.95, top-k 20,
  90,000 output tokens and 98,304 total context. Keep terminal/thinking markers for final-answer extraction.
  Do not discard hard-performing easy questions or resample capped responses to manufacture favorable groups.
- Algorithm: retain GRPO advantages and CISPO loss, Adam at `1e-6`, constant learning rate and the v2 sampling-support replay profile.
  Use global batch 32, rollout batch one, async concurrency 32, the existing buffer and staleness settings,
  two trainer GPUs and six rollout GPUs, with tensor parallelism two.
- Reward: Luna medium uses each question's supplied seven-point binary rubric in both anonymous answer orders.
  Grade any nonempty final normally. For an empty final at a recorded model token limit, judge the empty answer in both orders.
  Both empty scores must be zero.
  An unfinished Luna verdict may retry with 16,384 then 32,768 completion tokens, under the same shared spending cap.
  Keep the 8,192-token initial budget and medium effort. Preserve all receipts. Never retry a completed grade. Unclassified missing finals, provider failures and parser failures remain errors.
  A question-specific router shares one concurrency limit and one $5 judge ledger across all 20 questions.
  Every reference/empty/irrelevant preflight control must receive 7/0/0 in both orders.
- Metrics: actual optimizer updates, finite losses and gradient norms, skipped steps, exact question coverage,
  per-question reward distributions by visit and policy version, full-credit and zero rates, cap rates,
  output lengths, weight publication, checkpoint state, elapsed time and judge cost.
- Baselines and controls: compare observed early and later training visits on the same question.
  Early visits already occur during optimization and are not a frozen-base evaluation.
  The existing frozen easy diagnostic covers one question only. There is no separate before/after evaluation in this budget.
  Human calibration and independent-provider verification remain pending, so reward changes are provisional.
- Seed: 20261118. One training seed. No significance test, error bars over seeds or generalization claim.
- Plots: mean provisional reward and cap rate versus optimizer update, loss/gradient norm versus update,
  per-question first-versus-later visit rewards, and each delivered group's reward and token-count histograms.
  Include a seed-selected raw answer and judge links. Mark groups without an observed optimizer update separately.
- Resources and saving: one MI350 eight-GPU node for at most four hours, at most 32 GPU-hours for this attempt.
  Save every five updates and retain the latest completed checkpoint. During a save, two full checkpoints may coexist.
  Preserve exact replay fields in gzip JSON artifacts. Native PyTorch snapshots stay on AMD.
  Transfer compressed traces in batches up to 512 MiB. Evict only VM cache copies after a successful immutable private upload.
  Preserve the AMD originals and record each archived path, checksum and upload commit.
- Logging: native offline W&B and TensorBoard, structured optimizer logs, generation/judgment traces and compressed delivered groups.
  Upload W&B to the verified `geney/tcs-async-rl` destination. Credentials remain on the VM.
  The existing guarded Slurm submission and owned-job monitor record real scheduler receipts and terminal state.

## Results

Not submitted at plan creation. Data preparation produced 20 template-validated rows, with a maximum prompt length of 761 tokens.
Eight actual qualification judge calls scored the four recorded empty caps zero in both orders.
The resulting qualification group has 15 positive and 17 zero grades, with mean 2.5625 of seven points.
No optimizer update has run yet.

## Global takeaways

Reserved for the user. This pilot measures training-set fitting behavior within a bounded budget.

Pre-submission inference repair: the ninth question’s irrelevant control exhausted all 8,192 judge tokens without a verdict.
Twenty-six prior cases passed. The bounded token retry above repairs inference completion without changing scoring.
Evidence: `evidence/judge_length_repair_20261121.json`.

## Next steps

1. Finish CPU tests, route controls and the source/config review.
2. Freeze, push, submit on AMD and monitor conversion, generation, backward computation, saving and exit.
3. Preserve failures, record exact repairs before replacements, and report actual training coverage and rewards.

## Questions

The human calibration packet and independent judge access remain open for interpreting learning.
