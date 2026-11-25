# Final judge and planned training setup

The user accepted the remaining grading disagreements and retained prompt v5.
The measured validation remains 34/40 exact scores and 96.17% item agreement, with 132/132 controls passing.
The [acceptance record](evidence/judge_v5_user_acceptance_20261125.json) preserves the failed original numeric gate.
No provider call or GPU job ran while preparing this setup.

## Exact judge request

The complete system message is [rubric_judge_reference_v5.txt](rubric_judge_reference_v5.txt), unchanged from final validation.
W&B Files contains an exact copy and a rendered request template.
The user message is JSON with precisely these fields:

```json
{
  "problem": "<question text>",
  "rubric": "<original numbered, weighted rubric>",
  "reference_solution": "<explicitly identified supplied reference>",
  "candidate_solution": "<Qwen's extracted final answer>"
}
```

The generator sees the problem through Qwen's chat template, with thinking enabled.
The judge sees the final answer, not Qwen's thinking, model identity or experiment condition.
There is one labeled-reference presentation, without reversed-order averaging.
There is no new lower-score tie-break instruction.

| Judge setting | Value |
|---|---|
| Model / effort |`gpt-6-luna` /`high`|
| API |OpenAI Chat Completions, JSON response mode|
| Initial output budget |8192 tokens, including judge reasoning|
| Unfinished-output expansion |Bounded expansion to 32768 tokens, with every attempt recorded|
| Concurrency / attempts |8 concurrent calls, at most 3 malformed/provider attempts per completion budget|
| Output |One Boolean and short reason per original scored rubric item|
| Reward |Code sums satisfied-item weights into the original 0–7 score|
| Empty answer at a recorded model token cap |Send the empty answer through the same rubric judge|
| API failure |Surface failure after bounded retries; never substitute reward 0|
| Provider changes |No silent fallback or mixing with Gemini|

## Planned continuation

These are the already approved settings. They are not measurements from a new training run.
The versioned candidates are [easy](rl_long_high_easy_v1.json) and [mixed](rl_long_high_mixed_v1.json).
Those preserved candidates still name the earlier grader qualification.
A new execution snapshot must bind v5 and the recorded user acceptance before submission.

| Training setting | Value |
|---|---|
| Model |Qwen3.5-4B, revision`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`|
| Resume |Saved iteration 108 from`easy-overfit-v17`, after 109 historical optimizer updates|
| State |Preserve model, optimizer, scheduler and random-number state; reset rollout queues/data position|
| Target allocation |One AMD`mi3508x` node, 8 GPUs, up to 12 hours of RL|
| GPU division |2 actor GPUs, 6 rollout GPUs across 3 tensor-parallel engines|
| Parallelism |Tensor parallel 2, pipeline parallel 1, context parallel 1|
| Optimizer batch |4 problem groups × 32 answers = 128 answers per update|
| Gradient accumulation |Dynamic token batches, 98304 tokens/GPU; log actual microbatch count per update|
| Sampling |Thinking enabled, temperature 1, top-p 0.95, top-k 20|
| Length limits |90000 new tokens, including thinking; 98304 total context tokens|
| Algorithm |Fully asynchronous GRPO/CISPO, rollout log-probabilities, clipping settings 10 and 4|
| Optimizer |Adam, constant learning rate 1e-6, beta1=0.9, beta2=0.98, weight decay 0.1|
| Precision / auxiliary loss |BF16 model, FP32 gradient accumulation, MTP loss weight 0.2|
| Async limits |96 concurrent samples, maximum weight staleness 6, retry unused samples|
| Diagnostics |Raw reward, per-question visits, entropy, reference divergence, clipping, gradient norm, lengths, caps and weight lag|
| Regularization |Reference-divergence coefficient 0; entropy coefficient 0 in the validated parser configuration|
| Saves / early stop |Every 5 updates; checkpoint-and-stop near walltime, shared judge threshold or unsafe storage peak|
| Paid judging |Shared $25 cap, with $1.7725486 already recorded before this setup review|

Data is pinned to `ian-and-friends/tcs_v3` revision`8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
FOCS 2026 and STOC 2026 remain sealed and unread.
The remaining preparation is to grade the existing 384 screening answers with unchanged v5.
The declared rule then selects 20 easy questions, or 20 easy plus all 6 sampled medium questions.
It includes medium only if at least 2 of 6 medium groups have positive, genuinely varying rewards.
No new solver generation is needed for this choice.

## What W&B and Hugging Face contain

W&B Config and Files hold the exact judge prompt, API contract, safe training hyperparameters and source identities.
Future training's single online writer prints the full setup once to its console.
Actual parser defaults, code/container versions, dataset/model revisions, seed, resume point and limits are included.
The x-axis is the native optimizer step. Dynamic accumulation is `audit/train/num_microbatches`.
Validation artifacts and planned training metadata are labeled separately from actual training.

One graded API exchange was chosen from the 40-case validation packet with fixed seed 20261125.
It is case 01, the clipping-to-correlation proof, graded 7/7. It includes the full input and all five item reasons.
This is a random example from a stratified diagnostic packet, not an estimate of training accuracy.
The complete source exchanges are in the [verified private validation archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/d4d68e07f0de4ebe2c91597d8eff2b51d608a75c/miles/judge-comparison-item-v5-20261124).
All 384 screening generations and their full saved text are in the [verified screen archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/7a258e0947401b05d2207f63592bea799e9284bd/miles/difficulty-screen-high-v3).
Future full rollouts, extracted finals, rubric decisions, request/response receipts, usage and failures follow that private archive policy.
Large traces stay out of the VM after immutable upload and checksum verification. Checkpoints remain on AMD.

## Verified publication

The [W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/judge-item-v5-20261124) now contains the reviewed setup under Config, Logs and Files.
All 9 attached files passed byte-for-byte checksum verification, and every nonempty prompt line appeared in the console.
The original single validation history point and finished state are unchanged. No new provider calls ran.
The [setup archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/460538cb5c244b0a074560ac03b0d77fb99578b9/miles/setup-review-20261125) passed immutable verification for 18 files.
The [complete graded example](https://huggingface.co/datasets/3N3G/internalization-runs/blob/460538cb5c244b0a074560ac03b0d77fb99578b9/miles/setup-review-20261125/graded-example.md) is available there.
The logging changes passed 18 CPU tests plus formatting and lint checks. No GPU training test ran for this metadata-only change.
Receipts: [W&B](evidence/setup-review-20261125-wandb-setup-receipt.json), [HF](evidence/setup-review-20261125-archive.json), [CPU checks](evidence/setup-review-20261125-logging-validation.json).
