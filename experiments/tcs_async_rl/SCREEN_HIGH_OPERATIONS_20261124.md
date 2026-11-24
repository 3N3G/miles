# Approved checkpoint screen operations

The high-effort judge passed its frozen forty-answer gate and 51 controls.
See [the standalone result](HIGH_EFFORT_JUDGE_RESULTS_20261124.md).
This record prepares the already approved eight-GPU, two-hour inference screen.
It does not authorize a new experiment or extend that allocation.

## Frozen inputs and recipe

- Configuration: `checkpoint_screen_high_v1.json`.
- Solver: saved Qwen3.5-4B iteration108 from `easy-overfit-v17`, with 109 historical updates.
- Dataset: revision `8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
  FOCS2026 and STOC2026 remain sealed. Preparation read zero held-out shards.
- Seed20261124. Shuffle eligible easy questions and take six with distinct papers.
  Then do the same for medium, excluding those papers where possible.
  All twelve selected papers are distinct. Exact identities are in `evidence/difficulty_screen_selection_20261124.json`.
- Input bundle SHA256: `4ecded3e9db2947b19eb13ca8f1589a239b206f6607c684eae88f04da8fbbea0`.
  Prompt lengths are186–985 tokens, below the8,304-token prompt allowance.
  All twenty easy identities match the previous training run.
- Inference:32 answers/question, thinking on,90,000 output,98,304 context, temperature1,top-p0.95,top-k20.
- Judge: Luna high, labeled reference promptv2, supplied binary rubric and original weights, final-only extraction.
  Runtime fingerprints pin the prompt, training judge and qualification evidence.
- Reuse the actual previous topology: two actor GPUs with tensor parallel2/context parallel1,
  and three tensor-parallel2 rollout engines on six GPUs.
  The saved v17 `resolved-config.json` verifies tensor parallel2/context parallel1.
- Screen engine admission is96 requests, using the existing per-engine concurrency32 over three engines.
  There are384 exact eval samples. The fully asynchronous training producer never starts.

## Runtime repair and validation

The old reward server always graded both anonymous orders and loaded medium effort.
The selected runtime now reads the frozen judge settings and grades the labeled candidate once.
It records one score and omits the unmeasured answer-order gap.
Empty capped answers still receive an actual judgment. Provider errors raise rather than become zeros.
Legacy paired runs retain their old contract and diagnostics.

The new `--reset-rollout-state` launch flag skips rollout queue/data-position restoration.
Trainer model, optimizer, scheduler and random state still restore through the existing trainer path.
The inference-only entrypoint loads those trainers and publishes their weights, then invokes native evaluation.
It never calls an actor train method or a checkpoint save.
All three restored engine checksums must match the source checkpoint's saved publication before generation.
The completed artifact must contain exactly twelve pinned question identities and32 answers each.

CPU checks cover selected question identity, explicit dataset contracts, source checksum rejection,
single-reference reward validation, and actual single-call empty-cap grading.
Native argument parsing and checkpoint restore are checked inside the approved GPU allocation.
Preparation initially lacked Jinja2. Adding that tokenizer dependency fixed preparation before any GPU submission or API call.

## Spending, artifacts and supervision

Before service startup, copy the exact settled comparison ledger into this sequential phase's judging directory.
The incoming ledger has536 settled calls, zero pending, total$0.5342506,
SHA256`0d070e4f48f50f05b56a30c4c92f21470fe1461379ab01081540bb426ee16e7e`.
Record that transfer. Only one active owner may spend against the shared$25 cap.
Run reference, empty and irrelevant controls for all twelve questions before submitting GPUs.
Stop this service before transferring the updated ledger to RL.

The source checkpoint and all prior outputs remain protected.
The later twelve-hour run may keep its latest completed checkpoint, using the existing own-run retention hook.
Periodic saves every five updates require space for two new checkpoints while a save completes.
It will stop admissions at$22 and reserve at least30 minutes for a final save.
Screening saves no checkpoint.

Monitor scheduler state, native logs, engine identity, service/tunnel health, generated counts and disk space.
After completion, inspect one seeded raw answer per question and its item decisions.
Upload results with `push-run miles`, verify immutable hashes, and record the receipt in the job registry.
Apply the approved medium inclusion rule without choosing only successful questions.
