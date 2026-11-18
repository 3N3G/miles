# Reward signal, output precision, and recovery audit

No TCS model rollout, judge call, or learning update has run. Code support is not measured readiness.
The training configuration now enables Qwen thinking. The user explicitly kept Luna at medium effort.
Qwen's pinned template exposes a Boolean thinking switch, without a high-effort setting.
The 90,000-token generation ceiling remains unchanged.

## Reward signal is unmeasured

There is no measured nonzero-reward rate for this model and dataset. Do not assign a confidence percentage.
Of 2,519 eligible questions, 2,283 (90.6%) are labeled hard. A 4B solver without paper context could struggle.
Those labels do not establish its success rate. Long thinking does not establish TCS proof ability.
The supplied weighted rubrics allow partial credit, but the adapter and its calibration are unfinished.

GRPO centers rewards within each prompt's group in
[`train_data_conversion.py`](../../miles/ray/rollout/train_data_conversion.py#L328).
An identical score on every response yields zero reward-derived advantages, even if that score is positive.
MTP can still update its auxiliary objective. This does not establish a useful policy-reward signal.

Illustration only: independent binary rewards with per-response success probability p have
all-zero group probability `(1-p)^32`. At p=1%, it is 72.5%. At p=5%, it is 19.4%.
Real responses share a problem and need not be independent. These are arithmetic examples, not forecasts.

Before learning, propose a frozen-base check on randomly selected eligible papers, preserving the final test boundary.
Report zero-score and uniform-score group fractions, per-rubric decisions, parse failures, completion status, and lengths.
Inspect raw cases beside totals. Use checked positive and empty/irrelevant negative controls, plus human calibration.
Reward API and parser failures must remain errors, not zero scores. This audit authorizes no GPU or paid baseline.
Do not respond to zero signal by silently changing the rubric, data, model, or prompt.

## FP32 logits are already implemented

The trainer supports `--lm-head-logits-dtype fp32` with the selected `fused_lm_head` backend.
See [`arguments.py`](../../miles/utils/arguments.py#L636) and
[`_Fp32LogitsLinear`](../../miles/backends/megatron_utils/lm_head.py#L154).
It keeps BF16 weights and hidden inputs, accumulates/returns FP32 logits, and rounds the logits gradient once for backward.
It also covers the multi-token prediction loss. This is not full-model FP32 training.
Casting BF16 logits after projection cannot restore bits already rounded away.

The rollout counterpart is `--sglang-enable-fp32-lm-head`. Ordinary CISPO does not force it on.
The current proposal leaves both precision flags at their defaults. No precision change is approved yet.
Existing GPU correctness tests and the long-head benchmark already exercise trainer model-dtype and FP32 modes.
Neither has run on this AMD installation. The pinned container's serving implementation also needs verification.

FP32 addresses one source of trainer/inference probability mismatch. It does not make a weak solver earn rewards.
It cannot remove every kernel difference or stale policy weights during async training.
There is no evidence yet that it is necessary for this run. Proposed choice: qualify matched FP32 logits on both sides.
Compare fixed-weight token probabilities, gradient correctness, memory, and speed before selecting the learning configuration.
A stable short test would not establish stability at 90k generated tokens.

A 512-row vocabulary tile at TP2 occupies about 121 MiB in BF16 or 243 MiB in FP32.
The extra tile does not require making the whole model FP32. Workspace and serving costs need measurement.
Independent evidence shows rounding differences outside the head can matter:
[Where Rounding Happens Matters](https://arxiv.org/abs/2610.32951).
That paper does not validate this model, AMD layout, or async protocol.

## Logging exists, but recovery is not qualified

| Capability | Existing support | Current readiness |
| --- | --- | --- |
| Reward signal | Uniform-score counts and all-zero group fraction | Default metrics. The all-one fraction assumes full reward equals one |
| Generation | Lengths, truncation, repetition, speed, speculative acceptance | Existing rollout metrics |
| Async | Weight-version spans and prefill lag | Existing rollout metrics |
| Training | Loss, gradient norm, clipping, effective sample size, trainer/rollout probability difference and KL estimate | Existing metrics, including with the skipped scoring pass |
| Persistent metrics | TensorBoard, W&B, event logs | Learning proposal has no destination yet |
| Raw outputs | Optional sample and metadata dumps | Need a bounded private output path |
| Judge audit | Transport retries and explicit exceptions | Per-item grades, provider usage/cost, and parser logs need the adapter |
| Health and recovery | Optional health checks and fault-tolerance controller | Disabled by default. AMD behavior is untested |
| Checkpoints | Training and rollout-state save paths exist | Interval, retention, and restart qualification remain unfinished |

Metric sources: [`rollout/metrics.py`](../../miles/ray/rollout/metrics.py#L99),
[`losses.py`](../../miles/backends/training_utils/loss/hub/losses.py#L356), and
[`log_utils.py`](../../miles/backends/training_utils/metrics/log_utils.py#L503).
Logs reveal faults. They do not prevent memory exhaustion, kernel failures, node loss, or statistical collapse.
Dashboard metrics are not automatic stop conditions.

The qualification runner now persists a started event before each stage, followed by success or failure.
It records duration, return status, command, and one combined stdout/stderr log per stage.
A hard-killed process can leave an unmatched started event. It cannot record its termination after SIGKILL.
No automatic resubmission or cancellation of other tasks is enabled.

Before learning, wire persistent metrics, private raw samples and grades, finite-value diagnostics,
GPU-memory/progress monitoring, explicit judge failures, and tested restartable checkpoints.
Choose stop thresholds from the base-model check and user review. No arbitrary scientific threshold is selected here.

The updated CPU suite passes 16 checks. Tokenizer validation of the seeded eligible example produces 374 prompt tokens
and ends with an open `<think>` prefix. The formatted private example remains outside the public repository.
