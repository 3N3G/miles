# Numerical reference check for the fixed Qwen serving policy

## Question and decision

Eight completed answers in group 534174 received zero in both Luna orders.
The original four-answer attempt also had zero grades. The full replacement group continues.
Check whether the working vLLM implementation agrees numerically with a standard Transformers forward pass.
A pass reduces uncertainty about loading and short-prefix implementation errors.
It does not establish long-generation parity, proof correctness, or corpus-wide model ability.
A failure requires numerical/runtime diagnosis before interpreting model quality.

## Fixed inputs and implementation

Keep Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` and the exact approved thinking prompt.
Use the same eligible SODA 2023 polygon-packing question, with strip/bin/perimeter factors 51/11/7.3.
The input hash, selection and compatible image hash remain owned by `group_check_v7_mi210_vllm8.json`.
FOCS 2026 and STOC 2026 remain sealed.

Use the original prompt and append zero, 128 or 2,048 re-encoded tokens from its first completed response.
That response is sample 2 from job 534174. A separate private fixture freezes its bytes and SHA-256.
Both implementations receive identical explicit token IDs for each teacher-forced prefix.
The private artifact saves all input IDs. No new scientific question or system prompt is selected.

The already qualified image supplies vLLM 0.29.0+rocm723, Torch 2.12 and Transformers 5.16.1.
Run BF16 weights, target-model vLLM decoding and one output token, with raw top-20 log probabilities.
Disable prefix caching and speculative acceleration. Use eager execution and one request at a time.
Shut down the owned vLLM engine before loading the standard HF conditional-generation model.
Use eager HF attention, evaluation mode and no gradient or past-key cache.
Capture checkpoint loading diagnostics. Missing checkpoint parameters fail the check.
There are no judge calls or optimizer updates. This is a software test, not another reward experiment.

## Metrics and acceptance declared before execution

For each prefix, report the largest absolute difference in its returned top-token log probabilities.
Also report the HF log-probability gap between the HF best token and the vLLM chosen token.
Require both values to be at most 0.2 nats, a coarse BF16 agreement check.
Near-tied tokens may have different greedy choices. Report that fact rather than requiring bit identity.
Reject missing/changed input IDs, changed decode length, missing model parameters and non-finite values.
The CPU control accepts a near tie and rejects a large discrepancy or non-finite value.
Record all measurements even on failure. No statistical significance or learning claim is made.
Seed: inherit 20261118. Plot: none, because three deterministic prefix checks need a small numerical table.

## Allocation and provenance

`policy_reference_v1.json` requests one MI210 for at most 15 minutes, or 0.25 GPU-hours.
Use a fresh pinned checkout and run directory. Reuse the immutable compatible SIF and pinned cached weights.
The reduced KV reservation is local to this short technical test. Neither active generation job is modified.
Record the config/source/image/input/fixture hashes, loader diagnostics and numerical results.
Upload artifacts privately with `push-run`. Preserve every failed attempt.
This diagnostic is within the user's authorized implementation repair/validation scope.
The RL gate still requires 32 genuine, complete, nonconstant rubric grades.

## Results

Not submitted yet.

## First attempt and exact runtime repair

Job 534297 ran for 168 seconds. vLLM produced all three raw top-token log-probability records.
Its owned engine shut down before the HF model loaded. HF reported no missing parameters.
The first HF convolution then failed while MIOpen created a lock directory under read-only home.
No numerical comparison completed, so this attempt establishes neither parity nor a discrepancy.

Set the documented `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` under the image/partition work cache.
The ROCm 7.2.3 lock-file source places locks beside the chosen database file.
Forward both variables to future Ray workers. Leave the system DB, solver selection and tuning modes unchanged.
Repeat from immutable `policy_reference_v2.json` with the same fixture, prefixes and 0.2-nat tolerance.
CPU path checks pass. Actual convolution and numerical parity still require the replacement job.

Sources:
- https://rocm.docs.amd.com/projects/MIOpen/en/docs-7.2.3/conceptual/tuningdb.html
- https://raw.githubusercontent.com/ROCm/MIOpen/rocm-7.2.3/src/lock_file.cpp

Private first-attempt evidence: https://huggingface.co/datasets/3N3G/internalization-runs/tree/3349f13d3c95c3ffc3285b472b415b20dbe6dfd6/miles/policy-reference-v1
Loading metadata used empty sets, serialized as strings in v1. V2 writes those sets as JSON arrays.
The live missing-parameter guard used the original empty set and did pass.
