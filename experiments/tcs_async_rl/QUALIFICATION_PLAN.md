# Experiment: AMD environment qualification for long TCS rollouts

Status: proposed technical check. Requires approval before GPU submission.
No GPU result exists. This check makes no claim about learning or proof quality.

## TLDR

Test the AMD software stack before spending on a learning run.
Use one eight-GPU node for at most two hours. Use public model weights and synthetic token IDs.
Make no judge calls, read no private dataset, and perform no optimizer update.

## Background / motivation

The intended solver is Qwen3.5-4B, with a 90,000-token generation limit and 98,304 total tokens.
The available AMD partition has whole nodes with eight MI350-class GPUs.
The current memory calculations are component estimates. They are not measurements of this software stack.
The container packages SGLang for generation and Megatron for training. Neither has run here with this fork.

## Concrete question we are answering

Can the pinned container run this fork's tiled output head and Qwen long-context inference on AMD?
If yes, measure full-model training, checkpoint conversion, and simultaneous training/generation next.
If no, diagnose the failing dependency or kernel before requesting a learning allocation.
A failure here does not test whether reinforcement learning can improve TCS proofs.

The first likely failure is container compatibility. Check imports before downloading the model or running kernels.

## Methodology

- Model: Qwen/Qwen3.5-4B, pinned by the linked training configuration.
  Use its supplied multi-token prediction layer with EAGLE decoding.
- Data: synthetic token IDs only. Each ID lies between 1,000 and 19,999.
  Store each complete input and output with the measurements.
  No TCS question, answer, rubric, or held-out example enters this check.
- Controls: a 4,096-token input checks the shorter path first.
  A 98,272-token input plus 32 generated tokens reaches the planned total context.
  Report these as prefill and short-decode checks. They do not test 90,000 consecutive decode steps.
- Communication: check a 2 MiB all-reduce across all eight GPUs.
  Each rank starts with ones. Every result must equal eight.
- Output-head correctness: run the existing materialized-versus-tiled gradient tests, including the prediction-layer loss.
  Then benchmark the full planned token count and vocabulary shard with a 512-row tile.
  Treat any reported out-of-memory case as a failure even if the benchmark returns success.
- Seed: fixed in the qualification configuration for reproducible synthetic inputs.
- Metrics: pass/fail, dependency versions, device memory, head runtime and peak memory,
  input/output token counts, and inference elapsed time. Keep each raw case beside its aggregate timing.
- Statistical test: none. This is a technical qualification, without an estimated learning effect.
- Exact plots: none. There are two input lengths and no repeated performance study.
- Limits: one inference engine and one request at a time. This does not test the proposed production concurrency.
  It does not test full-model backward, optimizer memory, checkpoint conversion, async weight publication,
  pause/resume, save/reload, the reward adapter, or judge calibration.

The canonical allocation and test settings are in [qualification.proposed.json](qualification.proposed.json).
Operational commands and the submission gate are in [AMD_RUN_REVIEW.md](AMD_RUN_REVIEW.md).

## Results

Pending approval and execution. CPU checks cover the job renderer, allocation guard,
credential exclusion, error handling, and token-length assertions.

## Global takeaways

Pending measurements. Passing these checks qualifies only their stated scope.

## Next steps

1. Review and approve this bounded job.
2. Record every stage and upload its results after execution, including failures.
3. Use measured results to prepare the remaining training integration check.

## Questions

Approve the bounded technical allocation before submission. Learning design decisions remain in QUESTIONS.md.
