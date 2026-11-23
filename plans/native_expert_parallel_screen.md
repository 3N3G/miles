# Parallel causal screen and prospective text confirmation

## Reason and authorization

The user approved autonomous expert research within five GPU hours. AMD's
single-GPU estimate is several hours away. An exclusive eight-MI350 node is
idle. Replace the still-pending single-GPU job with one eight-GPU allocation,
at most12 minutes. Count all eight GPUs:96 GPU-minutes reserved. Cancel only
this session's old pending job, record its terminal state, and preserve its
submitted protocol. This is a scheduling change, with independent prompt shards.

## Calibration and development protocol

Use the exact model, formats, calibration and discovery questions, old text,
expert edits and control selections from `native_expert_calibration.md`.
One worker selects format using calibration only. All workers then use that
format. Each prompt runs on one GPU. No prompt is split across GPUs.
Expert IDs always refer to physical layer12 unless a derivative record gives
a different physical layer. Matched controls from old fit routing are89,41,90.

Add an explicitly exploratory derivative screen on the32 discovery questions.
For every loop-core layer6..17 and each of its three visits, differentiate the
gold four-choice log probability with respect to selected routing weights.
Multiply weight by its derivative and sum by expert. Later routers remain in
the graph. Freeze model weights. Top-k choices remain piecewise fixed. This
predicts infinitesimal removal effects only. Validate candidates by exact
removal before interpreting them. Rank candidates using discovery questions
alone. Do not read reserved test question outputs in this stage.

## Prospective text protocol fixed before inference

Use48 new WikiText-2 test articles,128 tokens per article, one paragraph each.
Exclude prior fit/development/final articles. Freeze IDs and file hashes in
`evidence/native_expert_text.json`. Evaluate prediction positions16..126.
Example: the first frozen paragraph and complete tokenized input are in that
manifest and printed before submission.

Primary test: does removing expert105 at all visits increase the loss more on
numeric targets than other targets? A target is numeric when its decoded token
contains a digit or exactly matches the previously fixed number/ordinal/time
word list. Secondary primary contrast: numeric input token versus other input
token. These distinguish producing a numeric token from processing one.
For each article calculate mean loss change within both strata, then their
difference. Exclude articles without both strata from that contrast only and
report the count. Bootstrap articles10,000 times, seed1729, Bonferroni-adjusted
97.5% intervals for the two contrasts. No token-level independent-sample test.

Report expert105 routing enrichment on this new set, with article bootstrap
and Bonferroni over three passes. Show full raw examples near the median
article effect plus largest effects, labeled as such. Show all other experts
and pass-specific removals descriptively. Renormalization and frequency-matched
experts control some mass/frequency confounds. Matched output-norm noise is
reserved for a later stage. This test can establish selective sensitivity,
not an abstract arithmetic algorithm or a unique expert function.

## Outputs and positive/negative controls

Save per-token loss, routes and weights, full text, question scores, gradients,
native identity and absent-expert exact checks, format selection, source hashes,
memory and runtime. Same controls as the original screen. Read clean accuracy
before interpreting changes in task answers. All initial data and hypotheses
are preserved even if the new test fails.

## Results

The eight-GPU allocation was cancelled while pending, using0 GPU seconds. The frozen forward stages ran on one GPU under native_expert_screen_single.md. Forward outputs completed before the later derivative-range assertion failed. Results: evidence/native_expert_screen_result.json. Archive: expert-screen-v3.
