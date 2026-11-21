# Frozen final evaluation: mean/scale calibration

## Question and decision

Does the simple unlabeled mean/scale correction improve recovery of unspoken
intermediate facts on the 43 source-bank items reserved before development capture?
This is the only primary final comparison. New cycle methods remain development
exploration and cannot replace this candidate after test outputs are read.
A positive result supports this small readout correction on this source bank.
A negative result leaves the development gain unconfirmed. Neither result proves
causal faithfulness or official challenge performance.

## Setup

Use the existing pinned Inkling 2 Mini checkpoint and Transformers implementation.
Use all 43 names in `evidence/transport_holdout_ids.json`, in that order, from the
same pinned WorkspaceBench JSON. Copy their exact prompts and labels. No prompt
edits, new training, new calibration paragraphs, or method tuning. Example of a
reserved source item: `violin-strings`; its exact source prompt is used as stored.
The source bank includes single- and multi-token intermediate names. Preserve both.

Capture the final prompt-token active state at visits 11,23,35,41. K=10 per visit.
Capture a native greedy continuation of at most twelve tokens with the same EOS
handling as development. Verify all 43 native final-head reconstructions exactly.
Do not use a chat template for these raw-completion prompts.

Mean/scale candidate: h_hat = mu_final + (sigma_final/sigma_visit)*(h-mu_visit).
Use the already fitted seed-0 active statistics from `transport-fit-v1` unchanged.
Seeds 1 and 2 are frozen replication checks. The readout uses the model's own
final RMS gain and vocabulary head. The comparator is raw active-state logit lens.

## Metrics and statistics

Primary: recovery on single-token-eligible, behavior-correct, unspoken cases, using
the identical letter/digit token filter for both methods. A hit means an entire
case-folded intermediate name appears in at least one of the four top-ten lists.
No fragments count. Eligibility and behavior/unspoken rules remain exactly those
in `prepare_items` used for development. These are determined independently of
which lens succeeds. Print the denominator and every excluded case with reason.

Primary inferential test: seed-0 paired exact two-sided McNemar test on the binary
per-item hit outcomes. Also show a paired bootstrap 95% percentile interval for
the hit-rate difference, resampling prompts 10,000 times with seed 20261121.
There is one primary hypothesis, so no multiple-comparison adjustment is needed.
Show seeds 1 and 2 individually as descriptive sensitivity checks, not new tests.

Secondary descriptive views: unfiltered readouts, all eligible cases regardless
of behavior, next-token KL, and per-visit recovery. These cannot overturn the
primary test. Report all wins and losses with the complete prompt, target bridge,
model continuation, raw top ten, and corrected top ten. Include an unchanged
case selected by the pre-existing test order. No LLM judge is used.

## Controls, plots, budget, and records

Must-work: all native final-head equalities and mean/scale algebra at each site.
Must-not-work: the raw comparator must not depend on the fit seed. Assert this.
The earlier development experiments already included mean-only and shuffled-pair
controls. Do not rerun or select those on the final test.

Plot paired item recovery and the hit-rate difference with the bootstrap interval.
Axes state the fixed read budget. Save all captured states, raw inputs and outputs,
readouts, learned offset vectors, summary counts, and exact code/data identities.
Budget: one MI210 GPU for at most 20 minutes, after the country-transfer sequence.
Upload the run privately. The 43 outputs may be opened only after this plan commits.
