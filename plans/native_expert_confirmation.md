# Independent confirmation of native expert influence

## Question

Which native experts contribute to answering math questions, beyond numeric
routing and generic output perturbation? Does a distributed family matter when
removing one number-routed expert has little effect? This is an adaptive next
experiment under the user's autonomous five-hour authorization. The48-article
text result has already been read and remains a primary null result. The
independent MMLU question cohorts have not been run.

## Model and frozen data

Use the pinned native Inkling-2-Mini BF16 conversion and unchanged weights.
Use64 mathematics and64 biology questions in the confirmation cohort of
`evidence/native_expert_inputs.json`, selected before any new inference.
Keep the separate64+64 replication questions unread. Three answer permutations,
seeds0,1,2, give repeated views of each question. Average within question before
statistics. Chat plus an assistant text prefix won earlier calibration. All
questions require only a letter answer. Save every complete rendered prompt.
No truncation. The earlier manifest excludes plain prompts longer than480
tokens, and the worker rejects rendered length above512 before inference.

Example calibration question: a gallon of paint costs$13 and two brushes cost
$9 each. The correct total is$31, represented by the letter attached to that
choice after permutation. Biology calibration asks about genetic differences
between fish populations that became separate species. This controls answer
format across subjects. Public MMLU may have appeared in pretraining. This
tests mechanisms of existing behavior, not unexposed task capability.

## Candidate identities frozen before confirmation

1. Preselected physical layer12, expert105, regardless of development effect.
2. Up to two derivative candidates, selected by the recorded rule and retained
   only if exact development mean math loss increase is positive and exceeds
   biology. Preserve failed candidates and their values. The first selected
   candidate can owe its contrast mostly to biology inhibition. Positive math
   loss is therefore a separate requirement below.
3. A numeric-routing family: one expert per repeated-core physical layer.
   Use old fitting paragraphs only. Pool three visits, require at least200
   selections, and maximize P(selected|numeric input)-P(selected|other input).
   Numeric means the fixed digit/number/ordinal/time-unit cue rule. For each
   layer choose a frequency-matched control with smoothed enrichment below1.5.
   Select closest total route count, tie by expert ID. This family is a routing
   hypothesis, not an asserted arithmetic circuit.

Save exact IDs, fit counts, development outcomes, source hashes and all
condition definitions to `evidence/native_expert_confirmation_spec.json` before
submission. Expert identities always include their layer. Family selection
must not use performance on confirmation questions or new article states.

## Conditions

Clean; each retained candidate/family removed at every visit; and a matched
random output direction for each. Each random perturbation matches the native
local deletion norm at that token/visit and current incoming state. Keep model
weights fixed. Also test105 on each visit alone, at numeric question tokens
only, at nonnumeric tokens only, and with remaining router weights normalized.
Test the frequency-matched family as a secondary control. All per-position
edits are restricted to valid prompt tokens, excluding padding.

Use random direction seeds tied to permutation seed and batch identity, fixed
before execution. Repeated seeds remain repeated views, not extra questions.
Record intended and actual BF16 norms. The all-visit random control matches
conditional local dose. Later incoming states can differ between conditions.

## Capability and inference validation

Before any test question, compare batched and individual forwards on eight
calibration questions, four from each subject. Use right padding, causal masks,
and gather each actual final token. Match the original one-row output-head
matrix shape. Require identical predicted labels, maximum correct-choice log
probability difference<=0.05 nats and deletion-effect difference<=0.05 nats.
Identity and the two native deletion implementations must match exactly.
If validation fails, abort before test questions and run separate single-prompt
seed jobs under the same protocol. Do not select a tolerance using test data.
Save clean accuracy, letter probability mass and majority/chance baselines.

## Metrics and preregistered tests

The primary score is correct-answer negative log probability among A/B/C/D,
summing all exact decoded-letter token aliases. Also report full-vocabulary
correct-letter probability and total letter mass to inspect format effects.

For question q, define D(q)=NLL(deletion)-NLL(clean) and
R(q)=NLL(matched random)-NLL(clean). C(q)=D(q)-R(q).
For each of m candidate/family identities (m<=4), report three primary intervals:

- Mean D on math: actual harm on the target task.
- Mean C on math: harm beyond equal-norm random directions.
- Mean C on math minus mean C on biology: relative task selectivity.

Average the three permutations within each question first. Bootstrap64 math
and64 biology questions independently,10,000 draws, seed1729. Use Bonferroni
intervals over3*m primary intervals, nominal family coverage95%. A candidate
qualifies for a math-directed influence claim only if all three lower bounds
exceed zero. This is a conditional two-task claim, not broad expert modularity
or proof of an arithmetic algorithm. Report all candidates, including failures.
Accuracy changes and all pass/location/control-family comparisons are secondary.
Do not select only baseline-correct questions for the primary outcome.

## Plots, examples and allocation

Plot clean/deleted accuracy by subject, and primary loss contrasts with corrected
question intervals. Plot105 per-pass effects descriptively. Include a complete
prompt and before/after scores nearest the median math effect, plus one largest
positive and one largest negative effect, labeled as extremes. Keep operational
records separate from the standalone human report.

One MI210 GPU on AMD devel, at most30 minutes for all three batched seeds.
Reserve1800 GPU seconds before submission and replace with observed allocation.
Keep total program use plus reservations under18,000 GPU seconds.

## Results

Batched calibration failed before any confirmation inference, with maximum choice-log-probability error0.4015 and effect error0.4796 nats. The cause was not isolated. The unchanged single-prompt repair completed128 questions, three answer orders and16 conditions, giving6144 scored condition views. None of four identities passed all three corrected primary requirements. The separate replication cohort remains unused. See evidence/native_expert_confirmation_result.json and the three expert-confirmation-s0/s1/s2 archives plus expert-confirmation-analysis-v1.
