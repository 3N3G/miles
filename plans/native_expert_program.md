# Native experts: autonomous research program

## Authorization and scope

On 2026-11-23 the user requested autonomous investigation of the native experts,
with up to five GPU hours. This covers the adaptive stages below and their
repairs. The model and checkpoint remain pinned in `evidence/source_pins.json`.
Count the previous 350-second pilot conservatively within 18,000 GPU seconds.
Run one allocation at a time. Count every allocated GPU, including loading
and waiting within a job. Reserve each queued job's entire GPU time limit before
submission. Record actual allocation seconds after completion. Stop submitting
when the reserved plus used time reaches the cap.

## Question and decision rules

Does a recurring expert perform a reusable operation, beyond recognizing a
lexical cue? Start with physical layer 12, expert 105, selected by earlier
development analysis. IDs are local to each physical layer. Experts 45 and 3
are secondary candidates. A routing preference is an association. Exact expert
removal establishes sensitivity. Task and dose controls are needed to interpret
that sensitivity. Shared routing across passes need not mean a shared role.

If native removal has an effect, test whether it is task-specific, where it acts,
and whether later passes repair it. If it is weak, inspect other native experts
using development-only attribution and exact ablation. Do not scale a surrogate
merely because its features have readable examples.

## Adaptive stages and maximum allocations

These are ceilings, not a requirement to spend the budget. Each stage gets a
frozen specific plan before it runs. Actual time freed by an early completion
can fund another stage within the same total cap.

1. Calibration and native intervention validation: 20 minutes.
2. Prospective natural-text routing and removal: 30 minutes.
3. Development expert discovery by router derivatives and exact removal: 40 minutes.
4. Task confirmation, three answer-order seeds: up to 120 minutes total.
5. Targeted replication, matched-dose and token-location controls: 50 minutes.
6. Repair reserve: 34 minutes. Prior pilot: 350 seconds.

## Data and analysis

Use real public data. MMLU elementary mathematics and high-school biology
provide questions with four letter-scored choices. This reduces the confound
between computing an answer and emitting a number. Validation supplies
calibration and discovery. Freeze independent test question IDs before running
confirmation. Three answer-order seeds are repeated views of the same question,
not independent samples. Known possible pretraining exposure limits capability
claims. WikiText-2 test articles provide prospective ordinary-language routing
and next-token loss. Existing articles are development data.

Report accuracy and gold-answer log probability for MMLU. Report next-token
negative log likelihood, numeric-target and numeric-input contrasts for text.
Use paired 10,000-sample bootstrap intervals over questions or articles, seed
1729. Average answer-order seeds within each question before resampling.
Correct preregistered primary families using Bonferroni intervals. Treat all
other searches as exploratory. Preserve full prompts, raw examples, routing
mass removed and per-pass effects. No language-model judge is used.

## Controls and validation

Identity and an absent expert must leave matched-shape logits exactly unchanged.
Check that edits target exactly three visits to the chosen physical module.
Remove weights without rerouting or renormalization first. Separately
renormalize the remaining selected experts to the original sum. Compare
similarly frequent experts selected using old fitting text only. A whole-sublayer
zero condition is a positive disruption check. Add matched output-norm noise
before claiming a semantic direction rather than a generic perturbation effect.
Inspect capability and rendered inputs before interpreting task scores.

## Reporting

Keep code, frozen manifests, results and scheduler receipts. Upload every run
with `push-run`. Keep self-contained human reports local for user review.
Plots show one claim, article/question intervals and at least one raw example.

## Results

Pending. Stage plans and evidence files hold the measured results. Scientific
interpretations in the final report are drafts for the user.
