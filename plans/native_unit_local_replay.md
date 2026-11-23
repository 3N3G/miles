# Explain the unit-routing result with exact local replay

## Question

The preregistered native-unit test found a small routing sign reversal on16
unused articles. On260 paired positions, unit45:1127's output changes from
negative to positive. Its up input changes sign in all260 cases, while its
gate input does not. Why does deleting this unit change block reading?
Separate the reader's dot product from the root-mean-square normalization.

This is an exploratory explanation of the already measured16 articles.
It is not another independent confirmation, and it cannot change the original
test, uncertainty intervals or MMLU conditions. MMLU efficacy remains unread
until the three fixed orders finish.

## Fixed replay and controls

Use the same pinned model, backend,128-token inputs and exact single-prompt
API. Capture the physical layer12 input, post-attention convolution output,
expert inputs/routes/weights, expert output and post-MLP convolution output.
Capture the immediately following layer13 reader's inputs.

Replay only the native experts, convolution, residual addition, BF16 cast and
block reader. Check the unmodified expert output, convolution and Block2 state
exactly against their full-forward values. Check reader weights against the
original saved forward. Then replay single-pass deletions of unit45:1127,
unit3:664, their pair and control unit45:130, on passes1 and3. Compare every
replayed reader-weight vector with the existing full-intervention result.
Require maximum error<=1e-6 before interpreting the decomposition. Never
replace the grouped expert backend with an eager implementation.

The reader score is a query dot product divided by the block's RMS. Evaluate
the two algebraic counterfactuals with only the numerator or only the RMS
changed. Average both orders of attribution, so the two contributions sum to
the actual attention-weight change. These factors are an algebraic
decomposition, not two separately realizable neural interventions.

## Outputs and analysis

Save all replay errors, per-token numerator/RMS contributions and raw examples.
Report article means and descriptive article-bootstrap intervals. Do not add
new primary significance claims. Keep the small absolute routing effect and
its native baselines visible. Preserve the existing measured effects.

## Allocation

One MI210 GPU on AMD devel, at most5 minutes,300 GPU seconds reserved before
submission. Run between fixed confirmation orders without reading their
efficacy. Count actual allocation toward the18,000-second program cap.

## Results

Pending.
