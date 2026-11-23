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

Also decompose the unit's gate and up projections into the current depth path
and direct previous-loop message path. Recompute the native message keep/gain
coefficients and require exact reconstruction of the mixed input. Apply the
same observed RMS normalization to both terms. Keep normalization/cast and
projection-rounding residuals as separate terms. The current depth path can
already contain earlier-loop information, so this is a direct-path accounting,
not a separation of all information by its original loop. Record all three
passes. No new intervention or significance test is added.

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

Completed on all16 existing articles. All128 edited reader-weight vectors
matched the saved full-model interventions exactly. Unmodified expert,
convolution and residual block reconstruction were exact. Both algebraic
attribution terms summed to the total with maximum error0.

For unit45:1127, the raw mean pass1 weight change+0.00007939 decomposed into
numerator-0.00028288 and RMS+0.00036227. Pass3's-0.00017851 decomposed into
numerator+0.00014100 and RMS-0.00031951. These are descriptive attributions
on the existing sample, without random subtraction.

At pass3 the selected-token mean up projection was-1.21052, with current-depth
contribution-0.70043 and direct-message contribution-0.50999. Among260 paired
up-sign changes,242 current-depth contributions were already negative.
This decomposition does not separate all information by its original loop.
Full results: `evidence/native_unit_replay_result.json`. Private archive stage:
`native-unit-replay-v1` in `evidence/artifact_archives.json`.
