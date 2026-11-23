# Test a normalization-aware native unit atlas on new articles

## Question and fixed selection

Does a local first-order reader lens predict native unit deletion effects
better when it includes root-mean-square normalization? Do the four new
phase-sensitive units selected on development data show causal sign reversals
on new articles?

The candidate selection rule is frozen in `plans/native_unit_atlas.md`.
Use all six selected identities and both known units45:1127 and3:664.
Do not choose units using the new articles. Pin the resulting candidate file's
SHA256 before submitting this stage. If the development gradient implementation
fails its native guards, repair it or report the failure before proceeding.

## Models and data

Same pinned Inkling 2 Mini BF16 conversion and native grouped expert backend.
Same physical layer12 writer and physical layer13 attention-input block reader.
Use32 previously unused WikiText-2 training-split articles, one128-token paragraph each, selection
seed31415. Exclude every previously used article and matching64-token prefix.
Data and one complete formatted example are recorded before inference in
`evidence/unit_atlas_test_inputs.json`. This selection is independent of unit
identities and their new-article effects. These articles are touched once.

A preparation-only availability check found only12 unused eligible validation
articles. Before any candidate inference or new-article outcome, switch to the
corpus training split. These32 articles are held out from this investigation
and all earlier surrogate fitting. No model is trained in this stage. The
original preparation draft is preserved in `plans/archive/`.

## Predictions, actual interventions and controls

For each clean article and each pass1/3, predict the mean Block2 reader-weight
change from deleting each unit as negative activation times local gradient.
The full lens differentiates the native convolution, residual sum, BF16 cast,
RMS normalization and reader softmax. The dot-product-only lens holds the block
RMS fixed at its observed value. Include a zero-change prediction baseline.
No lens is fitted or tuned on the new articles.

Then compute exact native local unit deletions. Replay is a causal shortcut:
the measured reader follows the edited layer, before any later feedback.
On the first article, compare all16 unit/pass local interventions against full
model forwards, with maximum reader-weight error<=1e-6. Require baseline expert
and block states to match exactly for every gradient computation. Abort at any
failed fidelity guard.

For each of the four selected phase units and known unit45:1127, also run three
equal-output-norm random directions per pass. Save BF16 norm errors. For the
first article, check the first random direction against a full native forward.
Identity and absent-expert edits must leave all logits exactly unchanged.
Save native unit activations and both gate/up factors on all three passes.

## Statistics fixed before inference

The sampling unit is the article, N=32. Average over the eight fixed units and
two visits within each article. The two primary forecasting contrasts are
MAE(full)-MAE(dot only) and MAE(full)-MAE(zero). Bootstrap articles10,000 times,
seed1729, with Bonferroni intervals for these two contrasts. Claim a forecasting
advantage only if both upper bounds are below zero. Report all MAEs, including
exact-zero effects, and descriptive sign agreement restricted to effects with
absolute magnitude>1e-7. Unit/pass pairs are not independent samples.

The separate phase-replication family contains the four newly selected phase
units, each at two visits: eight intervals. For each article and visit, subtract
the mean of its three equal-norm random changes from the deletion effect.
Bootstrap articles with the same fixed seed and Bonferroni adjustment for
eight intervals. A unit passes only if its two intervals exclude zero in
opposite directions. Report all four units. The known units and interactions
are descriptive. Do not add or remove a target after this run.

## Figures and raw examples

Plot the eight frozen unit identities against exact pass1/pass3 effects, with
article intervals clearly labeled. Plot prediction MAEs for full, dot-only
and zero lenses. Use one article nearest the median forecasting advantage as
a representative example, plus labeled extremes. Include a short raw context,
unit identity, predicted changes and actual change. Give full inputs in the
private archive. Keep article counts separate from unit and token counts.

## Compute and interpretations

One AMD MI210 GPU, at most10 minutes,600 GPU seconds, after development finishes.
Count allocation time and stay within the overall18,000-second cap.
If both forecasting intervals favor the full lens, this supports its usefulness
on the fixed selected units and this reader. Otherwise report the limit of
local linear forecasts at native deletion size. Neither result establishes
an arithmetic expert, a global iteration counter or improved final answers.

## Frozen identities

Development selected phase units83:1107,52:585,58:285 and44:492, plus
stable comparators38:1033 and111:310. Keep known comparators45:1127 and3:664.
All are in physical layer12. The exact selected-unit file is
`evidence/native_unit_atlas_spec.json`, SHA256
`912eef5da0ec779a58861c60f86c1276fcf6f49bcdaaa8cf356855f1c9b94c28`.
This was recorded before starting any new-article inference.

## Results

Pending.
