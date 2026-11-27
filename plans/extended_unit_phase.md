# Native unit roles beyond the trained three passes

## Question and authorization

Do the previously found unit effects recur over further passes? Does the
normalization-aware local reader predictor work beyond passes one and three?
The user requested this follow-up and authorized autonomous expert research,
with a total budget of five GPU hours. This test uses that existing budget.
It fits no model. An extended checkpoint trajectory is an intervention, not
the model's trained deployment setting.

If state and causal effects both recur at lag three, a recurring phase is a
better account than first-versus-final behavior. If only state recurs, the
unit interpretation does not extend to this phase description. If the local
predictor fails outside the selected visits, do not recommend it generally.

## Model, data and setup

Use the pinned production Inkling 2 Mini revision, the existing BF16 conversion,
native grouped expert backend and short convolution. Disable cache. Leave
all weights fixed. Change only the loop stopping count for the extension.
Use all eight fixed units in evidence/native_unit_atlas_spec.json at physical
layer12. Read the same native physical layer13 attention-input block reader.

Use16 previously unused WikiText-2 training articles, one128-token paragraph
per article, selection seed271828. Exclude all prior atlas, operator and text
articles and64-token prefixes. Exact inputs, source hashes, article identities
and one complete example are frozen in evidence/extended_phase_inputs.json.
These articles are new investigation data, not held out from model pretraining.
No prompts, data, units or stopping counts will be tuned after this test.

For each article, run24 passes and the native three. Save both recurrent
block vectors at every token and pass, exact native unit factors and reader
weights. Save BF16 vectors losslessly as uint16 bit patterns. Compute exact
single-unit deletions through the existing local replay. Predict each deletion
with the full local derivative and with the frozen-RMS derivative. Average
reader effects over positions16 through127, as in the previous study.

## Controls and early failure checks

Require all articles' first three states, expert inputs and reader weights to
match native execution exactly. On the first article, check stopping counts
one, two and six against the corresponding24-pass prefixes. Check identity
and absent-expert edits leave logits and reader weights unchanged. Validate
local versus full forwards for units45:1127 and52:585 at passes1,2,3,4,6,12,24.
Require maximum reader error<=1e-6. Reconstruct the native final output head
exactly. Local gradients must leave native logits unchanged and weight
gradients absent. Abort on guard failure before interpreting results.

No new claim of unit-specific semantic content is tested. Previous dose-matched
random controls established the initial selected effects. This test asks about
the same fixed units' trajectories, so does not add new random directions.

## Metrics and fixed statistics

The independent sampling unit is the article, N=16. Use10,000 article bootstrap
resamples, seed1729. Bonferroni-correct the following four primary intervals.

1. Mean late full-state normalized lag-three distance minus lag-one distance.
   RMS-normalize each token/block separately with epsilon1e-6. Average over
   both recurrent blocks, all prompt positions and coordinates, then square
   root. Average endpoint passes19 through24 within each article.
2. Mean late absolute lag-three deletion-effect difference minus lag-one
   difference. Average the eight fixed units and endpoint passes19 through24.
3. Mean absolute error(full predictor) minus error(frozen-RMS predictor),
   using passes2 and4 through24. Average all eight units within each article.
4. The same mean absolute error(full predictor) minus error(zero predictor).

Claim recurring causal phase only if intervals1 and2 have upper bounds below
zero. Report effect amplitude to distinguish recurrence from all-zero decay.
Claim prediction advantage only if both intervals3 and4 have negative upper
bounds. Report pass-one/three results separately as descriptive replication.
Lag-two/six distances and all unit trajectories are descriptive. Do not select
a best period or treat units, tokens or passes as independent samples.

## Exact plots and examples

Plot pass number against each unit's exact mean deletion effect in percentage
points, with article-bootstrap95% bands. Plot full-state lag-one/three/six
normalized distances versus pass. Plot full/frozen-RMS/zero prediction mean
absolute errors in percentage points, starting the bar axis at zero. Show the
first article's complete prompt and unit45:1127 pass-by-pass values next to
the aggregates. Report failed controls and null results.

## Compute and archive

One MI210 devel job, one GPU, at most15 minutes (900 GPU seconds). Reserve the
whole limit before submission from11477 remaining seconds. Preserve this
plan byte-for-byte before adding results. Archive code identity, frozen inputs,
results, controls, states and logs in the private run store after completion.

## Results

Completed on16 new articles. All36 recorded controls passed, including14
local/full-forward deletion comparisons with maximum error0. Native first-three
states and reader weights matched exactly for all16 articles.

Late full-state distance was0.18646 at lag3 and0.69467 at lag1. Mean absolute
causal effect difference was0.001863 percentage points at lag3 versus0.008701
at lag1. Both preregistered contrasts excluded zero after four-test correction.
The full predictor's MAE on22 new visit numbers was0.000425 percentage points,
versus0.005013 for frozen RMS and0.007920 for predicting zero. Both prediction
contrasts also passed the four-test correction. Detailed intervals and the
first frozen article example are in evidence/extended_unit_phase_result.json.

The original unit45:1127 had positive deletion effects at passes1,4,7,...,22,
and negative effects at passes3,6,9,...,24. Some other units changed between
initial and later cycles, so this is not a universal per-unit phase counter.
The extension is outside the trained three-pass horizon. It does not establish
a stable limit cycle or better semantic token recovery.

Private immutable run archive:
https://huggingface.co/datasets/3N3G/internalization-runs/tree/d92b6e8633d32f1210e02869cf958b7a47e3eb25/inkling-interp/extended-unit-phase-v2

Exploratory component controls and static-template checks were added afterward.
They preserve the primary analysis. Their augmented private archive is:
https://huggingface.co/datasets/3N3G/internalization-runs/tree/d9eca5bbab9c88adf2fc75c5df459403a22dd28d/inkling-interp/extended-unit-phase-v2
