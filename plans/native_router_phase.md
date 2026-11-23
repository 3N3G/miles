# Offline native-router phase diagnostic

## Question and scope

Do native experts respond mainly to content, pass-dependent offsets, or both?
Reconstruct only the exact physical layer12 router from checkpoint tensors.
Use stored native inputs and routes from the completed pilot. This needs no
GPU and makes no new full-model answer claim. The user authorized autonomous
expert research. All analysis is development-only, with no final article states.

## Fixed procedure

- Fit means on128 previous fit paragraphs, positions16..63, all three passes.
- Inspect16 previous development paragraphs. They include three articles also
  present in fitting and therefore are not an independent confirmation set.
- Extract BF16 router weight and FP32 correction bias from the pinned original
  checkpoint. Follow the official sigmoid, correction-for-top-k and normalized
  selected-score formula. Verify set equality of replayed top-eight routes and
  agreement of per-expert selected weight against stored native values.
- Report per-pass route reuse for the same token and per-expert content cues.
- Baselines: unchanged source-pass input and target-pass constant mean input.
- Translate each source-pass input by the fit mean difference to a target pass.
  Compare route intersection/8 with that token's true target-pass routes.
- Compute the two-dimensional span of fit pass mean differences. Project only
  this span out of the router input, preserving the grand mean. Measure route
  changes and expert105 numeric-cue enrichment before and after this edit.
- Compare edits with three random rank-two subspaces, seeds0,1,2. Match each
  token's perturbation norm to the phase projection before comparing changes.
- Show a full token context and route sets for all three passes. Choose the
  development token nearest the median improvement in translation overlap.
- All comparisons are descriptive. No post hoc significance test or population
  claim. This probes the routing function, not the whole native recurrent state.
- Plots: observed versus mean-translated route overlap by pass transition, and
  numeric-cue routing before/after removing pass-mean variation. Name all edits.

## Results

Pending.
