# Matched-normalization country-direction baselines

The independent Germany/China j_unit transfer has produced capital swaps.
Complete its frozen site ablations unchanged. This separate diagnostic asks whether
exact derivative directions outperform equally normalized simpler directions.

Use all fifteen original Germany/China cases, all four original sites, all three
fit seeds, doses 0/.5/1/2, and matched random doses 1 and 2. Preserve prompts and
baseline/counterfactual validity gates. The four apparently valid capital cases
are not selected into a new dataset. All fifteen cases remain visible.

Fixed methods:
- raw_unit: normalize each final-norm-folded country unembedding row, then subtract.
- ridge_unit: pull each folded country row through the existing active ridge map,
  normalize each resulting direction separately, then subtract. Omit the affine
  offset to match j_unit. Positive scalar source/target rescaling cancels.

The previous raw and ridge swaps did not normalize each country row separately.
These controls remove that difference. The primary comparison remains the frozen
j_unit transfer, with these two methods reported as added development diagnostics.
Use case-level paired bootstrap intervals. Report all three fit seeds, all doses,
and raw successful and failed examples. No new held-out source-bank use or tuning.

Budget: one sequential MI210 job, at most 20 minutes, after the pass-average stage.
Must-work zero-dose equality and exact reflection follow the existing checks.
Archive all outputs privately with push-run.
