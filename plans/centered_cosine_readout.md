# Does a state predictor hide concepts in its output prior?

The fitted final-state predictor has low hidden-state error but weak bridge
readout. This does not prove that it discarded the concepts. A constant output
mean or unequal word-vector norms could hide a useful direction in token ranking.
Test those two mechanisms on the frozen DEVELOPMENT data only.

Use the pinned source trajectories, all 55 development prompts, all three existing
fits, four read sites, K=10, and both identical vocabulary-filter conditions.
The 43-item primary test remains fixed and these methods cannot replace it.
No new fit labels, calibration source, prompts, or hyperparameters are introduced.

For source centered activation x=h-mu_source, use four fixed readers:
1. centered_raw: score each folded native vocabulary row against x.
2. centered_raw_cosine: normalize each folded native vocabulary row first.
3. centered_ridge: pull each folded vocabulary row back through the frozen ridge
   matrix, then score x. This omits the fitted constant final mean.
4. centered_ridge_cosine: normalize each pulled-back row before scoring x.

All four readers use RMS-normalized x. This common scalar does not affect ranks.
The head's learned RMS gain is folded into vocabulary rows exactly once. The
similarity heads use BF16 matrix products, matching the existing numerical budget.
These scores are for ranking, not calibrated probabilities. Ignore their output
KL when interpreting results. Record this distinction in the run manifest.

The comparison isolates a simple readout confound. A gain would not establish
that this is a new method or that it improves the unopened primary test.
Report complete concepts and all eligible/solved-unspoken denominators. Inspect
wins AND losses. The mean-only and shuffled-map failures remain existing controls.
Must-work: normalized row norms are one before casting. Source-head shape and
folded final gain must match. Every source mean is fit on unlabeled WikiText only.

Budget: one sequential MI210 job, at most 15 minutes, after single-phase transfer.
Save raw readouts, row-norm diagnostics, and exact fitting-artifact identity.
Archive privately. Add a separate report section only if the result is useful.
