# Transport across complete loop passes

This is a bounded continuation of the user's request to extrapolate rotations.
It separates complete-pass dynamics from the earlier, nonstationary layer steps.
No reserved test outputs are opened.

Question: does comparing the same physical layer across loop passes expose useful
facts, either by continuing a shared transformation or by isolating changes?
If yes, keep the method for final selection. If no, the present calibration does
not support these simple dynamical approximations.

Use the pinned model, saved calibration trajectories, 55 development prompts,
three existing 96-paragraph seeds, and the fixed four-site, top-ten read budget.
At visits 11 and 23, fit an orthogonal map from physical-layer-11 pass one to pass
two. Continue it for two and one passes, respectively, then restore the final
reader mean and scale. At visits 35 and 41 this method equals mean/scale correction.
Validate rotation prediction on separate text, including actual pass-three states.
A second control uses a ridge map from pass one to pass two, with lambda selected
from the original grid by pass-three geometric error on the sixteen validation
paragraphs. Compose its centered, standardized map for the remaining passes, matching the
rotation normalization. Select lambda by standardized pass-three prediction error.

Additional fixed methods use only already-seen states from the same physical layer:
- cycle_change: read the latest standardized pass-to-pass difference.
- cycle_removed: read its negative, to inspect information removed by refinement.
- projected_fixed_point: extrapolate three pass states using a scalar geometric
  series. Let d1=x2-x1 and d2=x3-x2, r=<d1,d2>/||d1||². Clip r to [-0.9,0.9],
  then predict x3+r/(1-r)*d2. This is a projected-state heuristic, not a complete
  recurrent-state fixed point. Use mean/scale correction where fewer than three
  states are available. Record unclipped ratios, step norms, and alignment.

Standardize each source visit with its own unlabeled mean and one global scale.
Restore the final-reader mean and scale before the vocabulary head. For the change
methods, use a zero update where only the first pass is available. No label-based
selection of signs, clipping, visits, or rank cutoffs.

Metrics: complete single-token bridge recovery, behavior-correct/unspoken recovery,
identical filtered/unfiltered vocabulary, next-token KL, and future-state error.
Show three fit seeds separately. Bootstrap by prompt/document for any final contrast.
An exact synthetic scalar-contraction control must recover its fixed point.
A fixed point alone does not prove that this model computes by convergence.

Plot recovery and pass-three prediction error. Save raw readouts and all settings.
Budget: one MI210 GPU, at most 20 minutes, after the previous causal stage.
Archive with push-run. This is development exploration, not a final evaluation.
