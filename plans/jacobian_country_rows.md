# Exact Jacobian rows for country concepts

The mean/scale readout improves development recovery, but its reflected country scores have not redirected answers in the first checked cases.
Do not treat readable labels as proof that the corresponding unembedding axes carry the argument.
Test the actual Jacobian-lens directions before rejecting a causal country representation.

## Method

Use the same 128 unlabeled 64-token calibration paragraphs and the same three 96-paragraph fit seeds.
Use the model's own leading-space vocabulary rows for France, Japan, Germany, and China.
No task answers enter fitting.

At visits 35 and 41, add an independent zero-valued vector to every source position 16..63.
Differentiate the mean final-reader activation at target positions 16..63 against each source vector.
Contract the final activation with each country's unembedding row and the learned final-norm weight.
This computes the country rows of the position/context-averaged Jacobian without materializing the whole matrix.
Uniform averaging over causal source/target pairs differs only by a fixed positive scale, which cancels in direction normalization.

Reuse visit-35 directions at earlier visits of physical layer 11, and visit-41 directions at its own core-end read.
Keep this reuse explicit. It does not measure each early visit's complete downstream Jacobian.

Primary causal variant: normalize each country row before constructing the France/Japan exchange direction.
Also retain the original unnormalized rows and a centered unit-row diagnostic.
The centered variant subtracts the corresponding unlabeled source mean only.

## Controls and scope

Verify a zero-vector insertion leaves the first calibration paragraph's final state unchanged.
Save every per-paragraph gradient, its norm, the averaged vectors, and their coherence.
Use the frozen country-swap cases, band, doses, and matched-norm controls unchanged.
Treat these cases as mechanism development after earlier readout selection.
If a direction works, validate transfer on a separately frozen Germany/China case set.
No reserved WorkspaceBench item may be evaluated as part of this fit.

This is a specialized four-country lens, not a full-vocabulary Jacobian lens.
Compute is capped at one GPU for 30 minutes for row fitting, followed by the bounded causal check.
