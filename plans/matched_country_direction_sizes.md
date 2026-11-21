# Compare country directions at exactly matched edit size

Jacobian swaps succeeded where raw and state-prediction directions did not.
The reflection rule gives different edit sizes when activations have different
loadings along each direction. Remove this possible explanation before interpreting
the direction comparison. This is a final diagnostic of the same four capital cases.

Use the existing fifteen-case Germany/China file and native original/counterfactual
validity gate. Use only visit 35, where the third-pass Jacobian alone swapped 4/4.
Keep the same model, calibration artifacts, fit seeds 0,1,2, and dose-two reference.
No task-dependent fitting or new prompts are added.

For each token's untouched activation h and the unit Germany-minus-China Jacobian
contrast u_J, compute c=-4<h,u_J>. The original reflection edit is c*u_J.
Compare c*u_raw and c*u_ridge, with their directions independently unit normalized.
This matches the SIGNED coefficient and perturbation norm at every token, using
exactly one source site. Also compare three random unit directions with the same c.
Only the direction changes. Do not retune c per method or case.

Record pre-cast and actual BF16 perturbation norms. Assert matched pre-cast norms
within 1e-5 relative tolerance. Native and Jacobian outputs must reproduce existing
forwards exactly. Report all four native-valid cases, all fits, random controls,
raw target probabilities, and any rounding discrepancy. These are related cases,
not an estimate of broad semantic intervention accuracy.

Budget: one sequential MI210 GPU, at most ten minutes, after recovery tracing.
Archive the full result. This diagnostic fixes a comparison confound without
changing the original experiments or promoting another reader on the final test.
