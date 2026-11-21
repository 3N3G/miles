# Track what happens after an early country swap

The reused third-pass direction reads all four countries correctly in the first
pass, but first-pass-only swaps leave the final answer unchanged. Single-site
swaps succeed in 0/4, 2/4, and 4/4 cases in passes one, two, and three. These counts
do not show that early country information is unused. Later passes may recompute
it from the unchanged prompt. Inspect the intervened trajectory directly.

Use only the existing Germany/China inputs, native validity gate, fit seed zero,
and dose two. Preserve all original conditions. Observe the third-pass country's
unit contrast at physical layer 11 in every pass. Normalize the observed activation
for a cosine score. This fixed probe already distinguishes the two countries in
the untouched states. Also measure the whole active bank's normalized distance to
the clean bank at each pass, at every token position.

Fixed conditions: clean; third-pass direction at visits 11,23,35 separately;
third-pass direction at all three; each early site's own fitted direction at visit
11 or 23; and a norm-matched random direction at visit 11 with random seed zero.
These are diagnostic traces, not new method/dose/prompt selection.

Save the probe value before and after each edit, subsequent values, raw activation
snapshots, and final top ten. A return to the clean probe sign is evidence of probe
recovery, not proof of a unique semantic correction mechanism. The unchanged
prelude provides a plausible alternative source; it is not removed in this test.

Must-work: clean logits equal the original forward exactly. The three single-site
third-row interventions must reproduce the original function's logits exactly.
The snapshot observer must not change outputs. Include counterfactual input controls.

Budget: one sequential MI210 job, at most ten minutes, after early-horizon validation.
This reuses the approved experiment's model, data, prompts, metrics, and doses.
Archive the diagnostic trajectories and distinguish observation from intervention.
