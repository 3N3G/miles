# Does the actual recurrence approach a fixed point?

This is a bounded continuation of the user's fixed-point and rotation question.
The scalar projected-state acceleration gave a small filtered development gain.
Three passes cannot establish a convergence trend. Test the actual full recurrence.

Use the same pinned Inkling model and twelve development prompts chosen by a
seed-1729 shuffle of the already frozen 55 items. The exact list is in
`evidence/long_loop_items.json`. No reserved test outputs enter this experiment.
No prompts, weights, gates, temperatures, or caches are fitted or changed.
Only the configured number of shared-layer passes changes during inference.
This is extrapolation outside the trained three-pass setting, not a new model.

Run pass counts 1,2,3,4,8,16,32 without KV caching. At 32 passes, save both complete
recurrent banks after every pass, at every prompt position. The prelude is fixed.
A recurrent state comprises the physical-layer-11 and physical-layer-17 endpoint
banks, not the last token alone. The first pass has no message input. Passes two
and later apply the same message-conditioned transition.

Must-work controls: the first three bank states from the long forward exactly
match the native three-pass forward. The reconstructed native head matches exact
logits at every stopping count. Restore the original pass count after each item.

Primary diagnostic: normalized full-state distance between successive passes.
RMS-normalize each saved bank separately, matching message-reader invariance.
Also measure raw relative distance and distance to two passes earlier.
A small two-step distance with a persistent one-step distance suggests a cycle.
A falling one-step distance is evidence on these inputs, not a contraction proof.
Show all twelve curves, not only a median. Bootstrap 95% intervals by prompt for
aggregate distances and native-output agreement. The forward computation is
deterministic, so random model seeds would not provide independent runs.

Secondary: exact native next-token top ten at all seven stopping counts, agreement
with the three-pass model, and whether the source's expected answer remains top-1.
Keep punctuation/format failures visible. Read one complete input and output from
every stopping condition. Do not choose a stopping count to maximize accuracy.

Compare the earlier projected-fixed-point method with two simple cached-state
baselines: a 50/50 and a 75/25 mixture of the latest and preceding standardized
same-phase states. Use the existing four-site/K=10 development protocol, with
mean/scale fallback where fewer than two passes exist. This diagnoses whether the
adaptive heuristic adds anything beyond averaging. Keep it out of the final test.

Plots: full-state one- and two-step distance against loop pass, and native answer
changes against stopping count. One plot, one claim. Save raw states, token IDs,
all native readouts, code/config pins, and the prior config value.
Budget: one sequential MI210 GPU job, at most 30 minutes, after the frozen final
readout evaluation. A separate cached-mixture calculation may share that job.
Archive results privately with push-run. Human-facing conclusions remain drafts.
