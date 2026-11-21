# Confirm the three-pass orbit on separate unlabeled text

Exploratory 32-pass runs on twelve development prompts showed persistent movement.
Lag-three distances were much smaller than lag-one and lag-two distances. This
suggests an approximate three-cycle rather than a fixed point. It is not yet a
proof of a stable limit cycle, and extra passes are outside the trained setting.

Confirm on the FIRST EIGHT of the sixteen existing WikiText calibration-validation
paragraphs, indices 128..135. None was used to fit any readout. Use the first 64
tokens, as in the frozen calibration capture. Do not choose paragraphs by result.
Use the same model and implementation with no weight, message, or prompt changes.

Run native three-pass and extended 64-pass forwards without KV caching. Save both
recurrent banks at all 64 token positions after every pass. The first three states
must match the unmodified three-pass forward bit-for-bit. Verify native final-head
reconstruction. The shared transition has no loop-counter input after initialization.

Primary diagnostic, fixed before this confirmatory run: per paragraph, the mean
normalized state distance from pass n to n-3 divided by the mean distance to n-1,
for n=49..64. Normalize each bank separately by RMS. Report all eight ratios and
a bootstrap 95% interval for their mean, resampling paragraphs 10,000 times with
seed 20261121. Also report fixed lags 1..8, raw distances, and the final-token view.
A small ratio supports approximate three-pass recurrence on this text, not exact
periodicity or a spectral stability proof.

Secondary analysis: separate a shared phase pattern from prompt-dependent content.
At the final token, average states by pass modulo three over passes 49..64.
Center each paragraph's states across those phases. Fit the rank-two shared phase
subspace from seven paragraphs and measure explained phase variance on the eighth,
rotating the held-out paragraph. Use RMS-normalized bank states, not raw norms.
This tests a shared clock-like component. It does not by itself show a semantic
clock or establish which coordinates are necessary. Compare lag distances before
and after subtracting the shared phase templates fitted on the other paragraphs.

Show all eight trajectory/lag curves and one complete rendered input with native
and extended output. No benchmark accuracy or lens gain is inferred from this.
Budget: one MI210 GPU, at most 15 minutes, after the queued cosine-reader diagnostic.
Archive raw state arrays, input IDs, model/code pins, and all native controls.
