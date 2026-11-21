# Does one semantic direction transfer across loop passes?

The frozen country-pair transfer changes all four valid capital cases when all
four sites are changed. The late-bank-only condition has changed none. The
three-loop-site condition is still running. These observations do not identify
which loop pass carries the effect.

Question: can a country direction fitted after physical layer 11 on the THIRD
pass cause the correct country exchange at that physical layer on the FIRST or
SECOND pass? A positive result supports reuse of this phase-specific decoder.
A negative result prevents that claim even if multi-site interventions work.

Use the exact pinned model, Germany/China case file, and previously fitted rows.
At each single site 11,23,35 separately, use the unit-normalized visit-35 rows.
Run fit seeds 0,1,2, doses 1 and 2, and three random directions matched at EACH dose.
Compare all native original/counterfactual inputs first. Run the interventions only
on non-unrelated cases where both controls have their expected answer as top token.
This validity gate is independent of intervention success. Keep the original
fifteen case inputs and outputs visible. Do not rewrite or supplement their prompts.

Primary descriptive comparison: dose-2 swap counts by single site. All three sites
and both doses are fixed now. Report all fits and raw changes in output probability.
These related cases cannot support a broad statistical claim. Case bootstrap
intervals may be degenerate. Never treat seeds or token positions as extra cases.

Must-work: zero-dose shams equal native logits at each site. Must-not-work: same-
norm random directions should not reproduce task-specific exchanges. This checks
causal steering, not necessity of the country direction in an untouched model.

Budget: one MI210 GPU, at most 15 minutes, after the normalized baseline controls.
Save all native and intervened outputs, single sites, row provenance, and seeds.
Archive privately with push-run. No reserved bridge-test readouts are added.
