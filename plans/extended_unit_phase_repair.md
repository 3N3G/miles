# Output-head validation repair

The first extended-unit run stopped before writing any article results.
The manually reconstructed final head passed an FP32 tensor into its BF16
linear weights. The native model explicitly casts between these operations.
Add that same cast to the validation expression. This is the expression
already used by the earlier long-loop control. Preserve the failed run.

Run the unchanged frozen protocol and inputs from plans/extended_unit_phase.md
in a new output directory, extended-unit-phase-v2. No research condition,
metric, data, native model operation or selected unit changes. The first
guard verifies this repair against the native final logits exactly.

Reserve another900 GPU seconds, one MI210, maximum15 minutes. Count the
failed allocation in the existing five-hour program. This repair and rerun
are covered by the user's standing inference-repair authorization.
