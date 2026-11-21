# Separate decoder mismatch from an early causal failure

The single-site reuse check has so far produced no first-pass swaps. A direction
fitted on the third pass may be wrong for an earlier pass with more computation
remaining. Fit the actual early Jacobian rows before drawing a conclusion about
whether early country content can control the answer.

Use exactly the existing 128 unlabeled WikiText fit paragraphs, first 64 tokens,
positions 16..63, and existing three 96-paragraph fit subsets. Fit ordinary
autograd rows for Germany and China at visits 11 and 23. The source and final
reader definitions remain those in `jacobian_country_rows.md`. Do not change
normalization backward rules or gate derivatives. No task answers enter fitting.

Then repeat the existing single-site protocol on the SAME fifteen Germany/China
cases and the SAME native validity gate. Only the four previously valid capital
cases are expected to enter the intervention stage, but retain all native controls.
Compare each early visit's own unit-normalized rows at its own site. Keep doses
1 and 2 and exact matched-norm random controls for each dose, three fit seeds.
The reused third-pass row results remain the comparator. Do not tune a dose,
prompt, row normalization, or fitting set after seeing this check.

Primary descriptive question: does its own Jacobian support a causal swap at a
site where the reused third-pass Jacobian does not? Show cases and fit seeds.
If both fail, that rules out these two ordinary-Jacobian implementations on these
cases, not the presence of a country representation. A new backward rule would
need its own later test. No further method is selected on the 43-item final test.

Must-work: zero insertion equals the native final reader in fitting; intervention
shams equal native outputs. Log mean-gradient coherence across paragraphs to
identify cancellation. Preserve and diagnose any numerical/inference failure.

Budget: one MI210 fit job at most 20 minutes and one sequential causal check at
most 15 minutes, after period confirmation. Archive both stages. The conditional
native validity gate avoids spending forwards on unanswerable formatting variants.
