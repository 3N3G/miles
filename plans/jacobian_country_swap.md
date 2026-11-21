# Exact Jacobian country-direction validation

This continues the user-authorized search and `jacobian_country_rows.md`.
The original country-swap results remain unchanged.

Question: do exact averaged Jacobian rows support country exchanges when ordinary
unembedding and fitted state-prediction rows do not?

Use the same pinned model, twelve cases, visits, doses, shams, and random controls
as `country_swap.md`. Use only the existing development country tasks.
The 43 reserved source-bank prompts remain unopened.

Three fixed methods use the rows fitted on unlabeled WikiText:
- j_raw: France row minus Japan row.
- j_unit: normalize each country row separately, then subtract.
- j_centered: j_unit with a contrast offset from the unlabeled source mean.

The primary new method is j_unit. The other two diagnose scale and offset effects.
Use all three existing 96-paragraph fit seeds. The visit-35 row is reused at visits
11, 23, and 35. The visit-41 row is used at visit 41. This tests decoder reuse
through the shared loop, with the same fixed four-site intervention budget.

A successful swap must produce the task-specific alternate answer, not merely
change the output. Report baseline-correct cases separately. Report random controls
and unrelated tasks. Use paired case-level bootstrap 95% intervals if any swaps
succeed. With these related development cases, claims remain exploratory.

Must-work: zero-dose logits equal native logits exactly and the feature contrast
reflects with small rounding error. Must-not-work: matched-norm random changes
should not reproduce task-specific exchanges. Inspect all twelve raw cases.

Budget: one MI210 GPU, at most 30 minutes, after the Jacobian fit succeeds.
Save every forward and archive privately with push-run.
