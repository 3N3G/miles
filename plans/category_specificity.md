# Development readout specificity check

Bridge recovery can reward a reader that lists many names of the right type.
Use the source's existing multiple-choice distractors as a diagnostic of this risk.
This is a development-only analysis, not another primary test on the reserved set.

For each eligible, solved/unspoken development item with at least one representable
single-token distractor, union the four top-ten reads. Score gold presence minus
the fraction of representable distractor names present. Report gold and distractor
rates separately. Keep the same word filter for every method and report both views.
Use seed zero as the fixed illustration and show other fit seeds as sensitivity.
Bootstrap prompts 10,000 times with seed 20261121 for paired differences versus raw.

A distractor may be a legitimate final answer or considered alternative. Therefore
this is a specificity diagnostic, not a definitive false-positive count. Show all
newly recovered cases with their raw lists and detected alternatives. No labels,
method parameters, or test-set candidates are changed.
