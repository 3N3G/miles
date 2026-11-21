# Saved-bank readout stage

This implements PLAN.md's exploratory question about loop differences and a prompt-independent component.
It does not modify the model or rerun its forward pass.

Freeze the mean normalized bank for each loop and block using the four engineering controls.
Apply this mean to the 12 already selected development items.
Compare raw banks, centered banks, loop differences, and centered loop differences.
Compare only loops 2 and 3 and both banks: four read opportunities per method.
For every method show both unfiltered and letter/digit-filtered top-10 lists.
Count only complete single-token bridge names with case and surrounding whitespace ignored.
Mark multi-token bridges ineligible, rather than treating a fragment as a recovered concept.
No benchmark judge or language model makes scoring decisions.

This is exploratory development evidence. Do not select a method from it and call that selection a held-out result.
Record all examples, including the cases where native next-token predictions were incorrect or incomplete.
