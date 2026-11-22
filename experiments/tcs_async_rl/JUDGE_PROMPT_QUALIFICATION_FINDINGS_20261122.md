# Grading-prompt clarification before resumed training

The revised Luna-medium prompt passes all 60 basic controls on the existing 20-question training bundle.
It also yields nonconstant scores on the same saved 32-answer diagnostic: 17 zeros, five fives, four sixes, and six sevens.
No new solver generation was needed. This qualifies a provisional resumed run, not a claim of calibrated proof correctness.

The question was whether explicit grading rules resolve the observed disagreements about implicit facts, typos, and partial credit.
The original prompt, data, references, rubric weights, and all old scores remain intact.
The user authorized assistant mathematical adjudication instead of waiting for human labels.
The diagnostic labels are explicitly assistant judgments. Human agreement and independent-provider validation remain unmeasured.

The first revision matched all six known development cases in both orders.
Five separate saved answers exposed ten item disagreements, mainly about repeated deductions for a missing derivation.
The next revision explicitly defines follow-through credit while retaining requirements for valid final counterexamples and quantitative estimates.

On that revision, 11 development answers match **99/102 item decisions** across two presentation orders.
Five fresh answers from four questions match **43/48 item decisions**.
Those are repeated decisions on five answers, not 48 independent examples.
Remaining differences concern explicitness requirements, elementary bit-complexity arguments, and a rooted-forest invariant proof.
They prevent treating agreement with the assistant as a reliable estimate of true judge accuracy.

For one concrete case, the same incomplete unary counterexample originally received four and zero points.
It now receives two in both calls: credit for its valid unary calculation, without credit for missing parameter or density checks.
A false theorem assertion remains zero in both calls. The rate-error examples retain six points and lose the rate item.
The private [qualification evidence](https://huggingface.co/datasets/3N3G/internalization-runs/tree/b69851eb463907e85d0bb760e97d4acf48f63742/miles/easy-overfit-v14)
contains the raw proofs, item labels, both judgments, controls, and exact prompt fingerprints.

Both prompt-qualification phases cost an estimated **$0.3973**, carried into the continuation's combined $5 ledger.
All 99 TCS CPU checks pass, including the archive-rate integration check.
The [frozen continuation config](rl_easy_overfit_v14.json) resumes after 65 saved updates with unchanged solver and optimizer settings.
Compare early and later visits scored by the new prompt. A changed reward scale is not model improvement.
