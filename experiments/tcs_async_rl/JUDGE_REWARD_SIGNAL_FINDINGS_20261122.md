# Judge disagreement after centering training rewards

This check reads the completed Qwen3.5-4B easy-only continuation: 40 groups, 32 answers per group, 20 questions, one seed.
It asks whether the original Luna-medium judge's two presentation orders change which answers rank above their group mean.
Training averaged both orders. This analysis makes no new inference calls and changes no training settings.

Of 1,280 answers, **338 (26.4%)** received different scores between calls.
For **233 (18.2%)**, subtracting each order's group mean gives opposite signs.
Another 32 answers equal their group mean in exactly one order.
The mean cosine between the centered score vectors is **0.690** across 38 groups where both vectors vary.
The minimum is 0.033. Two other groups have at least one constant vector.

For a concrete auction proof, the calls award 7 and 5 points.
The corresponding group means are 6.875 and 6.750, giving centered scores +0.125 and −1.750.
Their actual average gives a centered reward of −0.8125.
Assistant review found its payment argument valid under the supplied monotonicity assumption.
The repaired prompt later graded this saved answer 7 in both calls. Its other group members were not regraded here.

These hypothetical single-order contrasts precede token whitening. They do not reconstruct actual parameter gradients or measure correctness.
Order effects and stochastic variation remain confounded. One seed supplies no seed-level error bars.
The result does not establish that judge disagreement caused slow learning.
Apply the same diagnostic to the resumed run when new training groups become available.

The [verified private evidence](https://huggingface.co/datasets/3N3G/internalization-runs/tree/7ab813e7b201e8aef7fa3daae36684ed863789ff/miles/easy-overfit-v12-judge-signal-review-20261122)
contains the raw auction answer, all group and answer calculations, source fingerprints, and five analytic controls.
The [analysis code](judge_signal.py) reads the existing checksum-verified sample summaries.
