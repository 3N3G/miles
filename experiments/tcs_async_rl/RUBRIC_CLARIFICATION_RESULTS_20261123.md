# Easy-question training after rubric clarification

Reward did not increase across repeated questions in this bounded continuation. Across six questions, mean reward fell
from **4.06 to 3.52 out of seven**. Two questions improved and four declined. These are descriptive training visits from one seed.

Qwen3.5-4B generated with thinking enabled. Luna graded final answers at medium reasoning effort, using the clarified prompt,
the original weighted binary rubrics, and the average of both anonymous answer orders. The configured dataset contains
20 easy TCS questions. FOCS and STOC 2026 remained untouched. Two hours on eight MI350 GPUs completed 24 optimizer updates:
768 answers across 18 visited questions. The final checkpoint preserves 20 updates. Four later updates were not saved.

Across matched visits, mean response length rose from **60,847 to 68,756 tokens**, or 13%. The fraction hitting the 90,000-token
cap rose from **25.0% to 40.1%**. The four comparisons entirely inside the saved trajectory also declined, by 0.45 points on average.
For the rooted-forest question, mean reward fell from 2.67 to 1.20. Its uniformly selected later answer assumes an unproved
submodularity lemma and hits the cap; both judges give zero. The earlier selected answer constructs the required edge deletions
and receives seven. [Raw later example and grades](https://huggingface.co/datasets/3N3G/internalization-runs/blob/848b571b4f6b29c4dc7a2f21930dbe3d6c6082fa/miles/easy-overfit-v15-diagnostic-20261122/example-step-86.json).

All 24 completed updates have finite metrics and matching weight publications. All four measured update-direction gains
are positive. Full-policy entropy ranges from 0.35 to 0.64 nats. Sampled initialization-divergence estimates range from
0.0054 to 0.0118. These checks support numerical operation, not improved mathematical performance.

The judge orders disagree on **101/768 answers (13.2%)**, including one seven-versus-zero case involving a valid endpoint
argument and an incorrect intermediate line. Assistant review finds full credit defensible. This is not human accuracy calibration.
[Exact paired receipts](https://huggingface.co/datasets/3N3G/internalization-runs/blob/848b571b4f6b29c4dc7a2f21930dbe3d6c6082fa/miles/easy-overfit-v15-diagnostic-20261122/maximum-gap-paired-actual-receipts.json).

The remaining approved continuation resumes the saved state under the same compute and judge caps. No cause of the reward
decline has been isolated. Longer capped generations and residual grading errors remain separate diagnostic targets.
[Full audit, matched comparisons, plots and uniformly selected examples](https://huggingface.co/datasets/3N3G/internalization-runs/tree/848b571b4f6b29c4dc7a2f21930dbe3d6c6082fa/miles/easy-overfit-v15-diagnostic-20261122).
Operational details belong in [the run record](RUN_V15_OPERATIONS_20261122.md).
