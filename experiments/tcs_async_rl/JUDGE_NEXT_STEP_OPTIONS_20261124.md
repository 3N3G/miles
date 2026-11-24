# Decision after the unsuccessful grading gate

The [approved plan](JUDGE_COMPARISON_LONG_RUN_PLAN_20261124.md) fixes Luna medium and allows one prompt repair.
Both original recipes and that repair have now failed its observed agreement gate.
No new solver generations or GPU jobs ran. The training checkpoint remains iteration 108.

## Option A: test higher judge reasoning effort

Recommended if the user wants another automatic diagnostic before reviewing labels.
Change only Luna's reasoning effort from medium to high.
Keep the repaired reference prompt, original rubrics and the already frozen fresh forty-answer packet.
Reuse the existing paired-medium comparison as the baseline on those same answers.
Run reference, empty and irrelevant controls on the same seventeen questions.
This requires 91 base requests. Preserve all unfinished-verdict retries and their charges.
Keep the same append-only ledger and its $2 comparison cap.
Only $1.5483058 remains under that cap. All stages still share the user's $25 allowance.
There is no automatic cap extension if the requests cannot finish within that remainder.

Use the same predeclared observed agreement gate, including review of false full-credit decisions.
Do not change the existing labels to fit high-effort outputs or claim independent human accuracy.
This is a new reasoning-effort comparison on a reused audit packet, not a fresh corpus evaluation.
If it passes, the already approved easy/medium screen and twelve-hour training plan can continue.
If it fails, resolve the rubric interpretation before GPU training. Do not add another automatic variant.

This option needs a user decision because the current plan holds the judge at medium effort.
The user's earlier instruction explicitly selected that effort.

## Option B: resolve rubric interpretation first

Review ten saved cases selected to expose disagreements and clear controls.
The question is whether equivalent proofs earn the mathematical item despite omitting its named route,
and which later items can earn independent credit when an earlier item is incomplete.
Keep these interpretation cases separate from an accuracy estimate on randomly sampled answers.
This option needs no additional API or GPU spending.
It may reveal errors in assistant labels as well as judge errors.

No GPU submission is authorized by this options file itself.
