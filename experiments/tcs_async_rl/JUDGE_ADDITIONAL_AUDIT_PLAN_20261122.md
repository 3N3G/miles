# Additional saved-proof audit during the queue wait

The user authorized careful assistant adjudication of grader ambiguities and continued RL supervision.
This additional diagnostic keeps the queued training job, model, prompt, rubrics, data and compute limits unchanged.

The question is whether the frozen revised grader incorrectly awards a false proof or rejects a sound proof over harmless wording.
Select eight distinct training questions with seed 20261125, then one existing seed-selected answer for each.
Exclude the sixteen explicit earlier prompt-adjudication cases. Selection uses question identity, not the original reward.
The resulting saved steps are 29, 36, 48, 54, 63, 37, 34 and 59.
They cover eight of the same twenty easy training questions. FOCS/STOC 2026 remain sealed.

Before making revised-judge calls, record a Boolean decision and mathematical reason for every supplied rubric item.
Freeze those assistant labels by SHA-256. Preserve later corrections separately.
The frozen file contains 35 item labels with SHA-256 `ebceaa8fea99eabe8949fab3afc4272784dfaa5e14bfac3291834a470120b187`.
The expected weighted totals are seven for the first seven cases and zero for the final false-theorem assertion.
Several cases contain unnecessary inaccurate wording after an independently sufficient argument. Record those explicitness ambiguities.

Call the existing Luna-medium VM service in both anonymous orders: 16 calls before any bounded unfinished-verdict retries.
Use the unchanged prompt v3 and its existing shared $5 ledger, including all previous qualification and training costs.
The service owns credentials and concurrency. The review client loads no credentials and submits no GPU jobs.
Keep original labels, source answers, receipts, item disagreements and reference grades.

Report both score pairs and criterion disagreements. A mathematically false full-credit award requires investigation.
Do not tune the prompt to force agreement on a defensible borderline case.
These assistant checks are not human calibration, independent questions or a seed-level accuracy estimate.
No significance test or new primary training metric is introduced.
Foreground GPU supervision takes priority if the owned job becomes active.

Source: [review client](review_saved_judge.py), reusing the existing [per-item review](qualify_judge_repair.py).
Private inputs and pre-call labels are under `easy-overfit-v14/additional-proof-audit` in the VM run.
