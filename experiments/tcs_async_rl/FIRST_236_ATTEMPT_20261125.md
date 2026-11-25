# First attempt on the full easy and medium corpus

This attempt cannot establish improvement because every sampled question was different. It also ended before the approved runtime.
The experiment asks whether the fixed grader and broader corpus improve comparable proof scores.

Qwen3.5-4B continued from checkpoint108 on236 question rows:20 easy and216 medium.
Each optimizer update consumed four questions with32 answers each.
Thinking was enabled, with a90,000-token response limit and the accepted Luna/high reference-guided rubric judge.
The FOCS2026 and STOC2026 cohorts remained sealed.

The run produced19 finite updates and2,432 consumed answers across76 distinct question rows.
Checkpoint124 preserves16 updates. Three later updates were unsaved when a malformed judge reply terminated generation.
The replacement resumes124 within the original compute limit.

Raw batch reward ranged from1.000 to5.352/7. Question mix alone can create large movements:
two batches generated under the same published policy scored3.109 and5.258.
The centered reward metric stays near zero by construction.
Use raw reward by difficulty and the planned fixed-question checkpoint comparison to assess changes.

The optimizer showed71–127 accumulated microbatches per update, gradient norms0.058–0.102, and full-policy entropy0.381–0.499.
Four local before/after probes increased the surrogate objective. These checks support correct update direction on consumed samples.
They do not establish better proofs or rule out slow learning.

One final-batch question averaged0.063/7. Its sampled answer counted individual matching executions instead of entire counterfactual functions.
Another averaged6.313/7. Its sampled answer correctly derived the estimator variance but omitted the all-zero input case, scoring6/7.
These examples illustrate question difficulty and rubric sensitivity. They are seeded diagnostic samples, not independent human calibration.

The same12-question final screen remains pending. One continuation seed cannot measure variation between training seeds.
Full measurements and raw examples are linked in the [final-batch audit](evidence/long236_failed_final_review_20261125.json)
and [policy checks](evidence/long236_failed_attempt_policy_health_20261125.json).
