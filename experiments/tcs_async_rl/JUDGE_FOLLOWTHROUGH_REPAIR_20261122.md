# Make partial-credit dependencies explicit before resuming

This is a prompt repair within the user's request to resolve grading ambiguities and resume RL.
The model, medium reasoning, solver prompt, data, rubric weights, and optimizer stay fixed.
The prior [plan](JUDGE_REPAIR_RESUME_PLAN_20261122.md) governs the 40-update, four-hour, combined $5 continuation.
The next frozen GPU configuration is `rl_easy_overfit_v14.json`, resuming iteration 64.

Prompt v2 matched all assistant item decisions on six development answers in both orders.
All 20 supplied-reference, 20 empty-answer, and 20 irrelevant-answer controls passed.
Five separate saved answers exposed ten item disagreements across the two orders.
The judge identified substantive gaps, but also deducted later components for gaps charged in earlier items.
For example, a rate error could lose the rate, conversion, and scope items in one call but only one item in another.
The original labels and all v2 decisions remain intact. No GPU job used v2.

Prompt v3 states the follow-through policy explicitly. A correct later operation can earn its component's points
when it uses a mathematically valid intermediate claim whose proof was incomplete in an earlier component.
It does not give completed-counterexample credit for unspecified hypotheses, or permit an actually false premise.
It also clarifies the existing elementary-inference rule for polynomially many operations on bounded-size rational numbers.
This interpretation is an assistant adjudication authorized by the user. It is not human calibration.

The earlier five validation cases become development cases for v3.
Five fresh saved answers, sampled with seed 20261124, are labeled before any v3 calls.
They include a recursion-count omission, a discrete interpolation proof, a zero-propagation sign slip,
a direct matching-sensitivity witness, and a uniform row-subset construction.
Their complete mathematical assessments remain private beside the saved answers.
The [adjudication manifest](evidence/judge_prompt_v3_adjudication_plan_20261122.json) pins their hashes and roles.

All qualification calls share the continuation's combined $5 allowance.
The v14 ledger will inherit the settled v13 qualification ledger byte-for-byte before starting.
The earlier prompt is not retried until a favorable score appears. V3 has an explicitly different scoring instruction,
and all old receipts remain available. Source examples are not copied into the solver's training prompt.

The shared filesystem now has 160,078,282,752 free bytes, above the unchanged 151,265,361,920-byte preflight requirement.
Five public cache revisions were evicted only after verifying their immutable remote revisions were available.
Both original checkpoints remain: iteration 24 from the first run and iteration 64 from the completed continuation.
The [cache record](evidence/amd_cache_space_recovery_v14_20261122.json) links the exact revisions and restoration commands.

Qualification must precede GPU submission. The run will retain all original runtime diagnostics and independent native W&B observation.
The watcher now limits live archive snapshots to once every ten minutes while checking metrics each minute.
Its CPU integration check verifies that archive throttling preserves metric checks and terminal finalization.
All 99 TCS CPU checks passed before the first prompt-qualification call.
