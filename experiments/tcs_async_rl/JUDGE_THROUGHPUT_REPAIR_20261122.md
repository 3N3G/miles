# Drain and repair the grade-service throughput bottleneck

At 06:41:42 UTC, the grade ledger recorded 1,218 settled calls. Mean wall time
per call was 17.49 seconds and the median was 15.83 seconds. All four permitted
API slots were occupied for 5,325.58 of 5,335.90 observed seconds. Four calls
remained in flight. No new API call was needed for this measurement.

The native scheduler counts a sample until `generate_and_rm` returns. This run
uses per-sample grading, so waiting for Luna also holds an admission slot.
At 06:37:57 UTC, a read-only Ray snapshot found 65 samples in flight, but only
32 generation semaphore slots occupied. Two pending groups had all 32 samples
marked generation-complete and were still waiting inside their sample tasks.
The router reported all three engines healthy. Their request loads were unequal.
This identifies a grade-service throughput bottleneck. It does not establish
why per-question rewards have not consistently increased.

The repair changes only simultaneous independent API calls, from four to eight.
It preserves the model, medium reasoning, request bodies and fingerprints,
answer-order swapping, rubric, extraction, retries, and shared $5 ledger.
The four-core VM's default executor can run eight blocking I/O calls.
There is no new GPU allocation and no additional training or judging budget.
GPU code, rollout admission 96, sampling, optimizer, data, prompts, and held-out
split remain unchanged. This is a service repair within the approved run.

Before restarting, pause only new rollout submissions through the owned
executor's thread-safe event-loop callback. Let existing generation and scoring
finish. Require both zero sample tasks in flight and zero unsettled judge calls.
Archive every old service manifest byte-for-byte. Stop the drained owned service,
then start its replacement with `judge_group_v3_api_concurrency_repair.json`.
Keep the same URL, input collection, result directory and budget ledger.
Verify health, identity, and a cached scored answer without a new judge call.
Resume producer submissions only after those checks pass.

Record pause, drain, restart, replay, and resume times in the run artifacts.
Report the phase change with final throughput and per-question results.
The used four-call configuration and all scores remain preserved.
