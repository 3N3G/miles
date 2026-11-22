# Keep metric reporting independent of artifact upload

At 06:08 UTC, AMD job 538154 remained running and decoding. Its native log had
completed ten new updates. The main VM supervisor's last heartbeat was 05:55 UTC.
Its child `hf upload` was waiting during an incremental private result upload.
The upload had not failed, and training and grading remained active.

`watch_metrics.py` now copies only native event files and the training log each
minute. It uses the existing native metric reader and the existing W&B writer
lock, delivery acknowledgement, and API readback. It writes its own heartbeat
before calling W&B. It neither manages training or grading processes nor
uploads or evicts traces. The original supervisor retains those responsibilities.

The live job keeps its frozen source and approved configuration. This repair
changes only VM observation. Existing upload timeout and retry remain intact.
The observer stops after publishing the scheduler's terminal state.

Validation: the actual rsync dry run selected only the native event files and
`rl-smoke.log`. Python compilation, Black, isort, and Ruff passed. Live W&B
readback and fresh optimizer coordinates will qualify the observer after launch.

At 06:10:59 UTC, the observer verified all eleven completed updates through
global step 35 in W&B, with state `running`. It repeated that verification at
06:12:04 UTC. The first readback lagged SDK delivery, then caught up without
resending acknowledged points. The archive uploader was still waiting.

## Private archive commit rate limit

At 06:15:37 UTC, I terminated only the owned upload client. Its captured log
identified HTTP 429 at the Hub's dataset `commit/main` endpoint. The client
requested a 3,601-second delay. Its streamed upload had checked 2,304 files and
committed 1,703 files in three partial commits. It had not produced the final
immutable receipt. All four local trace copies remained intact and unindexed.

At 06:16:32 UTC, I paused only the original artifact supervisor, PID 1114406.
Its reward service and tunnel remain active. The independent native metric
observer and GPU telemetry remain active. Foreground supervision checks those
processes and new grading receipts while the original supervisor waits.

The separate `watch_archive.py` worker respects a conservative backoff until
07:16 UTC, one hour after observing the error. It then takes live snapshots
at most once every five minutes. The ordinary Hugging Face upload path makes
one commit per snapshot, rather than splitting snapshots across several commits.
The worker drains the remaining traces after the GPU job ends. Resume the
original supervisor only after that worker exits, so two uploaders never race.

The replacement eight-call judge service has its own process receipt. After
the GPU job reaches a terminal state, let its remaining provider requests drain.
Stop that replacement service after the drain, then resume the original
supervisor once the archive worker has exited. Its final upload can then include
all grade receipts and the completed cleanup record. The original supervisor
still owns its old service handle and tunnel. Keep it paused during live training.

Upload timeouts now stop the owned shell wrapper and its client descendants.
They retain output for diagnosis and preserve all unarchived traces. A real
parent/child timeout test verifies that neither process continues running.
The live GPU source, experiment settings, score rules, and limits are unchanged.

At 07:16:42 UTC, the bounded-rate worker completed its first ordinary upload in
one commit: `5597f043bebc1bdac88e42e4a79fe84b8f22ce47`. It archived all 24 consumed
groups available at that snapshot. Three exact metadata files, including the
before/after judge-service manifests, were downloaded from that revision and
matched their local SHA-256 hashes. Generated-sample cache files remain queued
for later bounded transfers. AMD originals remain intact.

## Trace-audit retry-chain repair

The resumed full trace audit stopped at a preserved judge length retry. It had
assumed exactly one HTTP-200 receipt per answer and presentation order. The real
case kept an 8,192-token unfinished verdict and its completed 16,384-token retry.
The training grader had already selected the completed verdict correctly.

The audit now verifies request fingerprints, identities, anonymous orders,
the original and maximum judge budgets, unchanged scoring payloads, and the
recorded expansion chain. It accepts one completed endpoint after unfinished
verdicts. It rejects regrading a completed verdict, a changed payload, an
unrecorded expansion, or an unfinished endpoint. Original receipts stay intact.
Ninety-eight TCS CPU tests pass. The actual 8,192-to-16,384 case also passed.
This changes only artifact validation. Scores and GPU execution are unchanged.
