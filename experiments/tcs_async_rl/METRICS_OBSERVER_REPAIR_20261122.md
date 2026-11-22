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
