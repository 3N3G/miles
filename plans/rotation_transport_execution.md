# Allocation change for the rotation capture

At 2026-11-21 15:48 UTC, job 536932 remained pending on mi3501x.
The scheduler estimated a start around 16:36 UTC. All seven available mi3501x nodes were allocated.
Live `sinfo` showed nine idle mi2101x nodes, each with one 64 GB MI210 GPU.

Move this session's pending job to mi2101x using `scontrol update JobId=536932 Partition=mi2101x`.
The model's roughly 53 GB BF16 weights and bounded capture batches should fit within 64 GB, but the job will verify this.
Keep the exact script, checkpoint, source revision, prompts, selected positions, and one-hour limit.
This changes the execution host only. Preserve the original submission receipt and record the scheduler update.
No other session's jobs may be changed.
