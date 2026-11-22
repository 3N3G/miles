# Allow a shorter allocation for the owned queued replacement

The transport-repair job 539594 remains pending. Its forecast moved from 23:59 UTC on November 22
to 11:48 UTC on November 23. At 20:50 UTC, all three usable MI350 nodes and both MI300 nodes were occupied.
The fourth MI350 node was drained. These are scheduler snapshots, not promises about actual completion times.

The existing continuation already requests at most 40 new updates or its allocation limit, within a combined
32 GPU-hour ceiling. Keep the submitted source/configuration unchanged. This operational amendment changes
only the pending job's scheduling flexibility, under the authorization to monitor and repair this experiment.

Before changing Slurm, verify job 539594 is still pending, owned by `geney`, named `tcs-easy-overfit-v15`,
with maximum time 3h43m and automatic requeue disabled. Set `TimeMin=02:00:00` on that job only.
Do not change its maximum time, priority, GPU count, partition or any other user's allocation.
Skip the update if the job has already started. Save the exact before/after scheduler readbacks.

AMD uses `sched/backfill`. The [official Slurm documentation](https://slurm.schedmd.com/sbatch.html#OPT_time-min)
states that a minimum time can permit an earlier start by allowing a shorter slot. The allocated time remains
between that minimum and the requested maximum. The [scontrol documentation](https://slurm.schedmd.com/scontrol.html#OPT_TimeMin)
supports changing `TimeMin`. No earlier start is guaranteed.

If Slurm grants less than 3h43m, the same configuration targets global updates 65–104 until it reaches the actual time limit.
Preserve every result, exact scheduler timing and the last atomically completed checkpoint. Do not call a walltime exit a training bug.
If 40 updates are unfinished, any authorized repair continuation must use the remaining combined GPU budget and same $5 ledger.
Deduct all elapsed allocation time, including initialization and failed work. Keep the target global update at 104.
Use the checkpoint tracker, rather than the highest directory name, when selecting a valid resume point.
Do not create an automatic requeue or a second full allowance.

Models, data, prompts, reward, sampling, optimizer, metrics, seed and held-out split stay fixed.
The same MI350 layout remains the only qualified target. MI300 and Flame were not submitted as fallbacks.
The two MI300 nodes were occupied. Its 90k layout remains unqualified, and full activation recomputation is already enabled.
The preceding MI350 allocation observed 248.5–249.2 GiB on its two highest-use devices, including allocator reservations.
That observation does not prove the minimum active-memory requirement on MI300. Flame's changed host key still needs trusted verification.

Continue the [owned-run supervision](RUN_V15_OPERATIONS_20261122.md), including first-update verification,
actual proof reviews, matched-question metrics, terminal checkpoint inspection, immutable uploads and W&B readback.
