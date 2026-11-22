# Queue and hardware fallback check

Operational snapshot, 2026-11-22 19:28–19:33 UTC. Recheck live state before acting.
AMD job 538647 remains pending. No GPU time or new optimizer updates have been used.

## Current capacity

| Partition | Allocated nodes | Drained nodes | Idle nodes |
| --- | ---: | ---: | ---: |
| mi3508x | 1 | 3 | 0 |
| mi3008x | 2 | 0 | 0 |

Earlier, mi3508x had three allocated nodes and one drained node. Two more nodes are now drained.
Its remaining running job has a scheduled limit ending at 2026-11-23 04:48:58 UTC.
The scheduler gives that same tentative start for our queued job: 11:48:58 p.m. Eastern on November 22.
The two running MI300 jobs have later scheduled limits, 05:46:29 and 06:17:54 UTC on November 23.
Actual job completions, repairs and priorities can change these estimates. No other user's job was modified.

## Compatibility limits

The preceding run observed up to 267,561,508,864 bytes of device memory use, about 249.2 GiB.
The earlier MI300 control reports 206,141,652,992 bytes of physical memory, about 192.0 GiB.
The observed use includes allocator reservations. It is not an exact minimum active-memory requirement.
The current 90k-token training layout has not been qualified on MI300.

Increasing tensor parallelism is also a checkpoint question.
The preceding launch used tensor parallelism two and `dist_ckpt_optim_fully_reshardable=False`.
Its restore log identifies `dp_reshardable` optimizer metadata.
An opcode-only read of iteration 64's 544,950-byte checkpoint metadata confirms buffer/bucket optimizer keys.
The [Megatron Core optimizer documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.optimizer.distrib_optimizer.html)
states that `dp_reshardable` supports resharding in the data-parallel dimension.
Inference: a four-way tensor-parallel resume is not a validated direct replacement for this checkpoint.
No checkpoint conversion or four-way load was attempted.

The Qwen linear-attention implementation contains a context-parallel path.
A layout preserving tensor parallelism two and adding context parallelism would need memory, numerical and restore validation.
No alternative GPU job was submitted, and the existing compute limit stays unchanged.

## Current action

Keep the qualified MI350 job queued and supervise its services and scheduler state.
The optional Flame inventory check still awaits trusted verification of its changed SSH host key, tracked in [QUESTIONS.md](../../QUESTIONS.md).
Use the [run supervision procedure](RUN_V14_OPERATIONS_20261122.md) as soon as an owned allocation starts.
