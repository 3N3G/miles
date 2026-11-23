# Operational plan: finish the remaining bounded continuation

Submitted as AMD job **540286**, `tcs-easy-overfit-v16`, at **2026-11-23 01:07:42 UTC**.
It started at **01:07:51 UTC** on `k007-004`. Read Slurm again before reporting current state.
The actual limit is 102 minutes, ending **02:49:51 UTC**, with no shorter minimum and `Requeue=0`.
The fifteen-minute checkpoint request deadline is **02:34:51 UTC**.
Use the same workstream `tcs-async-rl-recon`, session `01a1ed24-1032-7731-910b-824f484f5f71`, home VM, original directory `/home/ubuntu/miles`.
No subagents were requested or used.

- Config: `rl_easy_overfit_v16.json`. Resume the terminal v15 tracker at iteration 84.
- Only 20 target updates remain, global IDs 85–104. The v15 branch 85–88 was not saved. Preserve it separately.
- Request exactly 102 minutes on eight MI350 GPUs, with no shorter minimum and no automatic requeue.
- Charge prior allocations 996 + 7,211 seconds. Requested time 6,120 seconds plus 60-second termination reserve remains below 14,400 seconds total.
- Judge ledger inherits every settled receipt and byte-exact budget prefix: 2,135 calls, $3.1215173, no pending calls. The total cap remains $5.
- Models, prompts, data, rubric, sampling, optimizer and final global target are unchanged. `native-argument-change-audit.json` lists only paths and checkpoint timing changes.
- Native `single_checkpoint=True`, positive save interval 105, owned sentinel `<run>/checkpoint-request`. The native mode overrides epoch/periodic saves.
- No training hook is installed. Validate the real native args before model startup. Check disk at submission and allocation.
- The new output is separate from all source checkpoints. Every old checkpoint remains preserved.
- At actual remaining walltime 15 minutes, request the one checkpoint if unfinished. Also request it at shared judge spend/reservations $4.50.
  Persist the request reason and issue it once. Verify a tracked checkpoint and all files, then stop only this owned allocation if the checkpoint precedes the final target.
  Do not reset or raise the cost cap. A budget rejection must not become a fabricated zero reward.
- Preserve one final or deadline/budget checkpoint. Do not issue a second request. A final completed checkpoint lets the job exit normally.
- Watch the job and all services in the foreground through terminal, checkpoint inspection, immutable uploads and full W&B coordinate verification.

VM run: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-overfit-v16`.
VM controller checkout: `/home/ubuntu/.cache/miles/tcs-async-rl/checkouts/easy-overfit-controller-v16`.
Persistent service Python remains `service-env-v15/bin/python`. Credentials stay inside the authorized VM client.
AMD run: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v16`.
AMD checkout: `/work1/aviralkumar/geney/tcs-async-rl/checkouts/easy-overfit-v16`.
AMD source checkpoint: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v15/checkpoints`.

At 00:49 UTC, 91 regenerable LLVM cache files predating both current allocations were removed from the TCS compiler cache.
Their total size is 1,630,392,828 bytes. No TCS job was active. No result, model or checkpoint was removed.
The shared filesystem's later 00:51 readback has 91,779,805,184 free bytes. Keep rechecking, since other jobs also write there.
`compiler-cache-cleanup.json` contains every removed filename and size.

Before submission: verify v15 closure/readbacks, inherited budget, clean remote source, available disk, and no duplicate TCS job.
After submission: immediately record the job and disable requeue after owner/name validation. Start independent metrics, telemetry and hourly reviews.
Use the actual producer ID from `resolved-config.json` for W&B. Never merge AMD offline streams into its VM-native history.
A successful first optimizer step does not end supervision. Do not claim reliable learning from one seed or mixed questions.

## Verified startup and first update

Training/controller source is frozen at **`4fb3ca73fe9667fe19766d1626671df8d275283e`**.
Producer ID **`261123-010857-293`** owns the verified
[native W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/261123-010857-293-native).
At 01:24 UTC, its first update 85 matched all 79 native scalar values. W&B reported `running`.
The initial three-engine weights equal the predecessor's saved iteration-84 publication exactly.
Update 85 changed 464 of 822 language tensors, with all three rollout engines agreeing.

The read-only 01:17:59 probe completed after update 85. Both ranks have scheduler counter 2,752,
constant learning rate 1e-6, `single_checkpoint=True` and save interval 105.
The counter equals 86 completed global updates × 32 answers, rather than the pre-update expectation 2,720.
Separate multi-token prediction gradient norm is 0.067516 on both ranks at this snapshot.
Every reported worker source hash matches the frozen checkout. Container package-origin metadata is not a worker source hash.

The first group scores 0.09375/7, with 28 zeros, two half-points and two ones. Twelve answers hit the cap.
It was restored from saved rollout state, with decode weight versions 80–84. Do not label these as fresh-policy answers.
The native rollout checkpoint saves completed and pending generation state as well as model state.
Keep this resumed trajectory separate from the predecessor's unsaved branch, even when restored answers repeat.
Finite entropy, divergence and gradient metrics verify numerical execution, not learning.

Startup controls pass 60/60 from cache, with no extra judge spend. Physical GPU memory is 309,220,868,096 bytes per GPU.
The 01:22 minute sample reports 89,379,962,880 shared free bytes. Keep checking this shared resource.
At allocation, the independent checkpoint guard verified 90,886,299,648 free bytes against a 90,696,450,048-byte requirement.
Receipts: `optimizer-and-rollout-bootstrap-probe.json`, `restored-weight-publication-verification.json`,
`first-native-wandb-all-scalars-verification.json`, `checkpoint-storage-at-allocation.json` and `slurm-budget-guard.json`.

Session-owned VM processes: controller **1276272**, judge **1276276**, metrics observer **1276761**,
GPU telemetry **1276762**, hourly review **1276763**, and active reverse tunnel **1276570**.
Verify process receipts and `/proc` before managing them. Do not start a second controller against this submission.

## Update-90 review and guards

At 01:47 UTC, all 487 scalar values for updates 85–90 match the sole W&B history, without duplicate coordinates.
Update 90's fixed-batch surrogate gain is +0.00041719. Positive-advantage mean log-probability change is +0.00038898,
versus −0.00035641 for negative advantages. This checks update direction, not future answer quality.
`six-updates-native-wandb-all-scalars-verification.json` and `update-90-direction-check.json` preserve these observations.

The first three complete immutable audits cover 96 answers and 6,411,855 response tokens.
Three answers have judge-order disagreement, with maximum gap one. Every paired reference score is seven.
Seed-selected examples include a defensible one-point incomplete counterexample, a valid seven-point matrix construction,
and a valid seven-point argument that the stated parameter range is empty. These assistant checks are not human calibration.
The later local summaries remain provisional until their immutable full-group checks finish.

At five new groups, the combined saved trajectory has eight repeated questions: three increase and five decrease.
Their first-to-latest mean change is −0.3046875/7. This includes the earlier durable prefix, not just recent positive pairs.
Always exclude the predecessor's unsaved updates 85–88 from this combined path.

The foreground disk alert is 85,000,000,000 free bytes. It requests inspection rather than changing the training protocol.
One observed checkpoint plus 20 GiB workspace requires 82,052,199,083 bytes. Other users also write to this filesystem.
The cost/deadline request thresholds above remain authoritative. `foreground-resource-guard.json` records the live guard.
