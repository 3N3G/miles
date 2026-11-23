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

## Terminal budget stop and verified closure

At 02:05:50 UTC, shared judge spend plus reservations reached $4.5033907.
The one native checkpoint sentinel was written at **02:06:11 UTC**, after recording its intent and verifying ownership.
New producer admissions were paused at 02:10 UTC. Existing groups continued, allowing the active update to finish.
No model, prompt, data, reward or optimizer setting changed during this stop.

The final tracker is **iteration 98**, containing **99 total saved updates**, with **14 new updates 85–98**.
Its four model/optimizer files total **60,577,362,731 bytes**. Its ten rollout-state files total **49,819,659 bytes**.
The source iteration-84 model remains intact at 60,577,362,603 bytes.
The saved iteration's actual three-engine publication is verified at **02:14:42 UTC**.
After these checks, only owned job 540286 was cancelled at **02:16:47 UTC**.
Slurm reports `CANCELLED by 3414`, parent `0:0`, batch `0:15`.
This was the planned cost stop, not an uncaught training exception.

Charge 4,137 seconds conservatively, including batch termination. Combined charges are 12,344 seconds on eight GPUs,
or **27.4311 GPU-hours**, below 32. The unused 2,056 seconds do not authorize a changed protocol or larger judge budget.
The shared ledger ends with **3,189 settled calls, zero pending calls and $4.6894719** estimated spend.
All controller, service, tunnel, metrics, telemetry and hourly-review processes are closed.
The hourly reviewer was stopped only after the main supervisor completed its final upload.

All **14 actual publications** agree across three engines. W&B has **14 unique points and 1,127 matching scalar values**.
Its terminal state is `failed`, because the native adapter maps Slurm cancellation to failure.
Its summary explicitly records the controlled budget stop and the saved/target counts.
The original 20-update runtime acceptance remains false because only 14 were completed. Do not relabel it successful.

The raw audit verifies all **448 answers and 21,182,426 response tokens** against immutable consumed batches and both judge receipts.
The combined saved path includes only predecessor 65–84 and continuation 85–98: 34 groups, 1,088 answers, all 20 questions.
Fourteen repeated-question means change from 5.066964 to 4.968750: six increase, seven decrease, one is unchanged.
See the [standalone results](BOUNDED_CONTINUATION_RESULTS_20261123.md) for scientific limits and raw examples.

Primary archive: **`2ea9cf54a7e52fb27ceb1a0e4c70c513dd0f999b`**, with seven exact immutable file readbacks.
Diagnostic archive: **`6eb6ab1a9781ec1392d18cf6144e21401c93d5db`**.
Its eight exact readbacks include the complete comparison, both stress cases and the final matched-question plot.
The final manual closure archive is **`c971823603e3efbe35c7fe78a51e2aa303fa146d`**.
Nine exact readbacks verify its terminal supervision, W&B scalar audit and stop provenance, saved/source checkpoint checks,
weight publication, judge ledger, both earlier readback receipts and owned-process closure.
`closure-archive-verification.json` preserves these hashes. W&B's terminal stop summary was also independently read back.
The VM run preserves `primary-archive-verification.json`, `diagnostic-archive-verification.json`,
`terminal-supervision.json`, `terminal-native-wandb-all-scalars-verification.json`,
`budget-stop-checkpoint-and-source-verification.json`, `budget-stop-saved-weight-publication.json` and `owned-process-closure.json`.

## Reporting fixes found during the stop

The short-lived W&B publisher let its server heartbeat expire during a long rollout gap, falsely displaying `crashed`.
A serialized, no-history SDK refresh repaired the live status. The permanent two-minute refresh is in commit
**`8508403ed04041cac1bdb6e0f0ad55feb8949c55`**, together with its delivery acknowledgement.
It passed simulated delayed-update and lost-acknowledgement tests. The new periodic behavior still needs observation in a future live run.

Actor timing metrics also arrived before the current transfer during checkpoint saving. They can describe the preceding transfer.
The corrected reader waits for the driver's actual checksum publication event, rather than treating timing as proof of publication.
All final counts above were independently checked against those actual events.
All 111 TCS CPU tests pass, and Black, isort and Ruff checks pass.
Training source and every used configuration remain frozen at their recorded revisions.
AMD's clean main checkout has pulled the reporting repairs. No TCS allocation remains active or queued.
No further GPU job was submitted. The next research decision is recorded in `QUESTIONS.md`.
