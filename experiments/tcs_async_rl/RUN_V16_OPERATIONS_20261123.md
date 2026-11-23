# Operational plan: finish the remaining bounded continuation

Prepared, not submitted. Check `submission.json` and Slurm before claiming a live job.
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
