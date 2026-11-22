# Preserve source checkpoints during the remaining-budget continuation

The current two-hour replacement is healthy. A separate continuation would need about 151.3 GB free under the existing
two-checkpoint space guard. Recent shared free space is about 94–96 GB. Every original checkpoint must remain preserved.
This operational amendment uses the existing repair authorization and changes only checkpoint timing, not the experiment.
No new GPU job has been submitted for this amendment. The running v15 snapshot remains unchanged.

## One checkpoint, using existing Miles machinery

Use the new `checkpoint_policy.request_final_checkpoint` before-training-step hook. It reads no model or optimizer state.
Disable `save_interval` by omitting its native flag, whose verified default is `None`. This also disables epoch-boundary saves.
A large interval alone would still save at epoch boundaries and would not solve the storage problem.
On the final global rollout, the hook touches this run's `checkpoint-request` sentinel before the optimizer step.
Miles checks it after training that rollout, saves synchronously with the rollout state, and removes the sentinel.
The hook does nothing if a completed checkpoint already exists. The source checkpoint is never modified.
Do not trigger from the rollout logger: prefetch can run it before the preceding optimizer step finishes.

The explicit sentinel path must be `<new-run>/checkpoint-request`, and the native periodic-save interval must be disabled.
Preflight validates both before allowing the one-checkpoint storage estimate. Other configurations keep the old estimate.
For 4,326,350,848 text-plus-MTP parameters, the guarded requirement is 90,696,450,048 bytes:
one model/master/Adam checkpoint, the existing conversion allowance, and 20 GiB workspace margin.
The measured checkpoint is 60,577,362,603 bytes. Recheck free space immediately before submission and allocation.

## Deadline safeguard and fixed limits

Read actual elapsed allocation times after v15 terminates. The combined ceiling remains four hours on eight GPUs and $5
for the shared judge ledger. Resume the atomically tracked v15 checkpoint and retain the original final global target 104.
Use an exact remaining-duration request without a shorter backfill minimum. Preserve all old configurations and results.

Keep foreground supervision. If the final update is unfinished with 15 minutes left in the actual allocation, and no output
checkpoint exists, write the owned sentinel once. Let the current optimizer rollout finish and verify its tracker and all
checkpoint files. If that checkpoint precedes the final target, stop only this owned allocation after verification.
A single completed checkpoint prevents the final hook from requesting another save, even if monitoring is delayed.
Never issue a second external request. Record a deadline stop as a bounded operational stop, not a training exception.
If the final update completes first, its sentinel requests the one final checkpoint and the job exits normally.
If no checkpoint finishes before the limit, preserve the original source and report the missing new durable checkpoint.
This policy trades intermediate recovery points for source preservation and a bounded storage peak. It does not enlarge compute.

## Validation before submission

- All 106 TCS CPU tests pass in the existing service environment, with no provider calls or GPU allocation.
- New checks cover the final rollout boundary, an existing deadline checkpoint, absence of training-state access, native flag omission,
  the unchanged global target, and rejection of periodic saves or a foreign sentinel before lowering the space guard.
- Black, isort and Ruff pass. The native before-step hook signature and post-training synchronous sentinel save were read directly.
- A first attempt with system Python lacked `aiohttp`; the established service environment passes both HTTP tests and the full suite.

The earlier deletion-permission question is withdrawn. This alternative preserves every old checkpoint and needs no retention exception.
