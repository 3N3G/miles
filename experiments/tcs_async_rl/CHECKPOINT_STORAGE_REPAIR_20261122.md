# Preserve source checkpoints during the remaining-budget continuation

A separate continuation needs about 151.3 GB free under the existing two-checkpoint space guard. The terminal shared-space
snapshot has 91.5 GB free. Every original checkpoint stays preserved. This operational amendment changes checkpoint timing
only. The current v15 allocation is already terminal. No GPU job has used the new option yet.

## Native single-checkpoint option

Use `--single-checkpoint` with a positive save interval of 105 and the new run's external `checkpoint-request` sentinel.
The native option overrides periodic and epoch-boundary saves in both training drivers. It saves at the final rollout or
at the first earlier external request. After a completed checkpoint exists, it skips the final save and rejects another
external request. Both final and external saves are synchronous. Source checkpoints remain read-only.

The first CPU-only proposal omitted `save_interval`. Inspection of the installed Megatron validator found that this fails
its positive-interval assertion. No GPU job used that proposal. Its exact draft and document are preserved privately.
The corrected native option retains a positive interval and changes the checkpoint predicate explicitly, after validation.
A large interval alone would still save at epoch boundaries and would not solve the problem. No training hook is needed.

Preflight validates a positive interval and this run's sentinel before using the one-checkpoint space estimate. Other
configurations keep the old estimate. For 4,326,350,848 text-plus-MTP parameters, the requirement is 90,696,450,048 bytes:
one model/master/Adam checkpoint, the existing conversion allowance, and 20 GiB workspace margin. The measured checkpoint
is 60,577,362,603 bytes. The same storage guard now runs before submission and again at allocation, before model startup.
The native flag rejects unsupported FSDP use and requires a fresh output directory with a separate resume source.

## Deadline safeguard and fixed limits

The first failed allocation used 996 seconds. The second used 7,209 seconds, with its batch ending after 7,211 seconds.
Charge the larger value conservatively. Of the original combined 14,400 seconds on eight GPUs, 6,193 seconds remain.
Reserve 60 seconds for scheduler termination overhead and request 102 minutes, leaving 73 seconds outside that request.
The shared judge ledger contains 2,135 settled calls and $3.1215173 estimated spend, with zero pending reservations.
Its cap stays $5. Do not reset the ledger. Cost may stop the continuation before its target.

Resume tracked iteration 84, containing 85 total saved updates. Target only the remaining global updates 85–104. The prior
allocation executed 65–88, but 85–88 were not saved and belong to a rolled-back branch. Keep that branch's results separate.
Use the exact remaining-duration request without a shorter backfill minimum. Keep all model, data, prompt, rubric, sampling,
optimizer, metric, seed and held-out settings fixed.

Keep foreground supervision. If the final update is unfinished with 15 minutes left in the actual allocation and no new
checkpoint exists, write the owned sentinel once. Let the current rollout finish and verify its tracker and checkpoint
files. Stop only this owned allocation after verifying an early checkpoint. Never issue a second external request.
Also inspect judge cost and pending reservations. Request the one checkpoint before a budget rejection when feasible.
Never fabricate a zero reward or raise the cap. If the cap blocks completion first, preserve the source and report lost
unsaved progress. Record a managed deadline stop separately from a training exception.
This policy trades intermediate recovery points for source preservation and a bounded storage peak. It does not enlarge compute.

## Validation before submission

- All 111 TCS CPU tests pass in the existing service environment, with no provider calls or GPU allocation.
- Tests execute the actual async driver's checkpoint block and actual epoch predicate. Normal mode saves at 89, 94, 99
  and 104. Single-checkpoint mode saves only at 104, or only at an earlier external request. Data saves precede model saves.
- The tests also cover a completed deadline checkpoint, a rejected second request, positive native argument serialization,
  the unchanged final target, and a foreign sentinel rejected before reducing the space estimate.
- The installed Megatron save-validation clause is tested directly from its source. Interval 105 passes. None fails.
  Its source SHA is in `native-checkpoint-policy-compatibility.json`. Full GPU startup validation remains mandatory.
- Black, isort and Ruff pass. Existing used snapshots are unchanged.

The earlier deletion question is withdrawn. No source checkpoint is removed, and no retention exception is needed.

## Cached container gate

The old host preflight also required 120 GiB before checking whether the container was already ready. That blanket gate
would reject this continuation despite its smaller checkpoint requirement. The corrected preflight reuses a cached SIF
and ready sandbox for RL, then applies the independent checkpoint guard before model startup. Missing or incomplete image
assets still require the original provisioning allowance. Other qualification modes keep their original threshold.
The new test covers all three cases. No GPU job used either rejected preparation proposal.
