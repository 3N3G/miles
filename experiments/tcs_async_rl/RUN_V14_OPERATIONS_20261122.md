# Operational handoff: resumed easy-question RL with clarified grading

Keep foreground supervision through terminal scheduler state, checkpoint inspection, private result upload, and W&B readback.
The existing background processes do not replace diagnosis. Do not claim reliable learning from a small mixed-question score change.

## Authority and current allocation

- User requested assistant adjudication of grading ambiguities, prompt repair, and rerun/resume with close supervision.
- Plan: [judge repair and resume](JUDGE_REPAIR_RESUME_PLAN_20261122.md), with [the follow-through clarification](JUDGE_FOLLOWTHROUGH_REPAIR_20261122.md).
- AMD job **538647**, `tcs-easy-overfit-v14`, submitted 2026-11-22 09:34:36 UTC.
- Actual allocation started **2026-11-22 19:57:31 UTC** on `k007-002`, verified against Slurm owner/name.
  The current four-hour limit ends at 23:57:31 UTC. Producer run ID is `261122-195832-676`.
  The VM-native W&B writer will use `261122-195832-676-native` after the first complete new update.
- At 20:01 UTC the owned node could reach the expected Luna-medium service through the loopback tunnel.
  All three rollout engines returned health HTTP 200. Checkpoint restoration and the first new update were still pending.
- The 20:08:54 UTC read-only probe verified both trainer ranks at scheduler counter 2,080 and learning rate 1e-6.
  The runtime loaded iteration 64 and checked the restored weight broadcast to all three rollout engines.
  The producer was active with three groups, no completed buffer group, and weight version 65. No optimizer update had completed.
  Probe evidence is `optimizer-and-rollout-bootstrap-probe.json` in the run directory.
- Direct scheduler check at 2026-11-22 09:52 UTC: PENDING, priority wait, start estimate unavailable.
  The earlier 23:36 UTC estimate was withdrawn. Recheck before reporting a live state or ETA.
- Limit: 40 new updates, global steps 65–104, or four hours on eight MI350 GPUs. Combined repair ceiling 32 GPU-hours.
- Slurm automatic requeue was disabled on this owned pending job. Any repair must respect the remaining combined budget.
- Judge: Luna medium, original weighted binary rubrics, anonymous order swaps, final-only extraction. Shared cap $5.
- Models, data, solver prompt, sampling, optimizer, seed, and held-out split stay fixed. FOCS/STOC 2026 remain sealed.
- Human calibration and independent-provider validation remain absent. The user authorized assistant diagnostic labels.

## Immutable source and paths

- GPU/controller source: `3f33bbd0d370d31126e792750f3a363f83a5c5e6`.
- Frozen config: `experiments/tcs_async_rl/rl_easy_overfit_v14.json`.
- Judge config: `judge_group_v5_followthrough_repair.json`, prompt `rubric_judge_prompt_v3.txt`.
- Resume: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v12/checkpoints/iter_0000064`, 65 saved updates.
- Original iteration 24 checkpoint also remains. Never alter either source checkpoint or old run results.
- VM run: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-overfit-v14`.
- VM frozen checkout: `/home/ubuntu/.cache/miles/tcs-async-rl/checkouts/easy-overfit-controller-v14`.
- AMD run: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v14`.
- AMD frozen checkout: `/work1/aviralkumar/geney/tcs-async-rl/checkouts/easy-overfit-v14`.
- VM diagnostic gate: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-qualification-v14`.
  It regrades the original saved easy-check-v14 generations, job 535067. No replacement generation job ran.
- Source bundle SHA-256: `18ed636017d16c3af5c1448ee4967b4aa6b996037c73f5304bf4ce95469e5929`.

## Owned processes

Read their process receipts and `/proc` state before managing them. They were started in this session.

| Role | PID | Receipt/log in the VM run |
|---|---:|---|
| Controller | 1154899 | `controller-process.json`, `controller.log`, `handoff-state.json` |
| Judge service | 1154901 | `reward-service-process.json`, `reward-service.log` |
| Independent metrics observer | 1155113 | `metrics-observer-process.json`, `metrics-observer.json` |
| GPU telemetry | 1155114 | `gpu-telemetry-process.json`, `gpu-telemetry.jsonl` |
| Hourly detailed review | 1155115 | `hourly-review-process.json`, `latest-detailed-review.json` |

The controller owns a separate reward-tunnel watcher. It connects to a compute node only while this owned job runs.
Do not run another `auto_rl` against the same receipt. It must not create a duplicate submission.
Credential loading remains inside the authorized VM service. Never inspect or transfer credential values.

## Qualification and remaining limits

- Prompt v2: six development answers matched all assistant item labels. Five fresh answers exposed ten item disagreements.
  No GPU job used v2. Its settled estimated cost was $0.1849453.
- Prompt v3: 11 development answers matched 99/102 decisions. Five fresh answers matched 43/48 decisions.
  These repeated decisions are not independent examples or human accuracy estimates.
- All 60 reference/empty/irrelevant controls passed. The regraded 32-answer gate has 17 zeros, five fives, four sixes, six sevens.
- Both qualification phases settled with zero pending calls and $0.3972903 estimated total cost.
  The v14 ledger inherits the entire v13 ledger prefix. Do not reset it or add another allowance.
- A later [eight-proof audit](JUDGE_ADDITIONAL_AUDIT_FINDINGS_20261122.md) matches all 70 repeated assistant item decisions.
  Its sixteen new calls bring the same ledger to $0.4150215, with 422 settled call IDs and zero pending reservations.
  The prompt, queued job and training settings remain unchanged. Keep earlier borderline disagreements and calibration limits.
- [Preserved v2 qualification](https://huggingface.co/datasets/3N3G/internalization-runs/tree/ddcfe3aa37d888f7376835682baa7dcdde8f7e0f/miles/easy-overfit-v13).
- [Verified v3 pre-submission archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/b69851eb463907e85d0bb760e97d4acf48f63742/miles/easy-overfit-v14).
- The VM run also contains a manual algebra/boundary check of all 20 references, without data changes or extra judge calls.
  It is not formal certification. One reference invokes the standard fractional spanning-tree-packing bound.
- The iteration-64 rollout checkpoint has zero completed scored groups and zero executor replay batches.
  Exact ZIP/pickle-opcode inspection is recorded in `resume-old-reward-buffer-inspection.json` in the VM run.
  Source checkpoint bytes remain unchanged. Temporary inspection copies were removed before archival.
- A post-qualification review supports full credit for the forest proof's component invariant: the cut already confines each
  sampled vertex's component to its descendants. The earlier assistant denial was too strict under the permitted implicit-step policy.
  `assistant-adjudication-followup.json` preserves this correction separately. Original pre-call labels and validation counts stay intact.
- Five reproducible public cache revisions were evicted after verifying their immutable remote availability.
  Shared free space then became 160,078,282,752 bytes, above the unchanged 151,265,361,920-byte submission preflight.
  See the cache plan/recovery evidence for restoration commands. All research results and checkpoints remain.

## Monitoring and closure

Inspect scheduler state, native updates, finite losses and gradients, publication hashes, service/tunnel health, and disk space.
Expected restored scheduler counter is 2,080 with constant learning rate 1e-6. The final target counter is 3,360.
Scheduled fixed-batch update-direction observations are at 65, 66, 70, 80, 90, and 100.
Inspect full/sampling entropy, sampled initialization drift, rollout mismatch, clipping, staleness, caps, length, and judge gaps.
Native history exports `train/grad_norm` and `train/mtp_loss`, without a separate MTP gradient-norm scalar.
Separate MTP norms come from [read-only optimizer probes](optimizer_probe.py) inside a verified owned running allocation.
Record each probe's timestamp and expected run ID. These snapshots are not a continuous per-update MTP norm history.
For an external probe, use the same container/runtime bind and `RAY_ADDRESS`, but omit `--pid` from its prefix.
Ray 2.58 discovers local raylets by scanning process arguments. A fresh private PID namespace hides the running raylet
and caused the preserved first probe attempt to fail before reaching any actor. The host-PID read-only retry succeeded.
Read actual seeded proofs behind matched-question changes at least hourly. Use `audit_live.py`, then `summarize_continuation.py`
and `plot_continuation.py` on a new diagnostic output directory. Keep prior diagnostics unchanged.
Use `judge_signal.py` on audited samples to inspect agreement after group centering.
Its [retrospective findings](JUDGE_REWARD_SIGNAL_FINDINGS_20261122.md) describe the original judge, without measuring new-prompt accuracy.

`baseline-groups.json` is deliberately empty. Never compare old-prompt rewards directly with new-prompt rewards.
Use within-continuation matched question visits. The independent metrics observer uses the existing locked single writer.
Tag the new online W&B run with the judge prompt SHA once its first native update creates the run.
Do not merge AMD offline writers into that online history. Verify all scalar coordinates, not just point count.

Live archive snapshots are throttled to ten minutes. Native checks remain minute-level. If upload stalls, keep watching native
progress and the service separately. Preserve unarchived traces; use the existing recovery instructions for actual failures.

At termination, drain provider calls, verify checkpoint inventory, archive all traces and metadata, verify immutable readbacks,
finish W&B at the actual terminal state, and stop owned observers/services after their work completes. Commit/push outcomes and
workstream state through `workstate`. Do not mark the research goal achieved unless the evidence supports it.

Session: Codex `01a1ed24-1032-7731-910b-824f484f5f71`, machine `home`, original directory `/home/ubuntu/miles`.
Workstream: `tcs-async-rl-recon`. No subagents were authorized or spawned.
