# Operational handoff: final-answer transport repair

Keep foreground supervision through scheduler termination, checkpoint inspection, immutable upload and native W&B readback.
The user requested ambiguity repairs and a supervised resume. The [transport repair](INFERENCE_TRANSPORT_REPAIR_20261122.md)
uses standing inference-repair authorization. No subagents were authorized or spawned.

## Frozen run and resource limits

- AMD job **539594**, `tcs-easy-overfit-v15`, submitted **2026-11-22 20:23:43 UTC**.
- Last direct scheduler check at 20:24 UTC: PENDING for resources, no start estimate. Recheck before reporting live state.
- Actual allocation started **2026-11-22 22:16:44 UTC** on `k007-003`, with **2h** granted by backfill.
  Its scheduled limit is **2026-11-23 00:16:44 UTC**. Source configuration retains its original maximum unchanged.
  `owned-allocation-start.json` preserves the exact scheduler readback and a successful service check from the owned node.
- Producer run ID: **`261122-221746-947`**. The [native W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/261122-221746-947-native)
  is running. At 22:45 UTC, all 88 scalar values for completed update 65 matched their exact native coordinates.
  `first-native-wandb-all-scalars-verification.json` records that readback. Judge and resume provenance was added to its config/tags.
- The 22:22:47 UTC read-only probe verifies both ranks at scheduler counter 2,080 and learning rate 1e-6.
  Iteration 64 loaded successfully, and the restored weight checks cover all three rollout engines at version 65.
  The producer was active with 96 in-flight samples, three running groups and zero completed buffer groups.
  `optimizer-and-rollout-bootstrap-probe.json` preserves the snapshot. Separate MTP norms were unset before any new update.
- First published update 65 is finite: reward 4.5625/7, 14/32 caps, gradient norm 0.16368, full entropy 0.51334,
  sampling entropy 0.42736, sampled initialization divergence 0.0067501, rollout mismatch 0.00027755,
  effective sample-size ratio 0.999445, policy clipping zero and positive surrogate gain 0.00009912.
  This is a numerical startup check, not a learning trend. Its communication-protocol group took 902.7 seconds to generate.
  The raw batch has 17 sevens, three sixes, one five, one four and ten zeros. A seed-selected proof is preserved
  in `first-batch-provisional-raw-review.json`, pending its immutable batch-archive readback.
- Training/controller source **`1620b3a0569f9192696406d51332b513f3d793e0`**.
- Frozen configuration `rl_easy_overfit_v15.json`, unchanged judge configuration `judge_group_v5_followthrough_repair.json`.
- Resume the original iteration-64 checkpoint, containing 65 updates. Target new global updates 65–104.
- Replacement limit **3h43m on eight MI350 GPUs**, no automatic Slurm requeue.
  `slurm-budget-guard.json` records the owner/name check and `Requeue=0` readback.
- At 20:51 UTC the owned pending job accepted `TimeMin=02:00:00`, with its maximum unchanged at 3h43m.
  See the [scheduling amendment](SCHEDULING_REPAIR_20261122.md) and `slurm-backfill-minimum-time.json` before/after readbacks.
  Slurm may grant a shorter slot. Read its actual time limit on allocation, and charge every continuation to the same remaining budget.
- The failed predecessor used 16m36s. The combined ceiling is 31.9467 GPU-hours, below the approved 32.
- Judge ledger inherits **552 settled calls, $0.6374188 estimated spend, zero pending reservations**, within the same $5 cap.
  Earlier qualification, proof audits and failed-run calls remain included. Do not reset the ledger.
- All 60 startup reference/empty/irrelevant controls passed. Their cached receipts left the inherited ledger byte-for-byte unchanged.
- Protocol 3 sends exact extracted final answers. Full raw thinking remains in generated artifacts.
  The exact failed 4,671,119-byte sample now produces a 133-byte request and the same cached paired zero verdicts.
- Models, questions, solver and judge prompts, rubric weights, seed, sampling, optimization and GPU layout remain fixed.
  FOCS/STOC 2026 stay sealed. Do not claim a changed grading scale as learning.
- The actual qualified `log_probs_backend` is **`torch`**, with chunk size 512. Bounded sampling requires this path.
  Do not infer a fused-head backend from Ian's example or the separate tiled-head benchmark.

## Paths and owned processes

- VM run: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-overfit-v15`.
- VM frozen checkout: `/home/ubuntu/.cache/miles/tcs-async-rl/checkouts/easy-overfit-controller-v15`.
- VM persistent controller/service Python: `/home/ubuntu/.cache/miles/tcs-async-rl/service-env-v15/bin/python`, aiohttp 3.14.5.
- AMD run: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v15`.
- AMD frozen checkout: `/work1/aviralkumar/geney/tcs-async-rl/checkouts/easy-overfit-v15`.
- Resume root: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v12/checkpoints`.
- Gate source remains the saved `easy-qualification-v14` regrade of original generation job 535067.
  No new solver qualification GPU job ran. Dataset bundle SHA is `18ed636017d16c3af5c1448ee4967b4aa6b996037c73f5304bf4ce95469e5929`.

Read each receipt and `/proc` before managing these session-owned processes.

| Role | PID | Receipt/log in the VM run |
|---|---:|---|
| Controller | 1234498 | `controller-process.json`, `controller.log`, `handoff-state.json` |
| Judge service | 1234500 | `reward-service-process.json`, `reward-service.log` |
| Independent native metrics observer | 1234737 | `metrics-observer-process.json`, `metrics-observer.json` |
| GPU telemetry | 1234738 | `gpu-telemetry-process.json`, `gpu-telemetry.jsonl` |
| Hourly review | 1234739 | `hourly-review-process.json`, `latest-detailed-review.json` |

The controller owns the reward-tunnel watcher. Never start a second `auto_rl` against an existing submission receipt.
Provider credentials remain in the authorized VM service. Do not inspect or transfer them.
The verified running-node reverse tunnel uses PID **1250672**, recorded in `reward-tunnel-state.json`.

## Startup, learning checks and closure

Verify iteration 64, scheduler counter 2,080, constant learning rate 1e-6 and initial restored weight publication.
Expected final scheduler counter is 3,360. New native optimizer points must begin at 65, filtering inherited event history.
For read-only Ray probes, verify allocation owner/name first, bind the same run runtime, set the actual `RAY_ADDRESS`,
and omit `--pid` so Ray can discover the existing raylet. Never run a forward pass or change the producer without cause.
Separate MTP norms are probe snapshots, not continuous native scalar history.

Inspect finite losses/gradients, full and sampling entropy, initialization drift, rollout mismatch, ratios/effective sample size,
clipping, learning rate, update direction, publication hashes, staleness, answer lengths/caps and judge discrepancies.
Fixed-batch update-direction observations are scheduled at 65, 66, 70, 80, 90 and 100. Zero gain is possible for constant rewards.
Check actual seeded proofs and matched question visits at least hourly. `baseline-groups.json` is deliberately empty.
Use `audit_live.py`, `summarize_continuation.py`, `plot_continuation.py` and `judge_signal.py` on fresh diagnostic outputs.
All comparisons remain descriptive, single-seed training observations. Judge calibration uses assistant labels, without human accuracy.

The W&B writer is the sole locked VM native publisher under `geney/tcs-async-rl` in organization `geney-org`.
Its actual run ID is unknown before allocation. Verify the URL and all native scalar coordinates after the first complete update.
Do not sync the independent AMD offline histories into the online run. Preserve them privately.
Watch service/tunnel health, monitor heartbeats and disk free bytes as well as the GPU job.
Ten-minute archive throttling does not change minute-level native checks.

At termination, drain provider requests, inspect checkpoints, finish archival and exact immutable readbacks,
verify W&B's actual terminal state and all native scalar coordinates, then stop owned observers/services.
If the bounded run does not show reliable improvement, report the evidence and needed decision. Do not invent a success or enlarge budgets.

Session `01a1ed24-1032-7731-910b-824f484f5f71`, home VM, original directory `/home/ubuntu/miles`.
Workstream `tcs-async-rl-recon`. Failed-run closure and causal validation remain in the linked repair document and private evidence.
