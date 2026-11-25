# All236 continuation operations

This file records execution. The [approved plan](EASY_MEDIUM_236_PLAN_20261125.md) owns the scientific setup.

- AMD job549728, `tcs-easy-medium236-v1`, node`k007-003`, eight GPUs on`mi3508x`.
- Submitted2026-11-25T13:26:51.891616Z. Started13:26:52Z,08:26:52 EST.
- Source commit`6f9279f25b19eb47f0ddd435859b279cc88fc17c`.
- Frozen config`rl_easy_medium_236_v1.json`. All20 easy and216 medium questions, without score filtering.
- Resume iteration108, preserving109 historical updates. New optimizer coordinates begin109.
- Hard allocation limit12h. Request final save at elapsed11.5h, shared judge spending/reservations$22, or unsafe storage.
- The shared$25 judge cap includes$2.2082024 settled before RL. No budget reset.
- VM run`/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-medium-236-v1`.
- AMD run`/work1/aviralkumar/geney/tcs-async-rl/runs/easy-medium-236-v1`.
- Detached AMD checkout`/work1/aviralkumar/geney/tcs-async-rl/checkouts/easy-medium-236-v1`.
- Initial VM supervisor PID1578963. The current identity is in`supervisor-process.json`. Session`01a1ed24-1032-7731-910b-824f484f5f71`.
- Supervisor owns the reward service, tunnel and resource guard. Its process receipts list exact child identities.
- Online W&B ID`261125-132805-437-native`, created after the first verified complete optimizer row.
- Expected [W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/261125-132805-437-native).
  This link does not prove that history exists. Readback verification remains required.

Both startup publications matched the source iteration108 on every one of three engines.
Native arguments and reward tunnel passed. The first sysfs snapshot showed six GPUs at100% and two at0% before backward training.
Full receipts are [startup](evidence/long236_startup_20261125.json) and [GPU snapshot](evidence/long236_gpu_startup_20261125.json).
`rocm-smi` was unavailable at the initially attempted host path. The read-only sysfs probe runs inside the existing allocation.
The older `gpu_telemetry` interface is preserved. New snapshots use`allocation_gpu_snapshot` and`gpu-allocation-snapshots.jsonl`.

The baseline regrade uses the existing live service and its sole budget ledger, with two client requests at most.
The [frozen baseline plan](evidence/long236_baseline_regrade_plan_20261125.json) specifies exact-request reuse and unchanged generation provenance.
Its output is`/home/ubuntu/.cache/miles/tcs-async-rl/runs/checkpoint108-screen-v5-baseline`.
It produces no solver generation or optimizer update. Original validation receipts keep their original identities and bytes.

The live run stores seeded raw-proof reviews under`live-proof-reviews/`.
Early easy and medium examples were inspected against their original items. Both scored7 under the accepted prompt.
These are assistant diagnostic reviews, not independent human calibration or proof that corpus-wide judging is accurate.
Minute-level checks cover both job and supervisor, actual updates, finite metrics, publications, service health and space.
Foreground supervision continues through terminal state, checkpoint inspection and immutable HF/W&B verification.
The12-hour cap may permit fewer than236 distinct questions to be consumed. Report actual coverage separately from prepared corpus size.

Finalization repair,2026-11-25: the VM publisher now waits for outstanding judge attempt receipts and60 seconds of ledger quiet before closing W&B. The supervisor retries while retaining its scorer. Completed provider failures retain conservative cost reservations. This reporting change does not restart the live service, alter the frozen GPU checkout, or change grading. Behavioral tests cover pending calls, completed failure receipts, retry quiet time, and closure without duplicate optimizer rows.

Checkpoint reserve repair,2026-11-25: update115 took32.30 minutes in the trainer and1.07 minutes waiting. The original30-minute reserve is insufficient for that measured cycle. The separate`operational-save-reserve.json` now requests the final save at elapsed11h15m, reserving45 minutes within the unchanged12-hour allocation. The submitted config and frozen GPU checkout remain byte-identical. The VM supervisor and guard will reload this operational policy while adopting the same running scorer and tunnel. See[evidence](evidence/long236_operational_save_reserve_20261125.json).

Saved-support diagnostic: `rollout_support` reads already archived consumed batches. It bins each response into ten relative-position intervals and counts singleton sampling shortlists, admission of the closing thinking token, and sampled surprisal. It records exact trace and tokenizer hashes. These are token-weighted descriptive diagnostics conditioned on observed completion status, not full-policy entropy estimates or controlled learning comparisons. No model generation, grader call, reward change or training setting changes. The next hourly review will inspect the new batches before evicting their verified replay copies.

VM cache recovery,18:42Z:24,703 raw order/attempt receipt copies from five terminal older runs were removed after fresh immutable HF hash verification. The original remote objects, local scores, ledgers, inputs, manifests and seeded examples remain. Each old judging directory now has a restoration marker. Restore exact receipts before reusing those old diagnostics. See[evidence](evidence/long236_vm_old_judge_cache_20261125.json). The active service cache and protected AMD checkpoints were not evicted.
