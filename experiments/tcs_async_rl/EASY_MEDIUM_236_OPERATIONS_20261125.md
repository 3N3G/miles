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

Repetition diagnostic: `rollout_repetition` verifies immutable trace hashes and separately measures exact20-word window repetition in the full response and extracted final answer. It distinguishes model stopping, caps with final text, and caps without final text. A hand-computed three-answer fixture validates the denominators and separation of thinking from final text. This is an offline descriptive check with no new generation or judging.

## Bounded inference repair after the first allocation failed

Job549728 ended FAILED1:0 at20:25:55Z after25,143 allocation seconds. All19 observed updates109–127 were finite.
Checkpoint124 preserves16 updates. Unsaved125–127 remain failed-attempt evidence and are excluded from the durable lineage.
W&B readback verified19 rows and1762 exact scalar values. The archive is
`3N3G/internalization-runs@1e1d18a3613dcb38557b34921d7381115480c285/miles/easy-medium-236-v1`.

The failed question has seven problem tasks but six weighted rubric items. Luna split the final rubric item into two decisions.
Three malformed replies exhausted the parser retries. The service returned502 and terminated the async producer.
The affected identities are`online-5e6e89467583f8516b866293` and`online-6d22296e0d3e1a0d9bbf8cf5`.
Question key:`fd32b919a83950d5b7358c6a4864a3725d7d8283c8b08c04e059c5240b49e040`.

Repair the inference output format only after the original three attempts are unusable.
Append an`output_contract` to the existing user JSON. It lists original rubric item IDs and weights and says:
“Return exactly one candidate decision for each scored rubric item listed here. Numbered tasks in the problem are not additional scored rubric items.”
Keep the v5 system prompt, original problem, reference, candidate, rubric, weights, model and reasoning effort unchanged.
Allow at most three additional attempts. Preserve raw replies and both request fingerprints.
Never alter malformed verdicts into scores, substitute zero, or retry an existing valid grade.
Validate the two failed candidates and reuse a valid cached control without a new request before submission.
The auditor verifies every failed attempt and the exact clarification before accepting its provenance.
CPU tests pass for both format/length retry orders, a one-permit semaphore, cache reuse and bounded exhaustion.

Resume iteration124 in a new`easy-medium-236-v2` run. Restore saved rollout/data state because the corpus is unchanged.
The checkpoint contains data-source, fully asynchronous queue and executor state. Preserve source checkpoints108 and124.
Use a05:00:00 allocation:18,000 seconds plus25,143 already used is43,143 seconds, below the43,200-second limit.
All scientific hyperparameters remain unchanged. Save every update for recovery, retain one new checkpoint and reserve45 minutes for final saving.
Transfer the exact settled/reserved shared ledger only after the old supervisor and scorer exit. Do not reset the$25 cap.
Continue the original minute checks and hourly proof reviews through finalization, then the approved same12-question final screen.
