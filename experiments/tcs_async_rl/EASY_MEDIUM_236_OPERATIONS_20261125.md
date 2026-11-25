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
- VM supervisor PID1578963. Session`01a1ed24-1032-7731-910b-824f484f5f71`.
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
