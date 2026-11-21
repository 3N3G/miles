---
paths:
  - "experiments/tcs_async_rl/**"
  - "tests/fast/test_tcs*.py"
---

# TCS experiment supervision

The user requested supervision after a run reached its time limit and its monitor failed during finalization.

- Keep foreground supervision through the scheduler's terminal state, checkpoint inspection, result upload and W&B verification.
  A successful first optimizer step does not complete supervision. A background poller does not diagnose unknown training failures.
- Check both the GPU job and its monitor: native optimizer progress, finite metrics, weight publication, service/tunnel health,
  disk space and the monitor heartbeat. Investigate failures within the already approved repair scope.
- Verify W&B reward and timing coordinates against native optimizer rollout IDs, as well as training-point counts.
  Multiple offline writers with one run ID have independent history counters. Follow the single-writer recovery in
  `experiments/tcs_async_rl/WANDB.md` before interpreting merged charts.
- Distinguish a training exception, the approved walltime limit, and a failed reporting/upload process.
  Reaching the time limit does not authorize another full GPU allocation or a larger experiment budget.
- Preserve used configurations and results. Retry recoverable archive/storage failures without losing the terminal scheduler state.
  Recovery must work after an abnormal monitor exit, without relying on a clean completion marker.
- Keep transient Ray/dashboard caches on AMD. Evict a VM result copy only after a successful immutable upload and checksum verification.
  Claim final completion only after the result report and uploads are verified. Report remaining work explicitly.
