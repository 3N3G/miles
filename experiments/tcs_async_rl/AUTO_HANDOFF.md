# Automatic medium-to-RL handoff

The user requested automatic submission and monitoring after the approved medium diagnostic.
This implements the existing two-update authorization in `RL_SMOKE_PLAN.md`.
It does not select a corpus training mixture or change any rubric, model, prompt or sampling setting.

## Conditional snapshot

`rl_smoke_v7_medium.json` binds the candidate to the exact medium input and its qualified grade file.
It inherits the same target-only sampling profile and trainer topology as v6.
Only run identity, diagnostic provenance, the dedicated VM reward port and the selected W&B team differ.
Old snapshots remain unchanged. Audit events now live outside excluded checkpoint directories.

`auto_rl.py` runs on the VM from a frozen Git checkout.
It waits for the existing medium watcher to finish grading and archiving job 535170.
It can restart a stopped diagnostic watcher after acquiring that watcher's lock.
It never replaces a live grader or touches another Slurm job.

## Submission gate

- Require successful inference, grading and private result upload.
- Check the exact input hash, model judge/effort, sampling settings, generation indices and native support control.
- Require all 32 nonempty final answers, finite real grades, passing 7/0/0 controls and at least two reward values.
- Check both presentation-order verdicts against the summary. Stop on a paired-reference anomaly.
- Start the question-specific VM reward service and pass actual reference, empty and irrelevant controls before submission.

The recorded-cap question in `QUESTIONS.md` remains unresolved.
Capped missing finals do not silently become diagnostic zero scores.
A failed gate stops the handoff. It does not fall back to easy, draw another question or change training weights.

## One submission and monitoring

The controller pulls AMD source, creates a detached checkout at its frozen source revision,
checks disk capacity, and renders the existing Python Slurm launcher.
`rl_submission.py` writes an intent before calling `sbatch` and a receipt after acceptance.
An existing receipt is reused. An unresolved intent requires scheduler reconciliation, preventing duplicate submissions.
The helper revalidates source, private input and the complete reward gate on AMD before submission.

`watch_rl.py` polls Slurm, copies logs without checkpoints or PyTorch files, and keeps the question-specific service and tunnel alive.
It can restart its own stopped service or tunnel. SIGTERM cleanup also closes the tunnel's own SSH client.
It records the actual optimizer-step evidence, finite metrics, checkpoint inventory and exported consumed groups.
It deduplicates rank logs and preserves exact per-token weight-version spans in the portable artifacts.
It generates reward/token-count plots and seed-selected raw examples, uploads results privately, and syncs offline W&B.
The native initial weight-equality assertion and later weight-publication calls remain enabled.
Novel training failures are preserved for diagnosis under the standing repair authorization.
The controller does not invent training fixes or spend further GPU attempts automatically.

Validation: 65 fast TCS tests passed, including failed-gate blocking and uncertain-submission idempotence.
The existing native offline logging control also uploaded successfully to the selected W&B project.
See `evidence/wandb_sync_control_20261120.json` for its API readback.
The process manifest and current `handoff-state.json` live in the private RL run directory.
The job registry records an RL allocation only after an actual scheduler receipt exists.
