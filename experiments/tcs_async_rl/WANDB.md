# W&B destination for TCS runs

The user selected organization `geney-org` on 2026-11-20.
Its accessible project-owning team is `geney`.
The verified project is [geney/tcs-async-rl](https://forge.coreweave.com/wandb/geney/tcs-async-rl/workspace).
Project creation under `geney-org` failed. Creation under `geney` succeeded.
The SDK independently resolved the `geney` team to organization `geney-org`.
Exact metadata checks are in `evidence/wandb_destination_20261120.json`.

The VM login is authenticated. This checkout's local W&B defaults now select that team and project.
The account's global default may still name `iris-sobolmark`. It does not choose this repo's upload destination.
No old runs were moved. No experiment run or GPU job was created during this setup.

AMD saves offline W&B files. Credentials stay on the VM.
Historical configurations upload each completed run explicitly from the VM:

```bash
uvx --from wandb==0.31.0 wandb sync --entity geney --project tcs-async-rl <offline-run-directory>
```

Every fresh RL snapshot must set `wandb_team=geney` and `wandb_project=tcs-async-rl`.
The native adapter passes `wandb_team` as W&B's `entity` argument.
Keep historical snapshots unchanged.
The existing native offline logging control was uploaded and read back successfully.
Its [run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/offline-logging-control-v1) contains the actual control metric at one.
See `evidence/wandb_sync_control_20261120.json`. This control is not an RL training run.
Record each RL run's actual uploaded URL after synchronization.

## Verify offline metric axes

The easy-only audit found three offline writers using the same run ID with independent history counters.
Their uploads misassigned some `rollout/episode_raw_reward` coordinates, despite preserving all actor training points.
Count checks alone do not validate a chart. Native event JSON owns the optimizer rollout IDs.

For an already-terminal failed or finished run, append corrected metrics from one VM writer:

```bash
uv run --no-project --with wandb==0.31.0 python -m experiments.tcs_async_rl.native_metrics \
  --run-directory <original-run-directory> --output <separate-audit-directory>
```

This command resumes the existing ordinary run. CoreWeave rejects changing a run with history to shared mode.
It preserves original metrics and the terminal state. It verifies corrected reward points by API readback.
The verified series is `audit/rollout/episode_raw_reward` against `audit/optimizer_step`.
Other corrected native metrics use the same `audit/` prefix and explicit axis.
See [the full reward and optimizer audit](RL_NOISE_AUDIT_20261121.md).

## Publish new runs from one live writer

New configurations can set `wandb_metrics_source` to `native_vm`.
The monitor publishes complete native optimizer rows from one VM client during training.
Its online run ID appends `-native` to the allocation's producer run ID.
Checkpoint restoration also restores old event files. The approved resume iteration filters out these inherited rows.
It waits for optimizer and reward metrics plus the driver's actual three-engine weight-check event before publishing an active step.
Actor timing metrics can describe the preceding transfer. They do not prove the current transfer finished, especially during a save.
It uses the same `audit/` series and explicit optimizer-step axis described above.

The client refreshes its serialized SDK heartbeat at least every two minutes, even without a new optimizer point.
A long rollout gap otherwise lets the server mark an active GPU job as crashed. Heartbeats append no history rows.
The delivery acknowledgement owns the heartbeat time independently of the verified-history cursor.
The client closes the run only after Slurm ends.
It verifies reward coordinates and run state through fresh API reads before advancing its local cursor.
History indexing can lag a successful SDK upload. Reads bypass the SDK history cache.
The client records upload acknowledgement separately from verified history. Monitor retries do not resend acknowledged points.
After a lost acknowledgement, an identical repeated point is accepted. Conflicting values at one step are errors.

For these configurations, do not also sync the independent AMD offline histories into the online run.
Keep those files in the private result archive. The monitor handles active and terminal publication automatically.
The observer adds no provider credentials to AMD. Transient W&B client files stay outside each result directory.
