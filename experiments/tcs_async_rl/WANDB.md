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

The current AMD runtime plan saves offline W&B files.
Credentials stay on the VM. Upload each completed run explicitly from the VM:

```bash
uvx --from wandb==0.31.0 wandb sync --entity geney --project tcs-async-rl <offline-run-directory>
```

Every fresh RL snapshot must set `wandb_team=geney` and `wandb_project=tcs-async-rl`.
The native adapter passes `wandb_team` as W&B's `entity` argument.
Keep historical snapshots unchanged. No new question-bound RL snapshot has been selected while medium is pending.
The existing native offline logging control was uploaded and read back successfully.
Its [run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/offline-logging-control-v1) contains the actual control metric at one.
See `evidence/wandb_sync_control_20261120.json`. This control is not an RL training run.
Record each RL run's actual uploaded URL after synchronization.
