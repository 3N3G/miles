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

During an active job, an acknowledged row can remain unindexed beyond one bounded readback window.
The publisher reports `waiting_for_history_index` when indexed rows form the expected prefix and all previously verified rows remain.
It retains the delivery receipt and verified cursor, then checks again on the next poll without resending the row.
Conflicting values, loss of previously verified rows and incomplete terminal history remain errors.

For these configurations, do not also sync the independent AMD offline histories into the online run.
Keep those files in the private result archive. The monitor handles active and terminal publication automatically.
The observer adds no provider credentials to AMD. Transient W&B client files stay outside each result directory.

The online client's Config includes `training` with the actual safe launcher arguments and a resolved-config SHA256.
This includes learning rate, optimizer settings, global batch, answers per problem and dynamic token-batch settings.
`native_metrics.training_log_config` owns the positive list. Authentication fields and service URLs are excluded.
This publication does not rewrite historical run configurations.

New runs also freeze `judge-system-prompt.txt` and `judge-request-template.json` before training.
The online Config contains that exact prompt, checked against the resolved SHA256.
It includes model/data revisions, held-out cohorts, source checkpoint, code/container identity and runtime versions.
Safe parser-validated arguments include defaults such as gradient clipping and entropy coefficients.
`training_arguments_source` names the argument snapshot. Older runs without it use their resolved launcher arguments.
`run_setup.publish_run_setup` prints the full setup once to the online client's console.
It attaches `setup/RUN_SETUP.md`, `setup/training-hyperparameters.json` and the two judge files to W&B Files.
It never adds metric history. Heartbeats do not repeat the printout or file publication.
Full generated answers, extracted finals, per-item judge decisions and provider receipts remain in the private HF run archive.

The [review card](FINAL_JUDGE_SETUP_20261125.md) shows the accepted final prompt and planned continuation settings.
It distinguishes the completed judge validation from the not-yet-submitted training continuation.

New trainer processes emit `train/num_microbatches` from the actual schedule for each optimizer update.
The single writer publishes it as `audit/train/num_microbatches` on `audit/optimizer_step`.
With dynamic token batching, this is the accumulation count. Config does not invent one fixed count for the run.

## Key charts and summary status

The selected [personal workspace](https://forge.coreweave.com/wandb/geney/tcs-async-rl/workspace) now opens these sections:

- **Return:** mean raw rubric reward, on its original zero-to-seven scale.
- **Policy health:** full/sampling entropy, sampled initialization divergence, trainer–rollout mismatch and gradient norm.
- **Generation:** response length and the fraction reaching the token cap.
- **Run status:** summary cards for total saved updates, new saved updates and shared judge spend.

The complete automatic `audit` section remains available below, collapsed by default.
Every key time-series panel explicitly uses `audit/optimizer_step`. The audit section also selects that axis.
The original workspace filters, run selections and unrelated panel definitions are preserved.

`monitor/*` contains final/current summary values, not optimizer history. Automatic line plots of those fields have no points,
including when `_step` is selected. Summary cards read these fields directly.
The sole live writer derives saved-update cards from completed checkpoint-retention receipts.
It separates settled judge charges from pending reservations in the shared ledger.
These status updates append no optimizer history rows.
`_step` exists for training history, but counts publisher rows; optimizer updates are the research coordinate.
The raw rubric reward is the displayed return. Group-centered GRPO rewards/advantages average near zero and are unsuitable for this chart.
See [W&B's custom-axis documentation](https://docs.coreweave.com/products/wandb/track/log/customize-logging-axes).

`wandb_dashboard.py` prepares a reversible workspace proposal, then applies it with an unchanged-history check:

```bash
uv run --no-project --with wandb==0.31.0 --with wandb-workspaces==0.4.13 \
  python -m experiments.tcs_async_rl.wandb_dashboard \
  --entity geney --project tcs-async-rl --view-name nw-nwusergenejinyang-w \
  --run-id 261123-010857-293-native --output <new-backup-directory>
```

Inspect `view-before.json`, `spec-proposed.json` and `proposal-verification.json`.
Repeat with `--apply` and the same directory. A concurrent workspace edit blocks an outdated proposal.
The helper checks saved layout readback, aligned chart data, unchanged run history and unchanged terminal state.
It never calls `wandb.init`, publishes a metric or submits a training job.

For the all236-question continuation, add `--difficulty-breakdown` to the proposal and apply commands.
This adds separate easy/medium raw returns, corpus coverage and constant-reward group plots.
Wait until both difficulties have verified optimizer points before applying that proposal.
Sparse difficulty series contain points only when an update consumed that difficulty.
