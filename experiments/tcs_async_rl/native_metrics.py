"""Read one metric row per optimizer update from native events, before W&B merging.

Offline actor and rollout processes own separate W&B history counters. Syncing
those streams into one run can misassign the shared rollout/step field. Native
events carry the optimizer rollout ID independently of those history counters.
"""

import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.rl_smoke import initial_optimizer_updates


def read_native_metrics(directory, *, allow_pending=False, require_publication=False):
    # Restoring a checkpoint also restores its historical event files. The
    # reviewed checkpoint owns the boundary between inherited and new updates.
    config_path = directory / "approved-config.json"
    initial = initial_optimizer_updates(json.loads(config_path.read_text())) if config_path.exists() else 0
    rows = {}
    sources = ["actor_cell00000_rank00000.jsonl", "rollout_executor.jsonl"]
    for name in sources:
        path = directory / "events" / name
        if allow_pending and not path.exists():
            continue
        lines = path.read_text().splitlines()
        for position, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if allow_pending and position == len(lines) - 1:
                    continue
                raise
            if event["type"] != "metric":
                continue
            step = event["rollout_id"]
            if step < initial:
                continue
            metrics = event["metrics"]
            row = rows.setdefault(step, {"optimizer_step": step})
            for key, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite native metric at step {step}: {key}")
                if key.endswith("/step") and value != step:
                    raise ValueError(f"Native metric axis disagrees at step {step}: {key}")
                if key in row and row[key] != value:
                    raise ValueError(f"Conflicting native metric at step {step}: {key}")
                row[key] = value
    observed = sorted(rows)
    if allow_pending:
        complete = []
        required = {"train/pg_loss", "rollout/episode_raw_reward"}
        if require_publication:
            required.add("perf/update_weights_time")
        for step in observed:
            if not required <= rows[step].keys():
                break
            complete.append(step)
        observed = complete
        if not observed:
            return []
    if not observed or observed != list(range(observed[0], observed[-1] + 1)):
        raise ValueError("Native optimizer steps are missing or noncontiguous")
    for step in observed:
        if "train/pg_loss" not in rows[step] or "rollout/episode_raw_reward" not in rows[step]:
            raise ValueError(f"Missing optimizer or rollout metrics at step {step}")
    return [rows[step] for step in observed]


def compare_wandb(native, history, *, metric, step_key):
    expected = {(row["optimizer_step"], row[metric]) for row in native}
    observed = [(row.get(step_key), row[metric]) for row in history if row.get(metric) is not None]
    return {
        "metric": metric,
        "native_points": len(expected),
        "wandb_points": len(observed),
        "missing_or_misassigned_native_steps": sorted(
            step for step, value in expected if (step, value) not in observed
        ),
        "misassigned_points": [
            {"step": step, "value": value} for step, value in observed if (step, value) not in expected
        ],
    }


def publish_audit(native, run):
    """Append corrected, separately named metrics through one VM writer.

    The caller owns authentication and preserves the scheduler's terminal state.
    Never overwrite the original producer series or reuse their history counters.
    """
    run.define_metric("audit/optimizer_step")
    run.define_metric("audit/*", step_metric="audit/optimizer_step")
    for row in native:
        run.log({"audit/" + key: value for key, value in row.items()})


def sync_terminal_audit(directory, output):
    # W&B is optional for local evidence reads and CPU tests. Only this VM
    # publishing command loads the SDK and its normal existing authentication.
    import wandb

    config = json.loads((directory / "resolved-config.json").read_text())
    args = config["args"]
    run_path = f"{args['wandb_team']}/{args['wandb_project']}/{config['run_id']}"
    before = wandb.Api().run(run_path)
    if before.state not in ("failed", "finished"):
        raise ValueError("This recovery command requires a failed or finished W&B run whose state it can preserve")
    native = read_native_metrics(directory)
    digest = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    if before.summary.get("audit/native_metrics_sha256") != digest:
        client_directory = output.parent / ".wandb-clients" / output.name
        client_directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            entity=args["wandb_team"],
            project=args["wandb_project"],
            id=config["run_id"],
            resume="must",
            dir=str(client_directory),
            # CoreWeave rejects switching an existing ordinary offline history
            # to shared mode (HTTP 409). Resume one writer at the history tail.
            settings=wandb.Settings(mode="online"),
        )
        try:
            publish_audit(native, run)
            run.summary["audit/native_metrics_sha256"] = digest
            run.summary["audit/source"] = "Native event JSON, joined by optimizer rollout ID through one VM writer"
        finally:
            run.finish(exit_code=0 if before.state == "finished" else 1)
    expected = [(row["optimizer_step"], row["rollout/episode_raw_reward"]) for row in native]
    for attempt in range(6):
        # History indexing can lag a successful finish. Retry fresh reads,
        # without resending points or declaring an unverified upload successful.
        after = wandb.Api().run(run_path)
        observed = list(
            after.scan_history(keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"], use_cache=False)
        )
        actual = [
            (row["audit/optimizer_step"], row["audit/rollout/episode_raw_reward"])
            for row in observed
            if row.get("audit/optimizer_step") is not None and row.get("audit/rollout/episode_raw_reward") is not None
        ]
        if actual == expected and after.state == before.state:
            break
        if attempt < 5:
            time.sleep(2)
    else:
        raise RuntimeError("Corrected W&B history or original terminal state failed readback verification")
    receipt = {
        "url": after.url,
        "state_before": before.state,
        "state_after": after.state,
        "native_metrics_sha256": digest,
        "verified_points": len(actual),
        "reward_metric": "audit/rollout/episode_raw_reward",
        "x_axis": "audit/optimizer_step",
    }
    (output / "wandb-corrected-metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def sync_live_audit(directory, output, scheduler_state):
    """Publish only complete native steps, preserving live W&B run state.

    The VM is the sole online writer. AMD's independent offline streams stay
    archived without being merged into this history. A local lock serializes
    monitor retries. API readback owns the publication cursor.
    """
    import wandb

    config_path = directory / "resolved-config.json"
    terminal = scheduler_state not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "UNKNOWN")
    native = read_native_metrics(directory, allow_pending=True, require_publication=not terminal)
    if not config_path.exists() or not native:
        return {"phase": "waiting_for_first_complete_update", "verified_points": 0}
    config = json.loads(config_path.read_text())
    args = config["args"]
    # Keep the single online writer separate from AMD's offline producer IDs.
    online_id = config["run_id"] + "-native"
    path = f"{args['wandb_team']}/{args['wandb_project']}/{online_id}"
    receipt_path = output / "wandb-live-metrics.json"
    submitted_path = output / "wandb-submitted-metrics.json"
    with (output / "wandb-live-metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        submitted = json.loads(submitted_path.read_text()) if submitted_path.exists() else {}
        last_verified = previous.get("last_verified_optimizer_step", native[0]["optimizer_step"] - 1)
        last = max(last_verified, submitted.get("last_submitted_optimizer_step", last_verified))
        pending = [row for row in native if row["optimizer_step"] > last]
        if not pending and not terminal and last_verified == last:
            return previous
        if pending or terminal:
            client_directory = output.parent / ".wandb-clients" / output.name
            client_directory.mkdir(parents=True, exist_ok=True)
            run = wandb.init(
                entity=args["wandb_team"],
                project=args["wandb_project"],
                id=online_id,
                resume="allow",
                dir=str(client_directory),
                config={
                    "native_metrics_source": "VM single writer",
                    "approved_config": str(directory / "approved-config.json"),
                },
                settings=wandb.Settings(mode="online", x_update_finish_state=terminal),
            )
            if (previous or submitted) and not run.resumed:
                run.finish(exit_code=1)
                raise RuntimeError("Live metric publication did not resume its existing run")
            try:
                publish_audit(pending, run)
                run.summary["monitor/slurm_state"] = scheduler_state
                run.summary["monitor/completed_updates"] = len(native)
            finally:
                run.finish(exit_code=0 if not terminal or scheduler_state == "COMPLETED" else 1)
            # SDK success and API verification are separate acknowledgements.
            # Persist delivery before waiting for indexing, so monitor retries
            # do not resend a successful upload whose readback was delayed.
            _atomic_bytes(
                submitted_path,
                (json.dumps({"last_submitted_optimizer_step": native[-1]["optimizer_step"]}) + "\n").encode(),
            )
        expected = [(row["optimizer_step"], row["rollout/episode_raw_reward"]) for row in native]
        expected_state = ("finished" if scheduler_state == "COMPLETED" else "failed") if terminal else "running"
        # The server's history index can lag SDK upload by tens of seconds.
        # Keep one upload and retry reads instead of republishing its points.
        for attempt in range(12):
            remote = wandb.Api().run(path)
            observed = list(
                remote.scan_history(keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"], use_cache=False)
            )
            actual = [
                (row["audit/optimizer_step"], row["audit/rollout/episode_raw_reward"])
                for row in observed
                if row.get("audit/optimizer_step") is not None
                and row.get("audit/rollout/episode_raw_reward") is not None
            ]
            # A retry after a lost acknowledgement may repeat an identical
            # point. Verify unique coordinates and reject conflicting values.
            observed_values = {}
            for step, value in actual:
                if step in observed_values and observed_values[step] != value:
                    raise RuntimeError("Conflicting W&B values at one optimizer step")
                observed_values[step] = value
            if sorted(observed_values.items()) == expected and remote.state == expected_state:
                break
            if attempt < 11:
                time.sleep(3)
        else:
            raise RuntimeError(
                f"Live W&B readback failed: {len(observed_values)}/{len(expected)} steps, "
                f"state {remote.state!r}, expected {expected_state!r}"
            )
        receipt = {
            "url": remote.url,
            "verified_points": len(observed_values),
            "history_rows": len(actual),
            "last_verified_optimizer_step": native[-1]["optimizer_step"],
            "scheduler_state": scheduler_state,
            "wandb_state": remote.state,
            "native_metrics_sha256": hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest(),
        }
        _atomic_bytes(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode())
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-state")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    result = (
        sync_live_audit(cli.run_directory, cli.output, cli.scheduler_state)
        if cli.scheduler_state
        else sync_terminal_audit(cli.run_directory, cli.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
