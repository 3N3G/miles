"""Read one metric row per optimizer update from native events, before W&B merging.

Offline actor and rollout processes own separate W&B history counters. Syncing
those streams into one run can misassign the shared rollout/step field. Native
events carry the optimizer rollout ID independently of those history counters.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def read_native_metrics(directory):
    rows = {}
    sources = ["actor_cell00000_rank00000.jsonl", "rollout_executor.jsonl"]
    for name in sources:
        for line in (directory / "events" / name).read_text().splitlines():
            event = json.loads(line)
            if event["type"] != "metric":
                continue
            step = event["rollout_id"]
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
    if not observed or observed != list(range(len(observed))):
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
        client_directory = output / ".cache/wandb-client"
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
        observed = list(after.scan_history(keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"]))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(sync_terminal_audit(cli.run_directory, cli.output), indent=2))


if __name__ == "__main__":
    main()
