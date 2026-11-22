"""Keep native progress and W&B current when a private artifact upload stalls.

This observer never submits jobs, manages reward services, or evicts artifacts.
The existing W&B file lock serializes it with the main run supervisor.
"""

import argparse
import datetime
import fcntl
import json
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state
from experiments.tcs_async_rl.watch_rl import sync_native_wandb


def observe(job_id, remote_directory, directory):
    state = remote_state(job_id)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--include=/events/",
            "--include=/events/*.jsonl",
            "--include=/rl-smoke.log",
            "--exclude=*",
            f"amd:{remote_directory}/",
            f"{directory}/",
        ],
        check=True,
        timeout=60,
    )
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    row = {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "job_id": job_id,
        "scheduler": state,
        "completed_published_updates": len(native),
        "last_optimizer_step": native[-1]["optimizer_step"] if native else None,
    }
    # Preserve a fresh local heartbeat even if the W&B endpoint times out.
    heartbeat = directory / "metrics-observer.json"
    _atomic_bytes(heartbeat, (json.dumps(row, indent=2) + "\n").encode())
    row["wandb"] = sync_native_wandb(directory, state["state"])
    _atomic_bytes(heartbeat, (json.dumps(row, indent=2) + "\n").encode())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--remote-run-directory", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    cli = parser.parse_args()
    with (cli.run_directory / "metrics-observer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                row = observe(cli.job_id, cli.remote_run_directory, cli.run_directory)
                print(json.dumps(row), flush=True)
                if row["scheduler"]["state"] not in ACTIVE | {"UNKNOWN"}:
                    break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                print(json.dumps({"error_type": type(error).__name__, "reason": str(error)}), flush=True)
            time.sleep(60)


if __name__ == "__main__":
    main()
