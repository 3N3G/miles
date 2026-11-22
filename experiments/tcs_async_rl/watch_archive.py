"""Archive an existing run at a bounded rate while its original watcher waits.

The operator must pause the original artifact uploader first. This worker owns
no training, reward, tunnel, or W&B process. Resume the original watcher only
after this worker exits. It drains remaining traces before leaving a terminal
job. Provider backoff is explicit and is never shortened by a retry.
"""

import argparse
import datetime
import fcntl
import json
import re
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.rl_archive import archive_chunks
from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state
from experiments.tcs_async_rl.watch_rl import consumed_rollouts, sync_outputs


def save(directory, name, row):
    _atomic_bytes(directory / name, (json.dumps(row, indent=2) + "\n").encode())


def require_paused_supervisor(directory):
    owner = json.loads((directory / "controller-process.json").read_text())
    process = Path("/proc") / str(owner["pid"])
    argv = (process / "cmdline").read_bytes().split(b"\0")
    state = (process / "stat").read_text().split(") ", 1)[1].split()[0]
    if b"experiments.tcs_async_rl.auto_rl" not in argv or str(directory).encode() not in argv or state != "T":
        raise RuntimeError("Pause the recorded original supervisor before using the separate archive worker")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--remote-run-directory", type=Path, required=True)
    parser.add_argument("--not-before-utc", type=datetime.datetime.fromisoformat, required=True)
    parser.add_argument("--every-seconds", type=int, default=300)
    cli = parser.parse_args()
    if cli.not_before_utc.tzinfo is None or cli.every_seconds < 300:
        raise ValueError("Use an explicit UTC backoff and at least five minutes between live snapshots")
    receipt = json.loads((cli.run_directory / "submission.json").read_text())
    if receipt["job_id"] != cli.job_id:
        raise ValueError("Archive worker does not match the existing submission")
    require_paused_supervisor(cli.run_directory)
    backoff = cli.run_directory / "archive-provider-backoff.json"
    saved = (
        datetime.datetime.fromisoformat(json.loads(backoff.read_text())["not_before_utc"])
        if backoff.exists()
        else cli.not_before_utc
    )
    next_upload = max(cli.not_before_utc, saved)
    with (cli.run_directory / "archive-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            now = datetime.datetime.now(datetime.UTC)
            row = {"at_utc": now.isoformat(), "job_id": cli.job_id, "not_before_utc": next_upload.isoformat()}
            if now < next_upload:
                row["phase"] = "provider_backoff_or_snapshot_interval"
                save(cli.run_directory, "archive-worker.json", row)
                time.sleep(min(60, (next_upload - now).total_seconds()))
                continue
            try:
                require_paused_supervisor(cli.run_directory)
                state = remote_state(cli.job_id)
                row["scheduler"] = state
                remaining = sync_outputs(cli.remote_run_directory, cli.run_directory, bounded=True)
                consumed_rollouts(cli.run_directory, allow_pending=True)
                row["immutable_receipt"] = archive_chunks(cli.run_directory)
                row["remaining_remote_traces"] = remaining
                row["phase"] = "archived"
                save(cli.run_directory, "archive-worker.json", row)
                print(json.dumps(row), flush=True)
                if state["state"] not in ACTIVE | {"UNKNOWN"}:
                    if remaining:
                        continue
                    save(cli.run_directory, "archive-worker-complete.json", row)
                    break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                row.update(phase="archive_retry_required", error_type=type(error).__name__, reason=str(error))
                save(cli.run_directory, "archive-worker.json", row)
                print(json.dumps(row), flush=True)
                log = cli.run_directory / "incremental-upload.log"
                text = log.read_text() if log.exists() else ""
                if "HTTP Error 429" in text:
                    waits = [float(value) for value in re.findall(r"Waiting ([\d.]+)s before retry", text)]
                    # Keep a conservative full delay after observing the error.
                    delay = max([3601, *waits])
                    next_upload = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=delay)
                    save(
                        cli.run_directory,
                        backoff.name,
                        {"not_before_utc": next_upload.isoformat(), "reason": "HTTP 429"},
                    )
                    continue
            next_upload = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=cli.every_seconds)


if __name__ == "__main__":
    main()
