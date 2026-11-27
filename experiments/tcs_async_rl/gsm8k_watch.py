"""Observe one already-submitted GSM8K job. Never submit or cancel GPU jobs."""

import argparse
import datetime
import json
import os
import shutil
import time
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.rl_archive import archive_chunks, read_index, trace_paths
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state
from experiments.tcs_async_rl.watch_rl import sync_native_wandb, sync_outputs


def heartbeat(directory, phase, **values):
    previous = directory / "monitor-heartbeat.json"
    data = json.loads(previous.read_text()) if previous.exists() else {}
    data.update(
        phase=phase,
        checked_at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        pid=os.getpid(),
        **values,
    )
    write_json(previous, data)
    return data


def cycle(directory):
    submission = json.loads((directory / "submission.json").read_text())
    scheduler = remote_state(submission["job_id"])
    heartbeat(directory, "syncing", job_id=submission["job_id"], scheduler=scheduler)
    if shutil.disk_usage(directory).free < 2 * 2**30:
        raise RuntimeError("VM reserve below2 GiB. Foreground storage diagnosis required before syncing.")
    sync_outputs(submission["remote_run"], directory, bounded=False)
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    evaluations = [json.loads(p.read_text()) for p in (directory / "evaluations").glob("*.json")]
    latest = max(evaluations, key=lambda point: point["optimizer_updates"]) if evaluations else None
    heartbeat(
        directory,
        "publishing",
        verified_optimizer_updates=len(native),
        latest_evaluation={key: latest[key] for key in ("optimizer_updates", "accuracy")} if latest else None,
        free_vm_bytes=shutil.disk_usage(directory).free,
    )
    receipt = {"verified_points": 0}
    if native:
        # Freeze the local input across delivery and readback. Closing the first
        # SDK subprocess can precede server history indexing. The acknowledged
        # rows are not resent by this second, verification-only retry.
        for _ in range(2):
            receipt = sync_native_wandb(directory, scheduler["state"])
            if receipt.get("verified_points") == len(native):
                break
    heartbeat(directory, "archiving", wandb=receipt)
    if sum(path.stat().st_size for path in trace_paths(directory)) >= 64 * 2**20:
        archived = archive_chunks(directory)
        write_json(
            directory / "last-incremental-archive.json",
            {
                "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "receipt": archived,
                "archived_trace_files": len(read_index(directory)),
            },
        )
    terminal = scheduler["state"] not in ACTIVE and scheduler["state"] != "UNKNOWN"
    ready = terminal and receipt.get("verified_points") == len(native)
    result = heartbeat(directory, "terminal_observed" if ready else "waiting", scheduler=scheduler, last_error_type=None, archived_trace_files=len(read_index(directory)))
    if ready:
        write_json(directory / "monitor-complete.json", result)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=25)
    args = parser.parse_args()
    while True:
        try:
            if cycle(args.directory):
                return
        except Exception as error:
            # GPU failures, publication failures and storage failures remain
            # separate evidence. Foreground supervision owns their diagnosis.
            heartbeat(args.directory, "needs_foreground_diagnosis", last_error_type=type(error).__name__)
            print(json.dumps({"error_type": type(error).__name__}), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
