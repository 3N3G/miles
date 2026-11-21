"""Read native progress and finalize an existing run after its original monitor exits.

This VM-only repair never submits or changes a GPU job, service or tunnel.
It preserves the original report before correcting its native-log reader.
"""

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.auto_rl import save
from experiments.tcs_async_rl.rl_archive import upload_run
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state
from experiments.tcs_async_rl.watch_rl import finish, run_training_evidence


def controller_alive(directory):
    owner = json.loads((directory / "controller-process.json").read_text())
    try:
        argv = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return b"experiments.tcs_async_rl.auto_rl" in argv and str(directory).encode() in argv


def finalize(cli):
    receipt = json.loads((cli.run_dir / "submission.json").read_text())
    if receipt["job_id"] != cli.job_id:
        raise ValueError("Finalizer does not match this submitted job")
    config, _ = load_configs(cli.config)
    if json.loads((cli.run_dir / "approved-config.json").read_text()) != config:
        raise ValueError("Finalizer must preserve the submitted experiment configuration")
    while True:
        save(cli.run_dir / "live-training-evidence.json", run_training_evidence(cli.run_dir))
        try:
            terminal = json.loads((cli.run_dir / "handoff-state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            terminal = {}
        if not controller_alive(cli.run_dir):
            # The original monitor may stop before producing a finished marker.
            # Reconcile the existing receipt with Slurm. Never submit a replacement.
            scheduler = remote_state(cli.job_id)
            if scheduler["state"] not in ACTIVE | {"UNKNOWN"}:
                terminal = {"scheduler": scheduler, "job_id": cli.job_id}
                break
            save(
                cli.run_dir / "monitor-supervision-required.json",
                {
                    "job_id": cli.job_id,
                    "scheduler": scheduler,
                    "reason": "Primary monitor exited while the GPU job remains active. Foreground recovery is required.",
                },
            )
        time.sleep(30)
    backup = cli.run_dir / "analysis-before-log-reader-repair"
    backup.mkdir(exist_ok=True)
    names = ["runtime-acceptance.json", "REPORT.md", "question-trajectories.json", "handoff-state.json"]
    for path in [*(cli.run_dir / name for name in names), *cli.run_dir.glob("*.png")]:
        if path.exists() and not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
    finish(
        job_id=cli.job_id,
        config_path=cli.config,
        directory=cli.run_dir,
        remote_directory=cli.remote_run_dir,
        checkout=cli.remote_checkout,
        scheduler=terminal["scheduler"],
        state_repo=cli.state_repo,
    )
    evidence = run_training_evidence(cli.run_dir)
    receipt = upload_run(cli.run_dir, log_name="corrected-upload.log")
    save(
        cli.run_dir / "native-log-finalizer-complete.json",
        {
            "job_id": cli.job_id,
            "training": evidence,
            "gpu_submissions": 0,
            "old_analysis": str(backup),
            "hf_commit_url": receipt,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    for name in ("run-dir", "config", "remote-run-dir", "remote-checkout", "state-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    cli = parser.parse_args()
    with (cli.run_dir / "native-log-finalizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (cli.run_dir / "native-log-finalizer-complete.json").exists():
            while True:
                try:
                    finalize(cli)
                    return
                except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                    save(
                        cli.run_dir / "finalizer-retry.json",
                        {"job_id": cli.job_id, "error_type": type(error).__name__, "reason": str(error)},
                    )
                    time.sleep(60)


if __name__ == "__main__":
    main()
