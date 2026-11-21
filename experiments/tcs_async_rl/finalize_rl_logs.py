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
        if (
            terminal.get("phase") == "finished"
            and terminal.get("job_id") == cli.job_id
            and not controller_alive(cli.run_dir)
        ):
            break
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
    save(
        cli.run_dir / "native-log-finalizer-complete.json",
        {"job_id": cli.job_id, "training": evidence, "gpu_submissions": 0, "old_analysis": str(backup)},
    )
    uploaded = subprocess.run(
        ["push-run", "miles", str(cli.run_dir)], capture_output=True, text=True, timeout=1800, check=True
    )
    (cli.run_dir / "corrected-upload.log").write_text(uploaded.stdout + uploaded.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    for name in ("run-dir", "config", "remote-run-dir", "remote-checkout", "state-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    cli = parser.parse_args()
    with (cli.run_dir / "native-log-finalizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (cli.run_dir / "native-log-finalizer-complete.json").exists():
            finalize(cli)


if __name__ == "__main__":
    main()
