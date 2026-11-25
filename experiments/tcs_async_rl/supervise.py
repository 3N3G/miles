"""Attach monitoring to one recorded, approved RL job. Never submit a job."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.auto_rl import ROOT, save, start_service
from experiments.tcs_async_rl.watch_rl import watch


def supervise(cli):
    receipt = json.loads((cli.directory / "submission.json").read_text())
    config, training = load_configs(cli.config)
    if receipt["session_uuid"] != cli.session_uuid:
        raise ValueError("Only this session's recorded job may be supervised")
    if json.loads((cli.directory / "approved-config.json").read_text()) != config or not config.get("resource_stop"):
        raise ValueError("Expected the frozen approved configuration with resource guards")
    port = urlsplit(config["overrides"]["rm_url"]).port
    children = []
    try:
        service, _ = start_service(
            input_file=cli.directory / "input.json",
            directory=cli.directory,
            port=port,
            training=training,
            python=Path(sys.executable),
            judge_config=cli.config.parent / config["judge_config"],
        )
        children.append(service)
        for module, arguments, log_name in (
            (
                "reward_tunnel",
                ["--run-dir", str(cli.directory), "--input-file", str(cli.directory / "input.json")],
                "tunnel-controller.log",
            ),
            (
                "resource_guard",
                [
                    "--directory",
                    str(cli.directory),
                    "--remote-directory",
                    str(cli.remote_directory),
                    "--session-uuid",
                    cli.session_uuid,
                ],
                "resource-guard.log",
            ),
        ):
            argv = [
                sys.executable,
                "-m",
                f"experiments.tcs_async_rl.{module}",
                "--job-id",
                receipt["job_id"],
                "--config",
                str(cli.config),
                *arguments,
            ]
            with (cli.directory / log_name).open("a") as log:
                child = subprocess.Popen(
                    argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                )
            children.append(child)
            save(
                cli.directory / f"{module}-process.json",
                {"pid": child.pid, "argv": argv, "session_uuid": cli.session_uuid},
            )
        save(
            cli.directory / "supervisor-process.json",
            {
                "pid": os.getpid(),
                "job_id": receipt["job_id"],
                "session_uuid": cli.session_uuid,
                "children": [child.pid for child in children],
            },
        )
        watch(
            job_id=receipt["job_id"],
            config_path=cli.config,
            directory=cli.directory,
            remote_directory=cli.remote_directory,
            checkout=cli.checkout,
            state_repo=cli.state_repo,
            children=children,
        )
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "directory", "remote-directory", "checkout", "state-repo"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--session-uuid", required=True)
    cli = parser.parse_args()
    with (cli.directory / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervise(cli)
