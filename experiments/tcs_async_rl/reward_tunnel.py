"""Keep a loopback reward tunnel attached to one already submitted own RL job.

This VM process never submits, cancels, or modifies a Slurm job. It starts and
stops only its own SSH client. It connects to the allocated node through AMD's
known login alias and keeps provider credentials on the VM.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.reward_hook import REWARD_PROTOCOL_VERSION, question_key
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state


def allocation(job_id, expected_name, expected_user):
    if not job_id.isdigit():
        raise ValueError("Job ID must be numeric")
    result = subprocess.run(
        ["ssh", "amd", f"squeue --noheader --jobs={job_id} --format='%u|%j|%T|%N'"],
        check=True,
        capture_output=True,
        text=True,
        timeout=40,
    )
    if not result.stdout.strip():
        return None
    user, name, state, node = result.stdout.strip().split("|")
    if user != expected_user or name != expected_name:
        raise ValueError("Refuse a job outside this approved runtime check")
    if state != "RUNNING":
        return None
    if not re.fullmatch(r"k[0-9]{3}-[0-9]{3}(?:-v[0-9]+)?", node):
        raise ValueError("Expected exactly one known AMD compute node")
    return user, node


def ssh_command(target, *, forward_port=None):
    command = [
        "ssh",
        "-J",
        "amd",
        "-o",
        f"HostName={target[1]}",
        "-o",
        f"User={target[0]}",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if forward_port:
        command += ["-N", "-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{forward_port}:127.0.0.1:{forward_port}"]
    # Reuse AMD's existing identity configuration. Only the destination changes.
    return command + ["amd"]


def check_health(target, health_url):
    # This reports only public service identity. No provider request or key is sent.
    code = f"import json,urllib.request; r=urllib.request.urlopen({health_url!r},timeout=5); print(json.dumps(json.load(r)))"

    result = subprocess.run(
        ssh_command(target) + ["python3 -c " + shlex.quote(code)],
        check=True,
        capture_output=True,
        text=True,
        timeout=40,
    )
    return json.loads(result.stdout)


def stop_tunnel(tunnel):
    if tunnel is not None and tunnel.poll() is None:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()


def watch(*, job_id, expected_name, expected_user, run_dir, expected_health, port):
    run_dir.mkdir(parents=True, exist_ok=True)
    tunnel = None
    target = None
    try:
        with (run_dir / "reward-tunnel.log").open("a") as log:
            while True:
                state = remote_state(job_id)
                status = {
                    **state,
                    "checked_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "job_id": job_id,
                    "tunnel_pid": tunnel.pid if tunnel else None,
                }
                (run_dir / "reward-tunnel-state.json").write_text(json.dumps(status, indent=2) + "\n")
                print(json.dumps(status), flush=True)
                if state["state"] not in ACTIVE | {"UNKNOWN"}:
                    return
                if state["state"] != "RUNNING":
                    stop_tunnel(tunnel)
                    tunnel = None
                if state["state"] == "RUNNING":
                    current = allocation(job_id, expected_name, expected_user)
                    if current is not None:
                        if target != current:
                            stop_tunnel(tunnel)
                            tunnel = None
                        if tunnel is None or tunnel.poll() is not None:
                            target = current
                            tunnel = subprocess.Popen(
                                ssh_command(target, forward_port=port),
                                stdin=subprocess.DEVNULL,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                            time.sleep(3)
                        if tunnel.poll() is not None:
                            raise RuntimeError(
                                "Reward SSH connection failed; inspect its log without weakening host verification"
                            )
                        health = check_health(target, f"http://127.0.0.1:{port}/health")
                        if any(health.get(key) != value for key, value in expected_health.items()):
                            raise ValueError("Allocated node reached the wrong question or judge")
                        (run_dir / "reward-tunnel-health.json").write_text(
                            json.dumps(
                                {"checked_at": status["checked_at"], "target": "@".join(target), "health": health},
                                indent=2,
                            )
                            + "\n"
                        )
                time.sleep(30)
    finally:
        stop_tunnel(tunnel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("rl_smoke_v1.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config, training = load_configs(args.config)
    input_config, _ = load_configs(args.config.parent / config["input_config"])
    if hashlib.sha256(args.input_file.read_bytes()).hexdigest() != input_config["input_sha256"]:
        raise ValueError("Tunnel input does not match the approved question")
    address = urlsplit(config["overrides"]["rm_url"])
    if address.scheme != "http" or address.hostname != "127.0.0.1" or not address.port:
        raise ValueError("Only an explicit loopback reward endpoint is supported")
    ssh_settings = subprocess.check_output(["ssh", "-G", "amd"], text=True)
    user = next(line.split(maxsplit=1)[1] for line in ssh_settings.splitlines() if line.startswith("user "))
    expected_health = {
        "ready": True,
        "protocol_version": REWARD_PROTOCOL_VERSION,
        "question_key": question_key(json.loads(args.input_file.read_text())),
        "model": training["judge"]["primary_model"],
        "reasoning_effort": training["judge"]["reasoning_effort"],
    }
    with (args.run_dir / "reward-tunnel.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watch(
            job_id=args.job_id,
            expected_name=config["job_name"],
            expected_user=user,
            run_dir=args.run_dir,
            expected_health=expected_health,
            port=address.port,
        )


if __name__ == "__main__":
    main()
