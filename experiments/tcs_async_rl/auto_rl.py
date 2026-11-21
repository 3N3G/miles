"""Durable VM handoff from the approved medium diagnostic to one bounded RL job.

Run from a frozen checkout. This submits only after the original strict group
gate passes. It never relaxes missing-final handling or selects corpus weights.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.group_inference import sampling_parameters, validate_sampling_support_control
from experiments.tcs_async_rl.reward_hook import REWARD_PROTOCOL_VERSION, question_key
from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.rl_inputs import read_bundle, service_identity
from experiments.tcs_async_rl.rl_job_state import record
from experiments.tcs_async_rl.rl_smoke import validate_group_gate
from experiments.tcs_async_rl.watch_group import remote_state

ROOT = Path(__file__).resolve().parents[2]


def save(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def remote(argv, *, timeout=60):
    return subprocess.run(
        ["ssh", "amd", shlex.join([str(x) for x in argv])], capture_output=True, text=True, check=True, timeout=timeout
    )


def qualify_medium(directory, config_path):
    config, training = load_configs(config_path)
    input_config, _ = load_configs(config_path.parent / config["input_config"])
    complete = json.loads((directory / "watcher-complete.json").read_text())
    if complete["scheduler"]["state"] != "COMPLETED" or complete["scheduler"].get("grading_returncode") != 0:
        raise ValueError("Medium inference or grading did not complete successfully")
    if not complete.get("hf_results"):
        raise ValueError("Medium results were not archived")
    digest = hashlib.sha256((directory / "input.json").read_bytes()).hexdigest()
    if digest != input_config["input_sha256"]:
        raise ValueError("Medium input hash differs from the frozen RL snapshot")
    grade_file = Path(config.get("local_gate_file", directory / "group-grades.json"))
    grades = json.loads(grade_file.read_text())
    expected = sampling_parameters({"miles_args": {**training["miles_args"], **config["overrides"]}})
    count = training["miles_args"]["n_samples_per_prompt"]
    policy = config.get("diagnostic_gate_policy", "complete_only")
    validate_group_gate(grades, count, expected_sampling=expected, policy=policy)
    if (grades["model"], grades["reasoning_effort"]) != (
        training["judge"]["primary_model"],
        training["judge"]["reasoning_effort"],
    ):
        raise ValueError("Medium grades changed the judge or effort")
    cases = {f"sample-{i:02d}" for i in range(count)}
    if {k for k in grades["scores_by_case"] if k.startswith("sample-")} != cases:
        raise ValueError("Medium grades do not cover the exact generation indices")
    rows = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
    if len(rows) != count or {r["sample_index"] for r in rows} != set(range(count)):
        raise ValueError("Medium responses are incomplete or duplicated")
    if policy == "complete_only" and not all(r["final_answer"] for r in rows):
        raise ValueError("Recorded-cap qualification needs the pending user decision")
    if policy == "recorded_caps_judged_v1":
        if (
            grades.get("original_grades_sha256")
            != hashlib.sha256((directory / "group-grades.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Qualified grades do not preserve the original diagnostic")
        if (
            grades.get("original_responses_sha256")
            != hashlib.sha256((directory / "responses.jsonl").read_bytes()).hexdigest()
        ):
            raise ValueError("Cap provenance differs from the generated responses")
        caps = {cap["case"]: cap for cap in grades["recorded_cap_judgments"]}
        for item in rows:
            if not item["final_answer"]:
                cap = caps[f"sample-{item['sample_index']:02d}"]
                if cap["raw_response_sha256"] != hashlib.sha256(item["output"]["text"].encode()).hexdigest():
                    raise ValueError("Cap judgment is linked to a different raw generation")
    verdicts = grades["grades"]
    for case in cases | {"control-reference", "control-empty", "control-irrelevant"}:
        pair = [v for v in verdicts if v["identity"] == case]
        if len(pair) != 2 or {v["swapped"] for v in pair} != {False, True}:
            raise ValueError("Missing or duplicate presentation-order verdicts")
        if any(v["reference"]["score"] != 7 for v in pair):
            raise ValueError("A paired-reference anomaly needs review before automatic submission")
        mean = sum(v["candidate"]["score"] for v in pair) / 2
        if mean != grades["scores_by_case"][case]:
            raise ValueError("Grade summary differs from actual paired verdicts")
    support = json.loads((directory / "sampling-support-control.json").read_text())
    validate_sampling_support_control(support, expected_tokens=16, top_k=expected["top_k"])
    mode = json.loads((directory / "resolved-inference-mode.json").read_text())
    if mode["speculative_algorithm"] is not None:
        raise ValueError("The diagnostic enabled unsupported speculative support replay")
    return {**grades, "input_sha256": digest}


def wait_for_group(*, directory, job_id, remote_directory, state_repo, output):
    restarts = 0
    while not (directory / "watcher-complete.json").exists():
        try:
            state = remote_state(job_id)
        except (OSError, subprocess.SubprocessError) as error:
            state = {"state": "UNKNOWN", "error_type": type(error).__name__}
        save(
            output / "handoff-state.json",
            {
                "phase": "waiting_for_medium",
                "scheduler": state,
                "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )
        lock_path = directory / "watcher.lock"
        # A crashed watcher must not leave the handoff waiting forever. Its lock
        # also prevents replacing a live grader during long provider requests.
        if lock_path.exists():
            with lock_path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    if not (directory / "watcher-complete.json").exists():
                        if restarts >= 3:
                            raise RuntimeError("Diagnostic watcher repeatedly stopped; inspect its recovery log")
                        with (output / "watcher-recovery.log").open("a") as log:
                            subprocess.Popen(
                                [
                                    sys.executable,
                                    "-m",
                                    "experiments.tcs_async_rl.watch_group",
                                    "--job-id",
                                    job_id,
                                    "--remote-run-dir",
                                    remote_directory,
                                    "--run-dir",
                                    str(directory),
                                    "--state-repo",
                                    str(state_repo),
                                    "--continue-to-rl",
                                ],
                                cwd=ROOT,
                                stdin=subprocess.DEVNULL,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        restarts += 1
        time.sleep(60)


def start_service(*, input_file, directory, port, training, python):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    log = (directory / "reward-service.log").open("a")
    child = subprocess.Popen(
        [
            str(python),
            "-m",
            "experiments.tcs_async_rl.reward_service",
            "--input-file",
            str(input_file),
            "--directory",
            str(directory / "judging"),
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    expected = {
        "ready": True,
        "protocol_version": REWARD_PROTOCOL_VERSION,
        "question_key": service_identity(input_file),
        "model": training["judge"]["primary_model"],
        "reasoning_effort": training["judge"]["reasoning_effort"],
    }
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Owned reward service could not start; no occupied service was replaced")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                    health = json.load(response)
                if health != expected:
                    raise ValueError("Reward port belongs to a different question or judge")
                save(
                    directory / "reward-service-process.json",
                    {"pid": child.pid, "source": str(ROOT), "health": health},
                )
                return child, expected
            except (OSError, TimeoutError):
                time.sleep(1)
        raise RuntimeError("Reward service startup timed out")
    except Exception:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        raise


def service_controls(*, directory, input_file, port, key):
    if service_identity(input_file) != key:
        raise ValueError("Reward controls do not match the requested service collection")
    value = json.loads(input_file.read_text())
    rows = value["questions"] if "questions" in value else [value]
    outputs = {}
    for row in rows:
        identity = question_key(row)
        for name, text, truncated, score in (
            ("reference", "</think>\n" + row["answer"], False, 7),
            ("empty", "", True, 0),
            ("irrelevant", "</think>\nThe answer is 42.", False, 0),
        ):
            case = identity + "-" + name
            body = json.dumps({"question_key": identity, "response": text, "generator_truncated": truncated}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/score", data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=1300) as response:
                outputs[case] = json.load(response)
            save(directory / "online-controls.json", outputs)
            if outputs[case]["scores"] != [score, score]:
                raise ValueError("Online reward controls failed before GPU submission")


def stage_and_submit(
    *, config_path, source_commit, grades, directory, work_dir, checkout, remote_run, training_data=None
):
    config = json.loads(config_path.read_text())
    local_gate = directory / "qualified-medium-grades.json"
    save(local_gate, grades)
    main_repo = work_dir.parent / "miles-ian"
    if remote(["git", "-C", main_repo, "status", "--porcelain"]).stdout.strip():
        raise RuntimeError("AMD main checkout has local changes; no pull or submission attempted")
    remote(["git", "-C", main_repo, "pull", "--rebase"], timeout=120)
    # A deterministic checkout and output directory are unique to this attempt.
    check = subprocess.run(
        ["ssh", "amd", shlex.join(["test", "-d", str(checkout)])], capture_output=True, text=True, timeout=30
    )
    if check.returncode == 1:
        remote(["git", "-C", main_repo, "worktree", "add", "--detach", checkout, source_commit])
    elif check.returncode != 0:
        raise RuntimeError("Could not determine the remote checkout state")
    remote(["mkdir", "-p", remote_run, Path(config["gate_file"]).parent])
    subprocess.run(["scp", str(local_gate), f"amd:{config['gate_file']}"], check=True, timeout=60)
    if config.get("dataset_file"):
        subprocess.run(["scp", str(training_data), f"amd:{config['dataset_file']}"], check=True, timeout=60)
    result = remote(
        [
            "env",
            f"PYTHONPATH={checkout}",
            config["host_python"],
            "-m",
            "experiments.tcs_async_rl.rl_submission",
            "--config",
            checkout / config_path.relative_to(ROOT),
            "--run-dir",
            remote_run,
            "--work-dir",
            work_dir,
            "--expected-commit",
            source_commit,
        ],
        timeout=120,
    )
    (directory / "remote-submission.stdout").write_text(result.stdout)
    (directory / "remote-submission.stderr").write_text(result.stderr)
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    save(directory / "submission.json", {**receipt, "source_commit": source_commit, "remote_run_dir": str(remote_run)})
    return receipt


def run(cli):
    config, training = load_configs(cli.config)
    source = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if source != cli.expected_commit:
        raise ValueError("Handoff must use its frozen committed source")
    wait_for_group(
        directory=cli.diagnostic_dir,
        job_id=cli.group_job_id,
        remote_directory=cli.remote_diagnostic_dir,
        state_repo=cli.state_repo,
        output=cli.run_dir,
    )
    grades = qualify_medium(cli.diagnostic_dir, cli.config)
    save(cli.run_dir / "qualified-medium-grades.json", grades)
    service_file = cli.diagnostic_dir / "input.json"
    if config.get("dataset_file"):
        read_bundle(cli.training_data, expected_sha256=config["dataset_sha256"], training=training)
        service_file = cli.training_data
    (cli.run_dir / "input.json").write_bytes(service_file.read_bytes())
    address = urlsplit(config["overrides"]["rm_url"])
    if address.hostname != "127.0.0.1" or address.scheme != "http" or not address.port:
        raise ValueError("Only a loopback reward service is allowed")
    service, health = start_service(
        input_file=service_file,
        directory=cli.run_dir,
        port=address.port,
        training=training,
        python=cli.service_python,
    )
    children = [service]
    try:
        service_controls(
            directory=cli.run_dir,
            input_file=service_file,
            port=address.port,
            key=health["question_key"],
        )
        receipt = stage_and_submit(
            config_path=cli.config,
            source_commit=source,
            grades=grades,
            directory=cli.run_dir,
            work_dir=cli.work_dir,
            checkout=cli.remote_checkout,
            remote_run=cli.remote_run_dir,
            training_data=service_file,
        )
        event = {
            **receipt,
            "cluster": "hpcfund",
            "phase": "submitted",
            "status": "SUBMITTED",
            "config": str(cli.config.relative_to(ROOT)),
            "code_commit": source,
            "session_uuid": cli.session_uuid,
            "allocation": config["allocation"],
            "gate_sha256": hashlib.sha256((cli.run_dir / "qualified-medium-grades.json").read_bytes()).hexdigest(),
        }
        try:
            record(
                cli.state_repo,
                event,
                summary=f"Automatically submitted approved RL job {receipt['job_id']}",
                next_action="Monitor the owned RL job, reward tunnel, optimizer updates and saving.",
            )
        except subprocess.SubprocessError as error:
            # A repository transport error must not detach a real submitted job.
            save(
                cli.run_dir / "registry-error.json", {"error_type": type(error).__name__, "job_id": receipt["job_id"]}
            )
        with (cli.run_dir / "tunnel-controller.log").open("a") as log:
            tunnel = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "experiments.tcs_async_rl.reward_tunnel",
                    "--job-id",
                    receipt["job_id"],
                    "--config",
                    str(cli.config),
                    "--run-dir",
                    str(cli.run_dir),
                    "--input-file",
                    str(service_file),
                ],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(tunnel)
        from experiments.tcs_async_rl.watch_rl import watch

        watch(
            job_id=receipt["job_id"],
            config_path=cli.config,
            directory=cli.run_dir,
            remote_directory=cli.remote_run_dir,
            checkout=cli.remote_checkout,
            state_repo=cli.state_repo,
            children=children,
        )
    finally:
        for child in reversed(children):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "diagnostic-dir",
        "run-dir",
        "state-repo",
        "work-dir",
        "remote-checkout",
        "remote-run-dir",
        "service-python",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("group-job-id", "remote-diagnostic-dir", "expected-commit", "session-uuid"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--training-data", type=Path)
    cli = parser.parse_args()
    cli.run_dir.mkdir(parents=True, exist_ok=True)
    with (cli.run_dir / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Restarts cannot create another job or stop an unrelated existing service.
        if (cli.run_dir / "submission.json").exists():
            raise RuntimeError("An RL receipt already exists; attach a monitor instead of submitting again")
        try:
            run(cli)
        except Exception as error:
            result = {
                "phase": "stopped",
                "error_type": type(error).__name__,
                "reason": (
                    str(error) if isinstance(error, (ValueError, RuntimeError)) else "Inspect the controller log"
                ),
                "at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            save(cli.run_dir / "handoff-state.json", result)
            record(
                cli.state_repo,
                {"phase": "automatic_handoff_stopped", "status": "NEEDS_REVIEW", **result},
                summary="Automatic RL handoff stopped before acceptance",
                next_action="Inspect handoff-state.json and preserve the strict reward gate. Resolve any required research decision.",
                needs_user=isinstance(error, ValueError),
            )
            raise


if __name__ == "__main__":
    main()
