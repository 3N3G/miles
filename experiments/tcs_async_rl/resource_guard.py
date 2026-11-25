"""Request a final save for one explicitly owned job at its approved resource limits."""

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.auto_rl import remote, save
from experiments.tcs_async_rl.checkpoint_policy import storage_required
from experiments.tcs_async_rl.resource_estimate import model_inputs
from experiments.tcs_async_rl.reward_tunnel import allocation
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state


def duration_seconds(value):
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = map(int, clock.split(":"))
    if min(int(days), hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid scheduler duration")
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def resource_limits(config, training, config_dir, remote_directory):
    settings = config["resource_stop"]
    margin = settings["walltime_margin_seconds"]
    maximum = duration_seconds(config["allocation"]["walltime"])
    judge_config = json.loads((config_dir / config["judge_config"]).read_text())
    if (
        type(margin) is not int
        or type(settings["judge_stop_usd"]) not in (int, float)
        or not math.isfinite(settings["judge_stop_usd"])
        or not config["overrides"].get("stop_after_save_trigger")
        or config["overrides"]["save_trigger_sentinel"] != str(remote_directory / "checkpoint-request")
        or not 1800 <= margin < maximum
        or not 0 < settings["judge_stop_usd"] < judge_config["usd_cap"]
    ):
        raise ValueError("Resource guard requires a scoped graceful-stop recipe and final-save reserves")
    inputs = model_inputs(training, base_dir=config_dir)
    return {
        "elapsed_seconds": maximum - margin,
        "spent_reserved_usd": settings["judge_stop_usd"],
        "free_bytes": storage_required(
            config, remote_directory, inputs["text_parameters"] + inputs["mtp_parameters"]
        ),
    }


def stop_reason(limits, observed):
    for value in [*limits.values(), *observed.values()]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Resource observations must be finite and nonnegative")
    if observed["spent_reserved_usd"] >= limits["spent_reserved_usd"]:
        return "judge_budget"
    if observed["elapsed_seconds"] >= limits["elapsed_seconds"]:
        return "walltime_save_margin"
    if observed["free_bytes"] < limits["free_bytes"]:
        return "checkpoint_storage"
    return None


def operational_limits(directory, config, limits):
    """Allow a documented earlier save, without editing the submitted recipe."""
    path = directory / "operational-save-reserve.json"
    if not path.exists():
        return limits
    policy = json.loads(path.read_text())
    receipt = json.loads((directory / "submission.json").read_text())
    reserve = policy["walltime_save_reserve_seconds"]
    maximum = duration_seconds(config["allocation"]["walltime"])
    approved = config["resource_stop"]["walltime_margin_seconds"]
    if (
        policy["job_id"] != receipt["job_id"]
        or policy["session_uuid"] != receipt["session_uuid"]
        or policy["approved_config_sha256"]
        != hashlib.sha256((directory / "approved-config.json").read_bytes()).hexdigest()
        or type(reserve) is not int
        or not approved <= reserve < maximum
        or not policy.get("reason")
    ):
        raise ValueError("Operational save reserve must preserve ownership and only shorten the allocation")
    return {**limits, "elapsed_seconds": maximum - reserve}


def validated_stop(directory, config_path, remote_directory):
    path = directory / "resource-stop-request.json"
    if not path.exists():
        return None
    config, training = load_configs(config_path)
    marker = json.loads(path.read_text())
    receipt = json.loads((directory / "submission.json").read_text())
    limits = operational_limits(
        directory, config, resource_limits(config, training, config_path.parent, remote_directory)
    )
    if (
        marker["job_id"] != receipt["job_id"]
        or marker["session_uuid"] != receipt["session_uuid"]
        or marker["approved_config_sha256"]
        != hashlib.sha256((directory / "approved-config.json").read_bytes()).hexdigest()
        or marker["limits"] != limits
        or not marker["reason"]
        or stop_reason(limits, marker["observed"]) != marker["reason"]
    ):
        raise ValueError("Resource stop lacks matching ownership, configuration or a crossed limit")
    return marker


def watch(cli):
    config, training = load_configs(cli.config)
    limits = operational_limits(
        cli.directory, config, resource_limits(config, training, cli.config.parent, cli.remote_directory)
    )
    receipt = json.loads((cli.directory / "submission.json").read_text())
    if receipt["job_id"] != cli.job_id or receipt["session_uuid"] != cli.session_uuid:
        raise ValueError("Refuse to request a save for a job outside this session")
    settings = subprocess.check_output(["ssh", "-G", "amd"], text=True)
    user = next(line.split(maxsplit=1)[1] for line in settings.splitlines() if line.startswith("user "))
    while True:
        try:
            state = remote_state(cli.job_id)
            status = {
                "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "scheduler": state,
                "job_id": cli.job_id,
            }
            if state["state"] not in ACTIVE | {"UNKNOWN"}:
                save(cli.directory / "resource-guard-state.json", status)
                return
            if state["state"] == "RUNNING":
                lines = (cli.directory / "judging/budget.jsonl").read_text().splitlines(keepends=True)
                spent = sum(json.loads(line)["usd_delta"] for line in lines if line.endswith("\n"))
                output = remote(["df", "-B1", "--output=avail", cli.remote_directory]).stdout
                observed = {
                    "elapsed_seconds": duration_seconds(state["elapsed"]),
                    "spent_reserved_usd": spent,
                    "free_bytes": int(output.splitlines()[-1]),
                }
                reason = stop_reason(limits, observed)
                status.update(observed=observed, limits=limits, reason=reason)
                marker = cli.directory / "resource-stop-request.json"
                if reason and not marker.exists():
                    if allocation(cli.job_id, config["job_name"], user) is None:
                        raise RuntimeError("The owned running allocation changed before the save request")
                    request = {
                        **status,
                        "session_uuid": cli.session_uuid,
                        "approved_config_sha256": hashlib.sha256(
                            (cli.directory / "approved-config.json").read_bytes()
                        ).hexdigest(),
                    }
                    save(cli.directory / "resource-stop-intent.json", request)
                    remote(["touch", cli.remote_directory / "checkpoint-request"])
                    save(marker, request)
                    print(json.dumps(request), flush=True)
            save(cli.directory / "resource-guard-state.json", status)
        except (OSError, subprocess.SubprocessError) as error:
            save(
                cli.directory / "resource-guard-state.json",
                {
                    "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "job_id": cli.job_id,
                    "error_type": type(error).__name__,
                },
            )
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "directory", "remote-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("job-id", "session-uuid"):
        parser.add_argument("--" + name, required=True)
    watch(parser.parse_args())
