"""Monitor one submitted own RL job, preserve evidence and sync its offline W&B.

This does not submit replacements, change rewards or declare a learning effect.
It separates a completed scheduler job from verified optimizer updates.
"""

import ast
import datetime
import hashlib
import json
import math
import random
import re
import subprocess
import time
from collections import Counter
from urllib.parse import urlsplit

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.rl_job_state import record
from experiments.tcs_async_rl.watch_group import ACTIVE, remote_state

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEPS = re.compile(r"train op=train_step rollout=(\d+) step=(\d+) attempt=(\d+) outcome=(\w+) valid_step=(true|false)")
METRICS = re.compile(r"(?<![\w-])step (\d+): (\{.*\})")


def training_evidence(text):
    valid, skipped, metrics = set(), set(), {}
    for raw in text.splitlines():
        line = ANSI.sub("", raw)
        if match := STEPS.search(line):
            rollout, step, attempt, outcome, ok = match.groups()
            identity = tuple(map(int, (rollout, step, attempt)))
            (valid if ok == "true" and outcome == "NORMAL" else skipped).add(identity)
        if match := METRICS.search(line):
            try:
                values = ast.literal_eval(match[2])
                numeric = {k: v for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                grad = [v for k, v in numeric.items() if k.endswith("grad_norm")]
                loss = [v for k, v in numeric.items() if k.endswith("pg_loss")]
                metrics[int(match[1])] = {
                    "finite": bool(grad and loss) and all(math.isfinite(v) for v in numeric.values()),
                    "values": numeric,
                }
            except (ValueError, SyntaxError, TypeError, AttributeError):
                metrics[int(match[1])] = {"finite": False}
    return {
        "valid_updates": [list(x) for x in sorted(valid)],
        "skipped_or_discarded_updates": [list(x) for x in sorted(skipped)],
        "metrics": metrics,
    }


def consumed_rollouts(directory):
    manifest = directory / "rollout-export.json"
    if not manifest.exists():
        return []
    groups = []
    for entry in json.loads(manifest.read_text()):
        path = directory / "rollout_data" / entry["path"].rsplit("/", 1)[-1]
        digest = hashlib.sha256()
        rewards, versions, indices, lengths = [], Counter(), [], []
        selected = random.Random(20261118).randrange(entry["samples"])
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                value = row["reward"]
                score = value["reward"] if isinstance(value, dict) else value
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("Consumed rollout contains an invalid real reward")
                rewards.append(score)
                lengths.append(row["response_length"])
                for call in row["weight_versions"] or []:
                    for span in call["spans"]:
                        versions[span["version"]] += span["abs_end"] - span["abs_start"]
                indices.append(row["optimizer_rollout_id"])
                if len(rewards) - 1 == selected:
                    example = {
                        key: row.get(key)
                        for key in (
                            "index",
                            "prompt",
                            "response",
                            "final_answer",
                            "response_length",
                            "reward",
                            "status",
                            "weight_versions",
                            "judge_identity",
                            "optimizer_rollout_id",
                        )
                    }
                    (directory / f"raw-example-{path.stem}.json").write_text(
                        json.dumps(example, ensure_ascii=False, indent=2) + "\n"
                    )
        if digest.hexdigest() != entry["sha256"] or len(rewards) != entry["samples"]:
            raise ValueError("Portable rollout differs from its allocation export")
        groups.append(
            {
                "file": path.name,
                "rollout_ids": sorted(set(indices)),
                "samples": len(rewards),
                "reward_histogram": dict(Counter(rewards)),
                "varied_rewards": len(set(rewards)) > 1,
                "response_lengths": lengths,
                "example_file": f"raw-example-{path.stem}.json",
                "decode_tokens_by_weight_version": dict(versions),
            }
        )
    return groups


def acceptance(directory, scheduler, expected_updates):
    text = "\n".join(p.read_text(errors="replace") for p in directory.glob("slurm-*.log"))
    evidence = training_evidence(text)
    groups = consumed_rollouts(directory)
    inventory = directory / "checkpoint-inventory.json"
    checkpoints = json.loads(inventory.read_text()) if inventory.exists() else {"tracker": None, "iterations": []}
    checks = {
        "scheduler_completed": scheduler["state"] == "COMPLETED" and scheduler.get("exit_code") == "0:0",
        "launcher_completed": (directory / "training-finished.json").exists(),
        "optimizer_updates_recorded": len(evidence["valid_updates"]) == expected_updates,
        "no_skipped_updates": not evidence["skipped_or_discarded_updates"],
        "finite_step_metrics": len(evidence["metrics"]) == expected_updates
        and all(item["finite"] for item in evidence["metrics"].values()),
        "consumed_groups_exported": len(groups) == expected_updates and all(g["samples"] == 32 for g in groups),
        "real_reward_variation_in_consumed_groups": len(groups) == expected_updates
        and all(g["varied_rewards"] for g in groups),
        "both_checkpoints_exist": len([x for x in checkpoints["iterations"] if x["files"] and x["bytes"]])
        >= expected_updates,
        "final_checkpoint_tracker": str(checkpoints["tracker"]) == str(expected_updates - 1),
    }
    return {
        "runtime_acceptance_passed": all(checks.values()),
        "checks": checks,
        "training": evidence,
        "consumed_groups": groups,
        "checkpoints": checkpoints,
        "limits": "Runtime evidence only. Weight versions are reported as observed. No learning-effect or judge-calibration claim.",
    }


def sync_outputs(remote_directory, directory):
    subprocess.run(
        ["rsync", "-a", "--exclude=checkpoints/", "--exclude=*.pt", f"amd:{remote_directory}/", f"{directory}/"],
        check=True,
        timeout=180,
    )


def finish(*, job_id, config_path, directory, remote_directory, checkout, scheduler, state_repo):
    from experiments.tcs_async_rl.auto_rl import remote, save

    config, _ = load_configs(config_path)
    try:
        remote(
            [
                "env",
                f"PYTHONPATH={checkout}",
                config["host_python"],
                "-m",
                "experiments.tcs_async_rl.rl_submission",
                "--run-dir",
                remote_directory,
                "--audit-only",
            ]
        )
    except subprocess.SubprocessError as error:
        save(directory / "checkpoint-audit-error.json", {"error_type": type(error).__name__})
    sync_outputs(remote_directory, directory)
    try:
        result = acceptance(directory, scheduler, config["scope"]["optimizer_updates"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"runtime_acceptance_passed": False, "evidence_error_type": type(error).__name__}
    save(directory / "runtime-acceptance.json", result)
    plots = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "matplotlib",
            "python",
            "-m",
            "experiments.tcs_async_rl.analyze_rl",
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    (directory / "analysis.log").write_text(plots.stdout + plots.stderr)
    uploads = []
    for data in sorted((directory / "wandb").rglob("*.wandb")):
        try:
            output = subprocess.run(
                [
                    "uvx",
                    "--from",
                    "wandb==0.31.0",
                    "wandb",
                    "sync",
                    "--entity",
                    config["overrides"]["wandb_team"],
                    "--project",
                    config["overrides"]["wandb_project"],
                    str(data.parent),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            (directory / f"wandb-sync-{data.stem}.log").write_text(output.stdout + output.stderr)
            uploads.append({"run_directory": str(data.parent), "returncode": output.returncode})
        except subprocess.SubprocessError as error:
            uploads.append({"run_directory": str(data.parent), "error_type": type(error).__name__})
    save(directory / "wandb-sync.json", uploads)
    upload = subprocess.run(["push-run", "miles", str(directory)], capture_output=True, text=True, timeout=1800)
    (directory / "upload.log").write_text(upload.stdout + upload.stderr)
    upload.check_returncode()
    urls = [line for line in upload.stdout.splitlines() if line.startswith("https://huggingface.co/datasets/")]
    hf_url = urls[-1] if upload.returncode == 0 and urls else None
    terminal = {
        "phase": "finished",
        "job_id": job_id,
        "status": scheduler["state"],
        "scheduler": scheduler,
        "runtime_acceptance_passed": result["runtime_acceptance_passed"],
        "hf_results": hf_url,
        "wandb_sync": uploads,
    }
    save(directory / "handoff-state.json", terminal)
    record(
        state_repo,
        terminal,
        summary=f"RL job {job_id} ended {scheduler['state']}; acceptance {result['runtime_acceptance_passed']}",
        next_action="Review exact updates, reward traces, weight versions and checkpoint evidence. Diagnose failed acceptance checks before a repair.",
        needs_user=False,
    )


def watch(*, job_id, config_path, directory, remote_directory, checkout, state_repo, children):
    from experiments.tcs_async_rl.auto_rl import ROOT, save, start_service

    config, training = load_configs(config_path)
    port = urlsplit(config["overrides"]["rm_url"]).port

    previous = None
    while True:
        try:
            state = remote_state(job_id)
        except (OSError, subprocess.SubprocessError) as error:
            save(directory / "monitor-error.json", {"error_type": type(error).__name__})
            time.sleep(60)
            continue
        if state["state"] in ACTIVE:
            if children[0].poll() is not None:
                try:
                    children[0], _ = start_service(
                        input_file=directory / "input.json",
                        directory=directory,
                        port=port,
                        training=training,
                        python=children[0].args[0],
                    )
                except Exception as error:
                    save(directory / "reward-service-restart-error.json", {"error_type": type(error).__name__})
            if children[1].poll() is not None:
                with (directory / "tunnel-controller.log").open("a") as log:
                    children[1] = subprocess.Popen(
                        children[1].args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                    )
        status = {
            "phase": "watching_rl",
            "job_id": job_id,
            "scheduler": state,
            "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "reward_service_alive": children[0].poll() is None,
            "reward_tunnel_alive": children[1].poll() is None,
        }
        save(directory / "handoff-state.json", status)
        if state["state"] != previous:
            try:
                record(
                    state_repo,
                    {"job_id": job_id, "status": state["state"], "phase": "monitoring"},
                    summary=f"RL job {job_id} is {state['state']}",
                    next_action="Monitor owned RL logs, optimizer updates, weight publication and saving.",
                )
            except subprocess.SubprocessError as error:
                save(directory / "registry-error.json", {"error_type": type(error).__name__, "job_id": job_id})
            previous = state["state"]
        try:
            sync_outputs(remote_directory, directory)
        except subprocess.SubprocessError as error:
            save(directory / "sync-error.json", {"error_type": type(error).__name__})
        if state["state"] not in ACTIVE | {"UNKNOWN"}:
            try:
                finish(
                    job_id=job_id,
                    config_path=config_path,
                    directory=directory,
                    remote_directory=remote_directory,
                    checkout=checkout,
                    scheduler=state,
                    state_repo=state_repo,
                )
                return
            except (OSError, subprocess.SubprocessError) as error:
                save(directory / "archive-retry.json", {"error_type": type(error).__name__})
        time.sleep(60)
