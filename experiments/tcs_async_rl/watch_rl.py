"""Monitor one submitted own RL job, preserve evidence and sync its offline W&B.

This does not submit replacements, change rewards or declare a learning effect.
It separates a completed scheduler job from verified optimizer updates.
"""

import ast
import datetime
import gzip
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from urllib.parse import urlsplit

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.rl_archive import (
    archive_chunks,
    copy_trace_batch,
    exclusion_file,
    read_index,
    upload_run,
)
from experiments.tcs_async_rl.rl_artifacts import file_sha256
from experiments.tcs_async_rl.rl_job_state import record
from experiments.tcs_async_rl.rl_smoke import initial_optimizer_updates
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


def run_training_evidence(directory):
    # amd_preflight._run owns the native subprocess log. Slurm contains its
    # launch commands and failures, while optimizer metrics are in this file.
    paths = [directory / "rl-smoke.log", *directory.glob("slurm-*.log")]
    return training_evidence("\n".join(p.read_text(errors="replace") for p in paths if p.exists()))


def consumed_rollouts(directory, *, allow_pending=False):
    manifest = directory / "rollout-export.json"
    if not manifest.exists():
        return []
    groups = []
    cache_path = directory / "consumed-summary-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    archived = read_index(directory)
    for entry in json.loads(manifest.read_text()):
        path = directory / "rollout_data" / entry["path"].rsplit("/", 1)[-1]
        relative = str(path.relative_to(directory))
        if not path.exists():
            if (
                relative in archived
                and archived[relative]["sha256"] == entry["sha256"]
                and path.name in cache
                and cache[path.name]["sha256"] == entry["sha256"]
            ):
                groups.append(cache[path.name]["summary"])
                continue
            if allow_pending:
                continue
            raise ValueError("Missing trace has no verified archive and summary")
        digest = file_sha256(path)
        rewards, versions, indices, lengths = [], Counter(), [], []
        questions = {}
        selected = random.Random(20261118).randrange(entry["samples"])
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            for line in stream:
                row = json.loads(line)
                value = row["reward"]
                score = value["reward"] if isinstance(value, dict) else value
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("Consumed rollout contains an invalid real reward")
                rewards.append(score)
                lengths.append(row["response_length"])
                question = (row.get("metadata") or {}).get("question_key", "unknown")
                question_data = questions.setdefault(question, {"scores": [], "lengths": [], "capped": 0})
                question_data["scores"].append(score)
                question_data["lengths"].append(row["response_length"])
                question_data["capped"] += row.get("status") == "truncated"
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
                            "metadata",
                        )
                    }
                    (directory / f"raw-example-{path.stem}.json").write_text(
                        json.dumps(example, ensure_ascii=False, indent=2) + "\n"
                    )
        if digest != entry["sha256"] or len(rewards) != entry["samples"]:
            raise ValueError("Portable rollout differs from its allocation export")
        groups.append(
            {
                "file": path.name,
                "rollout_ids": sorted(set(indices)),
                "samples": len(rewards),
                "reward_histogram": {str(k): v for k, v in Counter(rewards).items()},
                "varied_rewards": len(set(rewards)) > 1,
                "response_lengths": lengths,
                "example_file": f"raw-example-{path.stem}.json",
                "decode_tokens_by_weight_version": dict(versions),
                "by_question": questions,
            }
        )
        cache[path.name] = {"sha256": entry["sha256"], "summary": groups[-1]}
    from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes

    _atomic_bytes(cache_path, (json.dumps(cache, indent=2) + "\n").encode())
    return groups


def acceptance(
    directory, scheduler, expected_updates, *, minimum_varied_groups=None, retained_checkpoints=None, initial_updates=0
):
    evidence = run_training_evidence(directory)
    groups = consumed_rollouts(directory)
    inventory = directory / "checkpoint-inventory.json"
    checkpoints = json.loads(inventory.read_text()) if inventory.exists() else {"tracker": None, "iterations": []}
    checks = {
        "scheduler_completed": scheduler["state"] == "COMPLETED" and scheduler.get("exit_code") == "0:0",
        "launcher_completed": (directory / "training-finished.json").exists(),
        "optimizer_updates_recorded": len(evidence["valid_updates"]) == expected_updates,
        "no_skipped_updates": not evidence["skipped_or_discarded_updates"],
        "finite_step_metrics": bool(evidence["valid_updates"])
        and {x[0] for x in evidence["valid_updates"]} <= set(evidence["metrics"])
        and all(item["finite"] for item in evidence["metrics"].values()),
        "consumed_groups_exported": len(groups) == expected_updates and all(g["samples"] == 32 for g in groups),
        "real_reward_variation_in_consumed_groups": sum(g["varied_rewards"] for g in groups)
        >= (expected_updates if minimum_varied_groups is None else minimum_varied_groups),
        "retained_checkpoints_exist": len([x for x in checkpoints["iterations"] if x["files"] and x["bytes"]])
        >= (expected_updates if retained_checkpoints is None else retained_checkpoints),
        "final_checkpoint_tracker": str(checkpoints["tracker"]) == str(initial_updates + expected_updates - 1),
    }
    return {
        "runtime_acceptance_passed": all(checks.values()),
        "stop_reason": "walltime_limit" if scheduler["state"] == "TIMEOUT" else scheduler["state"].lower(),
        "target_updates": expected_updates,
        "initial_optimizer_updates": initial_updates,
        "completed_updates": len(evidence["valid_updates"]),
        "checks": checks,
        "training": evidence,
        "consumed_groups": groups,
        "checkpoints": checkpoints,
        "limits": "Runtime evidence only. Weight versions are reported as observed. No learning-effect or judge-calibration claim.",
    }


def sync_outputs(remote_directory, directory, *, bounded=False):
    exclude = exclusion_file(directory)
    extra = ["--exclude=*.gz"] if bounded else []
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=checkpoints/",
            "--exclude=ray/",
            "--exclude=dashboard_columns/",
            "--exclude=.cache/",
            "--exclude=*.pt",
            "--exclude=*.partial",
            "--exclude=*.tmp",
            "--exclude-from",
            str(exclude),
            *extra,
            f"amd:{remote_directory}/",
            f"{directory}/",
        ],
        check=True,
        timeout=180,
    )
    return copy_trace_batch(remote_directory, directory) if bounded else 0


def sync_native_wandb(directory, scheduler_state):
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "wandb==0.31.0",
            "python",
            "-m",
            "experiments.tcs_async_rl.native_metrics",
            "--run-directory",
            str(directory),
            "--output",
            str(directory),
            "--scheduler-state",
            scheduler_state,
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    (directory / "wandb-native-sync.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    path = directory / "wandb-live-metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"verified_points": 0}


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
    while True:
        remaining = sync_outputs(remote_directory, directory, bounded=config.get("incremental_archive", False))
        if config.get("incremental_archive"):
            consumed_rollouts(directory, allow_pending=True)
            archive_chunks(directory)
        if not remaining:
            break
    try:
        result = acceptance(
            directory,
            scheduler,
            config["scope"]["optimizer_updates"],
            minimum_varied_groups=config["scope"].get("minimum_varied_groups"),
            retained_checkpoints=config["scope"].get("retained_checkpoints"),
            initial_updates=initial_optimizer_updates(config),
        )
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
    native_vm_metrics = config.get("wandb_metrics_source") == "native_vm"
    if native_vm_metrics:
        uploads.append({"source": "native_vm", **sync_native_wandb(directory, scheduler["state"])})
    # VM-native runs already have one authoritative online history. Retain
    # independent offline streams in the private archive without merging them.
    offline_streams = [] if native_vm_metrics else sorted((directory / "wandb").rglob("*.wandb"))
    for data in offline_streams:
        try:
            output = subprocess.run(
                [
                    "uvx",
                    "--from",
                    "wandb==0.31.0",
                    "wandb",
                    "sync",
                    "--include-synced",
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
    receipt = upload_run(directory)
    hf_url = receipt.replace("/commit/", "/tree/") + "/miles/" + directory.name
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
                        judge_config=(
                            config_path.parent / config["judge_config"] if config.get("judge_config") else None
                        ),
                    )
                except Exception as error:
                    save(directory / "reward-service-restart-error.json", {"error_type": type(error).__name__})
            if children[1].poll() is not None:
                with (directory / "tunnel-controller.log").open("a") as log:
                    children[1] = subprocess.Popen(
                        children[1].args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
                    )
        status = {
            "phase": "watching_rl" if state["state"] in ACTIVE | {"UNKNOWN"} else "finalizing",
            "job_id": job_id,
            "scheduler": state,
            "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "reward_service_alive": children[0].poll() is None,
            "reward_tunnel_alive": children[1].poll() is None,
            "monitor_pid": os.getpid(),
            "vm_free_bytes": shutil.disk_usage(directory).free,
        }
        progress = run_training_evidence(directory)
        status["training"] = {
            "valid_updates": len(progress["valid_updates"]),
            "skipped_updates": len(progress["skipped_or_discarded_updates"]),
            "nonfinite_steps": [step for step, row in progress["metrics"].items() if not row["finite"]],
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
            sync_outputs(remote_directory, directory, bounded=config.get("incremental_archive", False))
            if config.get("incremental_archive"):
                consumed_rollouts(directory, allow_pending=True)
                archive_chunks(directory)
            if config.get("wandb_metrics_source") == "native_vm" and state["state"] in ACTIVE:
                sync_native_wandb(directory, state["state"])
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
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
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
                save(
                    directory / "archive-retry.json",
                    {
                        "phase": "finalization_pending",
                        "job_id": job_id,
                        "scheduler": state,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                        "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    },
                )
        time.sleep(60)
