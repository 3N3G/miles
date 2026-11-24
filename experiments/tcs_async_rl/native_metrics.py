"""Read one metric row per optimizer update from native events, before W&B merging.

Offline actor and rollout processes own separate W&B history counters. Syncing
those streams into one run can misassign the shared rollout/step field. Native
events carry the optimizer rollout ID independently of those history counters.
"""

import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.rl_smoke import initial_optimizer_updates

PUBLIC_TRAINING_ARGUMENTS = frozenset("""
    accumulate_allreduce_grads_in_fp32 actor_num_gpus_per_node actor_num_nodes
    adam_beta1 adam_beta2 advantage_estimator apply_chat_template apply_chat_template_kwargs
    async_data_buffer_capacity_factor async_max_concurrent_samples async_unused_samples_handler
    attention_backend attention_dropout balance_data bf16 ckpt_format context_parallel_size
    enable_mtp_training eps_clip eps_clip_high fully_async global_batch_size group_rm
    hidden_dropout kl_loss_coef kl_loss_type linear_attention_backend log_probs_backend
    log_probs_chunk_size lr lr_decay_style max_position_embeddings max_tokens_per_gpu
    max_weight_staleness mtp_loss_scaling_factor mtp_num_layers n_samples_per_prompt
    num_gpus_per_node num_rollout observe_training_entropy observe_update_direction_steps
    optimizer pause_generation_mode pipeline_model_parallel_size policy_loss_type
    recompute_granularity recompute_method recompute_num_layers rollout_batch_size
    rollout_max_context_len rollout_max_response_len rollout_num_gpus rollout_num_gpus_per_engine
    rollout_shuffle rollout_skip_special_tokens rollout_temperature rollout_top_k rollout_top_p
    reset_rollout_state rollout_seed save_interval seed seq_length sequence_parallel sglang_context_length sglang_max_running_requests
    sglang_mem_fraction_static single_checkpoint skip_actor_forward_only tensor_model_parallel_size
    stop_after_save_trigger train_backend use_checkpoint_opt_param_scheduler use_distributed_optimizer use_dynamic_batch_size
    use_kl_loss use_rollout_logprobs weight_decay
    """.split())


def training_log_config(directory, resolved):
    """Publish actual launcher settings through a positive list, never credentials."""
    return {
        "native_metrics_source": "VM single writer",
        "training": {name: value for name, value in resolved["args"].items() if name in PUBLIC_TRAINING_ARGUMENTS},
        "resolved_config_sha256": hashlib.sha256((directory / "resolved-config.json").read_bytes()).hexdigest(),
        "judge": resolved.get("judge"),
        "judge_prompt_sha256": resolved.get("judge_prompt_sha256"),
        "dataset_sha256": resolved.get("dataset_sha256"),
        "question_set_key": resolved.get("question_key"),
        "question_count": resolved.get("question_count"),
        "gradient_accumulation": "Dynamic token batches; actual count is audit/train/num_microbatches per update",
    }


def published_optimizer_steps(directory, *, allow_pending):
    path = directory / "events/main.jsonl"
    if not path.exists():
        return set()
    lines = path.read_text().splitlines()
    published = set()
    for position, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if allow_pending and position == len(lines) - 1:
                continue
            raise
        if event["type"] == "inference_engine_weight_checksum":
            published.add(event["rollout_id"])
    return published


def read_native_metrics(directory, *, allow_pending=False, require_publication=False):
    # Restoring a checkpoint also restores its historical event files. The
    # reviewed checkpoint owns the boundary between inherited and new updates.
    config_path = directory / "approved-config.json"
    initial = initial_optimizer_updates(json.loads(config_path.read_text())) if config_path.exists() else 0
    rows = {}
    sources = ["actor_cell00000_rank00000.jsonl", "rollout_executor.jsonl"]
    for name in sources:
        path = directory / "events" / name
        if allow_pending and not path.exists():
            continue
        lines = path.read_text().splitlines()
        for position, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if allow_pending and position == len(lines) - 1:
                    continue
                raise
            if event["type"] != "metric":
                continue
            step = event["rollout_id"]
            if step < initial:
                continue
            metrics = event["metrics"]
            row = rows.setdefault(step, {"optimizer_step": step})
            for key, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite native metric at step {step}: {key}")
                if key.endswith("/step") and value != step:
                    raise ValueError(f"Native metric axis disagrees at step {step}: {key}")
                if key in row and row[key] != value:
                    raise ValueError(f"Conflicting native metric at step {step}: {key}")
                row[key] = value
    observed = sorted(rows)
    published = (
        published_optimizer_steps(directory, allow_pending=allow_pending) if require_publication else set(observed)
    )
    if allow_pending:
        complete = []
        required = {"train/pg_loss", "rollout/episode_raw_reward"}
        if require_publication:
            required.add("perf/update_weights_time")
        for step in observed:
            if not required <= rows[step].keys() or step not in published:
                break
            complete.append(step)
        observed = complete
        if not observed:
            return []
    if not observed or observed != list(range(observed[0], observed[-1] + 1)):
        raise ValueError("Native optimizer steps are missing or noncontiguous")
    for step in observed:
        if "train/pg_loss" not in rows[step] or "rollout/episode_raw_reward" not in rows[step]:
            raise ValueError(f"Missing optimizer or rollout metrics at step {step}")
        if step not in published:
            raise ValueError(f"Missing actual weight publication at step {step}")
    return [rows[step] for step in observed]


def compare_wandb(native, history, *, metric, step_key):
    expected = {(row["optimizer_step"], row[metric]) for row in native}
    observed = [(row.get(step_key), row[metric]) for row in history if row.get(metric) is not None]
    return {
        "metric": metric,
        "native_points": len(expected),
        "wandb_points": len(observed),
        "missing_or_misassigned_native_steps": sorted(
            step for step, value in expected if (step, value) not in observed
        ),
        "misassigned_points": [
            {"step": step, "value": value} for step, value in observed if (step, value) not in expected
        ],
    }


def publish_audit(native, run):
    """Append corrected, separately named metrics through one VM writer.

    The caller owns authentication and preserves the scheduler's terminal state.
    Never overwrite the original producer series or reuse their history counters.
    """
    run.define_metric("audit/optimizer_step")
    run.define_metric("audit/*", step_metric="audit/optimizer_step")
    for row in native:
        run.log({"audit/" + key: value for key, value in row.items()})


def sync_terminal_audit(directory, output):
    # W&B is optional for local evidence reads and CPU tests. Only this VM
    # publishing command loads the SDK and its normal existing authentication.
    import wandb

    config = json.loads((directory / "resolved-config.json").read_text())
    args = config["args"]
    run_path = f"{args['wandb_team']}/{args['wandb_project']}/{config['run_id']}"
    before = wandb.Api().run(run_path)
    if before.state not in ("failed", "finished"):
        raise ValueError("This recovery command requires a failed or finished W&B run whose state it can preserve")
    native = read_native_metrics(directory)
    digest = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
    if before.summary.get("audit/native_metrics_sha256") != digest:
        client_directory = output.parent / ".wandb-clients" / output.name
        client_directory.mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            entity=args["wandb_team"],
            project=args["wandb_project"],
            id=config["run_id"],
            resume="must",
            dir=str(client_directory),
            # CoreWeave rejects switching an existing ordinary offline history
            # to shared mode (HTTP 409). Resume one writer at the history tail.
            settings=wandb.Settings(mode="online"),
        )
        try:
            publish_audit(native, run)
            run.summary["audit/native_metrics_sha256"] = digest
            run.summary["audit/source"] = "Native event JSON, joined by optimizer rollout ID through one VM writer"
        finally:
            run.finish(exit_code=0 if before.state == "finished" else 1)
    expected = [(row["optimizer_step"], row["rollout/episode_raw_reward"]) for row in native]
    for attempt in range(6):
        # History indexing can lag a successful finish. Retry fresh reads,
        # without resending points or declaring an unverified upload successful.
        after = wandb.Api().run(run_path)
        observed = list(
            after.scan_history(keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"], use_cache=False)
        )
        actual = [
            (row["audit/optimizer_step"], row["audit/rollout/episode_raw_reward"])
            for row in observed
            if row.get("audit/optimizer_step") is not None and row.get("audit/rollout/episode_raw_reward") is not None
        ]
        if actual == expected and after.state == before.state:
            break
        if attempt < 5:
            time.sleep(2)
    else:
        raise RuntimeError("Corrected W&B history or original terminal state failed readback verification")
    receipt = {
        "url": after.url,
        "state_before": before.state,
        "state_after": after.state,
        "native_metrics_sha256": digest,
        "verified_points": len(actual),
        "reward_metric": "audit/rollout/episode_raw_reward",
        "x_axis": "audit/optimizer_step",
    }
    (output / "wandb-corrected-metrics.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def sync_live_audit(directory, output, scheduler_state):
    """Publish only complete native steps, preserving live W&B run state.

    The VM is the sole online writer. AMD's independent offline streams stay
    archived without being merged into this history. A local lock serializes
    monitor retries. API readback owns the publication cursor.
    """
    import wandb

    config_path = directory / "resolved-config.json"
    terminal = scheduler_state not in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "UNKNOWN")
    native = read_native_metrics(directory, allow_pending=True, require_publication=not terminal)
    if not config_path.exists() or not native:
        return {"phase": "waiting_for_first_complete_update", "verified_points": 0}
    config = json.loads(config_path.read_text())
    args = config["args"]
    # Keep the single online writer separate from AMD's offline producer IDs.
    online_id = config["run_id"] + "-native"
    path = f"{args['wandb_team']}/{args['wandb_project']}/{online_id}"
    receipt_path = output / "wandb-live-metrics.json"
    submitted_path = output / "wandb-submitted-metrics.json"
    with (output / "wandb-live-metrics.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        submitted = json.loads(submitted_path.read_text()) if submitted_path.exists() else {}
        last_verified = previous.get("last_verified_optimizer_step", native[0]["optimizer_step"] - 1)
        last = max(last_verified, submitted.get("last_submitted_optimizer_step", last_verified))
        pending = [row for row in native if row["optimizer_step"] > last]
        # Long rollout gaps can outlive the server's SDK heartbeat lease. Refresh
        # the same serialized client without appending another history point.
        heartbeat_due = not terminal and time.time() - submitted.get("last_heartbeat_unix", 0) >= 120
        if not pending and not terminal and last_verified == last and not heartbeat_due:
            return previous
        if pending or terminal or heartbeat_due:
            client_directory = output.parent / ".wandb-clients" / output.name
            client_directory.mkdir(parents=True, exist_ok=True)
            run = wandb.init(
                entity=args["wandb_team"],
                project=args["wandb_project"],
                id=online_id,
                resume="allow",
                dir=str(client_directory),
                config=training_log_config(directory, config),
                settings=wandb.Settings(mode="online", x_update_finish_state=terminal),
            )
            if (previous or submitted) and not run.resumed:
                run.finish(exit_code=1)
                raise RuntimeError("Live metric publication did not resume its existing run")
            try:
                publish_audit(pending, run)
                run.summary["monitor/slurm_state"] = scheduler_state
                run.summary["monitor/completed_updates"] = len(native)
            finally:
                run.finish(exit_code=0 if not terminal or scheduler_state == "COMPLETED" else 1)
            # SDK success and API verification are separate acknowledgements.
            # Persist delivery before waiting for indexing, so monitor retries
            # do not resend a successful upload whose readback was delayed.
            _atomic_bytes(
                submitted_path,
                (
                    json.dumps(
                        {
                            "last_submitted_optimizer_step": native[-1]["optimizer_step"],
                            "last_heartbeat_unix": time.time(),
                        }
                    )
                    + "\n"
                ).encode(),
            )
        expected = [(row["optimizer_step"], row["rollout/episode_raw_reward"]) for row in native]
        expected_state = ("finished" if scheduler_state == "COMPLETED" else "failed") if terminal else "running"
        # The server's history index can lag SDK upload by tens of seconds.
        # Keep one upload and retry reads instead of republishing its points.
        for attempt in range(12):
            remote = wandb.Api().run(path)
            observed = list(
                remote.scan_history(keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"], use_cache=False)
            )
            actual = [
                (row["audit/optimizer_step"], row["audit/rollout/episode_raw_reward"])
                for row in observed
                if row.get("audit/optimizer_step") is not None
                and row.get("audit/rollout/episode_raw_reward") is not None
            ]
            # A retry after a lost acknowledgement may repeat an identical
            # point. Verify unique coordinates and reject conflicting values.
            observed_values = {}
            for step, value in actual:
                if step in observed_values and observed_values[step] != value:
                    raise RuntimeError("Conflicting W&B values at one optimizer step")
                observed_values[step] = value
            if sorted(observed_values.items()) == expected and remote.state == expected_state:
                break
            if attempt < 11:
                time.sleep(3)
        else:
            raise RuntimeError(
                f"Live W&B readback failed: {len(observed_values)}/{len(expected)} steps, "
                f"state {remote.state!r}, expected {expected_state!r}"
            )
        receipt = {
            "url": remote.url,
            "verified_points": len(observed_values),
            "history_rows": len(actual),
            "last_verified_optimizer_step": native[-1]["optimizer_step"],
            "scheduler_state": scheduler_state,
            "wandb_state": remote.state,
            "native_metrics_sha256": hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest(),
        }
        _atomic_bytes(receipt_path, (json.dumps(receipt, indent=2) + "\n").encode())
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scheduler-state")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    result = (
        sync_live_audit(cli.run_directory, cli.output, cli.scheduler_state)
        if cli.scheduler_state
        else sync_terminal_audit(cli.run_directory, cli.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
