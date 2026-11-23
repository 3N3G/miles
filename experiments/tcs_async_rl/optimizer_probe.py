"""Observe an owned Ray job without a new forward pass.

Run this file as a script inside the allocation's container so Ray serializes
the observer functions by value. The caller verifies Slurm ownership first.
No gradient computation, collective, optimizer method or model tensor is read.
An explicit producer-action option pauses or resumes submission for a drained
reward-service repair. The default remains read-only.
"""

import argparse
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path


def worker_source_files(actor):
    modules = sorted({cls.__module__ for cls in type(actor).__mro__ if cls.__module__.startswith("miles.")})
    paths = {name: Path(sys.modules[name].__file__).resolve() for name in modules}
    return {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    }


def read_optimizer_snapshot(actor):
    optimizer = actor.optimizer
    scheduler = actor.opt_param_scheduler
    grouped = getattr(optimizer, "grad_norms_by_group", None)
    return {
        "rank": actor._rank,
        "run_id": actor.args.wandb_run_id,
        "worker_source_files": worker_source_files(actor),
        "single_checkpoint": actor.args.single_checkpoint,
        "save_interval": actor.args.save_interval,
        "scheduler_counter": scheduler.num_steps,
        "scheduler_decay_steps": scheduler.lr_decay_steps,
        "scheduler_style": scheduler.lr_decay_style,
        "restores_checkpoint_scheduler": actor.args.use_checkpoint_opt_param_scheduler,
        "last_started_rollout": getattr(actor, "_last_rollout_id", None),
        "active_model_tag": actor._active_model_tag,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "weight_decays": [group["weight_decay"] for group in optimizer.param_groups],
        "separate_gradient_norms": None if grouped is None else {k: float(v) for k, v in grouped.items()},
    }


def trainer_names(manager):
    return [
        (actor.name, actor.generation)
        for cell in manager._all_cells()
        if cell.alive and cell.actors is not None
        for actor in cell.actors
        if getattr(actor.spec, "worker_class", "").endswith(".MegatronTrainRayActor")
    ]


def rollout_names(manager):
    return [
        (actor.name, actor.generation)
        for cell in manager._all_cells()
        if cell.alive and cell.actors is not None
        for actor in cell.actors
        if getattr(actor.spec, "worker_class", "").endswith(".RolloutExecutor")
    ]


def await_chain(task):
    current = task.get_coro()
    chain = []
    for _ in range(20):
        if current is None:
            break
        frame = getattr(current, "cr_frame", getattr(current, "gi_frame", None))
        if frame is not None:
            chain.append({"function": frame.f_code.co_name, "line": frame.f_lineno})
        current = getattr(current, "cr_await", getattr(current, "gi_yieldfrom", None))
    return chain


def read_rollout_snapshot(actor):
    fn = actor.generate_rollout
    return {
        "run_id": actor.args.wandb_run_id,
        "worker_source_files": worker_source_files(actor),
        "published_weight_versions": actor._weight_versions_of_model_id,
        "last_requested_rollouts": actor.last_get_rollout_id_of_model_id,
        "samples_in_flight": fn._scheduler.samples_in_flight,
        "generation_semaphore_available": fn.state.generate_fn_semaphore._value,
        "completed_buffer_groups": len(fn._output._buffer),
        "completed_buffer_capacity": fn._output._capacity,
        "retry_groups": len(fn._retry_buffer),
        "producer_resumed": fn._producer_resumed.is_set(),
        "producer_done": fn._worker.done(),
        "producer_await_chain": await_chain(fn._worker),
        "running_groups": [
            {
                "first_sample_index": entry.prompt_group[0].index,
                "samples": len(entry.prompt_group),
                "statuses": dict(collections.Counter(sample.status.name for sample in entry.prompt_group)),
                "task_done": entry.task.done(),
                "await_chain": await_chain(entry.task),
            }
            for entry in list(fn._running_tasks)
        ],
    }


def change_producer_state(actor, action, expected_run_id):
    if actor.args.wandb_run_id != expected_run_id or action not in ("pause", "resume"):
        raise ValueError("Producer action does not match the owned run")
    fn = actor.generate_rollout
    callback = fn._producer_resumed.clear if action == "pause" else fn._producer_resumed.set
    fn._worker.get_loop().call_soon_threadsafe(callback)
    return {"producer_action_queued": action, "run_id": expected_run_id}


def main():
    # Ray belongs to the allocated AMD runtime, not the VM's review utilities.
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-workers", type=int, default=2)
    parser.add_argument(
        "--actor-read-timeout-seconds",
        type=int,
        default=300,
        help="Allow the read-only snapshot to queue behind an active long training update.",
    )
    parser.add_argument("--include-rollout", action="store_true")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--producer-action", choices=("pause", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.actor_read_timeout_seconds <= 0:
        parser.error("--actor-read-timeout-seconds must be positive")
    ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    try:
        managers = [x for x in ray.util.list_named_actors(all_namespaces=True) if x["name"] == "ray_worker_manager"]
        if len(managers) != 1:
            raise RuntimeError("Expected one worker manager in the owned allocation")
        manager = ray.get_actor(managers[0]["name"], namespace=managers[0]["namespace"])
        result = {
            "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "actor_read_timeout_seconds": cli.actor_read_timeout_seconds,
        }
        if not cli.producer_action and not cli.rollout_only:
            names = ray.get(manager.__ray_call__.remote(trainer_names), timeout=60)
            if len(names) != cli.expected_workers:
                raise RuntimeError("Expected the reviewed number of initialized training workers")
            handles = ray.get(
                [manager.get_actor_handle.remote(name, expected_generation=generation) for name, generation in names],
                timeout=60,
            )
            snapshots = ray.get(
                [handle.__ray_call__.remote(read_optimizer_snapshot) for handle in handles],
                timeout=cli.actor_read_timeout_seconds,
            )
            if any(r["run_id"] != cli.expected_run_id for r in snapshots):
                raise RuntimeError("Optimizer probe reached a different run")
            result["workers"] = snapshots
        if cli.include_rollout or cli.rollout_only or cli.producer_action:
            names = ray.get(manager.__ray_call__.remote(rollout_names), timeout=60)
            if len(names) != 1:
                raise RuntimeError("Expected one initialized rollout executor")
            name, generation = names[0]
            handle = ray.get(manager.get_actor_handle.remote(name, expected_generation=generation), timeout=60)
            if cli.producer_action:
                result.update(
                    ray.get(
                        handle.__ray_call__.remote(change_producer_state, cli.producer_action, cli.expected_run_id),
                        timeout=30,
                    )
                )
            rollout = ray.get(handle.__ray_call__.remote(read_rollout_snapshot), timeout=90)
            if rollout["run_id"] != cli.expected_run_id:
                raise RuntimeError("Rollout probe reached a different run")
            result["rollout"] = rollout
        cli.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
