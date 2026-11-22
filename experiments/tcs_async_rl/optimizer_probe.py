"""Read existing optimizer scalars from an owned Ray job without a new forward pass.

Run this file as a script inside the allocation's container so Ray serializes
the observer functions by value. The caller verifies Slurm ownership first.
No gradient computation, collective, optimizer method or model tensor is read.
"""

import argparse
import datetime
import json
from pathlib import Path


def read_optimizer_snapshot(actor):
    optimizer = actor.optimizer
    scheduler = actor.opt_param_scheduler
    grouped = getattr(optimizer, "grad_norms_by_group", None)
    return {
        "rank": actor._rank,
        "run_id": actor.args.wandb_run_id,
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


def main():
    # Ray belongs to the allocated AMD runtime, not the VM's review utilities.
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    try:
        managers = [x for x in ray.util.list_named_actors(all_namespaces=True) if x["name"] == "ray_worker_manager"]
        if len(managers) != 1:
            raise RuntimeError("Expected one worker manager in the owned allocation")
        manager = ray.get_actor(managers[0]["name"], namespace=managers[0]["namespace"])
        names = ray.get(manager.__ray_call__.remote(trainer_names), timeout=60)
        if len(names) != cli.expected_workers:
            raise RuntimeError("Expected the reviewed number of initialized training workers")
        handles = ray.get(
            [manager.get_actor_handle.remote(name, expected_generation=generation) for name, generation in names],
            timeout=60,
        )
        snapshots = ray.get([handle.__ray_call__.remote(read_optimizer_snapshot) for handle in handles], timeout=90)
        if any(r["run_id"] != cli.expected_run_id for r in snapshots):
            raise RuntimeError("Optimizer probe reached a different run")
        result = {"at_utc": datetime.datetime.now(datetime.UTC).isoformat(), "workers": snapshots}
        cli.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
