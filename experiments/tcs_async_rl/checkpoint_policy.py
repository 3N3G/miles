"""Bound checkpoint storage without deleting a continuation's source checkpoint.

The single-checkpoint mode disables periodic and epoch saves. A training hook
requests the final save without inspecting model state, using Miles' sentinel. The
foreground supervisor can request one earlier deadline save and stop the owned
job after verifying it. It must never request a second save in this mode.
"""

from pathlib import Path

from experiments.tcs_async_rl.resource_estimate import GIB

FINAL_CHECKPOINT_HOOK = "experiments.tcs_async_rl.checkpoint_policy.request_final_checkpoint"


def single_checkpoint(config, run_dir):
    overrides = config["overrides"]
    if overrides.get("custom_megatron_before_train_step_hook_path") != FINAL_CHECKPOINT_HOOK:
        return False
    if "save_interval" not in overrides or overrides["save_interval"] is not None:
        raise ValueError("Single-checkpoint mode must disable periodic and epoch saves")
    sentinel = overrides.get("save_trigger_sentinel")
    if sentinel != str(Path(run_dir) / "checkpoint-request"):
        raise ValueError("Checkpoint request must belong to this submitted run")
    return True


def storage_required(config, run_dir, parameters):
    copies = 1 if single_checkpoint(config, run_dir) else 2
    # Each model+master+Adam checkpoint costs about 14 bytes per parameter.
    # Preserve the existing conversion allowance and 20-GiB workspace margin.
    return (14 * copies + 2) * parameters + 20 * GIB


def request_final_checkpoint(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Request one synchronous final save without reading or changing training state."""
    if args.save_interval is not None:
        raise ValueError("Final-checkpoint request requires disabled periodic saves")
    root = Path(args.save)
    sentinel = Path(args.save_trigger_sentinel)
    if root.name != "checkpoints" or sentinel != root.parent / "checkpoint-request":
        raise ValueError("Checkpoint request must remain in the owned run")
    if rollout_id == args.num_rollout - 1 and step_id == 0:
        # A completed deadline checkpoint is the one permitted save. The
        # supervisor will stop that owned job. Never create a second disk peak.
        if not (root / "latest_checkpointed_iteration.txt").exists():
            sentinel.touch(exist_ok=True)
