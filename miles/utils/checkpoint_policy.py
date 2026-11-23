"""Choose checkpoint saves without creating a second checkpoint in single-save mode."""

from argparse import Namespace
from pathlib import Path


def should_save_checkpoint(args: Namespace, rollout_id: int, *, external_save: bool, periodic_save: bool) -> bool:
    if not args.single_checkpoint:
        return external_save or periodic_save
    completed = Path(args.save) / "latest_checkpointed_iteration.txt"
    if completed.exists():
        if external_save:
            raise RuntimeError("Single-checkpoint mode already has a completed checkpoint")
        return False
    return external_save or rollout_id == args.num_rollout - 1
