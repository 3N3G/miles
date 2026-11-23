"""Bound checkpoint storage without deleting a continuation's source checkpoint.

The native single-checkpoint mode overrides periodic and epoch saves. The
foreground supervisor can request one earlier deadline save through Miles'
sentinel and stop the owned job after verifying it.
"""

import shutil
from pathlib import Path

from experiments.tcs_async_rl.resource_estimate import GIB


def single_checkpoint(config, run_dir):
    overrides = config["overrides"]
    if not overrides.get("single_checkpoint", False):
        return False
    if not isinstance(overrides.get("save_interval"), int) or overrides["save_interval"] <= 0:
        raise ValueError("Megatron requires a positive save interval")
    sentinel = overrides.get("save_trigger_sentinel")
    if sentinel != str(Path(run_dir) / "checkpoint-request"):
        raise ValueError("Checkpoint request must belong to this submitted run")
    return True


def storage_required(config, run_dir, parameters):
    copies = 1 if single_checkpoint(config, run_dir) else 2
    # Each model+master+Adam checkpoint costs about 14 bytes per parameter.
    # Preserve the existing conversion allowance and 20-GiB workspace margin.
    return (14 * copies + 2) * parameters + 20 * GIB


def check_storage(config, run_dir, parameters, work_dir):
    required = storage_required(config, run_dir, parameters)
    free = shutil.disk_usage(work_dir).free
    if free < required:
        raise RuntimeError(f"Checkpoint saves require {required} free bytes; found {free}")
    return {"free_bytes": free, "required_bytes": required, "parameters": parameters}
