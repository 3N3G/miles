"""Exercise the native offline W&B adapter in our existing compute allocation."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this CPU-only control inside our existing compute allocation")
    cli.run_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        WANDB_MODE="offline",
        WANDB_CONFIG_DIR=str(cli.run_dir / "config"),
        WANDB_DATA_DIR=str(cli.run_dir / "data"),
        WANDB_CACHE_DIR=str(cli.work_dir / "cache" / "wandb"),
    )
    # The training image supplies W&B. No credential loader or online login is used.
    import wandb

    from miles.utils.tracking_utils.wandb_utils import init_wandb_primary

    args = SimpleNamespace(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_team=None,
        wandb_project="tcs-async-rl",
        wandb_group="offline-logging-control",
        wandb_run_id="offline-logging-control-v1",
        wandb_random_suffix=False,
        wandb_dir=str(cli.run_dir),
        env_report=None,
        rank=0,
    )
    init_wandb_primary(args)
    try:
        if not wandb.run.offline:
            raise RuntimeError("Logging control must remain offline")
        wandb.log({"diagnostic/offline_logging_control": 1.0}, step=0)
    finally:
        wandb.finish()
    files = sorted(cli.run_dir.rglob("*.wandb"))
    if not files or not all(path.stat().st_size > 0 for path in files):
        raise RuntimeError("Native W&B adapter did not produce a nonempty offline run")
    result = {
        "passed": True,
        "mode": "offline",
        "wandb_version": importlib.metadata.version("wandb"),
        "parent_job_id": os.environ["SLURM_JOB_ID"],
        "offline_files": [str(path) for path in files],
        "scope": "Native primary adapter and durable offline output only. No online-account authentication or RL update.",
    }
    (cli.run_dir / "logging-control.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
