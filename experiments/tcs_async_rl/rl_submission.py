"""Submit one approved, frozen RL snapshot, with a durable remote receipt.

An unresolved submission intent blocks a repeat. It never guesses that a lost
SSH response means sbatch did not accept the job. No credentials are accepted.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs, render_job
from experiments.tcs_async_rl.checkpoint_policy import storage_required
from experiments.tcs_async_rl.group_inference import load_input, sampling_parameters
from experiments.tcs_async_rl.resource_estimate import model_inputs
from experiments.tcs_async_rl.rl_inputs import read_bundle
from experiments.tcs_async_rl.rl_smoke import initial_optimizer_updates, validate_group_gate


def submit_once(run_dir, job_file, *, invoke=subprocess.run):
    receipt = run_dir / "submission.json"
    with (run_dir / "submission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            return json.loads(receipt.read_text())
        intent = run_dir / "submission-intent.json"
        if intent.exists():
            raise RuntimeError("Previous sbatch intent has no receipt. Reconcile Slurm before any retry.")
        intent.write_text(
            json.dumps({"at": datetime.datetime.now(datetime.UTC).isoformat(), "job_file": str(job_file)}) + "\n"
        )
        result = invoke(["sbatch", "--parsable", str(job_file)], capture_output=True, text=True, timeout=60)
        (run_dir / "sbatch.stdout").write_text(result.stdout)
        (run_dir / "sbatch.stderr").write_text(result.stderr)
        result.check_returncode()
        identifier = result.stdout.strip().split(";")[0]
        if not identifier.isdigit():
            raise RuntimeError("sbatch returned an unrecognized receipt; do not resubmit")
        value = {"job_id": identifier, "submitted_at_utc": datetime.datetime.now(datetime.UTC).isoformat()}
        # A temporary receipt cannot be mistaken for a completed scheduler response.
        temporary = receipt.with_suffix(".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(receipt)
        return value


def prepare(*, config_path, run_dir, work_dir, expected_commit):
    repo = Path(__file__).resolve().parents[2]
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if (
        actual != expected_commit
        or subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ):
        raise ValueError("RL source is not the requested clean frozen commit")
    config, training = load_configs(config_path)
    input_config, _ = load_configs(config_path.parent / config["input_config"])
    load_input(input_config, training)
    grades = json.loads(Path(config["gate_file"]).read_text())
    if grades.get("input_sha256") != input_config["input_sha256"]:
        raise ValueError("RL gate input hash differs")
    sampling = sampling_parameters({"miles_args": {**training["miles_args"], **config["overrides"]}})
    validate_group_gate(
        grades,
        training["miles_args"]["n_samples_per_prompt"],
        expected_sampling=sampling,
        policy=config.get("diagnostic_gate_policy", "complete_only"),
    )
    if config.get("dataset_file"):
        read_bundle(Path(config["dataset_file"]), expected_sha256=config["dataset_sha256"], training=training)
    if config.get("resume"):
        source = Path(config["resume"]["checkpoint_root"]).resolve()
        if source.name != "checkpoints" or not source.is_relative_to((work_dir / "runs").resolve()):
            raise ValueError("Resume source must be an existing run checkpoint under the approved work directory")
        tracker = source / "latest_checkpointed_iteration.txt"
        if tracker.read_text().strip() != str(initial_optimizer_updates(config) - 1):
            raise ValueError("Resume checkpoint no longer matches the reviewed iteration")
    if run_dir.resolve().parent != (work_dir / "runs").resolve():
        raise ValueError("RL output must be an immediate child of the approved work directory")
    run_dir.mkdir(exist_ok=True)
    if not (run_dir / "submission-intent.json").exists():
        inputs = model_inputs(training, base_dir=config_path.parent)
        parameters = inputs["text_parameters"] + inputs["mtp_parameters"]
        required = storage_required(config, run_dir, parameters)
        free = shutil.disk_usage(work_dir).free
        if free < required:
            raise RuntimeError("Insufficient shared-filesystem space for checkpoint saves and margin")
        body = render_job(
            repo=repo, work_dir=work_dir, run_dir=run_dir, expected_commit=expected_commit, config_path=config_path
        )
        (run_dir / "job.py").write_text(body)
        (run_dir / "submission-plan.json").write_text(
            json.dumps(
                {
                    "source_commit": actual,
                    "config": str(config_path),
                    "allocation": config["allocation"],
                    "gate_sha256": hashlib.sha256(Path(config["gate_file"]).read_bytes()).hexdigest(),
                    "input_sha256": input_config["input_sha256"],
                    "free_bytes": free,
                    "required_bytes": required,
                },
                indent=2,
            )
            + "\n"
        )
    return submit_once(run_dir, run_dir / "job.py")


def audit_checkpoints(run_dir):
    root = run_dir / "checkpoints"
    tracker = root / "latest_checkpointed_iteration.txt"
    result = {"tracker": tracker.read_text().strip() if tracker.exists() else None, "iterations": []}
    if root.exists():
        for path in sorted(root.iterdir()):
            if path.is_dir() and re.fullmatch(r"iter_[0-9]+", path.name):
                files = [p for p in path.rglob("*") if p.is_file()]
                result["iterations"].append(
                    {"name": path.name, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
                )
    (run_dir / "checkpoint-inventory.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--audit-only", action="store_true")
    cli = parser.parse_args()
    if cli.audit_only:
        audit_checkpoints(cli.run_dir)
    else:
        print(
            json.dumps(
                prepare(
                    config_path=cli.config,
                    run_dir=cli.run_dir,
                    work_dir=cli.work_dir,
                    expected_commit=cli.expected_commit,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
