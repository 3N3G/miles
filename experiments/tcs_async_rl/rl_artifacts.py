"""Preserve completed outputs and export trusted snapshots with exact policy replay inputs."""

import gzip
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

from experiments.tcs_async_rl.group_inference import final_answer

SAMPLE_FIELDS = (
    "index",
    "group_index",
    "prompt",
    "tokens",
    "response",
    "response_length",
    "label",
    "reward",
    "status",
    "metadata",
    "weight_versions",
    "rollout_log_probs",
    "loss_mask",
    "rollout_sampling_mask",
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def portable_sample(sample):
    row = {key: sample.get(key) for key in SAMPLE_FIELDS}
    mask = row["rollout_sampling_mask"]
    if mask is not None:
        # Borrow the framework-owned CPU CSR tensors for read-only transport.
        ids, offsets = mask._as_tensors()
        row["rollout_sampling_mask"] = {"encoding": "csr", "ids": ids.tolist(), "offsets": offsets.tolist()}
    answer = final_answer(row["response"] or "")
    candidate = answer if answer else None
    if candidate is None and row["status"] == "truncated":
        candidate = ""
    row["final_answer"] = answer
    row["judge_identity"] = (
        "online-" + hashlib.sha256(candidate.encode()).hexdigest()[:24] if candidate is not None else None
    )
    return row


def _atomic_bytes(path, data):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def save_generated_sample(output_pattern, sample, *, compress=False):
    if not output_pattern:
        return None
    data = json.dumps(portable_sample(sample), ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    identity = hashlib.sha256(data).hexdigest()
    directory = Path(output_pattern).parent / "completed_samples"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (f"{identity}.generated.json.gz" if compress else f"{identity}.generated.json")
    if not path.exists():
        _atomic_bytes(path, gzip.compress(data + b"\n", compresslevel=1, mtime=0) if compress else data + b"\n")
    return path


def save_judge_outcome(generated_path, *, result=None, error_type=None):
    if generated_path is None:
        return
    event = {"generated_sample": generated_path.name, "result": result, "error_type": error_type}
    path = generated_path.with_name(f"{generated_path.stem}.judge-{time.time_ns()}.json")
    _atomic_bytes(path, (json.dumps(event, allow_nan=False) + "\n").encode())


def export_rollouts(run_dir):
    if os.environ.get("TCS_COMPRESS_ARTIFACTS") == "gzip":
        manifest = Path(run_dir) / "rollout-export.json"
        return json.loads(manifest.read_text()) if manifest.exists() else []
    # Only the allocation runtime reads its own torch.save snapshots. No model
    # weights are loaded, and no PyTorch dependency is needed for VM grading.
    import torch

    exported = []
    for source in sorted((Path(run_dir) / "rollout_data").glob("*.pt")):
        snapshot = torch.load(source, map_location="cpu", weights_only=False)
        rows = []
        for sample in snapshot["samples"]:
            row = portable_sample(sample)
            row["optimizer_rollout_id"] = snapshot["rollout_id"]
            rows.append(json.dumps(row, ensure_ascii=False, allow_nan=False))
        target = source.with_suffix(".jsonl")
        temporary = target.with_suffix(".jsonl.partial")
        temporary.write_text("\n".join(rows) + "\n")
        temporary.replace(target)
        exported.append(
            {"path": str(target), "samples": len(rows), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        )
    (Path(run_dir) / "rollout-export.json").write_text(json.dumps(exported, indent=2) + "\n")
    return exported


def log_rollout(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    """Export the exact delivered group after the native snapshot, before training."""
    directory = Path(args.save_debug_rollout_data).parent
    target = directory / f"rollout-{rollout_id}.jsonl.gz"
    temporary = target.with_suffix(target.suffix + ".partial")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
        for sample in samples:
            row = portable_sample(sample.to_dict())
            row["optimizer_rollout_id"] = rollout_id
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    manifest = directory.parent / "rollout-export.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    entry = {"path": str(target), "samples": len(samples), "sha256": file_sha256(target)}
    entries = [x for x in entries if x["path"] != str(target)] + [entry]
    _atomic_bytes(manifest, (json.dumps(entries, indent=2) + "\n").encode())
    return False  # Preserve the framework's regular rollout metrics.


def retain_latest_checkpoint(args, rollout_id, checkpoint_dir, hf_checkpoint_dir):
    """Keep one completed checkpoint in this submitted run, with at most two during a save."""
    root = Path(args.save).resolve()
    current = Path(checkpoint_dir).resolve()
    receipt = root.parent / "submission.json"
    if current.parent != root or not current.is_dir() or not receipt.is_file() or root.name != "checkpoints":
        raise ValueError("Refuse checkpoint retention outside this submitted run")
    removed = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r"iter_[0-9]{7}", path.name):
            continue
        if path.resolve() != current and int(path.name.split("_")[1]) < rollout_id:
            shutil.rmtree(path)
            removed.append(path.name)
    with (root.parent / "checkpoint-retention.jsonl").open("a") as stream:
        stream.write(json.dumps({"rollout_id": rollout_id, "kept": current.name, "removed": removed}) + "\n")
