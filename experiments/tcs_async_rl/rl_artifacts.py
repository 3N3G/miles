"""Preserve completed outputs and export trusted snapshots with exact policy replay inputs."""

import hashlib
import json
import os
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


def save_generated_sample(output_pattern, sample):
    if not output_pattern:
        return None
    data = json.dumps(portable_sample(sample), ensure_ascii=False, allow_nan=False, sort_keys=True).encode()
    identity = hashlib.sha256(data).hexdigest()
    directory = Path(output_pattern).parent / "completed_samples"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identity}.generated.json"
    if not path.exists():
        _atomic_bytes(path, data + b"\n")
    return path


def save_judge_outcome(generated_path, *, result=None, error_type=None):
    if generated_path is None:
        return
    event = {"generated_sample": generated_path.name, "result": result, "error_type": error_type}
    path = generated_path.with_name(f"{generated_path.stem}.judge-{time.time_ns()}.json")
    _atomic_bytes(path, (json.dumps(event, allow_nan=False) + "\n").encode())


def export_rollouts(run_dir):
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
