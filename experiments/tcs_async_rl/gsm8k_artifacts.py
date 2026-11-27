"""Portable numeric-reward evidence for the approved GSM8K control."""

import collections
import gzip
import json
import statistics
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes, file_sha256, portable_sample


def write_json(path, value, *, default=None):
    _atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False, default=default) + "\n").encode())


def numeric_rows(samples):
    # Import the real production checker only inside the allocation runtime.
    from miles.rollout.rm_hub.math_utils import extract_answer, grade_answer_verl

    rows = []
    for sample in samples:
        row = portable_sample(sample.to_dict() if hasattr(sample, "to_dict") else sample)
        row.pop("judge_identity")
        if row["status"] not in ("completed", "truncated") or row["reward"] not in (0, 1):
            raise ValueError("Inference or reward failures cannot become zero-score examples")
        actual = int(grade_answer_verl(row["response"], row["label"]))
        if actual != row["reward"]:
            raise ValueError("Recorded reward differs from the native numeric checker")
        row["parsed_answer"] = extract_answer(row["response"])
        rows.append(row)
    return rows


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize_rows(rows):
    if not rows:
        raise ValueError("Empty control sample")
    return {
        "answers": len(rows),
        "correct": sum(row["reward"] for row in rows),
        "accuracy": statistics.mean(row["reward"] for row in rows),
        "valid_box_fraction": statistics.mean(row["parsed_answer"] is not None for row in rows),
        "capped_fraction": statistics.mean(row["status"] == "truncated" for row in rows),
        "mean_response_tokens": statistics.mean(row["response_length"] for row in rows),
    }


def log_training(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    root = Path(args.save_debug_rollout_data).parent.parent
    rows = numeric_rows(samples)
    output = root / "rollout_data" / f"rollout-{rollout_id}.jsonl.gz"
    write_rows(output, rows)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group_index"]].append(row["reward"])
    report = {
        "optimizer_step": rollout_id,
        **summarize_rows(rows),
        "groups": len(groups),
        "varied_groups": sum(len(set(scores)) > 1 for scores in groups.values()),
        "group_variation": [len(set(scores)) > 1 for scores in groups.values()],
        "path": str(output.relative_to(root)),
        "sha256": file_sha256(output),
    }
    folder = root / "training-summaries"
    folder.mkdir(exist_ok=True)
    write_json(folder / f"{rollout_id:06d}.json", report)
    initial = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))[:8]]
    first_eight = [varied for item in initial for varied in item["group_variation"]][:8]
    if len(first_eight) == 8 and not any(first_eight):
        write_json(root / "uninformative.json", {"reason": "No varied groups in the first eight groups"})
        raise RuntimeError("Approved control has no reward variation")
    return False


def export_evaluation(args, *, rollout_id, optimizer_updates):
    # Only the allocation reads its own trusted torch snapshot.
    import torch

    root = Path(args.save_debug_rollout_data).parent.parent
    source = Path(args.save_debug_rollout_data.format(rollout_id=f"eval_{rollout_id}"))
    rows = numeric_rows(torch.load(source, map_location="cpu", weights_only=False)["samples"])
    expected = {
        json.loads(line)["metadata"]["source_row"]
        for line in (root / "validation.jsonl").read_text().splitlines()
    }
    actual = [row["metadata"]["source_row"] for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Evaluation questions are incomplete, duplicated, or changed")
    folder = root / "evaluations"
    folder.mkdir(exist_ok=True)
    name = "baseline" if optimizer_updates == 0 else f"updates-{optimizer_updates:06d}"
    output = folder / f"{name}.jsonl.gz"
    write_rows(output, rows)
    result = {
        "optimizer_updates": optimizer_updates,
        "rollout_id": rollout_id,
        **summarize_rows(rows),
        "responses": str(output.relative_to(root)),
        "sha256": file_sha256(output),
        "scores": {str(row["metadata"]["source_row"]): row["reward"] for row in rows},
    }
    write_json(folder / f"{name}.json", result)
    source.unlink()
    return result
