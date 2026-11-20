"""Export our own trusted Miles rollout snapshots to portable private JSONL."""

import hashlib
import json
from pathlib import Path

from experiments.tcs_async_rl.group_inference import final_answer


def export_rollouts(run_dir):
    # Only the allocation runtime reads its own torch.save snapshots. No model
    # weights are loaded, and no PyTorch dependency is needed for VM grading.
    import torch

    exported = []
    fields = (
        "index",
        "group_index",
        "prompt",
        "response",
        "response_length",
        "label",
        "reward",
        "status",
        "metadata",
        "weight_versions",
    )
    for source in sorted((Path(run_dir) / "rollout_data").glob("*.pt")):
        snapshot = torch.load(source, map_location="cpu", weights_only=False)
        rows = []
        for sample in snapshot["samples"]:
            row = {key: sample[key] for key in fields}
            answer = final_answer(row["response"])
            row.update(
                optimizer_rollout_id=snapshot["rollout_id"],
                final_answer=answer,
                judge_identity="online-" + hashlib.sha256(answer.encode()).hexdigest()[:24] if answer else None,
            )
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
