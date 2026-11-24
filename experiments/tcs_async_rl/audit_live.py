"""Audit newly consumed training groups from exact private archive revisions.

Run with the Hugging Face CLI's Python environment. This reads saved outputs
and judge receipts. It makes no inference request and changes no training job.
"""

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import read_samples, save, validate_group_metrics
from experiments.tcs_async_rl.native_metrics import read_native_metrics


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def audit(directory, output, traces):
    # Optional SDK belongs to the VM's archive reader, not the GPU runtime.
    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    archived = load(directory / "archive-index.json", {})
    previous_rows = load(output / "samples.json", [])
    previous_groups = load(output / "groups.json", [])
    audited = {group["optimizer_step"] for group in previous_groups}
    pending = []
    for row in native:
        step = row["optimizer_step"]
        if step in audited:
            continue
        name = f"rollout_data/rollout-{step}.jsonl.gz"
        if name not in archived:
            break
        entry = archived[name]
        match = re.fullmatch(
            r"https://huggingface.co/datasets/3N3G/internalization-runs/commit/([0-9a-f]{40})",
            entry["hf_commit_url"],
        )
        if match is None:
            raise ValueError("Consumed trace lacks an immutable private archive revision")
        target = traces / name
        if not target.exists():
            source = Path(
                hf_hub_download(
                    "3N3G/internalization-runs",
                    f"miles/{directory.name}/{name}",
                    repo_type="dataset",
                    revision=match[1],
                )
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("Consumed trace differs from its immutable archive")
        pending.append(step)
    if pending:
        rows, groups = read_samples(directory, traces, output, optimizer_steps=pending)
        validate_group_metrics(groups, native)
        previous_rows.extend(rows)
        previous_groups.extend(groups)
        save(
            output / "samples.json",
            sorted(previous_rows, key=lambda row: (row["optimizer_step"], row["sample_index"])),
        )
        save(output / "groups.json", sorted(previous_groups, key=lambda row: row["optimizer_step"]))
    result = {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "newly_audited_optimizer_steps": pending,
        "audited_optimizer_steps": sorted({group["optimizer_step"] for group in previous_groups}),
        "audited_prompt_groups": len(previous_groups),
        "answers": len(previous_rows),
        "response_tokens": sum(row["response_tokens"] for row in previous_rows),
        "judge_order_disagreements": sum(
            row["order_gap"] > 0 for row in previous_rows if row["order_gap"] is not None
        ),
        "reference_scores_below_seven": sum(
            row["minimum_reference_score"] < 7 for row in previous_rows if row["minimum_reference_score"] is not None
        ),
        "reference_scores_measured": sum(row["minimum_reference_score"] is not None for row in previous_rows),
        "maximum_order_gap": max(
            (row["order_gap"] for row in previous_rows if row["order_gap"] is not None), default=None
        ),
        "new_inference_or_judge_calls": 0,
        "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    save(output / "live-audit-receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    cli = parser.parse_args()
    print(json.dumps(audit(cli.run_directory, cli.output, cli.trace_directory), indent=2))


if __name__ == "__main__":
    main()
