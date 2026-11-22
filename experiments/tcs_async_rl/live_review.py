"""Summarize every native metric and matched-question visits during an owned run.

Reviews are descriptive. They do not select a training variant, change a job or
turn mixed-question rewards into a learning curve.
"""

import argparse
import datetime
import json
import statistics
import time
from pathlib import Path

from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes


def group_rows(groups):
    rows = []
    for group in groups:
        step = group["rollout_ids"][0]
        if len(group["rollout_ids"]) != 1:
            raise ValueError("One consumed group has conflicting optimizer IDs")
        for key, data in group["by_question"].items():
            scores = data["scores"]
            rows.append(
                {
                    "optimizer_step": step,
                    "question_key": key,
                    "mean_reward": statistics.mean(scores),
                    "samples": len(scores),
                    "zero_answers": scores.count(0),
                    "full_credit_answers": scores.count(7),
                    "capped_answers": data["capped"],
                    "mean_response_tokens": statistics.mean(data["lengths"]),
                    "raw_example_file": group["example_file"],
                    "decode_tokens_by_weight_version": group["decode_tokens_by_weight_version"],
                }
            )
    return sorted(rows, key=lambda r: r["optimizer_step"])


def matched_changes(first_rows, later_rows):
    first, last = {}, {}
    for row in first_rows:
        first.setdefault(row["question_key"], row)
    for row in later_rows:
        last[row["question_key"]] = row
    pairs = []
    for key in sorted(first.keys() & last.keys()):
        a, b = first[key], last[key]
        pairs.append(
            {
                "question_key": key,
                "first_step": a["optimizer_step"],
                "last_step": b["optimizer_step"],
                "first_reward": a["mean_reward"],
                "last_reward": b["mean_reward"],
                "change": b["mean_reward"] - a["mean_reward"],
            }
        )
    return {
        "questions": len(pairs),
        "mean_change": statistics.mean(p["change"] for p in pairs) if pairs else None,
        "increased": sum(p["change"] > 0 for p in pairs),
        "decreased": sum(p["change"] < 0 for p in pairs),
        "pairs": pairs,
    }


def build_review(native, groups, baseline):
    completed = {r["optimizer_step"] for r in native}
    rows = [r for r in group_rows(groups) if r["optimizer_step"] in completed]
    repeated = [r for r in rows if sum(x["question_key"] == r["question_key"] for x in rows) > 1]
    metrics = {}
    for key in sorted({key for row in native for key in row} - {"optimizer_step"}):
        points = [(row["optimizer_step"], row[key]) for row in native if key in row]
        values = [v for _, v in points]
        metrics[key] = {
            "points": len(points),
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "last_step": points[-1][0],
        }
    flags = {
        "negative_observed_surrogate_gain": [
            r["optimizer_step"] for r in native if r.get("train/update_direction/surrogate_gain", 0) < 0
        ],
        "nonpositive_learning_rate": [
            r["optimizer_step"] for r in native if any(v <= 0 for k, v in r.items() if k.startswith("train/lr-pg_"))
        ],
        "missing_diagnostics": {
            str(r["optimizer_step"]): [
                k for k in ["train/full_policy_entropy", "train/sampling_entropy", "train/kl_loss"] if k not in r
            ]
            for r in native
        },
    }
    return {
        "completed_updates": len(native),
        "optimizer_steps": sorted(completed),
        "consumed_answers": sum(r["samples"] for r in rows),
        "questions": len({r["question_key"] for r in rows}),
        "metrics": metrics,
        "flags_to_inspect": flags,
        "groups": rows,
        "within_continuation_matched_visits": matched_changes(repeated, repeated),
        "prior_first_training_visit_vs_current_latest": matched_changes(baseline, rows),
        "comparison_limit": "Descriptive training visits from one seed, not frozen-policy evaluations or seed-level uncertainty.",
    }


def review(directory, baseline_path):
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    cache_path = directory / "consumed-summary-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    initial = json.loads((directory / "approved-config.json").read_text())["resume"]["iteration"] + 1
    baseline = [r for r in json.loads(baseline_path.read_text()) if r["optimizer_step"] < initial]
    result = build_review(native, [v["summary"] for v in cache.values()], baseline)
    result["at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    result["initial_saved_updates"] = initial
    state_path = directory / "handoff-state.json"
    result["monitor"] = json.loads(state_path.read_text()) if state_path.exists() else None
    budget = directory / "judging/budget.jsonl"
    result["judge_usd_spent_or_reserved"] = (
        sum(json.loads(line)["usd_delta"] for line in budget.read_text().splitlines()) if budget.exists() else 0
    )
    with (directory / "detailed-reviews.jsonl").open("a") as output:
        output.write(json.dumps(result, allow_nan=False) + "\n")
    _atomic_bytes(directory / "latest-detailed-review.json", (json.dumps(result, indent=2) + "\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--baseline-groups", type=Path, required=True)
    parser.add_argument("--every-seconds", type=int)
    cli = parser.parse_args()
    while True:
        result = review(cli.run_directory, cli.baseline_groups)
        print(json.dumps({k: v for k, v in result.items() if k not in ("metrics", "groups", "monitor")}), flush=True)
        if not cli.every_seconds or result["monitor"]["phase"] == "finished":
            break
        time.sleep(cli.every_seconds)


if __name__ == "__main__":
    main()
