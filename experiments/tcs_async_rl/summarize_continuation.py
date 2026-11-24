"""Summarize saved continuation evidence without sampling or changing training."""

import argparse
import collections
import csv
import datetime
import json
import statistics
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import (
    describe,
    per_question_steps,
    question_comparisons,
    save,
    validate_group_metrics,
)
from experiments.tcs_async_rl.native_metrics import read_native_metrics
from experiments.tcs_async_rl.rl_smoke import initial_optimizer_updates


def weight_checks(directory, native, initial):
    events = [json.loads(line) for line in (directory / "events/main.jsonl").read_text().splitlines()]
    publications = [event for event in events if event["type"] == "inference_engine_weight_checksum"]
    restored = [event for event in publications if event["rollout_id"] == initial - 1]
    steps = {row["optimizer_step"] for row in native}
    current = [event for event in publications if event["rollout_id"] in steps]
    if len(current) != len(native) or {event["rollout_id"] for event in current} != steps:
        raise ValueError("Completed updates and new weight publications do not match")
    changes = []
    previous = restored[-1]["engine_checksums"][0]
    for event in current:
        engines = event["engine_checksums"]
        if len(engines) != 3 or not all(engine == engines[0] for engine in engines):
            raise ValueError("The three rollout engines disagree")
        after = engines[0]
        names = [name for name in after if "/visual." not in name and "/mtp." not in name]
        changes.append(
            {
                "optimizer_step": event["rollout_id"],
                "language_tensors": len(names),
                "changed_language_tensors": sum(previous.get(name) != after[name] for name in names),
            }
        )
        previous = after
    return {
        "restored_coordinate": initial - 1,
        "restored_coordinate_publications": len(restored),
        "new_publications": len(current),
        "all_three_engines_agree": True,
        "new_language_tensor_changes": changes,
    }


def telemetry_summary(directory):
    path = directory / "gpu-telemetry.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    valid = [row for row in rows if "gpus" in row]
    cards = collections.defaultdict(list)
    for row in valid:
        for card, data in row["gpus"].items():
            cards[card].append(data)
    return {
        "samples": len(rows),
        "valid_samples": len(valid),
        "errors": sum("error_type" in row for row in rows),
        "minimum_observed_shared_free_bytes": min((row["shared_free_bytes"] for row in valid), default=None),
        "cards": {
            card: {
                "peak_observed_vram_bytes": max(int(row["VRAM Total Used Memory (B)"]) for row in data),
                "total_vram_bytes": int(data[0]["VRAM Total Memory (B)"]),
                "mean_sampled_gpu_use_percent": statistics.mean(float(row["GPU use (%)"]) for row in data),
            }
            for card, data in cards.items()
        },
        "limit": "Minute samples can miss peaks. Sampled utilization is not a continuous busy-time measurement.",
    }


def summarize(directory, output):
    config = json.loads((directory / "approved-config.json").read_text())
    initial = initial_optimizer_updates(config)
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    by_step = {row["optimizer_step"]: row for row in native}
    groups = json.loads((output / "groups.json").read_text())
    samples = json.loads((output / "samples.json").read_text())
    validate_group_metrics(groups, native)
    question_steps = per_question_steps(groups)
    visits = collections.defaultdict(list)
    for group in question_steps:
        visits[group["question_key"]].append(group)
    coverage = [
        {
            "question_key": key,
            "paper_id": entries[0]["paper_id"],
            "prompt_type": entries[0]["prompt_type"],
            "visits": [
                {k: group[k] for k in ("optimizer_step", "mean_reward", "cap_rate", "mean_tokens", "example")}
                for group in entries
            ],
        }
        for key, entries in visits.items()
    ]
    pairs = question_comparisons(groups)
    groups_by_step = {(group["optimizer_step"], group["question_key"]): group for group in question_steps}
    matched = {}
    for metric in ("full_credit_rate", "zero_rate", "mean_tokens", "cap_rate"):
        first = [groups_by_step[(pair["first_step"], pair["question_key"])][metric] for pair in pairs]
        last = [groups_by_step[(pair["last_step"], pair["question_key"])][metric] for pair in pairs]
        matched[metric] = {
            "questions": len(pairs),
            "first_mean": statistics.mean(first) if first else None,
            "latest_mean": statistics.mean(last) if last else None,
        }
    for pair in pairs:
        for key in ("train/full_policy_entropy", "train/sampling_entropy", "train/kl_loss"):
            pair[key] = {
                "first": by_step[pair["first_step"]][key],
                "last": by_step[pair["last_step"]][key],
                "change": by_step[pair["last_step"]][key] - by_step[pair["first_step"]][key],
            }
    tokens = sum(sample["response_tokens"] for sample in samples)
    caps = [sample for sample in samples if sample["status"] == "truncated"]
    budget = directory / "judging/budget.jsonl"
    result = {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "initial_saved_updates": initial,
        "completed_new_updates": len(native),
        "audited_groups": len(groups),
        "audited_answers": len(samples),
        "audited_response_tokens": tokens,
        "questions": len(visits),
        "question_visit_count_histogram": dict(collections.Counter(len(entries) for entries in visits.values())),
        "questions_with_repeated_visits": len(pairs),
        "matched_first_mean_reward": statistics.mean(pair["first_reward"] for pair in pairs) if pairs else None,
        "matched_latest_mean_reward": statistics.mean(pair["last_reward"] for pair in pairs) if pairs else None,
        "same_question_mean_change": statistics.mean(pair["difference"] for pair in pairs) if pairs else None,
        "questions_increased": sum(pair["difference"] > 0 for pair in pairs),
        "questions_decreased": sum(pair["difference"] < 0 for pair in pairs),
        "matched_visit_diagnostics": matched,
        "varied_groups": sum(group["reward_std"] > 0 for group in groups),
        "zero_answers": sum(sample["reward"] == 0 for sample in samples),
        "full_credit_answers": sum(sample["reward"] == 7 for sample in samples),
        "capped_answers": len(caps),
        "capped_answers_with_positive_reward": sum(sample["reward"] > 0 for sample in caps),
        "capped_token_fraction": sum(sample["response_tokens"] for sample in caps) / tokens,
        "tokens_with_singleton_support_fraction": sum(sample["tokens_with_support_size_one"] for sample in samples)
        / tokens,
        "judge_order_measurements": sum(sample["order_gap"] is not None for sample in samples),
        "judge_order_disagreements": sum(
            sample["order_gap"] > 0 for sample in samples if sample["order_gap"] is not None
        ),
        "maximum_judge_order_gap": max(
            (sample["order_gap"] for sample in samples if sample["order_gap"] is not None), default=None
        ),
        "reference_scores_measured": sum(sample["minimum_reference_score"] is not None for sample in samples),
        "reference_scores_below_seven": sum(
            sample["minimum_reference_score"] < 7
            for sample in samples
            if sample["minimum_reference_score"] is not None
        ),
        "judge_usd_spent_or_reserved": sum(json.loads(line)["usd_delta"] for line in budget.read_text().splitlines()),
        "native_metric_ranges": {
            key: describe([row[key] for row in native if key in row])
            for key in sorted({key for row in native for key in row})
        },
        "update_direction": [
            {
                key: value
                for key, value in row.items()
                if key == "optimizer_step" or key.startswith("train/update_direction/")
            }
            for row in native
            if "train/update_direction/surrogate_gain" in row
        ],
        "outside_training_fraction_of_summed_step_time": sum(row["perf/train_wait_time"] for row in native)
        / sum(row["perf/step_time"] for row in native),
        "weight_publication": weight_checks(directory, native, initial),
        "gpu_telemetry": telemetry_summary(directory),
        "limits": [
            "One seed and training visits, without frozen-policy evaluation or seed-level uncertainty.",
            "Human judge calibration and independent-provider verification remain pending.",
            "Sampled k3 is measured on rollout prefixes, not exact full-policy KL.",
            "Per-token effective sample size does not certify full-trajectory importance weights.",
            "Time outside the training function includes saving and weight publication.",
        ],
    }
    save(output / "native-metrics.json", native)
    save(output / "question-coverage.json", coverage)
    save(output / "same-question-comparisons.json", pairs)
    save(output / "summary.json", result)
    with (output / "native-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in native for key in row}))
        writer.writeheader()
        writer.writerows(native)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    result = summarize(cli.run_directory, cli.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("native_metric_ranges", "weight_publication", "gpu_telemetry")
            }
        )
    )


if __name__ == "__main__":
    main()
