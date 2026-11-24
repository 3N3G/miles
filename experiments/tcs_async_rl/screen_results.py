"""Validate a fixed checkpoint screen and summarize its per-question rubric rewards."""

import argparse
import collections
import gzip
import json
import random
import statistics
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import judge_receipts
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rl_artifacts import file_sha256


def summarize(directory):
    complete = json.loads((directory / "screen-complete.json").read_text())
    if complete["optimizer_updates"] != 0 or not complete["publication"]["matches_source"]:
        raise ValueError("Screen must evaluate the exact source checkpoint without updates")
    input_file = directory / "input.json"
    if file_sha256(input_file) != complete["dataset_sha256"]:
        raise ValueError("Screen input changed")
    questions = {question_key(row): row for row in json.loads(input_file.read_text())["questions"]}
    config = json.loads((directory / "approved-config.json").read_text())
    for key in questions:
        manifest = json.loads((directory / "judging" / key / "service-manifest.json").read_text())
        if (
            manifest["judge"]["primary_model"] != "gpt-6-luna"
            or manifest["judge"]["reasoning_effort"] != "high"
            or manifest["judge_config"]["grading_method"] != "reference_guided"
            or manifest["prompt_sha256"] != config["judge_qualification"]["prompt_sha256"]
        ):
            raise ValueError("Screen used a different grading recipe")
    source = directory / "rollout_data" / Path(complete["responses"]).name
    if file_sha256(source) != complete["responses_sha256"]:
        raise ValueError("Screen response checksum differs")
    samples = collections.defaultdict(list)
    indices, cache = set(), {}
    with gzip.open(source, "rt") as stream:
        for line in stream:
            sample = json.loads(line)
            key = sample["metadata"]["question_key"]
            if key not in questions or sample["index"] in indices:
                raise ValueError("Screen contains an unapproved question or duplicate sample")
            indices.add(sample["index"])
            if sample["status"] not in ("completed", "truncated"):
                raise ValueError("Inference failure cannot become a zero reward")
            cache_key = (key, sample["judge_identity"])
            if cache_key not in cache:
                cache[cache_key] = judge_receipts(directory, questions[key], sample)
            if sample["reward"]["reward"] != statistics.mean(cache[cache_key]["order_scores"]):
                raise ValueError("Screen reward differs from its actual judge receipts")
            samples[key].append(sample)
    if set(samples) != set(questions) or len(indices) != complete["answers"]:
        raise ValueError("Screen questions or answer count differ")
    rows = []
    examples = directory / "seeded-examples"
    examples.mkdir(exist_ok=True)
    rng = random.Random(20261124)
    for key, question in questions.items():
        group = sorted(samples[key], key=lambda row: row["index"])
        if len(group) != complete["samples_per_question"]:
            raise ValueError("Incomplete per-question response group")
        scores = [sample["reward"]["reward"] for sample in group]
        example = rng.choice(group)
        (examples / f"{key}.json").write_text(
            json.dumps(
                {
                    "selection": "One uniform answer per pinned question, Random(20261124), pinned question order",
                    "question": question,
                    "sample": example,
                    "judge": cache[(key, example["judge_identity"])],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        rows.append(
            {
                "question_key": key,
                "paper_id": question["paper_id"],
                "difficulty": question["difficulty"],
                "answers": len(group),
                "reward_counts": dict(sorted(collections.Counter(scores).items())),
                "mean_reward": statistics.mean(scores),
                "reward_std": statistics.stdev(scores),
                "positive": sum(score > 0 for score in scores),
                "zero": scores.count(0),
                "full_credit": scores.count(7),
                "capped": sum(sample["status"] == "truncated" for sample in group),
                "mean_response_tokens": statistics.mean(sample["response_length"] for sample in group),
                "varied_with_positive": len(set(scores)) > 1 and max(scores) > 0,
                "seeded_example": str((examples / f"{key}.json").relative_to(directory)),
            }
        )
    medium = [row for row in rows if row["difficulty"] == "medium"]
    qualifying = sum(row["varied_with_positive"] for row in medium)
    result = {
        "questions": rows,
        "answers": len(indices),
        "optimizer_updates": 0,
        "medium_questions_with_varied_positive_rewards": qualifying,
        "provisional_training_questions": 26 if qualifying >= 2 else 20,
        "rule": "Include all six sampled medium questions if at least two have positive varying rewards. Requires raw-proof audit before submission.",
        "grade_method": "reference_guided",
        "judge_reasoning_effort": "high",
        "limits": "One checkpoint and one sampling run. Assistant-audited grader, no independent human calibration. Easy questions were previously trained.",
    }
    (directory / "screen-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
