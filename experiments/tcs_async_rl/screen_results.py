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
from experiments.tcs_async_rl.rl_smoke import validate_judge_qualification


def verify_preserved_samples(original, regraded):
    """Only the reward may change. Preserve text, tokens, masks and weight versions."""
    count = 0
    with gzip.open(original, "rt") as before, gzip.open(regraded, "rt") as after:
        for old_line, new_line in zip(before, after, strict=True):
            old, new = json.loads(old_line), json.loads(new_line)
            old.pop("reward")
            new.pop("reward")
            if old != new:
                raise ValueError("Regrading changed a generated sample")
            count += 1
    return count


def summarize(directory):
    complete = json.loads((directory / "screen-complete.json").read_text())
    if complete["optimizer_updates"] != 0 or not complete["publication"]["matches_source"]:
        raise ValueError("Screen must evaluate the exact source checkpoint without updates")
    input_file = directory / "input.json"
    if file_sha256(input_file) != complete["dataset_sha256"]:
        raise ValueError("Screen input changed")
    questions = {question_key(row): row for row in json.loads(input_file.read_text())["questions"]}
    config = json.loads((directory / "approved-config.json").read_text())
    qualification = config["judge_qualification"]
    regrade_file = directory / "regrade-manifest.json"
    regrade = json.loads(regrade_file.read_text()) if regrade_file.exists() else None
    if regrade is not None:
        for name, expected in regrade["generation_file_sha256"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError("Offline regrading changed the original generation metadata")
        validate_judge_qualification(regrade["grading_config"], Path(__file__).parent)
        qualification = regrade["grading_config"]["judge_qualification"]
    args = json.loads((directory / "resolved-config.json").read_text())["args"]
    sampling_keys = (
        "seed",
        "rollout_seed",
        "eval_temperature",
        "eval_top_p",
        "eval_top_k",
        "eval_max_response_len",
        "rollout_max_context_len",
        "sglang_context_length",
        "apply_chat_template_kwargs",
        "rollout_skip_special_tokens",
        "n_samples_per_eval_prompt",
    )
    sampling = {key: args[key] for key in sampling_keys}
    if any(sampling[key] != value for key, value in config["overrides"].items() if key in sampling):
        raise ValueError("Screen sampling differs from the frozen configuration")
    if complete["checkpoint_rollout_id"] != config["resume"]["iteration"]:
        raise ValueError("Screen restored a different checkpoint iteration than its frozen plan")
    for key in questions:
        manifest = json.loads((directory / "judging" / key / "service-manifest.json").read_text())
        if (
            manifest["judge"]["primary_model"] != "gpt-6-luna"
            or manifest["judge"]["reasoning_effort"] != "high"
            or manifest["judge_config"]["grading_method"] != "reference_guided"
            or manifest["prompt_sha256"] != qualification["prompt_sha256"]
        ):
            raise ValueError("Screen used a different grading recipe")
    source = directory / "rollout_data" / Path(complete["responses"]).name
    if file_sha256(source) != complete["responses_sha256"]:
        raise ValueError("Screen response checksum differs")
    if regrade is not None:
        original = source
        source = directory / "rollout_data" / regrade["responses_file"]
        if file_sha256(source) != regrade["responses_sha256"]:
            raise ValueError("Regraded response checksum differs")
        if verify_preserved_samples(original, source) != complete["answers"]:
            raise ValueError("Regraded screen has an incomplete sample count")
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
                "prompt_type": question["prompt_type"],
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
        "checkpoint_rollout_id": complete["checkpoint_rollout_id"],
        "dataset_sha256": complete["dataset_sha256"],
        "responses_sha256": file_sha256(source),
        "judge_prompt_sha256": qualification["prompt_sha256"],
        "judge_config_sha256": qualification["runtime_config_sha256"],
        "sampling": sampling,
        "questions": rows,
        "answers": len(indices),
        "optimizer_updates": 0,
        "medium_questions_with_varied_positive_rewards": qualifying,
        "purpose": "Report fixed-question checkpoint scores. The frozen training configuration owns the training dataset.",
        "grade_method": "reference_guided",
        "judge_reasoning_effort": "high",
        "limits": "One checkpoint and one sampling run. Assistant-audited grader, no independent human calibration. Easy questions were previously trained.",
    }
    if regrade is not None:
        result["source_responses_sha256"] = complete["responses_sha256"]
        result["grading_provenance"] = (
            "Offline regrade of unchanged saved answers. Original GPU configuration preserved."
        )
    (directory / "screen-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
