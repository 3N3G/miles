"""Grade a saved baseline through the existing run's sole budget-owning service.

Reuse exact completed validation requests without changing their receipts. Keep
generation metadata byte-for-byte and store grades separately from solver traces.
"""

import argparse
import asyncio
import collections
import datetime
import gzip
import hashlib
import json
import random
import shutil
import statistics
from pathlib import Path

from experiments.tcs_async_rl.compare_screens import validate_summary
from experiments.tcs_async_rl.diagnose_rl import completed_judge_receipt
from experiments.tcs_async_rl.judge_comparison import digest, judge_settings, save
from experiments.tcs_async_rl.review_saved_judge import ExistingService
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rubric_judge import messages, parse_grade, rubric_items


def request_payload(question, candidate, judge, prompt):
    return {
        "model": judge["primary_model"],
        "reasoning_effort": judge["reasoning_effort"],
        "max_completion_tokens": judge["max_completion_tokens"],
        "response_format": {"type": "json_object"},
        "messages": messages(
            question, candidate, swapped=False, system_prompt=prompt, grading_method="reference_guided"
        ),
    }


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validation_receipts(directory):
    """Index only original, completed v5 requests. Never alias a receipt identity."""
    records = json.loads((directory / "verdicts.json").read_text())
    output = {}
    for methods in records.values():
        for grade in methods["reference_v5"]:
            path = Path(grade["receipt"])
            receipt = json.loads(path.read_text())
            key = fingerprint(receipt["payload"])
            if (
                receipt["request_sha256"] != key
                or receipt["http_status"] != 200
                or receipt["swapped"]
                or receipt["body"]["choices"][0]["finish_reason"] != "stop"
            ):
                raise ValueError("Reused validation receipt is incomplete or has a changed request")
            if key in output:
                raise ValueError("Multiple completed validation verdicts for one request require review")
            output[key] = path
    return output


def load_samples(source, output):
    complete = json.loads((source / "screen-complete.json").read_text())
    trace = source / "rollout_data" / Path(complete["responses"]).name
    if (
        complete["answers"] != 384
        or complete["optimizer_updates"] != 0
        or not complete["publication"]["matches_source"]
        or digest(source / "input.json") != complete["dataset_sha256"]
        or digest(trace) != complete["responses_sha256"]
    ):
        raise ValueError("Expected the complete, unchanged384-answer baseline")
    for name in ("input.json", "approved-config.json", "resolved-config.json", "screen-complete.json"):
        destination = output / name
        if destination.exists() and destination.read_bytes() != (source / name).read_bytes():
            raise ValueError("Baseline generation metadata changed")
        shutil.copyfile(source / name, destination)
    rows = []
    with gzip.open(trace, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(
                {
                    key: row[key]
                    for key in (
                        "index",
                        "metadata",
                        "final_answer",
                        "status",
                        "response_length",
                        "judge_identity",
                    )
                }
            )
    if len(rows) != 384 or len({row["index"] for row in rows}) != 384:
        raise ValueError("Baseline sample count or identities differ")
    return complete, rows


async def regrade(cli):
    cli.output.mkdir(parents=True, exist_ok=True)
    complete, samples = load_samples(cli.source, cli.output)
    service = ExistingService(cli.service_owner, cli.judge_config)
    config = json.loads(cli.judge_config.read_text())
    judge = judge_settings(cli.judge_config, config)
    prompt = (cli.judge_config.parent / config["prompt_file"]).read_text()
    questions = {
        question_key(row): row for row in json.loads((cli.source / "input.json").read_text())["questions"]
    }
    live = {
        question_key(row): row
        for row in json.loads((cli.service_owner / "input.json").read_text())["questions"]
    }
    if any(live.get(key) != row for key, row in questions.items()):
        raise ValueError("Baseline questions differ from the approved live corpus")
    reused = validation_receipts(cli.validation)
    provenance = {
        "source_archive": json.loads((cli.source / "full-archive-receipt.json").read_text()),
        "source_responses_sha256": complete["responses_sha256"],
        "judge_config_sha256": digest(cli.judge_config),
        "judge_prompt_sha256": service.prompt_sha256,
        "service_owner": str(cli.service_owner),
        "shared_budget": str(cli.service_owner / "judging/budget.jsonl"),
        "reuse_validation": str(cli.validation),
        "solver_generations": 0,
        "optimizer_updates": 0,
    }
    plan = cli.output / "regrade-plan.json"
    if plan.exists() and json.loads(plan.read_text()) != provenance:
        raise ValueError("Baseline grading provenance changed")
    save(plan, provenance)
    grades_dir = cli.output / "grades"
    grades_dir.mkdir(exist_ok=True)
    receipts_dir = cli.output / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(2)

    async def one(row):
        key = row["metadata"]["question_key"]
        question = questions[key]
        candidate = row["final_answer"] or ""
        payload = request_payload(question, candidate, judge, prompt)
        expected = fingerprint(payload)
        destination = grades_dir / f"sample-{row['index']:03d}.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            if result["request_sha256"] != expected:
                raise ValueError("Cached baseline grade request changed")
        else:
            if expected in reused:
                path = reused[expected]
                reused_valid_request = True
            else:
                async with semaphore:
                    score = await service.score(
                        {
                            "question_key": key,
                            "final_answer": candidate,
                            "generator_truncated": row["status"] == "truncated",
                        }
                    )
                if score["identity"] != row["judge_identity"]:
                    raise ValueError("Final-answer extraction changed")
                folder = cli.service_owner / "judging" / key
                receipt = completed_judge_receipt(folder, score["identity"], 0)
                path = folder / f"{score['identity']}-order0-{receipt['request_sha256'][:12]}.json"
                reused_valid_request = False
            archived_receipt = receipts_dir / f"{key}-{path.name}"
            if archived_receipt.exists() and digest(archived_receipt) != digest(path):
                raise ValueError("Preserved baseline receipt changed")
            shutil.copyfile(path, archived_receipt)
            result = {
                "index": row["index"],
                "question_key": key,
                "request_sha256": expected,
                "source_receipt": str(path),
                "receipt": str(archived_receipt),
                "receipt_sha256": digest(path),
                "reused_valid_validation_request": reused_valid_request,
            }
            save(destination, result)
        path = Path(result["receipt"])
        receipt = json.loads(path.read_text())
        actual = receipt["payload"]
        expected_payload = {**payload, "max_completion_tokens": actual["max_completion_tokens"]}
        if (
            digest(path) != result["receipt_sha256"]
            or actual != expected_payload
            or receipt["request_sha256"] != fingerprint(actual)
            or receipt["http_status"] != 200
            or receipt["swapped"]
            or not judge["max_completion_tokens"]
            <= actual["max_completion_tokens"]
            <= config["length_retry_max_completion_tokens"]
            or receipt["body"]["choices"][0]["finish_reason"] != "stop"
        ):
            raise ValueError("Baseline grade differs from its exact request and completed receipt")
        decisions = parse_grade(
            receipt["body"]["choices"][0]["message"]["content"],
            rubric_items(question["rubric"]),
            answer_labels=("candidate",),
        )["candidate"]
        return {**row, **result, "score": decisions["score"], "candidate_items": decisions["items"]}

    grades = await asyncio.gather(*(one(row) for row in samples))
    save(cli.output / "graded-answers.json", grades)
    original = json.loads((cli.source / "screen-summary.json").read_text())
    rng, groups = random.Random(20261125), []
    examples = cli.output / "seeded-examples"
    examples.mkdir(exist_ok=True)
    for key, question in questions.items():
        rows = [row for row in grades if row["question_key"] == key]
        scores = [row["score"] for row in rows]
        first = next(row for row in original["questions"] if row["question_key"] == key)
        example = rng.choice(sorted(rows, key=lambda row: row["index"]))
        save(
            examples / f"{key}.json",
            {
                "selection": "Random(20261125), one uniform answer per question",
                "question": question,
                "sample": example,
            },
        )
        groups.append(
            {
                **first,
                "reward_counts": dict(sorted(collections.Counter(scores).items())),
                "mean_reward": sum(scores) / len(scores),
                "reward_std": statistics.stdev(scores),
                "varied_with_positive": len(set(scores)) > 1 and max(scores) > 0,
                "full_credit": scores.count(7),
                "zero": scores.count(0),
                "positive": sum(score > 0 for score in scores),
                "seeded_example": f"seeded-examples/{key}.json",
            }
        )
    summary = {
        **{
            key: original[key]
            for key in ("checkpoint_rollout_id", "dataset_sha256", "sampling", "answers", "optimizer_updates")
        },
        "questions": groups,
        "responses_sha256": complete["responses_sha256"],
        "judge_prompt_sha256": service.prompt_sha256,
        "judge_config_sha256": digest(cli.judge_config),
        "grade_method": "reference_guided",
        "judge_reasoning_effort": judge["reasoning_effort"],
        "reused_validation_grades": sum(row["reused_valid_validation_request"] for row in grades),
        "limits": "Saved checkpoint108 answers regraded with the accepted v5 prompt. One generation per checkpoint, training-eligible questions, no held-out generalization or human calibration claim.",
        "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    validate_summary(summary)
    save(cli.output / "screen-summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "service-owner", "judge-config", "validation"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = asyncio.run(regrade(parser.parse_args()))
    print(
        json.dumps({key: result[key] for key in ("answers", "optimizer_updates", "reused_validation_grades")})
    )
