"""Compare two frozen single-reference prompts against pre-call assistant labels."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import completed_judge_receipt
from experiments.tcs_async_rl.grade_group import Budget, grade_one, usage_cost
from experiments.tcs_async_rl.judge_client import load_credentials
from experiments.tcs_async_rl.judge_comparison import comparison_metrics, digest, judge_settings, save
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rubric_judge import messages, parse_grade, rubric_items


def validate_labels(packet, labels):
    if len(packet["cases"]) != 40 or [r["case"] for r in labels["cases"]] != [r["case"] for r in packet["cases"]]:
        raise ValueError("Expected all forty frozen cases in label order")
    for case, label in zip(packet["cases"], labels["cases"], strict=True):
        if hashlib.sha256(case["candidate"].encode()).hexdigest() != label["candidate_sha256"]:
            raise ValueError("Labeled candidate text differs")
        if [i for i, _ in rubric_items(case["question"]["rubric"])] != [r["item"] for r in label["candidate_items"]]:
            raise ValueError("Labeled rubric items differ")
        if any(type(r["satisfied"]) is not bool or not r["reason"] for r in label["candidate_items"]):
            raise ValueError("Every original item needs a pre-call Boolean and reason")


async def compare(cli):
    packet = json.loads((cli.packet / "packet.json").read_text())
    labels = json.loads(cli.labels.read_text())
    if (
        digest(cli.labels) != cli.labels_sha256
        or digest(cli.packet / "packet.json")
        != json.loads((cli.packet / "packet-freeze.json").read_text())["packet_sha256"]
    ):
        raise ValueError("Packet or labels changed after freezing")
    validate_labels(packet, labels)
    config = json.loads(cli.config.read_text())
    prompts = config["prompt_files"]
    if set(prompts) != {"reference_v2", "reference_v3"}:
        raise ValueError("Expected exactly the frozen comparator and component repair")
    judge = judge_settings(cli.config, config)
    if judge["primary_model"] != "gpt-6-luna" or judge["reasoning_effort"] != "high":
        raise ValueError("Model and effort must remain fixed")
    incoming = cli.output / "incoming-budget.jsonl"
    if (
        not incoming.is_file()
        or not incoming.stat().st_size
        or not (cli.output / "judging/budget.jsonl").read_bytes().startswith(incoming.read_bytes())
    ):
        raise ValueError("The shared budget history must be transferred before calls")
    freeze = {
        "packet_sha256": digest(cli.packet / "packet.json"),
        "labels_sha256": cli.labels_sha256,
        "config_sha256": digest(cli.config),
        "incoming_ledger_sha256": digest(incoming),
        "prompt_sha256": {name: digest(cli.config.parent / path) for name, path in prompts.items()},
    }
    target = cli.output / "pre-call-label-freeze.json"
    if target.exists() and json.loads(target.read_text()) != freeze:
        raise ValueError("Comparison inputs changed after the pre-call freeze")
    save(target, freeze)
    key = load_credentials(Path.home() / ".config/api-keys.env")[judge["api_key_env"]]
    budget, semaphore = Budget(cli.output / "judging", config), asyncio.Semaphore(config["max_concurrency"])
    medium_sources = {
        row["case"]: row
        for row in json.loads((cli.packet / "selection-provenance-not-for-labeling.json").read_text())
        if row["kind"] == "medium"
    }

    async def one(case, recipe, identity, text):
        if recipe == "reference_v2" and identity == case["case"] and identity in medium_sources:
            # These exact v2/high verdicts already exist from generation.
            # Reuse immutable copies after labels freeze rather than regrade.
            source = medium_sources[identity]
            folder = cli.output / "baseline-v2-cache" / source["question_key"]
            receipt = completed_judge_receipt(folder, source["judge_identity"], 0)
            if receipt["payload"]["messages"] != messages(
                case["question"],
                text,
                swapped=False,
                system_prompt=(cli.config.parent / prompts[recipe]).read_text(),
                grading_method="reference_guided",
            ) or any(receipt["payload"][name] != judge[name] for name in ("reasoning_effort",)):
                raise ValueError("Cached comparator differs from the frozen request")
            if receipt["payload"]["model"] != judge["primary_model"]:
                raise ValueError("Cached comparator used another model")
            grade = parse_grade(
                receipt["body"]["choices"][0]["message"]["content"],
                rubric_items(case["question"]["rubric"]),
                answer_labels=("candidate",),
            )["candidate"]
            return {
                "identity": source["judge_identity"],
                "swapped": False,
                "candidate": grade,
                "reference": None,
                "receipt": str(folder / f"{source['judge_identity']}-order0-{receipt['request_sha256'][:12]}.json"),
                "reused_exact_baseline": True,
            }
        folder = cli.output / "judging" / identity / recipe
        folder.mkdir(parents=True, exist_ok=True)
        return await grade_one(
            identity=identity,
            candidate=text,
            row=case["question"],
            swapped=False,
            judge=judge,
            config={**config, "grading_method": "reference_guided"},
            prompt=(cli.config.parent / prompts[recipe]).read_text(),
            key=key,
            budget=budget,
            semaphore=semaphore,
            directory=folder,
        )

    tasks = [(case, recipe) for case in packet["cases"] for recipe in prompts]
    grades = await asyncio.gather(*(one(case, recipe, case["case"], case["candidate"]) for case, recipe in tasks))
    results = {case["case"]: {recipe: [] for recipe in prompts} for case in packet["cases"]}
    for (case, recipe), grade in zip(tasks, grades, strict=True):
        results[case["case"]][recipe].append(grade)
    save(cli.output / "verdicts.json", results)
    metrics = comparison_metrics(packet["cases"], labels, results)
    for recipe in prompts:
        receipts = [
            json.loads(path.read_text()) for path in (cli.output / "judging").glob(f"case-*/{recipe}/*-attempt*.json")
        ]
        metrics[recipe]["actual_usage_usd"] = sum(
            usage_cost(r["body"]["usage"], config) for r in receipts if r.get("body", {}).get("usage") is not None
        )
    save(cli.output / "comparison.json", metrics)
    questions = {question_key(case["question"]): case for case in packet["cases"]}
    controls = [
        (case, recipe, f"control-{qid}-{name}", text, expected)
        for qid, case in questions.items()
        for recipe in prompts
        for name, text, expected in (
            ("reference", case["question"]["answer"], 7),
            ("empty", "", 0),
            ("irrelevant", "The answer is 42.", 0),
        )
    ]
    verdicts = await asyncio.gather(
        *(one(case, recipe, identity, text) for case, recipe, identity, text, _ in controls)
    )
    checks = [
        {
            "identity": identity,
            "recipe": recipe,
            "expected": expected,
            "score": grade["candidate"]["score"],
            "passed": grade["candidate"]["score"] == expected,
        }
        for (_, recipe, identity, _, expected), grade in zip(controls, verdicts, strict=True)
    ]
    save(cli.output / "control-results.json", checks)
    return {
        "comparison": {name: {k: v for k, v in row.items() if k != "cases"} for name, row in metrics.items()},
        "controls": len(checks),
        "all_controls_pass": all(row["passed"] for row in checks),
        "cached_comparator_cases": len(medium_sources),
        "spent_reserved_usd": budget.total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "labels", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--labels-sha256", required=True)
    print(json.dumps(asyncio.run(compare(parser.parse_args())), indent=2))
