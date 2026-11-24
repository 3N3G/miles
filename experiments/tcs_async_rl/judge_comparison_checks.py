"""Run separate grading controls and question-clustered descriptive intervals."""

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

from experiments.tcs_async_rl.grade_group import Budget, grade_one
from experiments.tcs_async_rl.judge_client import load_credentials
from experiments.tcs_async_rl.judge_comparison import (
    SEED,
    digest,
    judge_settings,
    prompt_files,
    requested_methods,
    save,
)
from experiments.tcs_async_rl.reward_hook import question_key


def clustered_intervals(comparison, *, repetitions=5000, seed=SEED):
    """Resample questions, keeping every answer and both presentations together."""
    methods = list(comparison)
    grouped = {
        method: {
            key: [row for row in comparison[method]["cases"] if row["question_key"] == key]
            for key in sorted({row["question_key"] for row in comparison[method]["cases"]})
        }
        for method in methods
    }
    keys = list(grouped[methods[0]])
    if any(set(grouped[method]) != set(keys) for method in methods):
        raise ValueError("Compared methods must have the same question clusters")
    metrics = ("exact_score_match", "item_agreement", "absolute_score_error")
    draws = {method: {metric: [] for metric in metrics} for method in methods}
    differences = {metric: [] for metric in metrics}
    rng = random.Random(seed)
    for _ in range(repetitions):
        selected = rng.choices(keys, k=len(keys))
        means = {}
        for method in methods:
            rows = [row for key in selected for row in grouped[method][key]]
            means[method] = {metric: statistics.mean(row[metric] for row in rows) for metric in metrics}
            for metric, value in means[method].items():
                draws[method][metric].append(value)
        for metric in metrics:
            differences[metric].append(means[methods[1]][metric] - means[methods[0]][metric])

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {
        "kind": "descriptive_95_percent_question_cluster_percentile_bootstrap",
        "question_clusters": len(keys),
        "repetitions": repetitions,
        "seed": seed,
        "methods": {method: {metric: interval(v) for metric, v in rows.items()} for method, rows in draws.items()},
        "difference": {
            "direction": f"{methods[1]} minus {methods[0]}",
            "intervals": {metric: interval(v) for metric, v in differences.items()},
        },
        "limits": "Diagnostic stratified sample and assistant labels. Not human accuracy or a significance test.",
    }


async def controls(cli):
    packet = json.loads((cli.packet / "packet.json").read_text())
    assert (
        digest(cli.packet / "packet.json")
        == json.loads((cli.packet / "packet-freeze.json").read_text())["packet_sha256"]
    )
    questions = {question_key(case["question"]): case["question"] for case in packet["cases"]}
    config = json.loads(cli.config.read_text())
    judge = judge_settings(cli.config, config)
    key = load_credentials(Path.home() / ".config/api-keys.env")[judge["api_key_env"]]
    # Use the comparison's ledger directly. No second allowance is created.
    budget = Budget(cli.budget_directory or cli.output, config)
    semaphore = asyncio.Semaphore(config["max_concurrency"])

    async def one(identity, row, name, text, expected, method, swapped):
        directory = cli.output / "controls" / identity / method
        directory.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_files(config)[method]
        verdict = await grade_one(
            identity=name,
            candidate=text,
            row=row,
            swapped=swapped,
            judge=judge,
            config={**config, "grading_method": method},
            prompt=(cli.config.parent / prompt_file).read_text(),
            key=key,
            budget=budget,
            semaphore=semaphore,
            directory=directory,
        )
        return {"question_key": identity, "control": name, "expected": expected, "method": method, **verdict}

    tasks = [
        one(identity, row, name, text, expected, method, swapped)
        for identity, row in questions.items()
        for name, text, expected in (("reference", row["answer"], 7), ("empty", "", 0), ("irrelevant", "42", 0))
        for method in requested_methods(config)
        for swapped in ((False, True) if method == "anonymous_pair" else (False,))
    ]
    results = await asyncio.gather(*tasks)
    output = {
        "questions": len(questions),
        "presentations": len(results),
        "results": results,
        "failures": [r for r in results if r["candidate"]["score"] != r["expected"]],
        "spent_plus_reserved_usd": budget.total,
    }
    save(cli.output / "controls.json", output)
    return {name: value for name, value in output.items() if name != "results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("controls", "intervals"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-directory", type=Path)
    cli = parser.parse_args()
    if cli.command == "controls":
        print(json.dumps(asyncio.run(controls(cli)), indent=2))
    else:
        value = clustered_intervals(json.loads((cli.output / "comparison.json").read_text()))
        save(cli.output / "intervals.json", value)
        print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
