"""Qualify a versioned grader on saved answers using the future run's shared budget.

This makes judge calls only. It preserves old generations and grades, uses the
real online scorer, and exports a separately versioned generation gate.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from experiments.tcs_async_rl.auto_rl import save
from experiments.tcs_async_rl.diagnose_rl import completed_judge_receipt
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.reward_service import MultiScorer
from experiments.tcs_async_rl.rubric_judge import parse_grade, rubric_items


def verdicts(scorer, row, identity, result):
    folder = scorer.scorers[question_key(row)].directory
    output = []
    for swapped in (False, True):
        receipt = completed_judge_receipt(folder, result["identity"], int(swapped))
        grade = parse_grade(receipt["body"]["choices"][0]["message"]["content"], rubric_items(row["rubric"]))
        output.append(
            {
                "identity": identity,
                "swapped": swapped,
                "candidate": grade["B" if swapped else "A"],
                "reference": grade["A" if swapped else "B"],
                "receipt": str(
                    folder / f"{result['identity']}-order{int(swapped)}-{receipt['request_sha256'][:12]}.json"
                ),
            }
        )
    assert result["scores"] == [value["candidate"]["score"] for value in output]
    return output


async def score(scorer, row, candidate, *, truncated=False):
    return await scorer.score(
        {
            "question_key": question_key(row),
            "response": "</think>\n" + candidate if candidate else "",
            "generator_truncated": truncated,
        }
    )


async def review_labels(scorer, labels, directory):
    async def one(case):
        source = Path(case["source_file"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != case["source_sha256"]:
            raise ValueError("Saved validation answer differs from the adjudicated input")
        example = json.loads(source)
        if "original_example" in example:
            example = example["original_example"]
        row = example["question"]
        result = await score(scorer, row, example["final_answer"])
        pair = verdicts(scorer, row, case["case"], result)
        expected = {item["item"]: item for item in case["candidate_items"]}
        differences = []
        for order in pair:
            for item in order["candidate"]["items"]:
                target = expected[item["item"]]
                if item["satisfied"] != target["satisfied"]:
                    differences.append({"swapped": order["swapped"], "expected": target, "observed": item})
        return {**case, "result": result, "verdicts": pair, "differences": differences}

    results = await asyncio.gather(*(one(case) for case in labels["cases"]))
    save(directory / "assistant-label-validation.json", {"label_provenance": labels["provenance"], "cases": results})
    return results


async def control_all_questions(scorer, rows, directory):
    async def one(row, name, text, truncated, expected):
        result = await score(scorer, row, text, truncated=truncated)
        return {"question_key": question_key(row), "name": name, "expected": expected, **result}

    results = await asyncio.gather(
        *(
            one(row, name, text, truncated, expected)
            for row in rows
            for name, text, truncated, expected in (
                ("reference", row["answer"], False, 7),
                ("empty", "", True, 0),
                ("irrelevant", "The answer is 42.", False, 0),
            )
        )
    )
    save(directory / "all-question-controls.json", results)
    if any(result["scores"] != [result["expected"]] * 2 for result in results):
        raise ValueError("New prompt failed a reference, empty or irrelevant control; inspect preserved decisions")
    return results


async def generation_gate(scorer, source, output):
    output.mkdir(parents=True, exist_ok=False)
    names = (
        "input.json",
        "responses.jsonl",
        "sampling-parameters.json",
        "sampling-support-control.json",
        "resolved-inference-mode.json",
        "watcher-complete.json",
    )
    for name in names:
        shutil.copy2(source / name, output / name)
    row = json.loads((output / "input.json").read_text())
    responses = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    candidates = [(f"sample-{value['sample_index']:02d}", value["final_answer"] or "", value) for value in responses]
    controls = [
        ("control-reference", row["answer"], None),
        ("control-empty", "", None),
        ("control-irrelevant", "The answer is 42.", None),
    ]

    async def one(identity, candidate, sample):
        result = await score(scorer, row, candidate, truncated=not candidate)
        return identity, result, verdicts(scorer, row, identity, result), sample

    results = await asyncio.gather(*(one(*case) for case in controls + candidates))
    base = {
        "status": "provisional_assistant_reviewed_prompt_repair",
        "model": scorer.judge["primary_model"],
        "reasoning_effort": scorer.judge["reasoning_effort"],
        "generation_sampling": json.loads((output / "sampling-parameters.json").read_text()),
        "incomplete_or_empty_final_answers": sum(not value["final_answer"] for value in responses),
        "human_calibration": "not_human_labeled",
        "independent_judge_status": "not_run",
        "grades": [],
        "scores_by_case": {},
    }
    caps, qualified_pairs = [], []
    for identity, result, pair, sample in results:
        if sample is not None and not sample["final_answer"]:
            finish = sample["output"]["meta_info"]["finish_reason"]
            limit = base["generation_sampling"]["max_new_tokens"]
            assert finish == {"type": "length", "length": limit}
            assert len(sample["output"]["output_ids"]) == limit and result["scores"] == [0, 0]
            caps.append(
                {
                    "case": identity,
                    "finish_reason": finish,
                    "generated_tokens": limit,
                    "raw_response_sha256": hashlib.sha256(sample["output"]["text"].encode()).hexdigest(),
                    "scores": result["scores"],
                    "receipts": [value["receipt"] for value in pair],
                }
            )
            qualified_pairs.extend(pair)
        else:
            base["scores_by_case"][identity] = result["reward"]
            base["grades"].extend(pair)
    base["scored_responses"] = sum(name.startswith("sample-") for name in base["scores_by_case"])
    save(output / "group-grades.json", base)
    qualified = copy.deepcopy(base)
    qualified["grades"].extend(qualified_pairs)
    qualified["scores_by_case"].update({case["case"]: statistics.mean(case["scores"]) for case in caps})
    values = [v for k, v in qualified["scores_by_case"].items() if k.startswith("sample-")]
    qualified.update(
        scored_responses=len(values),
        all_zero=all(v == 0 for v in values),
        uniform_scores=len(set(values)) == 1,
        population_score_variance=statistics.pvariance(values),
        qualification_policy="recorded_caps_judged_v1",
        recorded_cap_judgments=caps,
        input_sha256=hashlib.sha256((output / "input.json").read_bytes()).hexdigest(),
        original_grades_sha256=hashlib.sha256((output / "group-grades.json").read_bytes()).hexdigest(),
        original_responses_sha256=hashlib.sha256((output / "responses.jsonl").read_bytes()).hexdigest(),
    )
    save(output / "qualified-group-grades.json", qualified)
    save(
        output / "generation-source.json",
        {
            "source_directory": str(source),
            "new_generations": 0,
            "original_responses_sha256": qualified["original_responses_sha256"],
        },
    )
    return qualified


async def qualify(cli):
    rows = json.loads(cli.bundle.read_text())["questions"]
    scorer = MultiScorer(
        input_file=cli.bundle, directory=cli.run_directory / "judging", config_path=cli.config, key_file=cli.key_file
    )
    labels = json.loads(cli.labels.read_text())
    reviewed = await review_labels(scorer, labels, cli.run_directory)
    controls = await control_all_questions(scorer, rows, cli.run_directory)
    gate = await generation_gate(scorer, cli.diagnostic_directory, cli.gate_output)
    result = {
        "reviewed_cases": len(reviewed),
        "item_mismatches": sum(len(case["differences"]) for case in reviewed),
        "controls": len(controls),
        "gate_samples": gate["scored_responses"],
        "gate_unique_scores": sorted(set(v for k, v in gate["scores_by_case"].items() if k.startswith("sample-"))),
        "labels_sha256": hashlib.sha256(cli.labels.read_bytes()).hexdigest(),
        "new_generations": 0,
    }
    save(cli.run_directory / "judge-repair-qualification.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "run-directory", "diagnostic-directory", "gate-output", "labels", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/api-keys.env")
    asyncio.run(qualify(parser.parse_args()))


if __name__ == "__main__":
    main()
