"""Add real rubric judgments for recorded caps to a separate qualification artifact."""

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.grade_group import Budget, grade_one
from experiments.tcs_async_rl.group_inference import final_answer
from experiments.tcs_async_rl.judge_client import load_credentials


async def qualify(*, directory, output, config_path, key_file):
    output.mkdir(parents=True, exist_ok=True)
    original = directory / "group-grades.json"
    result = copy.deepcopy(json.loads(original.read_text()))
    row = json.loads((directory / "input.json").read_text())
    responses = [json.loads(line) for line in (directory / "responses.jsonl").read_text().splitlines()]
    config = json.loads(config_path.read_text())
    judge = json.loads((config_path.parent / config["training_config"]).read_text())["judge"]
    prompt = (config_path.parent / config["prompt_file"]).read_text()
    key = load_credentials(key_file)[judge["api_key_env"]]
    budget, semaphore = Budget(output, config), asyncio.Semaphore(config["max_concurrency"])
    caps = []
    for sample in responses:
        if sample["final_answer"]:
            continue
        raw = sample["output"]["text"]
        reason = sample["output"]["meta_info"]["finish_reason"]
        limit = result["generation_sampling"]["max_new_tokens"]
        if final_answer(raw) or reason.get("type") != "length" or reason.get("length") != limit:
            raise ValueError("An unclassified missing final cannot receive a cap grade")
        if len(sample["output"]["output_ids"]) != limit:
            raise ValueError("Recorded cap does not match actual token count")
        case = f"sample-{sample['sample_index']:02d}"
        orders = await asyncio.gather(
            *(
                grade_one(
                    identity="qualified-" + case,
                    candidate="",
                    row=row,
                    swapped=swapped,
                    judge=judge,
                    config=config,
                    prompt=prompt,
                    key=key,
                    budget=budget,
                    semaphore=semaphore,
                    directory=output,
                )
                for swapped in (False, True)
            )
        )
        if [grade["candidate"]["score"] for grade in orders] != [0, 0]:
            raise ValueError("Actual empty-answer judge control failed")
        result["scores_by_case"][case] = statistics.mean(g["candidate"]["score"] for g in orders)
        result["grades"].extend(
            {**grade, "identity": case, "qualification_identity": grade["identity"]} for grade in orders
        )
        caps.append(
            {
                "case": case,
                "finish_reason": reason,
                "generated_tokens": limit,
                "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "scores": [g["candidate"]["score"] for g in orders],
                "receipts": [g["receipt"] for g in orders],
            }
        )
    scores = [v for k, v in result["scores_by_case"].items() if k.startswith("sample-")]
    result.update(
        scored_responses=len(scores),
        all_zero=all(v == 0 for v in scores),
        uniform_scores=len(set(scores)) == 1,
        population_score_variance=statistics.pvariance(scores),
        recorded_cap_judgments=caps,
        qualification_policy="recorded_caps_judged_v1",
        original_grades_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        original_responses_sha256=hashlib.sha256((directory / "responses.jsonl").read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((directory / "input.json").read_bytes()).hexdigest(),
        qualification_source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        qualification_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification_judge_cost_usd=budget.total,
    )
    (output / "qualified-group-grades.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "scored_responses": len(scores),
                "cap_cases": len(caps),
                "unique_rewards": sorted(set(scores)),
                "mean": statistics.mean(scores),
                "original_grades_unchanged": True,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("judge_group_v1.json"))
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/api-keys.env")
    args = parser.parse_args()
    asyncio.run(qualify(directory=args.directory, output=args.output, config_path=args.config, key_file=args.key_file))


if __name__ == "__main__":
    main()
