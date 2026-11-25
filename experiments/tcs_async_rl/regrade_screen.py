"""Regrade immutable screen answers without changing their generation provenance."""

import argparse
import asyncio
import gzip
import json
import os
import shutil
from pathlib import Path

from experiments.tcs_async_rl.judge_comparison import digest, save
from experiments.tcs_async_rl.reward_service import MultiScorer
from experiments.tcs_async_rl.rl_smoke import validate_judge_qualification
from experiments.tcs_async_rl.screen_results import summarize, verify_preserved_samples

GENERATION_FILES = ("input.json", "approved-config.json", "resolved-config.json", "screen-complete.json")


async def regrade(cli):
    root = Path(__file__).parent
    grading = json.loads(cli.config.read_text())
    validate_judge_qualification(grading, root)
    complete = json.loads((cli.source / "screen-complete.json").read_text())
    if complete["answers"] != 384 or complete["optimizer_updates"] != 0:
        raise ValueError("Expected the completed fixed384-answer screen")
    source = cli.source / "rollout_data" / Path(complete["responses"]).name
    if digest(source) != complete["responses_sha256"]:
        raise ValueError("Generated answers changed before offline regrading")
    cli.output.mkdir(parents=True, exist_ok=True)
    target = cli.output / "rollout_data" / source.name
    target.parent.mkdir(exist_ok=True)
    for name in GENERATION_FILES:
        if (cli.output / name).exists() and (cli.output / name).read_bytes() != (cli.source / name).read_bytes():
            raise ValueError("Offline regrade has different generation provenance")
        shutil.copyfile(cli.source / name, cli.output / name)
    if not target.exists():
        os.link(source, target)
    plan = {
        "generation_file_sha256": {name: digest(cli.output / name) for name in GENERATION_FILES},
        "source_responses_sha256": complete["responses_sha256"],
        "source_archive": json.loads((cli.source / "full-archive-receipt.json").read_text()),
        "grading_config": grading,
        "new_solver_generations": 0,
        "optimizer_updates": 0,
    }
    plan_file = cli.output / "regrade-plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Offline regrade plan changed after calls")
    save(plan_file, plan)
    incoming = cli.output / "incoming-budget.jsonl"
    if not incoming.is_file() or not (cli.output / "judging/budget.jsonl").read_bytes().startswith(
        incoming.read_bytes()
    ):
        raise ValueError("Transfer the settled shared ledger before regrading")
    scorer = MultiScorer(
        input_file=cli.output / "input.json",
        directory=cli.output / "judging",
        config_path=root / grading["judge_config"],
        key_file=Path.home() / ".config/api-keys.env",
    )

    async def one(question, text, truncated):
        return await scorer.score({"question_key": question, "final_answer": text, "generator_truncated": truncated})

    controls = [
        (key, name, text, truncated, expected)
        for key, item in scorer.scorers.items()
        for name, text, truncated, expected in (
            ("reference", item.row["answer"], False, 7),
            ("empty", "", True, 0),
            ("irrelevant", "The answer is 42.", False, 0),
        )
    ]
    grades = await asyncio.gather(*(one(key, text, capped) for key, _, text, capped, _ in controls))
    checks = [
        {"question_key": key, "name": name, "expected": expected, **grade}
        for (key, name, _, _, expected), grade in zip(controls, grades, strict=True)
    ]
    save(cli.output / "all-question-controls.json", checks)
    if any(row["reward"] != row["expected"] for row in checks):
        raise ValueError("New screen grader failed a reference or negative control")
    with gzip.open(source, "rt") as stream:
        samples = [json.loads(line) for line in stream]
    grades = await asyncio.gather(
        *(
            one(row["metadata"]["question_key"], row["final_answer"] or "", row["status"] == "truncated")
            for row in samples
        )
    )
    output = cli.output / "rollout_data/regraded.jsonl.gz"
    temporary = output.with_suffix(".partial")
    with gzip.open(temporary, "wt") as stream:
        for row, grade in zip(samples, grades, strict=True):
            if row["judge_identity"] != grade["identity"]:
                raise ValueError("Final-answer extraction changed during regrading")
            row["reward"] = {
                "reward": grade["reward"],
                "judge_presentations": 1.0,
                "generator_truncated": float(grade["generator_truncated"]),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    if verify_preserved_samples(target, output) != 384:
        raise ValueError("Regraded screen has an incomplete sample count")
    save(
        cli.output / "regrade-manifest.json",
        {**plan, "responses_file": output.name, "responses_sha256": digest(output)},
    )
    return summarize(cli.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = asyncio.run(regrade(parser.parse_args()))
    print(json.dumps({"answers": result["answers"], "optimizer_updates": result["optimizer_updates"]}))
