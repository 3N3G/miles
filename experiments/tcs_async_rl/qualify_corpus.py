"""Grade all reference/negative controls for the approved frozen corpus on the VM."""

import argparse
import asyncio
import json
from pathlib import Path

from experiments.tcs_async_rl.judge_comparison import save
from experiments.tcs_async_rl.reward_service import MultiScorer


async def qualify(directory, config):
    scorer = MultiScorer(
        input_file=directory / "input.json",
        directory=directory / "judging",
        config_path=config,
        key_file=Path.home() / ".config/api-keys.env",
    )
    cases = [
        (key, name, text, expected)
        for key, child in scorer.scorers.items()
        for name, text, expected in (
            ("reference", child.row["answer"], 7),
            ("empty", "", 0),
            ("irrelevant", "The answer is 42.", 0),
        )
    ]

    async def one(key, name, text, expected):
        result = await scorer.score(
            {"question_key": key, "final_answer": text, "generator_truncated": name == "empty"}
        )
        return {"question_key": key, "name": name, "expected": expected, **result}

    results = await asyncio.gather(*(one(*row) for row in cases))
    output = {
        "questions": len(scorer.scorers),
        "presentations": len(results),
        "results": results,
        "failures": [row for row in results if row["reward"] != row["expected"]],
    }
    save(directory / "corpus-controls.json", output)
    return {key: value for key, value in output.items() if key != "results"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    cli = parser.parse_args()
    print(json.dumps(asyncio.run(qualify(cli.directory, cli.config)), indent=2))
