"""Freeze the approved seeded six-easy/six-medium screen from eligible local shards."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rl_inputs import question_set_key
from experiments.tcs_async_rl.rubric_judge import rubric_items


def select_questions(rows, *, seed=20261124, per_difficulty=6):
    rng, selected, papers = random.Random(seed), [], set()
    for difficulty in ("easy", "medium"):
        pool = [row for row in rows if row["difficulty"] == difficulty]
        rng.shuffle(pool)
        chosen = []
        for row in pool:
            paper = (row["cohort"], row["paper_id"])
            if paper not in papers:
                chosen.append(row)
                papers.add(paper)
                if len(chosen) == per_difficulty:
                    break
        if len(chosen) < per_difficulty:
            chosen.extend(row for row in pool if row not in chosen)
            chosen = chosen[:per_difficulty]
        if len(chosen) != per_difficulty:
            raise ValueError("Insufficient eligible questions for the approved screen")
        selected.extend(chosen)
    return selected


def prepare(source_dir, training_file, tokenizer_dir, output):
    # Dataset preparation dependencies stay out of runtime validation.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    training = json.loads(training_file.read_text())
    audit = json.loads((source_dir / "audit.json").read_text())
    hashes = {row["cohort"]: row["sha256"] for row in audit["cohorts"]}
    rows = []
    for cohort in sorted(training["data"]["eligible_configs"]):
        if cohort in training["data"]["held_out_configs"]:
            raise ValueError("Held-out shards remain sealed")
        path = source_dir / cohort / "gpt_6_1_sol-00000-of-00001.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[cohort]:
            raise ValueError("Eligible source checksum differs")
        rows.extend(
            {**row, "cohort": cohort, "row": index} for index, row in enumerate(pq.read_table(path).to_pylist())
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    selected = select_questions(rows)
    easy = [row for row in rows if row["difficulty"] == "easy"]
    if len(easy) != 20:
        raise ValueError("Existing easy training pool changed")
    for row in selected + easy:
        row["formatted_prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
            **training["miles_args"]["apply_chat_template_kwargs"],
        )
        row["prompt_tokens"] = len(tokenizer.encode(row["formatted_prompt"], add_special_tokens=False))
        if row["prompt_tokens"] + 90000 > 98304:
            raise ValueError("A selected prompt cannot fit the approved output budget")
        rubric_items(row["rubric"])
    output.mkdir(exist_ok=True, parents=True)
    for name, data in (("screen-12-bundle.json", selected), ("easy-20-bundle.json", easy)):
        (output / name).write_text(
            json.dumps(
                {
                    "dataset_revision": training["data"]["revision"],
                    "question_set_key": question_set_key(data),
                    "questions": data,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    result = {
        "seed": 20261124,
        "selection": "Shuffle each eligible stratum, taking distinct papers where possible. Easy first, then medium.",
        "eligible_questions": len(rows),
        "eligible_easy": len(easy),
        "eligible_medium": sum(row["difficulty"] == "medium" for row in rows),
        "heldout_reads": 0,
        "optimizer_updates": 0,
        "model_generations": 0,
        "selected": [
            {k: row[k] for k in ("cohort", "row", "paper_id", "node_id", "difficulty", "prompt_tokens")}
            | {"question_key": question_key(row)}
            for row in selected
        ],
        "screen_bundle_sha256": hashlib.sha256((output / "screen-12-bundle.json").read_bytes()).hexdigest(),
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "training-file", "tokenizer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
