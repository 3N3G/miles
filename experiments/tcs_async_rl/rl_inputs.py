"""Prepare and validate the user-selected easy subset without reading held-out shards."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.tcs_async_rl.reward_hook import prepare_smoke_row, question_key
from experiments.tcs_async_rl.rubric_judge import rubric_items


def question_set_key(rows):
    keys = sorted(question_key(row) for row in rows)
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError("Training question identities must be nonempty and unique")
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def read_bundle(path, *, expected_sha256, training, contract=None):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Easy training bundle differs from the frozen input")
    bundle = json.loads(path.read_text())
    rows = bundle["questions"]
    contract = contract or {"questions": 20, "difficulties": ["easy"]}
    if len(rows) != contract["questions"] or bundle["dataset_revision"] != training["data"]["revision"]:
        raise ValueError("Expected the approved question count and dataset revision")
    for row in rows:
        if (
            row["difficulty"] not in contract["difficulties"]
            or row["cohort"] not in training["data"]["eligible_configs"]
        ):
            raise ValueError("Training contains an ineligible question")
        rubric_items(row["rubric"])
    if question_set_key(rows) != bundle["question_set_key"]:
        raise ValueError("Training question-set identity differs")
    return rows, bundle["question_set_key"]


def service_identity(path):
    value = json.loads(path.read_text())
    return question_set_key(value["questions"]) if "questions" in value else question_key(value)


def prepare(*, source_dir, manifest_file, training_file, tokenizer_dir, output):
    # These are preparation-only dependencies. Runtime validation uses the standard library.
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    manifest = json.loads(manifest_file.read_text())
    training = json.loads(training_file.read_text())
    audit = json.loads((source_dir / "audit.json").read_text())
    hashes = {row["cohort"]: row["sha256"] for row in audit["cohorts"]}
    selected = {(x["cohort"], x["row"]) for x in manifest["items"]}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    rows = []
    for cohort in sorted(training["data"]["eligible_configs"]):
        if cohort in training["data"]["held_out_configs"]:
            raise ValueError("Held-out cohorts must never be read")
        source = source_dir / cohort / "gpt_6_1_sol-00000-of-00001.parquet"
        if hashlib.sha256(source.read_bytes()).hexdigest() != hashes[cohort]:
            raise ValueError("Eligible Parquet checksum differs")
        for index, data in enumerate(pq.read_table(source).to_pylist()):
            if data["difficulty"] != "easy":
                continue
            if (cohort, index) not in selected:
                raise ValueError("Easy source differs from the frozen identity manifest")
            row = {**data, "cohort": cohort, "row": index}
            row["formatted_prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
                **training["miles_args"]["apply_chat_template_kwargs"],
            )
            row["prompt_tokens"] = len(tokenizer.encode(row["formatted_prompt"], add_special_tokens=False))
            if (
                row["prompt_tokens"] + training["miles_args"]["rollout_max_response_len"]
                > training["miles_args"]["seq_length"]
            ):
                raise ValueError("An easy prompt exceeds the approved context budget")
            rubric_items(row["rubric"])
            rows.append(row)
    if len(rows) != 20 or {(r["cohort"], r["row"]) for r in rows} != selected:
        raise ValueError("The easy subset is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "dataset_revision": training["data"]["revision"],
        "question_set_key": question_set_key(rows),
        "questions": rows,
    }
    bundle_file = output / "easy-20-bundle.json"
    bundle_file.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    (output / "miles-train.jsonl").write_text(
        "\n".join(json.dumps(prepare_smoke_row(row), ensure_ascii=False) for row in rows) + "\n"
    )
    result = {
        "questions": len(rows),
        "question_set_key": bundle["question_set_key"],
        "bundle_sha256": hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "heldout_reads": 0,
        "model_generations": 0,
        "optimizer_updates": 0,
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "manifest-file", "training-file", "tokenizer-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
