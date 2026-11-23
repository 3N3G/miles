"""Freeze real-text operator inputs without running the pretrained model."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    calibration = json.loads(Path("evidence/transport_calibration.json").read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    previous_ids = set()
    records = []
    for index, text in enumerate(calibration["texts"]):
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:64]
        assert len(ids) == 64
        previous_ids.add(digest(ids))
        records.append({"id": f"calibration-{index:03}", "split": "fit" if index < 128 else "validation",
                        "source_index": index, "text": text, "token_ids": ids,
                        "rendered_input": tokenizer.decode(ids), "input_sha256": digest(ids)})
    candidates = []
    texts = pq.read_table(args.parquet, columns=["text"]).column("text").to_pylist()
    article = None
    for index, text in enumerate(texts):
        heading = text.strip()
        if heading.startswith("= ") and not heading.startswith("= ="):
            article = heading
        if len(text) <= 500 or text.strip().startswith("="):
            continue
        ids = tokenizer.encode(text, add_special_tokens=False).ids[:64]
        if len(ids) != 64 or digest(ids) in previous_ids:
            continue
        if article is None:
            raise ValueError("Eligible paragraph has no article heading")
        candidates.append((index, text, ids, article))
    random.Random(1729).shuffle(candidates)
    selected = set()
    selected_articles = set()
    for index, text, ids, article in candidates:
        if digest(ids) in selected or article in selected_articles:
            continue
        selected.add(digest(ids))
        selected_articles.add(article)
        records.append({"id": f"final-{len(selected)-1:03}", "split": "final",
                        "source_index": index, "article": article, "text": text, "token_ids": ids,
                        "rendered_input": tokenizer.decode(ids), "input_sha256": digest(ids)})
        if len(selected) == 32:
            break
    assert len(records) == 176
    assert len({r["input_sha256"] for r in records}) == 176
    result = {"model_revision": "ed40e7da625f594fc9c4990f5f66be24873c22d7",
              "dataset": "Salesforce/wikitext", "subset": "wikitext-2-raw-v1",
              "final_source_split": "validation", "final_parquet_revision": "3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e",
              "final_parquet_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
              "selection_seed": 1729, "final_sampling": "At most one paragraph per article",
              "records": records}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for split in ["fit", "validation", "final"]:
        group = [r for r in records if r["split"] == split]
        print(json.dumps({"split": split, "paragraphs": len(group), "example": group[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
