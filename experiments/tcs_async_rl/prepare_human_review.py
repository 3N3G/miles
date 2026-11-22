"""Prepare ten blinded rubric cases from previously seed-selected audit examples."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.tcs_async_rl.rubric_judge import rubric_items


def prepare(audit, output):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20261122)
    paths = sorted(audit.glob("example-step-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    chosen = rng.sample(paths, 10)
    labels, provenance = {}, {}
    links = []
    for position, path in enumerate(chosen, 1):
        record = json.loads(path.read_text())
        row = record["question"]
        case = f"case-{position:02d}"
        swapped = bool(rng.getrandbits(1))
        candidate_label = "B" if swapped else "A"
        answers = {candidate_label: record["final_answer"] or "", "A" if swapped else "B": row["answer"]}
        items = rubric_items(row["rubric"])
        text = [f"# {case}", "## Problem", row["problem"], "## Rubric", row["rubric"]]
        for answer in ("A", "B"):
            text.extend([f"## Answer {answer}", answers[answer] or "[Empty answer]"])
        table = ["| Item | Points | A satisfied? | B satisfied? |", "|---|---:|---|---|"]
        table.extend(f"| {item} | {weight:g} | | |" for item, weight in items)
        text.extend(["## Your binary decisions", "\n".join(table)])
        (output / f"{case}.md").write_text("\n\n".join(text) + "\n")
        labels[case] = {
            answer: [{"item": item, "satisfied": None, "reason": ""} for item, _ in items] for answer in ("A", "B")
        }
        provenance[case] = {
            "source_file": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "question_key": record["trace"]["question_key"],
            "sample_index": record["sample_index"],
            "candidate_label": candidate_label,
            "machine_judgment": record["judge"],
            "original_reward": record["reward"],
        }
        links.append(f"- [{case}]({case}.md)")
    (output / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    (output / "provenance-and-existing-scores.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Ten blinded rubric cases\n\n"
        "Mark each rubric item true or false for both anonymous answers. Add a short reason when uncertain.\n"
        "The supplied item weights define the score. There is no separate subjective numeric score.\n"
        "Use the case files and labels.json for scoring. The provenance file contains answer origins and existing scores.\n\n"
        + "\n".join(links)
        + "\n\nSelection: ten training groups sampled with seed 20261122 from 27 audited groups.\n"
        "Each group's answer was already selected with seed 20261118 before this packet was prepared.\n"
        "Answer order is randomized. No model inference, judge call or held-out data was used.\n"
        "Human agreement remains unmeasured until the labels are completed.\n"
    )
    return {"cases": len(chosen), "distinct_questions": len({x["question_key"] for x in provenance.values()})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    print(json.dumps(prepare(cli.audit, cli.output)))


if __name__ == "__main__":
    main()
