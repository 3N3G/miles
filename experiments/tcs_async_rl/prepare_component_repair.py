"""Freeze unseen saved answers for the live component-credit prompt repair."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from experiments.tcs_async_rl.judge_comparison import SEED, choose_easy, digest, fetch_easy, save
from experiments.tcs_async_rl.rubric_judge import rubric_items


def prepare_easy(root, *, run, audit, previous, excluded):
    prior = [
        row
        for path in previous
        for row in json.loads((path / "selection-provenance-not-for-labeling.json").read_text())
        if row["kind"] == "easy"
    ]
    blocked = {(row["question_key"], row["judge_identity"]) for row in prior}
    review = json.loads(excluded.read_text())
    blocked.update((review["question_key"], row["identity"]) for row in review["audits"])
    live_reviews = Path(__file__).parent / "evidence/screen_partial_proof_reviews_20261124.jsonl"
    for line in live_reviews.read_text().splitlines():
        entry = json.loads(line)
        for row in entry.get("audits", [entry]):
            blocked.add((row.get("question_key", entry.get("question_key")), row["identity"]))
    selected = choose_easy(audit, random.Random(SEED), blocked)
    cases = fetch_easy(run, selected, root / "staging")
    random.Random(SEED).shuffle(cases)
    folder = root / "packet"
    folder.mkdir(exist_ok=False)
    rows, provenance = [], []
    for index, row in enumerate(cases, 1):
        case = f"case-{index:02d}"
        value = {"case": case, "question": row["question"], "candidate": row["candidate"]}
        rows.append(value)
        provenance.append({"case": case, **row["source"]})
        save(folder / f"{case}.json", value)
    save(folder / "easy-packet.json", {"seed": SEED, "cases": rows})
    save(folder / "selection-provenance-not-for-labeling.json", provenance)
    save(
        folder / "easy-freeze.json",
        {
            "packet_sha256": digest(folder / "easy-packet.json"),
            "answers": 32,
            "excluded_packets": [digest(path / "packet.json") for path in previous],
            "excluded_live_reviews_sha256": digest(live_reviews),
            "heldout_reads": 0,
            "new_solver_generations": 0,
            "new_judge_calls": 0,
        },
    )


def complete(root, *, screen, excluded, previous=(), exposed_files=()):
    import gzip

    from experiments.tcs_async_rl.reward_hook import question_key

    folder = root / "packet"
    packet = json.loads((folder / "easy-packet.json").read_text())
    if digest(folder / "easy-packet.json") != json.loads((folder / "easy-freeze.json").read_text())["packet_sha256"]:
        raise ValueError("Easy packet changed after freezing")
    complete = json.loads((screen / "screen-complete.json").read_text())
    source = screen / "rollout_data" / Path(complete["responses"]).name
    if digest(source) != complete["responses_sha256"] or complete["answers"] != 384:
        raise ValueError("Expected the complete immutable fixed screen")
    questions = {
        question_key(q): q
        for q in json.loads((screen / "input.json").read_text())["questions"]
        if q["difficulty"] == "medium"
    }
    review = json.loads(excluded.read_text())
    blocked = {(review["question_key"], row["identity"]) for row in review["audits"]}
    blocked.add((review["question_key"], review["additional_7_point_case"]["identity"]))
    for path in previous:
        blocked.update(
            (row["question_key"], row["judge_identity"])
            for row in json.loads((path / "selection-provenance-not-for-labeling.json").read_text())
            if row["kind"] == "medium" and "question_key" in row
        )
    exposed_file = Path(__file__).parent / "evidence/component_repair_exposed_grades_20261124.json"
    for path in (exposed_file, *exposed_files):
        blocked.update((row["question_key"], row["identity"]) for row in json.loads(path.read_text())["excluded"])
    groups = {key: [] for key in questions}
    with gzip.open(source, "rt") as stream:
        for line in stream:
            sample = json.loads(line)
            key = sample["metadata"]["question_key"]
            if key in groups and sample["final_answer"] and (key, sample["judge_identity"]) not in blocked:
                groups[key].append(sample)
    rng = random.Random(SEED)
    chosen, remaining = [], []
    for key in sorted(groups):
        # Exclude repeated text so one cached answer cannot become two audit cases.
        unique = {
            row["judge_identity"]: row for row in sorted(groups[key], key=lambda row: row["index"], reverse=True)
        }
        candidates = sorted(unique.values(), key=lambda row: row["index"])
        if candidates:
            first = rng.choice(candidates)
            chosen.append((key, first))
            remaining.extend((key, row) for row in candidates if row["judge_identity"] != first["judge_identity"])
    if len(chosen) + len(remaining) < 8:
        raise ValueError("Fewer than eight unaudited nonempty medium answers")
    chosen.extend(rng.sample(remaining, 8 - len(chosen)))
    provenance = json.loads((folder / "selection-provenance-not-for-labeling.json").read_text())
    for index, (key, sample) in enumerate(chosen, 33):
        case = f"case-{index:02d}"
        value = {
            "case": case,
            "question": {k: questions[key][k] for k in ("problem", "answer", "rubric")},
            "candidate": sample["final_answer"],
        }
        packet["cases"].append(value)
        save(folder / f"{case}.json", value)
        provenance.append(
            {
                "case": case,
                "kind": "medium",
                "question_key": key,
                "judge_identity": sample["judge_identity"],
                "sample_index": sample["index"],
                "responses_sha256": complete["responses_sha256"],
            }
        )
    save(folder / "packet.json", packet)
    save(folder / "selection-provenance-not-for-labeling.json", provenance)
    save(
        folder / "packet-freeze.json",
        {
            "packet_sha256": digest(folder / "packet.json"),
            "cases": 40,
            "easy_answers": 32,
            "medium_answers": 8,
            "medium_questions": len({key for key, _ in chosen}),
            "screen_responses_sha256": complete["responses_sha256"],
            "excluded_seen_grade_sha256": digest(exposed_file),
            "additional_exposed_grade_sha256": [digest(path) for path in exposed_files],
            "excluded_packets": [digest(path / "packet.json") for path in previous],
            "heldout_reads": 0,
            "new_solver_generations": 0,
            "new_judge_calls": 0,
        },
    )
    save(
        folder / "labels.template.json",
        {
            "kind": "assistant_not_human",
            "cases": [
                {
                    "case": row["case"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "confidence": None,
                    "candidate_items": [
                        {"item": i, "satisfied": None, "reason": ""}
                        for i, _ in rubric_items(row["question"]["rubric"])
                    ],
                }
                for row in packet["cases"]
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("easy", "complete"))
    for name in ("root", "run", "audit", "excluded", "screen"):
        parser.add_argument("--" + name, type=Path, required=name in ("root", "excluded"))
    parser.add_argument("--previous", type=Path, action="append", default=[])
    parser.add_argument("--exposed-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.stage == "easy":
        prepare_easy(args.root, run=args.run, audit=args.audit, previous=args.previous, excluded=args.excluded)
    else:
        complete(
            args.root,
            screen=args.screen,
            excluded=args.excluded,
            previous=args.previous,
            exposed_files=args.exposed_file,
        )
