"""Freeze a blinded saved-answer packet and compare two grading recipes.

Preparation reads immutable private traces one at a time. Comparison requires
pre-call labels and never generates a solver answer or changes an old grade.
"""

import argparse
import asyncio
import collections
import datetime
import gzip
import hashlib
import json
import os
import random
import shutil
import statistics
from pathlib import Path

from experiments.tcs_async_rl.grade_group import Budget, grade_one, usage_cost
from experiments.tcs_async_rl.judge_client import load_credentials
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rubric_judge import rubric_items

SEED = 20261124


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def choose_easy(audit, rng):
    excluded = {
        (row["question_key"], row["judge_identity"])
        for path in audit.glob("seeded-proofs-*.json")
        for row in json.loads(path.read_text()).values()
    }
    by_question = collections.defaultdict(list)
    for path in sorted(audit.glob("sample-audit-*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row["final_characters"] and (row["question_key"], row["judge_identity"]) not in excluded:
                by_question[row["question_key"]].append(row)
    selected = []
    for key in rng.sample(sorted(by_question), 16):
        candidates = by_question[key]
        first = rng.choice([r for r in candidates if r["reward"] == 7] or candidates)
        rest = [r for r in candidates if r["judge_identity"] != first["judge_identity"]]
        second = rng.choice([r for r in rest if r["reward"] < 7] or rest)
        selected.extend([first, second])
    return selected


def fetch_easy(run, selected, staging):
    # Only this preparation command needs the optional Hub SDK.
    from huggingface_hub import hf_hub_download

    questions = {question_key(q): q for q in json.loads((run / "input.json").read_text())["questions"]}
    archive = json.loads((run / "archive-index.json").read_text())
    wanted = {(r["optimizer_step"], r["sample_index"]): r for r in selected}
    found = []
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    staging.mkdir(parents=True, exist_ok=True)
    for step in sorted({r["optimizer_step"] for r in selected}):
        name = f"rollout_data/rollout-{step}.jsonl.gz"
        entry = archive[name]
        if shutil.disk_usage(staging).free < entry["bytes"] + 2**30:
            raise RuntimeError("Audit staging would violate its one-GiB reserve")
        path = Path(
            hf_hub_download(
                repo_id="3N3G/internalization-runs",
                repo_type="dataset",
                revision=entry["hf_commit_url"].rsplit("/", 1)[-1],
                filename=f"miles/{run.name}/{name}",
                local_dir=staging,
            )
        )
        assert digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
        with gzip.open(path, "rt") as stream:
            for line in stream:
                sample = json.loads(line)
                identity = step, sample["index"]
                if identity not in wanted:
                    continue
                row = wanted[identity]
                q = questions[row["question_key"]]
                found.append(
                    {
                        "question": {k: q[k] for k in ("problem", "answer", "rubric")},
                        "candidate": sample["final_answer"],
                        "source": {"kind": "easy", **row, "trace_sha256": entry["sha256"]},
                    }
                )
        path.unlink()
    assert len(found) == 32
    return found


def prepare(cli):
    cli.output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(SEED)
    selected = choose_easy(cli.audit, rng)
    cases = fetch_easy(cli.run, selected, cli.staging)
    medium = json.loads((cli.medium / "input.json").read_text())
    responses = [json.loads(line) for line in (cli.medium / "responses.jsonl").read_text().splitlines()]
    eligible = [r for r in responses if r["final_answer"] and r["sample_index"] != 13]
    for row in rng.sample(eligible, 8):
        cases.append(
            {
                "question": {k: medium[k] for k in ("problem", "answer", "rubric")},
                "candidate": row["final_answer"],
                "source": {
                    "kind": "medium",
                    "sample_index": row["sample_index"],
                    "responses_sha256": digest(cli.medium / "responses.jsonl"),
                },
            }
        )
    rng.shuffle(cases)
    packet, provenance, labels = [], [], []
    for i, row in enumerate(cases, 1):
        case = f"case-{i:02d}"
        value = {"case": case, "question": row["question"], "candidate": row["candidate"]}
        packet.append(value)
        provenance.append({"case": case, **row["source"]})
        save(cli.output / f"{case}.json", value)
        labels.append(
            {
                "case": case,
                "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                "confidence": None,
                "candidate_items": [
                    {"item": index, "satisfied": None, "reason": ""}
                    for index, _ in rubric_items(row["question"]["rubric"])
                ],
            }
        )
    save(cli.output / "packet.json", {"seed": SEED, "cases": packet})
    save(cli.output / "selection-provenance-not-for-labeling.json", provenance)
    save(cli.output / "labels.template.json", {"kind": "assistant_not_human", "cases": labels})
    save(
        cli.output / "packet-freeze.json",
        {
            "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "packet_sha256": digest(cli.output / "packet.json"),
            "cases": 40,
            "easy_questions": 16,
            "easy_answers": 32,
            "medium_questions": 1,
            "medium_answers": 8,
            "new_solver_generations": 0,
            "new_judge_calls": 0,
            "heldout_reads": 0,
        },
    )
    return {"cases": len(cases), "packet_sha256": digest(cli.output / "packet.json")}


def comparison_metrics(cases, labels, results):
    expected = {r["case"]: r for r in labels["cases"]}
    output = {}
    for method in ("anonymous_pair", "reference_guided"):
        rows = []
        for case in cases:
            identity = case["case"]
            weights = dict(rubric_items(case["question"]["rubric"]))
            truth = {x["item"]: x["satisfied"] for x in expected[identity]["candidate_items"]}
            if set(truth) != set(weights) or any(type(v) is not bool for v in truth.values()):
                raise ValueError("Audit must label every original item before grading")
            target = sum(weights[i] * truth[i] for i in weights)
            verdicts = results[identity][method]
            scores = [v["candidate"]["score"] for v in verdicts]
            agreement = statistics.mean(
                statistics.mean(x["satisfied"] == truth[x["item"]] for x in v["candidate"]["items"]) for v in verdicts
            )
            observed = statistics.mean(scores)
            rows.append(
                {
                    "case": identity,
                    "question_key": question_key(case["question"]),
                    "audit_score": target,
                    "judge_scores": scores,
                    "reward": observed,
                    "exact_score_match": observed == target,
                    "absolute_score_error": abs(observed - target),
                    "item_agreement": agreement,
                    "clear_false_full_credit": observed == 7 and target <= 5,
                }
            )
        exact = statistics.mean(r["exact_score_match"] for r in rows)
        item = statistics.mean(r["item_agreement"] for r in rows)
        output[method] = {
            "answers": len(rows),
            "exact_score_agreement": exact,
            "case_balanced_item_agreement": item,
            "mean_absolute_score_error": statistics.mean(r["absolute_score_error"] for r in rows),
            "clear_false_full_credit": sum(r["clear_false_full_credit"] for r in rows),
            "passes_observed_gate": exact >= 0.9
            and item >= 0.95
            and not any(r["clear_false_full_credit"] for r in rows),
            "cases": rows,
        }
    return output


async def compare(cli):
    packet = json.loads((cli.packet / "packet.json").read_text())
    assert (
        digest(cli.packet / "packet.json")
        == json.loads((cli.packet / "packet-freeze.json").read_text())["packet_sha256"]
    )
    assert digest(cli.labels) == cli.labels_sha256
    labels = json.loads(cli.labels.read_text())
    if {r["case"] for r in labels["cases"]} != {r["case"] for r in packet["cases"]}:
        raise ValueError("Audit labels differ from the frozen comparison cases")
    # Validate all labels before loading the already authorized VM credential.
    for case, label in zip(packet["cases"], labels["cases"], strict=True):
        assert case["case"] == label["case"]
        assert hashlib.sha256(case["candidate"].encode()).hexdigest() == label["candidate_sha256"]
        items = rubric_items(case["question"]["rubric"])
        assert [i for i, _ in items] == [x["item"] for x in label["candidate_items"]]
        assert all(type(x["satisfied"]) is bool and x["reason"] for x in label["candidate_items"])
    cli.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(cli.config.read_text())
    training = json.loads((cli.config.parent / config["training_config"]).read_text())
    judge = training["judge"]
    shared = Budget(cli.output, config)
    semaphore = asyncio.Semaphore(config["max_concurrency"])
    key = load_credentials(Path.home() / ".config/api-keys.env")[judge["api_key_env"]]
    freeze_path = cli.output / "pre-call-label-freeze.json"
    freeze = {
        "labels_sha256": cli.labels_sha256,
        "packet_sha256": digest(cli.packet / "packet.json"),
        "config_sha256": digest(cli.config),
        "prompt_sha256": {
            name: digest(cli.config.parent / name)
            for name in ("rubric_judge_prompt_v3.txt", "rubric_judge_reference_v1.txt")
        },
    }
    if freeze_path.exists():
        old = json.loads(freeze_path.read_text())
        if any(old[k] != v for k, v in freeze.items()):
            raise ValueError("Comparison inputs changed after their pre-call freeze")
    else:
        save(freeze_path, {**freeze, "at_utc": datetime.datetime.now(datetime.UTC).isoformat()})

    async def one(case, method, swapped):
        folder = cli.output / case["case"] / method
        folder.mkdir(parents=True, exist_ok=True)
        prompt_file = "rubric_judge_prompt_v3.txt" if method == "anonymous_pair" else "rubric_judge_reference_v1.txt"
        return await grade_one(
            identity=case["case"],
            candidate=case["candidate"],
            row=case["question"],
            swapped=swapped,
            judge=judge,
            config={**config, "grading_method": method},
            prompt=(cli.config.parent / prompt_file).read_text(),
            key=key,
            budget=shared,
            semaphore=semaphore,
            directory=folder,
        )

    tasks = [
        (case, method, swapped)
        for case in packet["cases"]
        for method in ("anonymous_pair", "reference_guided")
        for swapped in ((False, True) if method == "anonymous_pair" else (False,))
    ]
    grades = await asyncio.gather(*(one(*task) for task in tasks))
    results = {c["case"]: {"anonymous_pair": [], "reference_guided": []} for c in packet["cases"]}
    for (case, method, _), grade in zip(tasks, grades, strict=True):
        results[case["case"]][method].append(grade)
    save(cli.output / "verdicts.json", results)
    metrics = comparison_metrics(packet["cases"], labels, results)
    for method in metrics:
        receipts = [json.loads(p.read_text()) for p in cli.output.glob(f"case-*/{method}/*-attempt*.json")]
        metrics[method]["actual_usage_usd"] = sum(
            usage_cost(r["body"]["usage"], config) for r in receipts if r.get("body", {}).get("usage") is not None
        )
    candidates = [name for name in metrics if metrics[name]["passes_observed_gate"]]
    selected = (
        min(
            candidates,
            key=lambda k: (
                -metrics[k]["exact_score_agreement"],
                metrics[k]["mean_absolute_score_error"],
                metrics[k]["actual_usage_usd"],
            ),
        )
        if candidates
        else None
    )
    save(cli.output / "comparison.json", metrics)
    save(
        cli.output / "selection.json",
        {
            "selected_method": selected,
            "selection_rule": "Exact score agreement, then lower absolute error, then lower actual cost; observed gate required.",
            "labels_sha256": cli.labels_sha256,
            "human_calibration": False,
        },
    )
    return {k: {name: value for name, value in v.items() if name != "cases"} for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("run", "audit", "medium", "staging", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("compare")
    for name in ("packet", "labels", "config", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--labels-sha256", required=True)
    cli = parser.parse_args()
    print(json.dumps(prepare(cli) if cli.command == "prepare" else asyncio.run(compare(cli)), indent=2))


if __name__ == "__main__":
    main()
