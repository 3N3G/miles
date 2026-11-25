"""Measure exact repeated word windows in archived responses and final answers.

This descriptive diagnostic makes no inference or judge requests. Repetition is
not policy entropy, and outcome groups can contain different questions.
"""

import argparse
import collections
import datetime
import gzip
import json
import statistics
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import file_sha256


def repeated_windows(text, window_words):
    if type(window_words) is not int or window_words < 1:
        raise ValueError("The window must contain a positive integer number of words")
    words = text.split()
    total = max(0, len(words) - window_words + 1)
    distinct = len({tuple(words[start : start + window_words]) for start in range(total)})
    return {
        "words": len(words),
        "windows": total,
        "distinct_windows": distinct,
        "repeated_fraction": 1 - distinct / total if total else None,
    }


def completion_category(sample):
    if sample["status"] == "completed":
        return "completed"
    if sample["status"] != "truncated":
        raise ValueError("An inference error cannot become an analyzed model cap")
    return "capped_nonempty_final" if sample["final_answer"] else "capped_empty_final"


def summarize(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["completion_category"]].append(row)
    result = {}
    for category, group in sorted(grouped.items()):
        result[category] = {"answers": len(group), "full_credit": sum(row["reward"] == 7 for row in group)}
        for field in ("full_response", "final_answer"):
            values = [row[field]["repeated_fraction"] for row in group]
            values = [value for value in values if value is not None]
            result[category][field] = {
                "measured_answers": len(values),
                "median_repeated_fraction": statistics.median(values) if values else None,
                "answers_above_half": sum(value > 0.5 for value in values),
            }
    return result


def analyze(directory, traces, steps, *, window_words):
    index = json.loads((directory / "archive-index.json").read_text())
    sources, rows = [], []
    for step in steps:
        name = f"rollout_data/rollout-{step}.jsonl.gz"
        path, entry = traces / name, index[name]
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError("Repetition input differs from the immutable archived trace")
        sources.append({"optimizer_step": step, **entry})
        with gzip.open(path, "rt") as stream:
            for line in stream:
                sample = json.loads(line)
                if sample["optimizer_rollout_id"] != step:
                    raise ValueError("Sample and archived optimizer coordinates differ")
                rows.append(
                    {
                        "optimizer_step": step,
                        "sample_index": sample["index"],
                        "question_key": sample["metadata"]["question_key"],
                        "completion_category": completion_category(sample),
                        "response_tokens": sample["response_length"],
                        "reward": sample["reward"]["reward"],
                        "full_response": repeated_windows(sample["response"], window_words),
                        "final_answer": repeated_windows(sample["final_answer"] or "", window_words),
                    }
                )
    return {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "sources": sources,
        "window_words": window_words,
        "definition": "1 minus the number of distinct exact word windows divided by all word windows",
        "answers": len(rows),
        "by_completion_outcome": summarize(rows),
        "rows": rows,
        "limits": "Surface repetition, not policy entropy or a causal comparison. Questions differ between outcomes.",
        "new_inference_or_judge_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--optimizer-steps", nargs="+", type=int, required=True)
    parser.add_argument("--window-words", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    result = analyze(cli.run_directory, cli.trace_directory, cli.optimizer_steps, window_words=cli.window_words)
    result["analysis_source_commit"] = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
    ).strip()
    result["analysis_module_sha256"] = file_sha256(Path(__file__))
    cli.output.mkdir(parents=True, exist_ok=False)
    (cli.output / "repetition-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
