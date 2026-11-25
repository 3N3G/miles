"""Inspect saved sampling choices through long answers, without new inference.

Curves are conditioned on the observed completion outcome and question mix.
Sampled surprisal is not an independent full-policy entropy estimate. A token
outside the recorded shortlist cannot be sampled on that particular step.
"""

import argparse
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import trace_summary


def position_counts(sample, closing_token_id, bins):
    length = sample["response_length"]
    support = sample["rollout_sampling_mask"]
    ids, offsets = support["ids"], support["offsets"]
    counts = []
    for bucket in range(bins):
        start, end = bucket * length // bins, (bucket + 1) * length // bins
        counts.append(
            {
                "tokens": end - start,
                "single_choice_tokens": sum(offsets[i + 1] - offsets[i] == 1 for i in range(start, end)),
                "end_thinking_admitted_tokens": sum(
                    closing_token_id in ids[offsets[i] : offsets[i + 1]] for i in range(start, end)
                ),
                "sampled_surprisal_sum": -math.fsum(sample["rollout_log_probs"][start:end]),
            }
        )
    return counts


def aggregate(rows, bins):
    groups = {}
    for outcome in sorted({row["outcome"] for row in rows}):
        members = [row for row in rows if row["outcome"] == outcome]
        totals = [
            {key: sum(row["positions"][i][key] for row in members) for key in members[0]["positions"][i]}
            for i in range(bins)
        ]
        groups[outcome] = {
            "answers": len(members),
            "positions": [
                (
                    {
                        **bucket,
                        "single_choice_fraction": bucket["single_choice_tokens"] / bucket["tokens"],
                        "end_thinking_admitted_fraction": bucket["end_thinking_admitted_tokens"]
                        / bucket["tokens"],
                        "mean_sampled_surprisal_nats": bucket["sampled_surprisal_sum"] / bucket["tokens"],
                    }
                    if bucket["tokens"]
                    else {
                        **bucket,
                        "single_choice_fraction": None,
                        "end_thinking_admitted_fraction": None,
                        "mean_sampled_surprisal_nats": None,
                    }
                )
                for bucket in totals
            ],
        }
    return groups


def analyze(directory, traces, tokenizer_config, steps, bins=10):
    tokenizer = json.loads(tokenizer_config.read_text())
    closing = [
        int(key) for key, value in tokenizer["added_tokens_decoder"].items() if value["content"] == "</think>"
    ]
    if len(closing) != 1 or bins <= 0:
        raise ValueError("Expected one pinned closing-thinking token and positive position bins")
    index = json.loads((directory / "archive-index.json").read_text())
    rows, sources = [], []
    for step in steps:
        name = f"rollout_data/rollout-{step}.jsonl.gz"
        path, expected = traces / name, index[name]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError("Support diagnostic trace differs from its verified immutable archive")
        sources.append({"optimizer_step": step, **expected})
        with gzip.open(path, "rt") as stream:
            for line in stream:
                sample = json.loads(line)
                trace_summary(sample)
                outcome = "completed"
                if sample["status"] == "truncated":
                    outcome = "capped_with_answer" if sample["final_answer"] else "capped_without_answer"
                rows.append(
                    {
                        "optimizer_step": step,
                        "sample_index": sample["index"],
                        "question_key": sample["metadata"]["question_key"],
                        "outcome": outcome,
                        "positions": position_counts(sample, closing[0], bins),
                    }
                )
    return {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "sources": sources,
        "closing_thinking_token_id": closing[0],
        "tokenizer_config_sha256": hashlib.sha256(tokenizer_config.read_bytes()).hexdigest(),
        "position_bins": bins,
        "answers": len(rows),
        "by_completion_outcome": aggregate(rows, bins),
        "samples": rows,
        "limits": "Pooled token fractions by relative response position. Conditioned on outcome and changing question/policy mixtures. No causal comparison or full-policy entropy estimate. A singleton shortlist gives no sampling choice at that token; it does not prove an entire response cannot improve.",
        "new_inference_or_judge_calls": 0,
    }


def plot(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {
        "completed": ("Completed", "#0072B2"),
        "capped_with_answer": ("Capped, final answer present", "#009E73"),
        "capped_without_answer": ("Capped, no final answer", "#D55E00"),
    }
    for outcome, group in result["by_completion_outcome"].items():
        label, color = labels[outcome]
        ax.plot(
            [(i + 0.5) / result["position_bins"] * 100 for i in range(result["position_bins"])],
            [
                row["single_choice_fraction"] * 100 if row["single_choice_fraction"] is not None else math.nan
                for row in group["positions"]
            ],
            marker="o",
            color=color,
            label=f"{label} (n={group['answers']})",
        )
    ax.set(
        xlabel="Position within each response (%)",
        ylabel="Tokens with exactly one sampling choice (%)",
        ylim=(0, 100),
        xlim=(0, 100),
    )
    ax.legend(fontsize=8)
    ending = "; ".join(
        f"{labels[outcome][0]}: {group['positions'][-1]['single_choice_fraction']:.1%}"
        for outcome, group in result["by_completion_outcome"].items()
        if group["positions"][-1]["single_choice_fraction"] is not None
    )
    fig.text(0.5, 0.045, "Final tenth, single-choice fractions: " + ending, ha="center", fontsize=8)
    fig.text(
        0.5,
        0.015,
        "Pooled token fractions. Different questions and sampled policies; no inferential error bars.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"sampling-choices-by-position.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-directory", "trace-directory", "tokenizer-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    cli = parser.parse_args()
    result = analyze(cli.run_directory, cli.trace_directory, cli.tokenizer_config, cli.steps)
    cli.output.mkdir(parents=True, exist_ok=True)
    (cli.output / "support-position-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, cli.output)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))
