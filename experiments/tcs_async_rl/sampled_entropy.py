"""Retrospective sampler-surprisal estimates from already audited rollout traces.

This does not recover full-policy entropy or reference-policy divergence. Each
sample stores mean log-probability under the filtered behavior that generated it.
"""

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import file_sha256


def summarize(rows):
    by_step = collections.defaultdict(list)
    for row in rows:
        value = row["mean_sampled_log_probability"]
        if not math.isfinite(value) or value > 1e-5 or row["response_tokens"] <= 0:
            raise ValueError("Invalid audited sampled log-probability or response length")
        by_step[row["optimizer_step"]].append(row)
    groups = []
    visits = collections.defaultdict(list)
    for step, samples in sorted(by_step.items()):
        if len({r["question_key"] for r in samples}) != 1:
            raise ValueError("One group must contain one question")
        tokens = sum(r["response_tokens"] for r in samples)
        group = {
            "optimizer_step": step,
            "question_key": samples[0]["question_key"],
            "paper_id": samples[0]["paper_id"],
            "samples": len(samples),
            "answer_mean_sampled_surprisal_nats": statistics.mean(-r["mean_sampled_log_probability"] for r in samples),
            "token_weighted_sampled_surprisal_nats": sum(
                -r["mean_sampled_log_probability"] * r["response_tokens"] for r in samples
            )
            / tokens,
            "singleton_support_token_fraction": sum(r["tokens_with_support_size_one"] for r in samples) / tokens,
            "mean_reward": statistics.mean(r["reward"] for r in samples),
        }
        groups.append(group)
        visits[group["question_key"]].append(group)
    pairs = []
    for entries in visits.values():
        if len(entries) < 2:
            continue
        first, last = entries[0], entries[-1]
        a, b = (g["answer_mean_sampled_surprisal_nats"] for g in (first, last))
        pairs.append(
            {
                "question_key": first["question_key"],
                "paper_id": first["paper_id"],
                "first_step": first["optimizer_step"],
                "last_step": last["optimizer_step"],
                "first_nats": a,
                "last_nats": b,
                "change_nats": b - a,
                "first_reward": first["mean_reward"],
                "last_reward": last["mean_reward"],
            }
        )
    result = {"answers": len(rows), "groups": groups, "repeated_questions": pairs}
    if pairs:
        result["matched_first_nats"] = statistics.mean(p["first_nats"] for p in pairs)
        result["matched_last_nats"] = statistics.mean(p["last_nats"] for p in pairs)
        result["questions_increased"] = sum(p["change_nats"] > 0 for p in pairs)
        result["questions_decreased"] = sum(p["change_nats"] < 0 for p in pairs)
    return result


def plot_pairs(pairs, output):
    # Optional plotting dependency. Pure evidence reads and CPU tests need none.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.3, 5.3), constrained_layout=True)
    limit = max(max(p["first_nats"], p["last_nats"]) for p in pairs) * 1.15
    axis.plot([0, limit], [0, limit], color="#888888", linestyle="--", label="Unchanged")
    axis.scatter([p["first_nats"] for p in pairs], [p["last_nats"] for p in pairs], color="#0072B2")
    for pair in pairs:
        axis.annotate(
            f"{pair['first_step']}→{pair['last_step']}",
            (pair["first_nats"], pair["last_nats"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set(xlim=(0, limit), ylim=(0, limit))
    axis.set_xlabel("First visit: mean sampled surprisal (nats/token)")
    axis.set_ylabel("Later visit: mean sampled surprisal (nats/token)")
    axis.set_title(f"{len(pairs)} repeated questions retain nonzero sampler uncertainty")
    axis.legend()
    fig.savefig(output / "matched-sampled-surprisal.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    result = summarize(json.loads(cli.audited_samples.read_text()))
    result["source_file"] = str(cli.audited_samples)
    result["source_sha256"] = file_sha256(cli.audited_samples)
    (cli.output / "sampled-entropy.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_pairs(result["repeated_questions"], cli.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ("groups", "repeated_questions")}, indent=2))


if __name__ == "__main__":
    main()
