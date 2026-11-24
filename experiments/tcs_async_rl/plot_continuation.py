"""Plot the complete observed continuation without mixing question identities.

Inputs come from audit_live and the native metric reader. These are descriptive
training observations from one seed, without inferential error bars.
"""

import argparse
import json
import textwrap
from pathlib import Path


def plot(directory):
    # Optional plotting dependencies stay outside the GPU training environment.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    native = json.loads((directory / "native-metrics.json").read_text())
    groups = json.loads((directory / "groups.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    pairs = json.loads((directory / "same-question-comparisons.json").read_text())
    by_step = {row["optimizer_step"]: row for row in native}
    if any(group["optimizer_step"] not in by_step for group in groups):
        raise ValueError("A plotted group has no completed native update")
    steps = [row["optimizer_step"] for row in native]
    blue, orange, gray = "#0072B2", "#D55E00", "#666666"

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def series(ax, key, label, color=blue):
        points = [(row["optimizer_step"], row[key]) for row in native if key in row]
        ax.plot([x for x, _ in points], [y for _, y in points], "o-", color=color, label=label, markersize=3)

    questions = list(dict.fromkeys(group["question_key"] for group in groups))
    fig, axes = plt.subplots((len(questions) + 3) // 4, 4, figsize=(14, 3 * ((len(questions) + 3) // 4)))
    for ax, key in zip(axes.flat, questions, strict=False):
        visits = [group for group in groups if group["question_key"] == key]
        ax.plot([g["optimizer_step"] for g in visits], [g["mean_reward"] for g in visits], "o-", color=blue)
        title = visits[0]["paper_id"].replace("_", " ")
        ax.set_title("\n".join(textwrap.wrap(title, 34)) + "\n" + visits[0]["prompt_type"], fontsize=8)
        ax.set(
            xlabel="Global optimizer update",
            ylabel="Mean rubric points",
            ylim=(0, 7.15),
            xlim=(steps[0] - 0.5, steps[-1] + 0.5),
        )
    for ax in list(axes.flat)[len(questions) :]:
        ax.set_visible(False)
    fig.suptitle("All consumed question visits; one seed, descriptive training observations", y=1.01)
    finish(fig, "question-rewards.png")

    if pairs:
        groups_by_key = {group["question_key"]: group for group in groups}
        fig, ax = plt.subplots(figsize=(12, max(5, len(pairs) * 0.42)))
        labels = []
        for index, pair in enumerate(pairs):
            group = groups_by_key[pair["question_key"]]
            title = textwrap.shorten(pair["paper_id"].replace("_", " "), width=64, placeholder="…")
            kind = "proof plan" if group["prompt_type"] == "proof_strategy" else "proof"
            labels.append(f"{title} ({kind})")
            ax.plot([pair["first_reward"], pair["last_reward"]], [index, index], color=gray)
        ax.scatter([pair["first_reward"] for pair in pairs], range(len(pairs)), color=gray, label="First visit")
        ax.scatter(
            [pair["last_reward"] for pair in pairs], range(len(pairs)), color=blue, marker="s", label="Latest visit"
        )
        ax.set(
            yticks=range(len(pairs)),
            yticklabels=labels,
            xlabel="Mean rubric points (32 answers per training visit)",
            xlim=(0, 7.15),
            title=f"{len(pairs)} repeated questions; descriptive visits from one training seed",
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.legend()
        finish(fig, "matched-question-rewards.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    series(axes[0], "train/full_policy_entropy", "Full policy", blue)
    series(axes[0], "train/sampling_entropy", "After top-k/top-p filtering", orange)
    axes[0].set(ylabel="Entropy (nats/token)", ylim=(0, None))
    axes[0].legend()
    series(axes[1], "train/kl_loss", "To frozen initialization", blue)
    series(axes[1], "train/train_rollout_kl", "To rollout policy", orange)
    axes[1].set(xlabel="Global optimizer update", ylabel="Sampled k3 divergence\n(nats/token)", ylim=(0, None))
    axes[1].legend()
    fig.suptitle("Entropy and sampled policy drift on changing training questions")
    finish(fig, "entropy-divergence.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    series(ax, "train/reference_sampled_log_ratio", "Observed current/initial log-probability ratio")
    ax.axhline(0, color=gray, linestyle="--")
    ax.set(
        xlabel="Global optimizer update",
        ylabel="Mean log[p(current)/p(initial)]\n(nats/token)",
        title="Signed score change on observed rollout prefixes; changing questions",
    )
    finish(fig, "reference-signed-score-change.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    series(axes[0], "train/update_direction/surrogate_gain", "Observed CISPO surrogate gain")
    axes[0].axhline(0, color=gray, linestyle="--")
    axes[0].set(ylabel="Detached-weight surrogate gain")
    for advantage, color in [("positive", blue), ("negative", orange)]:
        key = f"train/update_direction/{advantage}_advantage_mean_logprob_change"
        series(axes[1], key, f"{advantage.capitalize()} advantages", color)
    axes[1].axhline(0, color=gray, linestyle="--")
    axes[1].set(xlabel="Global optimizer update", ylabel="Mean log-probability change\n(nats/token)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Scheduled before/after scores on the exact consumed batch")
    finish(fig, "update-direction.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, key, label in [
        (axes[0, 0], "train/grad_norm", "Main-policy gradient norm"),
        (axes[0, 1], "train/ess_ratio", "Effective sample-size ratio"),
        (axes[1, 0], "train/pg_clipfrac", "CISPO importance weights clipped"),
        (axes[1, 1], "rollout/fully_async/token_weighted_staleness", "Token-weighted policy lag (updates)"),
    ]:
        series(ax, key, label)
        ax.set(xlabel="Global optimizer update", ylabel=label, ylim=(0, None))
    axes[0, 0].axhline(1, color=orange, linestyle="--", label="Clipping threshold")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_ylim(0, 1.02)
    axes[1, 0].set_ylim(0, 1.02)
    fig.suptitle("Optimizer and asynchronous-policy diagnostics")
    finish(fig, "optimizer-health.png")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(steps, [r["rollout/response_lengths"] / 1000 for r in native], "o-", color=blue)
    axes[0].set(ylabel="Mean answer length\n(1,000 tokens)", ylim=(0, 90))
    series(axes[1], "rollout/truncated", "Output cap fraction")
    axes[1].set(ylabel="Answers reaching output cap", ylim=(0, 1))
    axes[2].bar(steps, [r["perf/train_time"] / 60 for r in native], color=blue, label="Training function")
    axes[2].bar(
        steps,
        [r["perf/train_wait_time"] / 60 for r in native],
        bottom=[r["perf/train_time"] / 60 for r in native],
        color=gray,
        label="Time outside training function",
    )
    axes[2].set(xlabel="Global optimizer update", ylabel="Minutes per step", ylim=(0, None))
    axes[2].legend(fontsize=8)
    fig.suptitle("Answer cost and logged step timing; outside time includes saving and publication")
    finish(fig, "generation-cost.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    gaps = [sample["order_gap"] for sample in samples if sample["order_gap"] is not None]
    if gaps:
        xs = sorted(set(gaps))
        ax.bar(xs, [gaps.count(x) for x in xs], width=0.4, color=blue)
        ax.set(
            xlabel="Absolute rubric-point difference after swapping answer order",
            ylabel="Consumed answers",
            ylim=(0, None),
            title=f"{sum(g > 0 for g in gaps)}/{len(gaps)} answers received different order scores",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Answer-order disagreement was not measured.\nThis recipe uses one labeled-reference presentation.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    finish(fig, "judge-order-disagreement.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)


if __name__ == "__main__":
    main()
