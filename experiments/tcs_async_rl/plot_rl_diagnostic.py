"""Plot saved descriptive diagnostics. Does not reread weights, sample or grade."""

import argparse
import json
from pathlib import Path


def plot(directory):
    # Plotting is optional; the full trace audit and CPU tests need only stdlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    native = json.loads((directory / "native-metrics.json").read_text())
    groups = json.loads((directory / "groups.json").read_text())
    pairs = json.loads((directory / "same-question-comparisons.json").read_text())
    samples = json.loads((directory / "samples.json").read_text())
    history = json.loads((directory / "wandb-full-history.json").read_text())["rows"]
    summary = json.loads((directory / "summary.json").read_text())
    if len(groups) != len(native):
        raise ValueError("Use plot_continuation for multiple prompt groups per optimizer update")
    steps = [row["optimizer_step"] for row in native]
    blue, orange, gray = "#0072B2", "#D55E00", "#666666"

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, [g["mean_reward"] for g in groups], "o-", color=blue, label="Native events: correct optimizer step")
    damaged = [r for r in history if r.get("rollout/episode_raw_reward") is not None]
    ax.scatter(
        [r["rollout/step"] for r in damaged],
        [r["rollout/episode_raw_reward"] for r in damaged],
        marker="x",
        color=orange,
        label="Original W&B episode series",
    )
    ax.axvline(summary["questions"] - 0.5, color=gray, linestyle="--", label="End of first question pass")
    ax.set(
        xlabel="Optimizer step (zero based)",
        ylabel="Group mean rubric points (0–7)",
        ylim=(0, 7.1),
        title="Offline writer collisions misassigned some W&B reward coordinates",
    )
    ax.legend(fontsize=8)
    finish(fig, "reward-axis-audit.png")

    labels = [
        "Adjacency labels",
        "Consumer surplus",
        "Rooted forest bound",
        "Binary perceptron",
        "Hypergraph Moore bound",
        "Dynamic minimum cut",
        "Low-degree testing",
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (pair, _label) in enumerate(zip(pairs, labels, strict=True)):
        ax.plot([pair["first_reward"], pair["last_reward"]], [i, i], color=gray)
    ax.scatter([p["first_reward"] for p in pairs], range(len(pairs)), color=gray, label="First training visit")
    ax.scatter(
        [p["last_reward"] for p in pairs], range(len(pairs)), color=blue, marker="s", label="Second training visit"
    )
    ax.set(
        yticks=range(len(pairs)),
        yticklabels=labels,
        xlabel=f"Mean rubric points ({groups[0]['samples']} answers per visit)",
        xlim=(0, 7.1),
        title=f"Repeated questions: {summary['questions_increased']} means rose; "
        f"{summary['questions_decreased']} fell (mean change {summary['same_question_mean_change']:+.3f}/7)",
    )
    ax.legend(fontsize=8)
    finish(fig, "same-question-rewards.png")

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, key, label in [
        (axes[0, 0], "train/grad_norm", "Pre-clipping gradient norm"),
        (axes[0, 1], "train/train_rollout_logprob_abs_diff", "Mean absolute log-probability difference"),
        (axes[1, 0], "train/train_rollout_kl", "Sampled rollout-to-trainer k3 KL\n(nats/token)"),
        (axes[1, 1], "train/pg_clipfrac", "Fraction of CISPO weights clipped"),
    ]:
        ax.plot(steps, [r[key] for r in native], color=blue)
        ax.set(ylabel=label, ylim=(0, None), xlabel="Optimizer step")
    axes[0, 0].axhline(1, color=orange, linestyle="--", label="Gradient clip threshold")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 1)
    fig.suptitle("No observed clipping, gradient explosion or growing trainer/rollout mismatch")
    finish(fig, "optimizer-health.png")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    train = [r["perf/actor_train_time"] / 60 for r in native]
    wait = [r["perf/train_wait_time"] / 60 for r in native]
    axes[0].bar(steps, train, color=blue, label="Actor training")
    axes[0].bar(steps, wait, bottom=train, color="#999999", label="Actor waiting for data")
    axes[0].set(ylabel="Minutes per optimizer step", ylim=(0, None))
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [g["mean_tokens"] / 1000 for g in groups], "o-", color=blue)
    axes[1].set(xlabel="Optimizer step", ylabel="Mean answer length (1,000 tokens)", ylim=(0, 90))
    waiting_fraction = sum(r["perf/train_wait_time"] for r in native) / sum(r["perf/step_time"] for r in native)
    fig.suptitle(f"Actor waiting accounts for {waiting_fraction:.0%} of summed step time")
    finish(fig, "generation-cost.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    gaps = [s["order_gap"] for s in samples if s["order_gap"] is not None]
    if gaps:
        xs = sorted(set(gaps))
        ax.bar(xs, [gaps.count(x) for x in xs], width=0.6, color=blue)
        disagreement = sum(gap > 0 for gap in gaps)
        ax.set(
            xlabel="Absolute change in rubric points after swapping answer order",
            ylabel="Consumed answers",
            ylim=(0, None),
            title=f"Judge scores differed between orders for {disagreement}/{len(gaps)} answers ({disagreement / len(gaps):.1%})",
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

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, [r["train/pg_loss"] for r in native], color=blue)
    axes[0].axhline(0, color=gray, linestyle="--")
    axes[0].set(ylabel="Logged CISPO policy loss")
    axes[1].plot(steps, [r["train/mtp_loss"] for r in native], color=blue)
    axes[1].set(xlabel="Optimizer step", ylabel="Logged auxiliary MTP loss", ylim=(0, None))
    fig.suptitle("Losses compare changing question groups; their values do not measure learning")
    finish(fig, "loss-diagnostics.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot(parser.parse_args().directory)


if __name__ == "__main__":
    main()
