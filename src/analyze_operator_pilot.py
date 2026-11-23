"""Compute the prespecified paired contrast and plots from completed artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"shared": "#0072B2", "specific": "#D55E00", "dense": "#009E73",
          "ridge": "#CC79A7", "mean": "#888888"}


def bootstrap_interval(values, seed=1729):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Expected at least two paired article values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(-1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(rows, seeds=(0, 1, 2)):
    lookup = {(r["model"], r["evaluation_pass"]): r for r in rows}
    differences = []
    individual = []
    for seed in seeds:
        for p in range(1, 4):
            a = lookup[(f"shared_seed{seed}", p)]
            b = lookup[(f"pass{p}_seed{seed}", p)]
            difference = np.array(a["per_paragraph_normalized_mse"]) - b["per_paragraph_normalized_mse"]
            differences.append(difference)
            individual.append({"seed": seed, "pass": p, "shared_nmse": a["normalized_mse"],
                               "specific_nmse": b["normalized_mse"], "difference": float(difference.mean())})
    # Seeds and passes are repeated measurements on the same held-out articles.
    by_article = np.mean(differences, axis=0)
    return {"primary_contrast": "Shared minus matching pass-specific normalized prediction error",
            "mean_difference": float(by_article.mean()), "paired_95_percent_ci": bootstrap_interval(by_article),
            "sampling_unit": "one paragraph from each of 32 distinct final articles",
            "bootstrap_resamples": 10000, "bootstrap_seed": 1729,
            "per_article_differences": by_article.tolist(), "per_seed_and_pass": individual,
            "scope": "One fixed sublayer and one sparse-surrogate configuration"}


def routing_summary(run):
    inputs = json.loads((run / "inputs.json").read_text())["records"]
    routing = [np.load(run / f"routing_{i:03}.npz")["indices"] for i in range(len(inputs))]
    result = {}
    for split in ["fit", "validation", "final"]:
        selected = np.stack([routing[i] for i, r in enumerate(inputs) if r["split"] == split])
        counts = [[int(x) for x in np.bincount(selected[:, :, p].reshape(-1), minlength=128)] for p in range(3)]
        overlap = {}
        for p, q in [(0, 1), (0, 2), (1, 2)]:
            intersection = (selected[:, :, p, :, None] == selected[:, :, q, None, :]).any(-1).sum(-1)
            overlap[f"{p+1}-{q+1}"] = (intersection / (16 - intersection)).mean(-1).tolist()
        result[split] = {"expert_selection_counts_by_pass": counts,
                         "per_article_or_paragraph_aligned_token_jaccard": overlap,
                         "limit": "Routing frequency and overlap do not establish an expert's semantic role."}
    return result


def plots(rows, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["model"], r["evaluation_pass"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(7.5, 4.3), layout="constrained")
    for label, color in COLORS.items():
        means, lower, upper = [], [], []
        for p in range(1, 4):
            if label in ["ridge", "mean"]:
                values = np.array(lookup[("ridge" if label == "ridge" else "specific_mean", p)]["per_paragraph_normalized_mse"])
            else:
                names = [f"pass{p}_seed{s}" if label == "specific" else f"{label}_seed{s}" for s in [0, 1, 2]]
                values = np.mean([lookup[(name, p)]["per_paragraph_normalized_mse"] for name in names], axis=0)
            mean = float(values.mean())
            lo, hi = bootstrap_interval(values)
            means.append(mean); lower.append(max(0, mean-lo)); upper.append(max(0, hi-mean))
        labels = {"shared":"Shared sparse model", "specific":"Separate sparse models", "dense":"Dense model", "ridge":"Ridge regression", "mean":"Pass mean"}
        ax.errorbar([1, 2, 3], means, yerr=[lower, upper], marker="o", color=color, label=labels[label], capsize=3)
    ax.set(xlabel="Loop pass", ylabel="Squared prediction error / target variance",
           xticks=[1, 2, 3], ylim=(0, None), title="One shared operator versus separate pass fits")
    ax.legend()
    fig.savefig(out / "operator_prediction.png", dpi=180)
    plt.close(fig)
    matrix = np.array([[np.mean([lookup[(f"pass{p}_seed{s}", q)]["normalized_mse"] for s in [0, 1, 2]]) for q in [1, 2, 3]] for p in [1, 2, 3]])
    fig, ax = plt.subplots(figsize=(5, 4.3), layout="constrained")
    im = ax.imshow(matrix, cmap="viridis", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] < matrix.max()*0.5 else "black")
    ax.set(xlabel="Evaluated pass", ylabel="Training pass", xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    fig.colorbar(im, ax=ax, label="Normalized squared prediction error")
    fig.savefig(out / "operator_transfer.png", dpi=180)
    plt.close(fig)
    selection = json.loads((out / "feature_selection.json").read_text())["features"]
    profiles = json.loads((out / "feature_profiles.json").read_text())
    contexts = json.loads((out / "feature_examples.json").read_text())
    fig, axes = plt.subplots(max(1, len(selection)), 2, figsize=(10, max(3, 2*len(selection))), squeeze=False, layout="constrained")
    for row, feature in enumerate(selection):
        values = [p["firing_rate"][feature] for p in profiles]
        axes[row, 0].plot([1, 2, 3], values, marker="o", color=COLORS["shared"])
        axes[row, 0].set(xlabel="Loop pass", ylabel="Training-token firing rate", xticks=[1, 2, 3], ylim=(0, None), title=f"Exploratory feature {feature}")
        examples = [c for c in contexts if c["feature"] == feature]
        import textwrap
        text = "\n\n".join(textwrap.fill(f"Pass {c['pass']}: {c['contexts'][0]['context']}", width=65) for c in examples)
        axes[row, 1].text(0, 0.95, text, va="top", fontsize=8)
        axes[row, 1].axis("off")
    if not selection:
        axes[0, 0].text(0.1, 0.5, "No active validation features")
        axes[0, 1].axis("off")
    fig.savefig(out / "operator_feature_firing.png", dpi=180)
    plt.close(fig)
    replacements = json.loads((out / "native_replacements.json").read_text())
    names = ["shared", "pass_specific", "pass_mean"]
    values = [[r["mean_kl_positions16_63"] for r in replacements if r["condition"] == name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.bar(["Shared surrogate", "Separate surrogates", "Mean only"], [np.mean(v) for v in values], color=[COLORS["shared"], COLORS["specific"], COLORS["mean"]])
    for i, group in enumerate(values):
        ax.scatter(np.full(len(group), i), group, color="black", s=12, alpha=0.6)
    ax.set(ylabel="Mean output KL divergence (nats)", ylim=(0, None), title="Native sublayer replacement: eight development paragraphs")
    fig.savefig(out / "operator_native_replacement.png", dpi=180)
    plt.close(fig)
    # Keep a complete raw input/output example beside the figure's aggregate data.
    (out / "plot_raw_examples.json").write_text(json.dumps([next(r for r in replacements if r["condition"] == name) for name in names], indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.run / "final_metrics.json").read_text())
    summary = summarize(rows)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run / "routing_summary.json").write_text(json.dumps(routing_summary(args.run), indent=2) + "\n")
    plots(rows, summary, args.run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
