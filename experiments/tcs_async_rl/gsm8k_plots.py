"""Static plots of the declared GSM8K outcomes and training diagnostics."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.tcs_async_rl.native_metrics import read_native_metrics

ARM_NAMES = {"reference": "Reference", "cispo": "CISPO"}
ARM_COLORS = {"reference": "#0072B2", "cispo": "#D55E00"}


def accuracy_change_plot(directories, destination):
    """Plot stored final-minus-baseline estimates without selecting a checkpoint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records, signatures, question_sets = {}, [], []
    for directory in directories:
        config = json.loads((directory / "approved-config.json").read_text())
        complete = json.loads((directory / "control-complete.json").read_text())
        result_file = directory / "control-results.json"
        result = json.loads(result_file.read_text())
        arm = config["arm"]
        if complete["uninformative"] or arm not in ARM_NAMES or arm in records:
            raise ValueError("Final accuracy plots require distinct completed, informative control arms")
        if (
            result["optimizer_updates"] != complete["optimizer_updates"]
            or result["final"]["optimizer_updates"] != complete["optimizer_updates"]
        ):
            raise ValueError("The stored result does not describe the final optimizer boundary")
        before = json.loads((directory / "evaluations/baseline.json").read_text())
        after_file = directory / f"evaluations/updates-{complete['optimizer_updates']:06d}.json"
        after = json.loads(after_file.read_text())
        if before["sha256"] != result["baseline"]["sha256"] or after["sha256"] != result["final"]["sha256"]:
            raise ValueError("The stored result and evaluation refer to different responses")
        if (
            before["scores"].keys() != after["scores"].keys()
            or len(before["scores"]) != result["paired"]["questions"]
        ):
            raise ValueError("Final accuracy plots require matching validation questions")
        signatures.append({key: config[key] for key in ("model", "data", "partitions")})
        question_sets.append(set(before["scores"]))
        records[arm] = {
            "run": directory.name,
            "optimizer_updates": result["optimizer_updates"],
            "paired": result["paired"],
            "result_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
        }
    if (
        not records
        or any(value != signatures[0] for value in signatures)
        or any(value != question_sets[0] for value in question_sets)
    ):
        raise ValueError("Control comparison requires the same model, data and validation membership")
    ordered = [arm for arm in ARM_NAMES if arm in records]
    fig, axis = plt.subplots(figsize=(8, 3 + 0.5 * len(ordered)), layout="constrained")
    extent = 0
    for index, arm in enumerate(ordered):
        paired = records[arm]["paired"]
        change = 100 * paired["change"]
        low, high = [100 * value for value in paired["paired_problem_bootstrap_95_interval"]]
        if not all(math.isfinite(value) for value in (change, low, high)) or low > high:
            raise ValueError("Invalid paired accuracy interval")
        color = ARM_COLORS[arm]
        axis.hlines(index, low, high, color=color, linewidth=2)
        axis.vlines([low, high], index - .06, index + .06, color=color)
        axis.plot(change, index, "o", color=color)
        axis.annotate(f"{change:+.2f} points [{low:.2f}, {high:.2f}]", (change, index),
                      xytext=(0, 14), textcoords="offset points", ha="center", fontsize=11)
        extent = max(extent, abs(change), abs(low), abs(high))
    extent = max(10, 5 * math.ceil(1.25 * extent / 5))
    axis.axvline(0, color="0.4", linewidth=1, linestyle="--", zorder=0)
    fig.suptitle("Final accuracy change", fontsize=14)
    axis.set(xlim=(-extent, extent), ylim=(-.5, len(ordered) - .4),
             xlabel="Final − baseline accuracy (percentage points)")
    axis.set_title("95% intervals from paired validation questions\nOne training seed", fontsize=10)
    axis.set_yticks(range(len(ordered)),
                    [f"{ARM_NAMES[arm]}\n{records[arm]['optimizer_updates']} updates" for arm in ordered])
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    caption = " ".join(
        f"{ARM_NAMES[arm]} changed by {100 * records[arm]['paired']['change']:+.2f} percentage points "
        f"(95% interval {100 * records[arm]['paired']['paired_problem_bootstrap_95_interval'][0]:.2f} "
        f"to {100 * records[arm]['paired']['paired_problem_bootstrap_95_interval'][1]:.2f})."
        for arm in ordered
    ) + " Intervals resample paired validation questions. One training seed. Estimates use final checkpoints."
    destination.with_suffix(".json").write_text(json.dumps({"caption": caption, "records": records}, indent=2) + "\n")
    return caption


def plot(directory, *, compare_with=None):
    # Matplotlib is an optional VM reporting dependency, never a GPU dependency.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = json.loads((directory / "control-complete.json").read_text())
    updates = complete["optimizer_updates"]
    config = json.loads((directory / "approved-config.json").read_text())
    arm = config["arm"]
    color = ARM_COLORS[arm]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    folder = directory / "figures"
    folder.mkdir(exist_ok=True)
    evaluations = sorted(
        [json.loads(p.read_text()) for p in (directory / "evaluations").glob("*.json")],
        key=lambda item: item["optimizer_updates"],
    )
    summaries = [json.loads(p.read_text()) for p in sorted((directory / "training-summaries").glob("*.json")) if int(p.stem) < updates]
    if [row["optimizer_step"] for row in summaries] != list(range(updates)):
        raise ValueError("Training plots require every actual optimizer batch summary")
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    if len(native) != updates:
        raise ValueError("Training plots and final checkpoint update counts differ")
    outputs, figure_captions = [], {}

    def save(name, x, y, label, *, probability=False, validation=False):
        fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
        axis.plot(x, y, color=color, marker="o" if len(x) < 20 else None, linewidth=1.4)
        axis.set(xlabel="Completed optimizer updates" if validation else "Optimizer update (one-based)", ylabel=label, title=ARM_NAMES[arm])
        axis.grid(alpha=0.2)
        if probability:
            axis.set_ylim(0, 1)
        else:
            axis.set_ylim(bottom=min(0, min(y)))
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path.relative_to(directory)))
        figure_captions[str(path.relative_to(directory))] = (
            f"{label}: {y[0]:.6g} at update {x[0]} and {y[-1]:.6g} at update {x[-1]}. "
            "One training seed; no seed error bars."
        )

    eval_steps = [point["optimizer_updates"] for point in evaluations]
    save("validation-accuracy", eval_steps, [x["accuracy"] for x in evaluations], "Exact boxed-answer validation accuracy", probability=True, validation=True)
    save("validation-box-format", eval_steps, [x["valid_box_fraction"] for x in evaluations], "Validation answers with a parsable box (fraction)", probability=True, validation=True)
    save("validation-capped", eval_steps, [x["capped_fraction"] for x in evaluations], "Validation answers reaching1,024 tokens (fraction)", probability=True, validation=True)
    if summaries:
        save("training-reward", [x["optimizer_step"] + 1 for x in summaries], [x["accuracy"] for x in summaries], "Consumed training reward (mixed groups only)" if arm == "reference" else "Consumed training reward", probability=True)
    diagnostics = [
        ("entropy", "train/full_policy_entropy", "Full policy entropy (nats/token)"),
        ("reference-divergence", "train/kl_loss", "Sampled reference KL proxy (nats/token)"),
        ("rollout-mismatch", "train/train_rollout_kl", "Sampled trainer–rollout KL proxy (nats/token)"),
        ("gradient-norm", "train/grad_norm", "Gradient norm before clipping"),
    ]
    for name, key, label in diagnostics:
        points = [row for row in native if key in row]
        if points:
            save(name, [row["optimizer_step"] + 1 for row in points], [row[key] for row in points], label)
    comparisons = [("final-accuracy-change", [directory])]
    if compare_with is not None:
        comparisons.append(("final-accuracy-change-comparison", [directory, compare_with]))
    for name, directories in comparisons:
        path = folder / f"{name}.png"
        caption = accuracy_change_plot(directories, path)
        outputs.append(str(path.relative_to(directory)))
        figure_captions[str(path.relative_to(directory))] = caption
    (folder / "captions.json").write_text(
        json.dumps(
            {
                "plots": outputs,
                "figure_captions": figure_captions,
                "seed_count": 1,
                "validation_questions": 256,
                "accuracy_uncertainty": "Line plots have no error bars. Final paired question-bootstrap uncertainty is in control-results.json.",
                "training_reward_scope": "The reference filters for mixed-reward groups. Use fixed validation to assess learning.",
                "optimizer_axis": "Training uses native rollout ID plus1, the one-based update index. Validation records completed updates.",
                "training_diagnostic_timing": "Training diagnostics are scored during the indicated update, before its optimizer step. They are not after-update measurements.",
            },
            indent=2,
        )
        + "\n"
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.directory, compare_with=args.compare_with)))
