"""Static plots of the declared GSM8K outcomes and training diagnostics."""

import argparse
import json
from pathlib import Path

from experiments.tcs_async_rl.native_metrics import read_native_metrics


def plot(directory):
    # Matplotlib is an optional VM reporting dependency, never a GPU dependency.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = json.loads((directory / "control-complete.json").read_text())
    updates = complete["optimizer_updates"]
    config = json.loads((directory / "approved-config.json").read_text())
    arm = config["arm"]
    color = "#0072B2" if arm == "reference" else "#D55E00"
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
    outputs = []

    def save(name, x, y, label, *, probability=False, validation=False):
        fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
        axis.plot(x, y, color=color, marker="o" if len(x) < 20 else None, linewidth=1.4)
        axis.set(xlabel="Completed optimizer updates" if validation else "Optimizer update (one-based)", ylabel=label, title=arm.capitalize())
        axis.grid(alpha=0.2)
        if probability:
            axis.set_ylim(0, 1)
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(str(path.relative_to(directory)))

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
    (folder / "captions.json").write_text(
        json.dumps(
            {
                "plots": outputs,
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
    print(json.dumps(plot(parser.parse_args().directory)))
