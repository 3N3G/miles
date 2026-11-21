"""Descriptive figures for the actual trainer-consumed groups, without learning claims."""

import argparse
import json
import statistics
from pathlib import Path


def analyze(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((directory / "runtime-acceptance.json").read_text())
    groups = result.get("consumed_groups", [])
    config_path = directory / "approved-config.json"
    config = (
        json.loads(config_path.read_text())
        if config_path.exists()
        else {"scope": {"optimizer_updates": 2, "prompt_count": 1}}
    )
    expected = config["scope"]["optimizer_updates"]
    trained_ids = {row[0] for row in result.get("training", {}).get("valid_updates", [])}
    trajectories = {}
    for group in groups:
        group["optimizer_update_observed"] = any(i in trained_ids for i in group["rollout_ids"])
        if not group["optimizer_update_observed"]:
            continue
        for key, values in group.get("by_question", {}).items():
            trajectories.setdefault(key, []).append(
                {
                    "rollout_id": group["rollout_ids"][0],
                    "mean_reward": statistics.mean(values["scores"]),
                    "samples": len(values["scores"]),
                    "cap_rate": values["capped"] / len(values["scores"]),
                    "mean_tokens": statistics.mean(values["lengths"]),
                    "decode_tokens_by_weight_version": group["decode_tokens_by_weight_version"],
                }
            )
    (directory / "question-trajectories.json").write_text(json.dumps(trajectories, indent=2) + "\n")
    paragraphs = [
        "# Initial easy-only training attempt\n" if expected > 2 else "# Asynchronous RL runtime check\n",
        f"The approved limit is {expected} optimizer updates, with {config['scope']['prompt_count']} planned training questions.",
        "The setup uses Qwen3.5-4B with thinking enabled and a 90,000-token output limit.",
        "CISPO trains on groups of 32 answers. Two MI350 GPUs train while six generate answers.",
        "GPT-6 Luna at medium effort grades final answers against the supplied seven-point rubric in both anonymous orders.\n",
        f"Automated runtime acceptance passed: **{result['runtime_acceptance_passed']}**.",
        f"The saved evidence contains {len(result.get('training', {}).get('valid_updates', []))} valid optimizer-step records",
        f"and {len(groups)} exported trainer-consumed groups. See [all checks](runtime-acceptance.json).\n",
        f"Observed updates cover {len(trajectories)} distinct question identities. Prepared inputs alone do not imply training coverage.",
        "[Per-question visits and model versions](question-trajectories.json) distinguish early and later training observations.\n",
    ]
    observed = [(group["rollout_ids"][0], group) for group in groups if group["optimizer_update_observed"]]
    if observed:
        observed.sort(key=lambda x: x[0])
        for label, filename, extract in [
            (
                "Mean provisional reward",
                "reward-by-update.png",
                lambda g: sum(float(k) * v for k, v in g["reward_histogram"].items()) / g["samples"],
            ),
            (
                "Recorded cap fraction",
                "caps-by-update.png",
                lambda g: sum(v["capped"] for v in g["by_question"].values()) / g["samples"],
            ),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([i for i, _ in observed], [extract(g) for _, g in observed], marker="o", color="#0072B2")
            ax.set(
                xlabel="Optimizer rollout ID", ylabel=label, title="Observed training groups; the question mix varies"
            )
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            fig.savefig(directory / filename, dpi=180)
            plt.close(fig)
        repeated = [
            sorted(visits, key=lambda x: x["rollout_id"]) for visits in trajectories.values() if len(visits) > 1
        ]
        if repeated:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(
                [v[0]["mean_reward"] for v in repeated], [v[-1]["mean_reward"] for v in repeated], color="#0072B2"
            )
            ax.plot([0, 7], [0, 7], linestyle="--", color="#777777")
            ax.set(
                xlim=(0, 7),
                ylim=(0, 7),
                xlabel="First observed visit: mean points",
                ylabel="Latest observed visit: mean points",
                title="Same-question training visits",
            )
            fig.tight_layout()
            fig.savefig(directory / "first-vs-latest.png", dpi=180)
            plt.close(fig)
        for metric, filename in [("pg_loss", "policy-loss.png"), ("grad_norm", "gradient-norm.png")]:
            pairs = []
            for step, record in result.get("training", {}).get("metrics", {}).items():
                values = [value for key, value in record.get("values", {}).items() if key.endswith(metric)]
                if values:
                    pairs.append((int(step), values[0]))
            if pairs:
                pairs.sort()
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot([x for x, _ in pairs], [y for _, y in pairs], color="#0072B2")
                ax.set(xlabel="Optimizer step", ylabel=metric)
                fig.tight_layout()
                fig.savefig(directory / filename, dpi=180)
                plt.close(fig)
        paragraphs.extend(
            [
                "[Rewards](reward-by-update.png), [cap rates](caps-by-update.png), and same-question trajectories describe the attempted fit.\n"
            ]
        )
    for index, group in enumerate(groups):
        scores = sorted((float(score), count) for score, count in group["reward_histogram"].items())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([x[0] for x in scores], [x[1] for x in scores], width=0.4, color="#0072B2")
        ax.set(
            xlabel="Provisional rubric points (0–7)",
            ylabel="Responses",
            xlim=(-0.5, 7.5),
            title=f"Consumed group {index}: observed reward distribution",
        )
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(directory / f"consumed-{index}-rewards.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(group["response_lengths"], bins=8, color="#0072B2")
        ax.axvline(90000, color="#D55E00", linestyle="--", label="Output limit")
        ax.set(xlabel="Generated tokens", ylabel="Responses", title=f"Consumed group {index}: generation cost")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"consumed-{index}-lengths.png", dpi=180)
        plt.close(fig)
        if len(groups) <= 2 or index == 0:
            paragraphs.extend(
                [
                    f"Group {index} consumed {group['samples']} answers. Score counts were {dict(scores)}.",
                    f"[Reward plot](consumed-{index}-rewards.png), [token-count plot](consumed-{index}-lengths.png),",
                    f"and [seed-selected raw answer with its reward]({group['example_file']}).\n",
                ]
            )
    paragraphs.extend(
        [
            "This is one training seed. It does not establish generalization or an RL-only causal effect.",
            "Early and later visits occur during optimization, with asynchronous model versions. They are not frozen-policy before/after evaluations.",
            "The plots describe individual responses, with no error bars or significance test.",
            "Human judge calibration and an independent-provider audit remain pending. Grade differences can include judge error.",
            "Inspect actual per-token weight versions before claiming generation used updated weights.",
            "The next step is to review failed checks or inspect the completed runtime evidence before selecting a corpus training plan.",
        ]
    )
    (directory / "REPORT.md").write_text("\n".join(paragraphs) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)


if __name__ == "__main__":
    main()
